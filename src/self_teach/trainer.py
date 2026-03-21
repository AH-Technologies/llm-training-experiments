"""Self-Teach tree-structured trainer and task runner.

Implements a tree-structured generation strategy:

  Q → A₁ → (F₁ ... Fₖ) → each Fᵢ → (A₂ᵢ₁ ... A₂ᵢₘ)

Step 1: Generate A₁ once per prompt.
Step 2: Branch k teacher feedbacks per prompt (all see same A₁).
Step 3: Branch m student A₂ responses per feedback (all see same A₁ + Fᵢ).

The m A₂ responses serve dual purpose:
  - Grade teacher quality: mean improvement rate across m A₂s → teacher reward.
  - Train student₂: GRPO groups of m sharing the same (Q, A₁, Fᵢ) context.

Every generation contributes to training — no wasted compute.

Based on verl v0.2+ RayPPOTrainer with AgentLoop-based generation.
"""

import json
import os
import re
import socket
import uuid
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from src.rlvr_grokking.rewards.deepscaler_reward import compute_score as grade_solution
from .prompts import (
    TEACHER_PROMPT_TEMPLATE,
    TEACHER_PROMPT_TEMPLATE_FILTERED,
    BLIND_TEACHER_PROMPT_TEMPLATE,
    BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED,
    STUDENT2_PROMPT_TEMPLATE,
)
from .rewards import compute_self_teach_rewards


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def extract_feedback(text: str) -> str:
    """Extract content inside <feedback>...</feedback> tags.

    Returns just the feedback portion so the student only sees the
    teacher's final guidance, not the internal reasoning. Falls back
    to the full text if tags are missing.
    """
    match = re.search(r"<feedback>(.*?)</feedback>", text, re.DOTALL)
    return match.group(1).strip() if match else text


@dataclass
class SelfTeachConfig:
    """Configuration for self-teach training."""

    enabled: bool = False
    blind_teacher: bool = False  # Teacher gets no ground truth (self-improvement variant)
    filter_a1_correct: bool | None = None  # Skip A₁-correct prompts; None = auto (True if blind_teacher)
    num_feedbacks: int = 6  # k: number of teacher feedbacks per prompt
    num_a2_per_feedback: int = 6  # m: number of A₂ responses per feedback
    max_a2_prompt_length: int | None = None  # Prompt length for A₂ generation; None = use default


def _patch_agent_loop_prompt_length():
    """Monkey-patch VERL's AgentLoopWorkerBase to support per-batch prompt length.

    When 'max_prompt_length' is present in non_tensor_batch, the agent loop
    will pad prompts to that size instead of the global config value. This
    allows A₂ generation to use a larger prompt budget than A₁/F without
    changing the global setting.
    """
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase

    if getattr(AgentLoopWorkerBase, "_prompt_length_patched", False):
        return  # already patched

    _original_postprocess = AgentLoopWorkerBase._agent_loop_postprocess

    async def _patched_postprocess(self, output, **kwargs):
        # Temporarily swap the config value if an override is provided
        override = kwargs.pop("max_prompt_length", None)
        if override is not None:
            original_value = self.config.actor_rollout_ref.rollout.prompt_length
            self.config.actor_rollout_ref.rollout.prompt_length = int(override)
        try:
            return await _original_postprocess(self, output, **kwargs)
        finally:
            if override is not None:
                self.config.actor_rollout_ref.rollout.prompt_length = original_value

    AgentLoopWorkerBase._agent_loop_postprocess = _patched_postprocess
    AgentLoopWorkerBase._prompt_length_patched = True
    print("[SelfTeach] Patched AgentLoopWorkerBase for per-batch prompt length override")


class SelfTeachRayPPOTrainer(RayPPOTrainer):
    """Tree-structured GRPO trainer for student-teacher self-play.

    Generates a tree per prompt: Q → A₁ → k feedbacks → m A₂s each.
    The m A₂s per feedback serve dual purpose: grading teacher quality
    (mean improvement rate) and training student₂ (GRPO groups of m).
    """

    def __init__(self, *args, self_teach_config: Optional[SelfTeachConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_teach_config = self_teach_config or SelfTeachConfig()
        # Resolve auto default for filter_a1_correct
        if self.self_teach_config.filter_a1_correct is None:
            self.self_teach_config.filter_a1_correct = self.self_teach_config.blind_teacher
        # A₂ prompt length — use dedicated config if set, otherwise fall back
        # to the global max_prompt_length.
        self.max_a2_prompt_tokens = (
            self.self_teach_config.max_a2_prompt_length
            or self.config.data.max_prompt_length
        )
        if self.self_teach_config.enabled:
            if self.config.reward_model.get("launch_reward_fn_async", False):
                raise ValueError(
                    "Self-teach is incompatible with launch_reward_fn_async=True."
                )
            _patch_agent_loop_prompt_length()
            mode = "blind" if self.self_teach_config.blind_teacher else "conditioned"
            filter_str = "yes" if self.self_teach_config.filter_a1_correct else "no"
            a2_len = self.self_teach_config.max_a2_prompt_length or "default"
            k = self.self_teach_config.num_feedbacks
            m = self.self_teach_config.num_a2_per_feedback
            assert k == m, (
                f"num_feedbacks ({k}) must equal num_a2_per_feedback ({m}) "
                f"for unified GRPO grouping"
            )
            print(
                f"[SelfTeach Tree] Initialized. teacher={mode}, "
                f"filter_a1_correct={filter_str}, max_a2_prompt_length={a2_len}, "
                f"k={k}, m={m}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_for_a2(self, a1_text: str, f_text: str, question: str) -> tuple[str, str]:
        """Truncate A1 and F texts so the A2 prompt fits within context.

        Fixed parts: question plus chat template overhead.
        The remaining budget is split evenly between A1 and F.
        """
        fixed_tokens = len(self.tokenizer.encode(
            question,
            add_special_tokens=False,
        )) + 50  # chat template overhead

        available = self.max_a2_prompt_tokens - fixed_tokens
        per_turn = available // 2

        a1_ids = self.tokenizer.encode(a1_text, add_special_tokens=False)
        f_ids = self.tokenizer.encode(f_text, add_special_tokens=False)

        if len(a1_ids) + len(f_ids) <= available:
            return a1_text, f_text

        # Truncate whichever is over budget, giving slack to the other
        if len(a1_ids) <= per_turn:
            f_ids = f_ids[:available - len(a1_ids)]
        elif len(f_ids) <= per_turn:
            a1_ids = a1_ids[:available - len(f_ids)]
        else:
            a1_ids = a1_ids[:per_turn]
            f_ids = f_ids[:per_turn]

        a1_out = self.tokenizer.decode(a1_ids, skip_special_tokens=True)
        f_out = self.tokenizer.decode(f_ids, skip_special_tokens=True)
        return a1_out, f_out

    def _build_gen_batch(
        self,
        raw_prompts: list,
        temperature: float,
        global_steps: int,
        data_sources: list[str] | None = None,
        ground_truths: list[str] | None = None,
        max_prompt_length: int | None = None,
    ) -> DataProto:
        """Build a DataProto for generation from raw prompt message lists.

        The agent loop reads raw_prompt from non_tensor_batch and handles
        tokenization via apply_chat_template internally.

        Args:
            raw_prompts: List of conversation message lists, e.g.
                [[{"role": "system", ...}, {"role": "user", ...}], ...]
            temperature: Sampling temperature.
            global_steps: Current training step (for tracing).
            data_sources: Per-entry data source labels (required by reward loop).
            ground_truths: Per-entry ground truth answers (required by reward loop).
            max_prompt_length: Override prompt padding length for this batch.
                When set, the agent loop pads prompts to this size instead of
                the global config value. Useful for A₂ batches that need more
                room for multi-turn context.
        """
        n = len(raw_prompts)
        non_tensors = {
            "raw_prompt": np.array(raw_prompts, dtype=object),
        }
        if data_sources is not None:
            non_tensors["data_source"] = np.array(data_sources, dtype=object)
        if ground_truths is not None:
            non_tensors["reward_model"] = np.array(
                [{"ground_truth": gt} for gt in ground_truths], dtype=object
            )
        if max_prompt_length is not None:
            non_tensors["max_prompt_length"] = np.array(
                [max_prompt_length] * n, dtype=np.int64
            )
        return DataProto.from_dict(
            # Dummy tensor so DataProto.batch is not None (required for chunk/len)
            tensors={"dummy_tensor": torch.zeros(n, dtype=torch.uint8)},
            non_tensors=non_tensors,
            meta_info={
                "temperature": temperature,
                "global_steps": global_steps,
            },
        )

    def _decode_responses(self, gen_output: DataProto) -> list[str]:
        """Decode response texts from generation output.

        Args:
            gen_output: DataProto from generate_sequences() containing
                'responses' and 'attention_mask' tensors.

        Returns:
            List of decoded response strings.
        """
        responses = gen_output.batch["responses"]  # [bs, response_length]
        prompt_length = gen_output.batch["prompts"].shape[1]
        attention_mask = gen_output.batch["attention_mask"]
        resp_attn = attention_mask[:, prompt_length:]  # response portion

        texts = []
        for i in range(len(responses)):
            real_ids = responses[i][resp_attn[i].bool()]
            text = self.tokenizer.decode(real_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    def _extract_real_tokens(self, gen_output: DataProto, idx: int):
        """Extract non-padded prompt and response token IDs for one entry.

        Returns:
            (prompt_ids, response_ids) as 1D tensors with no padding.
        """
        prompt_length = gen_output.batch["prompts"].shape[1]
        prompt_mask = gen_output.batch["attention_mask"][idx, :prompt_length]
        prompt_ids = gen_output.batch["prompts"][idx][prompt_mask.bool()]

        resp_mask = gen_output.batch["attention_mask"][idx, prompt_length:]
        response_ids = gen_output.batch["responses"][idx][resp_mask.bool()]

        return prompt_ids, response_ids

    def _build_combined_batch(
        self,
        entries: list[dict],
        pad_token_id: int,
    ) -> DataProto:
        """Construct a DataProto from sub-rollout entries.

        Each entry has: prompt_ids, response_ids, reward, uid.
        Returns a DataProto matching VERL's expected layout:
          - prompts:       [n, max_prompt_len]   LEFT-padded
          - responses:     [n, max_response_len] RIGHT-padded
          - input_ids:     [n, max_prompt_len + max_response_len]
          - response_mask: [n, max_response_len] aligned to responses
        """
        n = len(entries)

        max_prompt_len = max(len(e["prompt_ids"]) for e in entries)
        max_response_len = max(len(e["response_ids"]) for e in entries)
        max_seq_len = max_prompt_len + max_response_len

        input_ids = torch.full((n, max_seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(n, max_seq_len, dtype=torch.long)
        position_ids = torch.zeros(n, max_seq_len, dtype=torch.long)
        prompts = torch.full((n, max_prompt_len), pad_token_id, dtype=torch.long)
        responses = torch.full((n, max_response_len), pad_token_id, dtype=torch.long)
        response_mask = torch.zeros(n, max_response_len, dtype=torch.long)
        token_level_scores = torch.zeros(n, max_response_len, dtype=torch.float32)
        uids = np.empty(n, dtype=object)

        for i, entry in enumerate(entries):
            p_ids = entry["prompt_ids"]
            r_ids = entry["response_ids"]
            p_len = len(p_ids)
            r_len = len(r_ids)
            total_real = p_len + r_len
            prompt_pad = max_prompt_len - p_len

            prompts[i, prompt_pad:] = p_ids
            input_ids[i, prompt_pad:max_prompt_len] = p_ids
            input_ids[i, max_prompt_len : max_prompt_len + r_len] = r_ids
            attention_mask[i, prompt_pad : max_prompt_len + r_len] = 1
            position_ids[i, prompt_pad : max_prompt_len + r_len] = torch.arange(total_real)
            responses[i, :r_len] = r_ids
            response_mask[i, :r_len] = 1

            if r_len > 0:
                token_level_scores[i, r_len - 1] = entry["reward"]

            uids[i] = entry["uid"]

        combined = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "prompts": prompts,
            },
            non_tensors={"uid": uids},
        )
        combined.meta_info["self_teach_token_level_scores"] = token_level_scores
        return combined

    def _log_self_teach_metrics(self, metrics_dict: dict, total: int):
        """Log self-teach metrics to console and wandb."""
        # a1 accuracy uses its own total (one grade per prompt, not per rollout)
        a1_total = metrics_dict.get("a1_total", total)
        a1_acc = metrics_dict["a1_correct_count"] / max(a1_total, 1)
        a2_acc = metrics_dict["a2_correct_count"] / max(total, 1)
        imp_rate = metrics_dict["improvement_count"] / max(total, 1)
        reg_rate = metrics_dict["regression_count"] / max(total, 1)
        t_mean_reward = metrics_dict.get("teacher_mean_reward", 0.0)
        s2_mean_reward = metrics_dict.get("student2_mean_reward", 0.0)
        t_leak_count = metrics_dict.get("teacher_leak_count", 0)
        t_leak_rate = t_leak_count / max(total, 1)

        print(
            f"[SelfTeach step {self.global_steps}] "
            f"a1_acc={a1_acc:.3f}, a2_acc={a2_acc:.3f}, "
            f"improve={imp_rate:.3f}, regress={reg_rate:.3f}, "
            f"teacher_reward={t_mean_reward:.3f}, student2_reward={s2_mean_reward:.3f}, "
            f"teacher_leak_rate={t_leak_rate:.3f}"
        )

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "self_teach/a1_accuracy": a1_acc,
                        "self_teach/a2_accuracy": a2_acc,
                        "self_teach/improvement_rate": imp_rate,
                        "self_teach/regression_rate": reg_rate,
                        "self_teach/teacher_leak_rate": t_leak_rate,
                        "self_teach/teacher_mean_reward": t_mean_reward,
                        "self_teach/student2_mean_reward": s2_mean_reward,
                    },
                    step=self.global_steps,
                )
        except ImportError:
            pass

    def _log_sample_table(
        self,
        question_texts: list[str],
        a1_texts: list[str],
        a1_correct: list[bool],
        f_texts: list[str],
        a2_texts: list[str],
        a2_correct: list[bool],
        student_rewards: list[float],
        teacher_rewards: list[float],
        ground_truths: list[str],
        n: int,
        num_samples: int = 3,
    ):
        """Log a W&B table with sample prompts/responses for inspection.

        question_texts, a1_texts, a1_correct, f_texts, ground_truths are per-prompt (length bs).
        a2_texts, a2_correct, student_rewards, teacher_rewards are per-rollout (length bs*n).
        """
        try:
            import wandb

            if wandb.run is None:
                return
        except ImportError:
            return

        table = wandb.Table(
            columns=[
                "teacher_prompt", "a2_prompt",
                "question", "ground_truth", "a1", "a1_correct",
                "feedback", "a2", "a2_correct",
                "student_reward", "teacher_reward",
            ],
        )
        # Log up to num_samples prompts, using the first rollout (j=0) for a2
        filter_a1 = self.self_teach_config.filter_a1_correct
        if self.self_teach_config.blind_teacher:
            log_tpl = BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else BLIND_TEACHER_PROMPT_TEMPLATE
        else:
            log_tpl = TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else TEACHER_PROMPT_TEMPLATE
        for i in range(min(num_samples, len(question_texts))):
            if self.self_teach_config.blind_teacher:
                teacher_prompt = log_tpl.format(
                    question=question_texts[i],
                    student_attempt=a1_texts[i],
                )
            else:
                teacher_prompt = log_tpl.format(
                    question=question_texts[i],
                    student_attempt=a1_texts[i],
                    ground_truth=ground_truths[i],
                )
            a2_prompt = STUDENT2_PROMPT_TEMPLATE.format(
                question=question_texts[i],
                first_attempt=a1_texts[i],
                feedback=f_texts[i],
            )
            table.add_data(
                teacher_prompt,
                a2_prompt,
                question_texts[i],
                ground_truths[i],
                a1_texts[i],
                a1_correct[i],
                f_texts[i],
                a2_texts[i * n],
                a2_correct[i * n],
                student_rewards[i * n],
                teacher_rewards[i * n],
            )
        wandb.log({"self_teach/samples": table}, step=self.global_steps)

    def _log_teacher_jsonl(
        self,
        question_texts: list[str],
        a1_texts: list[str],
        a1_correct: list[bool],
        ground_truths: list[str],
        f_texts: list[str],
        teacher_rewards: list[float],
        a2_correct: list[bool],
        k: int,
        m: int,
        step: int,
    ):
        """Append one JSONL record per teacher feedback for reward-hacking analysis.

        f_texts has n_prompts * k entries. teacher_rewards has n_prompts * k entries.
        a2_correct has n_prompts * k * m entries (m A₂ outcomes per feedback).
        """
        if not hasattr(self, "_teacher_log_path"):
            from datetime import datetime
            log_dir = os.path.join(
                self.config.trainer.default_local_dir, "teacher_logs"
            )
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._teacher_log_path = os.path.join(log_dir, f"teacher_log_{ts}.jsonl")
        log_path = self._teacher_log_path

        blind = self.self_teach_config.blind_teacher
        filter_a1 = self.self_teach_config.filter_a1_correct
        if blind:
            log_tpl = BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else BLIND_TEACHER_PROMPT_TEMPLATE
        else:
            log_tpl = TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else TEACHER_PROMPT_TEMPLATE
        with open(log_path, "a") as f:
            for f_idx in range(len(f_texts)):
                i = f_idx // k  # prompt index
                if blind:
                    teacher_prompt = log_tpl.format(
                        question=question_texts[i],
                        student_attempt=a1_texts[i],
                    )
                else:
                    teacher_prompt = log_tpl.format(
                        question=question_texts[i],
                        student_attempt=a1_texts[i],
                        ground_truth=ground_truths[i],
                    )
                # Collect per-A₂ correctness for this feedback
                a2_start = f_idx * m
                a2_outcomes = [a2_correct[a2_start + off] for off in range(m)]
                record = {
                    "step": step,
                    "prompt_idx": i,
                    "feedback_idx": f_idx % k,
                    "blind_teacher": blind,
                    "question": question_texts[i],
                    "ground_truth": ground_truths[i],
                    "a1_text": a1_texts[i],
                    "a1_correct": a1_correct[i],
                    "teacher_prompt": teacher_prompt,
                    "teacher_response": f_texts[f_idx],
                    "teacher_reward": teacher_rewards[f_idx],
                    "a2_correct_count": sum(a2_outcomes),
                    "a2_total": m,
                }
                f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # fit() override — tree-structured generation flow
    # ------------------------------------------------------------------

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        n = self.config.actor_rollout_ref.rollout.n  # GRPO group size (e.g. 8)
        k = self.self_teach_config.num_feedbacks  # teacher feedbacks per prompt
        m = self.self_teach_config.num_a2_per_feedback  # A₂ responses per feedback
        temperature = self.config.actor_rollout_ref.rollout.temperature
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = temperature
                bs = len(batch)

                # Assign UIDs (one per original prompt)
                base_uids = [str(uuid.uuid4()) for _ in range(bs)]
                batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)

                # Extract question texts and system prompts from raw_prompt
                question_texts = []
                system_prompts = []
                for i in range(bs):
                    raw_prompt = batch.non_tensor_batch["raw_prompt"][i]
                    q = ""
                    sp = ""
                    for msg in raw_prompt:
                        if msg.get("role") == "system":
                            sp = msg.get("content", "")
                        elif msg.get("role") == "user":
                            q = msg.get("content", "")
                            break
                    question_texts.append(q)
                    system_prompts.append(sp)

                # Extract ground truths and data sources for grading
                ground_truths = []
                data_sources = []
                for i in range(bs):
                    gt = ""
                    ds = "math"
                    if "reward_model" in batch.non_tensor_batch:
                        rm = batch.non_tensor_batch["reward_model"][i]
                        if isinstance(rm, dict):
                            gt = rm.get("ground_truth", "")
                    if "data_source" in batch.non_tensor_batch:
                        ds = batch.non_tensor_batch["data_source"][i]
                    ground_truths.append(gt)
                    data_sources.append(ds)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ==========================================================
                    # Step 1: Generate A₁ — one per prompt (bs entries)
                    # ==========================================================
                    with marked_timer("gen_a1", timing_raw, color="red"):
                        a1_raw_prompts = list(batch.non_tensor_batch["raw_prompt"])
                        a1_gen_batch = self._build_gen_batch(
                            a1_raw_prompts, temperature, self.global_steps,
                            data_sources=data_sources, ground_truths=ground_truths,
                        )
                        a1_output = self.async_rollout_manager.generate_sequences(a1_gen_batch)
                        timing_raw.update(a1_output.meta_info.get("timing", {}))
                        a1_output.meta_info.pop("timing", None)

                    a1_texts = self._decode_responses(a1_output)

                    # Grade A₁
                    a1_correct = []
                    for i in range(bs):
                        score = grade_solution(
                            data_source=data_sources[i],
                            solution_str=strip_think_blocks(a1_texts[i]),
                            ground_truth=ground_truths[i],
                        )
                        a1_correct.append(score >= 0.5)

                    # Strip thinking from A₁ for use in all downstream prompts
                    a1_texts_clean = [strip_think_blocks(t) for t in a1_texts]

                    # ==========================================================
                    # Step 2: Generate F (teacher feedback) — n_teacher * k entries
                    # Each prompt gets k different feedback samples,
                    # all conditioned on the SAME A₁.
                    # When filter_a1_correct=True, skip prompts where A₁ is correct.
                    # ==========================================================
                    with marked_timer("gen_f", timing_raw, color="yellow"):
                        # Determine which prompts participate in teacher training
                        if self.self_teach_config.filter_a1_correct:
                            teacher_prompt_indices = [i for i in range(bs) if not a1_correct[i]]
                        else:
                            teacher_prompt_indices = list(range(bs))

                        # Pick the right template based on blind/filtered config
                        filter_a1 = self.self_teach_config.filter_a1_correct
                        if self.self_teach_config.blind_teacher:
                            tpl = BLIND_TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else BLIND_TEACHER_PROMPT_TEMPLATE
                        else:
                            tpl = TEACHER_PROMPT_TEMPLATE_FILTERED if filter_a1 else TEACHER_PROMPT_TEMPLATE

                        teacher_messages = []
                        for i in teacher_prompt_indices:
                            if self.self_teach_config.blind_teacher:
                                teacher_prompt = tpl.format(
                                    question=question_texts[i],
                                    student_attempt=a1_texts_clean[i],
                                )
                            else:
                                teacher_prompt = tpl.format(
                                    question=question_texts[i],
                                    student_attempt=a1_texts_clean[i],
                                    ground_truth=ground_truths[i],
                                )
                            msgs = [
                                {"role": "user", "content": teacher_prompt},
                            ]
                            teacher_messages.append(msgs)

                        n_teacher_prompts = len(teacher_prompt_indices)

                        # Repeat each prompt k times (k feedbacks per prompt)
                        f_raw_prompts = []
                        f_data_sources = []
                        f_ground_truths = []
                        for j, i in enumerate(teacher_prompt_indices):
                            for _ in range(k):
                                f_raw_prompts.append(teacher_messages[j])
                                f_data_sources.append(data_sources[i])
                                f_ground_truths.append(ground_truths[i])

                        if n_teacher_prompts > 0:
                            f_gen_batch = self._build_gen_batch(
                                f_raw_prompts, temperature, self.global_steps,
                                data_sources=f_data_sources, ground_truths=f_ground_truths,
                            )
                            f_output = self.async_rollout_manager.generate_sequences(f_gen_batch)
                            timing_raw.update(f_output.meta_info.get("timing", {}))
                            f_output.meta_info.pop("timing", None)
                        else:
                            f_output = None

                    if self.self_teach_config.filter_a1_correct:
                        n_filtered = bs - n_teacher_prompts
                        print(
                            f"[SelfTeach step {self.global_steps}] "
                            f"Filtered {n_filtered}/{bs} A₁-correct prompts from teacher training"
                        )

                    f_texts = self._decode_responses(f_output) if f_output is not None else []

                    # ==========================================================
                    # Step 3: Generate A₂ — n_teacher * k * m entries
                    # Each feedback Fᵢ gets m A₂ responses. These serve dual
                    # purpose:
                    #   1. Grade teacher quality (mean improvement rate across m)
                    #   2. Train student₂ (GRPO groups of m, same context)
                    # ==========================================================
                    teacher_rewards = []
                    student2_rewards = []
                    a2_correct = []
                    a2_texts = []
                    teacher_leak_count = 0
                    a2_output = None
                    # Store truncated texts per feedback for logging
                    truncated_a1_per_f = []
                    truncated_f_per_f = []

                    n_total_f = n_teacher_prompts * k  # total feedbacks
                    n_total_a2 = n_total_f * m  # total A₂ responses

                    if n_teacher_prompts > 0:
                        with marked_timer("gen_a2", timing_raw, color="blue"):
                            a2_raw_prompts = []
                            a2_data_sources = []
                            a2_ground_truths = []

                            for f_idx in range(n_total_f):
                                j = f_idx // k  # index into teacher_prompt_indices
                                i = teacher_prompt_indices[j]
                                a1_t, f_t = self._truncate_for_a2(
                                    a1_texts_clean[i], extract_feedback(f_texts[f_idx]), question_texts[i]
                                )
                                truncated_a1_per_f.append(a1_t)
                                truncated_f_per_f.append(f_t)
                                s2_prompt = STUDENT2_PROMPT_TEMPLATE.format(
                                    question=question_texts[i],
                                    first_attempt=a1_t,
                                    feedback=f_t,
                                )
                                base_msgs = [{"role": "system", "content": system_prompts[i]}] if system_prompts[i] else []
                                base_msgs.append({"role": "user", "content": s2_prompt})
                                for _ in range(m):
                                    a2_raw_prompts.append(base_msgs)
                                    a2_data_sources.append(data_sources[i])
                                    a2_ground_truths.append(ground_truths[i])

                            a2_gen_batch = self._build_gen_batch(
                                a2_raw_prompts, temperature, self.global_steps,
                                data_sources=a2_data_sources, ground_truths=a2_ground_truths,
                                max_prompt_length=self.self_teach_config.max_a2_prompt_length,
                            )
                            a2_output = self.async_rollout_manager.generate_sequences(
                                a2_gen_batch
                            )
                            timing_raw.update(a2_output.meta_info.get("timing", {}))
                            a2_output.meta_info.pop("timing", None)

                        a2_texts = self._decode_responses(a2_output)

                        # Grade all A₂ responses
                        for a2_idx in range(n_total_a2):
                            f_idx = a2_idx // m
                            j = f_idx // k
                            i = teacher_prompt_indices[j]
                            score = grade_solution(
                                data_source=data_sources[i],
                                solution_str=strip_think_blocks(a2_texts[a2_idx]),
                                ground_truth=ground_truths[i],
                            )
                            a2_correct.append(score >= 0.5)

                        # Compute teacher rewards: mean improvement rate per F
                        for f_idx in range(n_total_f):
                            j = f_idx // k
                            i = teacher_prompt_indices[j]
                            a2_start = f_idx * m
                            rewards_for_f = []
                            for a2_offset in range(m):
                                t_r, _ = compute_self_teach_rewards(
                                    a1_correct[i], a2_correct[a2_start + a2_offset]
                                )
                                rewards_for_f.append(t_r)
                            teacher_rewards.append(sum(rewards_for_f) / m)

                        # Penalize teacher for leaking answer (reward hacking detection)
                        # Only check inside <feedback> tags — reasoning section may
                        # legitimately reference the ground truth.
                        for f_idx in range(n_total_f):
                            f_visible = strip_think_blocks(f_texts[f_idx])
                            feedback_match = re.search(
                                r"<feedback>(.*?)</feedback>", f_visible, re.DOTALL
                            )
                            feedback_text = feedback_match.group(1) if feedback_match else f_visible
                            if "\\boxed{" in feedback_text:
                                teacher_rewards[f_idx] = -1.0
                                teacher_leak_count += 1

                        # Compute student₂ rewards: per A₂, binary
                        for a2_idx in range(n_total_a2):
                            f_idx = a2_idx // m
                            j = f_idx // k
                            i = teacher_prompt_indices[j]
                            _, s2_reward = compute_self_teach_rewards(
                                a1_correct[i], a2_correct[a2_idx]
                            )
                            student2_rewards.append(s2_reward)

                        # Log teacher data for reward-hacking analysis
                        self._log_teacher_jsonl(
                            question_texts=[question_texts[i] for i in teacher_prompt_indices],
                            a1_texts=[a1_texts_clean[i] for i in teacher_prompt_indices],
                            a1_correct=[a1_correct[i] for i in teacher_prompt_indices],
                            ground_truths=[ground_truths[i] for i in teacher_prompt_indices],
                            f_texts=f_texts,
                            teacher_rewards=teacher_rewards,
                            a2_correct=a2_correct,
                            k=k,
                            m=m,
                            step=self.global_steps,
                        )

                    # ==========================================================
                    # Build combined batch (teacher + student₂ sub-rollouts)
                    # Layout: teacher entries in groups of k (per prompt),
                    #         student₂ entries in groups of m (per feedback).
                    # Since k == m, compute_advantage uses a single num_repeat.
                    # ==========================================================
                    with marked_timer("self_teach_build_batch", timing_raw, color="cyan"):
                        # Metrics tracking
                        a2_correct_count = sum(1 for c in a2_correct if c)
                        improvement_count = sum(
                            1 for a2_idx in range(n_total_a2)
                            if not a1_correct[teacher_prompt_indices[a2_idx // m // k]] and a2_correct[a2_idx]
                        )
                        regression_count = sum(
                            1 for a2_idx in range(n_total_a2)
                            if a1_correct[teacher_prompt_indices[a2_idx // m // k]] and not a2_correct[a2_idx]
                        )

                        self_teach_metrics = {
                            "a1_correct_count": sum(1 for c in a1_correct if c),
                            "a1_total": bs,
                            "a2_correct_count": a2_correct_count,
                            "improvement_count": improvement_count,
                            "regression_count": regression_count,
                            "teacher_leak_count": teacher_leak_count,
                            "teacher_mean_reward": sum(teacher_rewards) / max(len(teacher_rewards), 1),
                            "student2_mean_reward": sum(student2_rewards) / max(len(student2_rewards), 1),
                        }
                        self._log_self_teach_metrics(self_teach_metrics, max(n_total_a2, 1))

                        test_freq = self.config.trainer.test_freq
                        if n_teacher_prompts > 0 and test_freq > 0 and (
                            is_last_step or self.global_steps % test_freq == 0
                        ):
                            # Log first feedback per prompt for the sample table
                            self._log_sample_table(
                                question_texts=[question_texts[i] for i in teacher_prompt_indices],
                                a1_texts=[truncated_a1_per_f[j * k] for j in range(n_teacher_prompts)],
                                a1_correct=[a1_correct[i] for i in teacher_prompt_indices],
                                f_texts=[truncated_f_per_f[j * k] for j in range(n_teacher_prompts)],
                                a2_texts=[a2_texts[j * k * m] for j in range(n_teacher_prompts)],
                                a2_correct=[a2_correct[j * k * m] for j in range(n_teacher_prompts)],
                                student_rewards=[student2_rewards[j * k * m] for j in range(n_teacher_prompts)],
                                teacher_rewards=[teacher_rewards[j * k] for j in range(n_teacher_prompts)],
                                ground_truths=[ground_truths[i] for i in teacher_prompt_indices],
                                n=1,  # one sample per prompt for the table
                            )

                        # Build teacher entries from F generation output
                        # Grouped by prompt: [p0_f0, p0_f1, ..., p0_f(k-1), p1_f0, ...]
                        teacher_entries = []
                        for f_idx in range(n_total_f):
                            j = f_idx // k
                            i = teacher_prompt_indices[j]
                            prompt_ids, response_ids = self._extract_real_tokens(f_output, f_idx)
                            teacher_entries.append(
                                {
                                    "prompt_ids": prompt_ids,
                                    "response_ids": response_ids,
                                    "reward": teacher_rewards[f_idx],
                                    "uid": f"teacher_{base_uids[i]}",
                                }
                            )

                        # Build student₂ entries from A₂ generation output
                        # Grouped by feedback: [p0_f0_a0, ..., p0_f0_a(m-1), p0_f1_a0, ...]
                        student2_entries = []
                        for a2_idx in range(n_total_a2):
                            f_idx = a2_idx // m
                            j = f_idx // k
                            i = teacher_prompt_indices[j]
                            prompt_ids, response_ids = self._extract_real_tokens(
                                a2_output, a2_idx
                            )
                            student2_entries.append(
                                {
                                    "prompt_ids": prompt_ids,
                                    "response_ids": response_ids,
                                    "reward": student2_rewards[a2_idx],
                                    "uid": f"student2_{base_uids[i]}_f{f_idx % k}",
                                }
                            )

                        all_entries = teacher_entries + student2_entries

                        if len(all_entries) == 0:
                            print(
                                f"[SelfTeach step {self.global_steps}] "
                                f"All prompts filtered (all A₁ correct). Skipping training step."
                            )
                            self.global_steps += 1
                            progress_bar.update(1)
                            continue

                        batch = self._build_combined_batch(all_entries, pad_token_id)

                    print(
                        f"[SelfTeach step {self.global_steps}] "
                        f"Combined batch: {len(all_entries)} entries "
                        f"({len(teacher_entries)} teacher + {len(student2_entries)} student₂) "
                        f"from {n_teacher_prompts}/{bs} prompts, k={k}, m={m}"
                    )

                    # ==========================================================
                    # Standard GRPO pipeline on combined batch
                    # ==========================================================

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Rewards already computed — extract from meta_info
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor = batch.meta_info["self_teach_token_level_scores"]
                        reward_extra_infos_dict = {}

                    # Recompute old_log_probs on the combined batch
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
                        "bypass_mode", False
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, (
                        f'"old_log_probs" not in {batch.batch.keys()=}'
                    )

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=k,  # k == m, both teacher and student groups use this size
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
                        )

                # Validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                from verl.experimental.dataset.sampler import AbstractCurriculumSampler

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


class SelfTeachTaskRunner(TaskRunner):
    """TaskRunner that uses SelfTeachRayPPOTrainer."""

    def run(self, config):
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"SelfTeachTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.trainer.ppo.reward import load_reward_manager

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Extract self-teach config
        st_cfg_raw = config.get("self_teach", {})
        filter_val = st_cfg_raw.get("filter_a1_correct", None)
        # Hydra may pass the string "null" or omit the key; normalize to None
        if filter_val == "null" or filter_val == "":
            filter_val = None
        self_teach_config = SelfTeachConfig(
            enabled=st_cfg_raw.get("enabled", False),
            blind_teacher=st_cfg_raw.get("blind_teacher", False),
            filter_a1_correct=filter_val,
            num_feedbacks=st_cfg_raw.get("num_feedbacks", 6),
            num_a2_per_feedback=st_cfg_raw.get("num_a2_per_feedback", 6),
            max_a2_prompt_length=st_cfg_raw.get("max_a2_prompt_length", None),
        )
        print(
            f"[SelfTeach] Config: enabled={self_teach_config.enabled}, "
            f"blind_teacher={self_teach_config.blind_teacher}, "
            f"filter_a1_correct={self_teach_config.filter_a1_correct}, "
            f"k={self_teach_config.num_feedbacks}, m={self_teach_config.num_a2_per_feedback}"
        )

        trainer = SelfTeachRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            self_teach_config=self_teach_config,
        )
        trainer.init_workers()
        trainer.fit()
