"""Self-Teach 2-phase trainer and task runner.

Implements a 2-phase generation strategy where branching happens at the right
point for each role:

Phase 1 (Teacher training):
  - Generate A₁ once per prompt (single-turn)
  - Branch: Generate n different F samples per prompt (all see same A₁)
  - Generate A₂ per F for teacher grading
  - Teacher reward = f(A₁ correctness, A₂ correctness)

Phase 2 (Student₂ training):
  - Select one F per prompt (from phase 1)
  - Branch: Generate n different A₂ samples (all see same A₁ + F)
  - Student₂ reward = f(A₁ correctness, A₂ correctness)

This fixes the credit assignment problem where student₂ was blamed/credited
for the teacher's quality, since all n copies now share the same context.

Based on verl v0.2+ RayPPOTrainer with AgentLoop-based generation.
"""

import os
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

from ..rewards.deepscaler_reward import compute_score as grade_solution
from .prompts import STUDENT1_SYSTEM, STUDENT2_PROMPT, TEACHER_PROMPT
from .rewards import compute_self_teach_rewards


@dataclass
class SelfTeachConfig:
    """Configuration for self-teach training."""

    enabled: bool = False


class SelfTeachRayPPOTrainer(RayPPOTrainer):
    """2-phase GRPO trainer for student-teacher self-play.

    Instead of generating all 3 turns as one trajectory, this trainer
    makes 4 separate single-turn generation calls per step, branching
    at the right point for each role to enable clean GRPO comparisons.
    """

    def __init__(self, *args, self_teach_config: Optional[SelfTeachConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_teach_config = self_teach_config or SelfTeachConfig()
        if self.self_teach_config.enabled:
            if self.config.reward_model.get("launch_reward_fn_async", False):
                raise ValueError(
                    "Self-teach is incompatible with launch_reward_fn_async=True."
                )
            print("[SelfTeach 2-Phase] Initialized.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_gen_batch(self, raw_prompts: list, temperature: float, global_steps: int) -> DataProto:
        """Build a DataProto for generation from raw prompt message lists.

        The agent loop reads raw_prompt from non_tensor_batch and handles
        tokenization via apply_chat_template internally.

        Args:
            raw_prompts: List of conversation message lists, e.g.
                [[{"role": "system", ...}, {"role": "user", ...}], ...]
            temperature: Sampling temperature.
            global_steps: Current training step (for tracing).
        """
        n = len(raw_prompts)
        return DataProto.from_dict(
            # Dummy tensor so DataProto.batch is not None (required for chunk/len)
            tensors={"dummy_tensor": torch.zeros(n, dtype=torch.uint8)},
            non_tensors={
                "raw_prompt": np.array(raw_prompts, dtype=object),
            },
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
        t_grading_a2_acc = metrics_dict.get("teacher_grading_a2_correct_count", 0) / max(
            metrics_dict.get("teacher_grading_total", 1), 1
        )
        t_mean_reward = metrics_dict.get("teacher_mean_reward", 0.0)
        s2_mean_reward = metrics_dict.get("student2_mean_reward", 0.0)

        print(
            f"[SelfTeach step {self.global_steps}] "
            f"a1_acc={a1_acc:.3f}, a2_acc={a2_acc:.3f}, "
            f"improve={imp_rate:.3f}, regress={reg_rate:.3f}, "
            f"grading_a2_acc={t_grading_a2_acc:.3f}, "
            f"teacher_reward={t_mean_reward:.3f}, student2_reward={s2_mean_reward:.3f}"
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
                        "self_teach/teacher_grading_a2_accuracy": t_grading_a2_acc,
                        "self_teach/phase1_teacher_mean_reward": t_mean_reward,
                        "self_teach/phase2_student2_mean_reward": s2_mean_reward,
                    },
                    step=self.global_steps,
                )
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # fit() override — 2-phase generation flow
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

                # Extract question texts from raw_prompt
                question_texts = []
                for i in range(bs):
                    raw_prompt = batch.non_tensor_batch["raw_prompt"][i]
                    q = ""
                    for msg in raw_prompt:
                        if msg.get("role") == "user":
                            q = msg.get("content", "")
                            break
                    question_texts.append(q)

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
                        a1_gen_batch = self._build_gen_batch(a1_raw_prompts, temperature, self.global_steps)
                        a1_output = self.async_rollout_manager.generate_sequences(a1_gen_batch)
                        timing_raw.update(a1_output.meta_info.get("timing", {}))
                        a1_output.meta_info.pop("timing", None)

                    a1_texts = self._decode_responses(a1_output)

                    # Grade A₁
                    a1_correct = []
                    for i in range(bs):
                        score = grade_solution(
                            data_source=data_sources[i],
                            solution_str=a1_texts[i],
                            ground_truth=ground_truths[i],
                        )
                        a1_correct.append(score >= 0.5)

                    # ==========================================================
                    # Step 2: Generate F (teacher feedback) — bs*n entries
                    # Each prompt gets n different feedback samples,
                    # all conditioned on the SAME A₁.
                    # ==========================================================
                    with marked_timer("gen_f", timing_raw, color="yellow"):
                        teacher_messages = []
                        for i in range(bs):
                            msgs = [
                                {"role": "system", "content": STUDENT1_SYSTEM},
                                {"role": "user", "content": question_texts[i]},
                                {"role": "assistant", "content": a1_texts[i]},
                                {
                                    "role": "user",
                                    "content": TEACHER_PROMPT.format(
                                        question=question_texts[i],
                                        student_answer=a1_texts[i],
                                    ),
                                },
                            ]
                            teacher_messages.append(msgs)

                        # Repeat each prompt n times (interleaved)
                        f_raw_prompts = []
                        for msgs in teacher_messages:
                            for _ in range(n):
                                f_raw_prompts.append(msgs)

                        f_gen_batch = self._build_gen_batch(f_raw_prompts, temperature, self.global_steps)
                        f_output = self.async_rollout_manager.generate_sequences(f_gen_batch)
                        timing_raw.update(f_output.meta_info.get("timing", {}))
                        f_output.meta_info.pop("timing", None)

                    f_texts = self._decode_responses(f_output)

                    # ==========================================================
                    # Step 3: Generate A₂ for teacher grading — bs*n entries
                    # Each F gets one A₂. Grade A₂ → teacher reward for that F.
                    # ==========================================================
                    with marked_timer("gen_a2_grading", timing_raw, color="blue"):
                        s2_grading_messages = []
                        for idx in range(bs * n):
                            i = idx // n  # prompt index
                            msgs = [
                                {"role": "system", "content": STUDENT1_SYSTEM},
                                {"role": "user", "content": question_texts[i]},
                                {"role": "assistant", "content": a1_texts[i]},
                                {
                                    "role": "user",
                                    "content": TEACHER_PROMPT.format(
                                        question=question_texts[i],
                                        student_answer=a1_texts[i],
                                    ),
                                },
                                {"role": "assistant", "content": f_texts[idx]},
                                {
                                    "role": "user",
                                    "content": STUDENT2_PROMPT.format(feedback=f_texts[idx]),
                                },
                            ]
                            s2_grading_messages.append(msgs)

                        a2_grading_gen_batch = self._build_gen_batch(
                            s2_grading_messages, temperature, self.global_steps
                        )
                        a2_grading_output = self.async_rollout_manager.generate_sequences(
                            a2_grading_gen_batch
                        )
                        timing_raw.update(a2_grading_output.meta_info.get("timing", {}))
                        a2_grading_output.meta_info.pop("timing", None)

                    a2_grading_texts = self._decode_responses(a2_grading_output)

                    # Grade A₂ → compute teacher rewards
                    teacher_rewards = []
                    a2_grading_correct_count = 0
                    for idx in range(bs * n):
                        i = idx // n
                        score = grade_solution(
                            data_source=data_sources[i],
                            solution_str=a2_grading_texts[idx],
                            ground_truth=ground_truths[i],
                        )
                        a2_correct_flag = score >= 0.5
                        if a2_correct_flag:
                            a2_grading_correct_count += 1
                        t_reward, _ = compute_self_teach_rewards(a1_correct[i], a2_correct_flag)
                        teacher_rewards.append(t_reward)

                    # ==========================================================
                    # Step 4: Generate A₂ for student₂ training — bs*n entries
                    # Pick one F per prompt (first from step 2), then branch
                    # n times for different A₂ samples.
                    # ==========================================================
                    with marked_timer("gen_a2_training", timing_raw, color="magenta"):
                        # Select first F per prompt
                        selected_f_texts = [f_texts[i * n] for i in range(bs)]

                        s2_training_messages = []
                        for i in range(bs):
                            msgs = [
                                {"role": "system", "content": STUDENT1_SYSTEM},
                                {"role": "user", "content": question_texts[i]},
                                {"role": "assistant", "content": a1_texts[i]},
                                {
                                    "role": "user",
                                    "content": TEACHER_PROMPT.format(
                                        question=question_texts[i],
                                        student_answer=a1_texts[i],
                                    ),
                                },
                                {"role": "assistant", "content": selected_f_texts[i]},
                                {
                                    "role": "user",
                                    "content": STUDENT2_PROMPT.format(
                                        feedback=selected_f_texts[i]
                                    ),
                                },
                            ]
                            s2_training_messages.append(msgs)

                        # Repeat each prompt n times (interleaved)
                        s2_raw_prompts = []
                        for msgs in s2_training_messages:
                            for _ in range(n):
                                s2_raw_prompts.append(msgs)

                        a2_training_gen_batch = self._build_gen_batch(
                            s2_raw_prompts, temperature, self.global_steps
                        )
                        a2_training_output = self.async_rollout_manager.generate_sequences(
                            a2_training_gen_batch
                        )
                        timing_raw.update(a2_training_output.meta_info.get("timing", {}))
                        a2_training_output.meta_info.pop("timing", None)

                    a2_training_texts = self._decode_responses(a2_training_output)

                    # Grade A₂ → compute student₂ rewards
                    student2_rewards = []
                    a2_training_correct = []
                    for idx in range(bs * n):
                        i = idx // n
                        score = grade_solution(
                            data_source=data_sources[i],
                            solution_str=a2_training_texts[idx],
                            ground_truth=ground_truths[i],
                        )
                        a2_correct_flag = score >= 0.5
                        a2_training_correct.append(a2_correct_flag)
                        _, s2_reward = compute_self_teach_rewards(a1_correct[i], a2_correct_flag)
                        student2_rewards.append(s2_reward)

                    # ==========================================================
                    # Build combined batch (teacher + student₂ sub-rollouts)
                    # ==========================================================
                    with marked_timer("self_teach_build_batch", timing_raw, color="cyan"):
                        # Metrics tracking (uses cached a2_training_correct from grading)
                        a2_correct_count = sum(1 for c in a2_training_correct if c)
                        improvement_count = sum(
                            1 for idx in range(bs * n)
                            if not a1_correct[idx // n] and a2_training_correct[idx]
                        )
                        regression_count = sum(
                            1 for idx in range(bs * n)
                            if a1_correct[idx // n] and not a2_training_correct[idx]
                        )

                        self_teach_metrics = {
                            "a1_correct_count": sum(1 for c in a1_correct if c),
                            "a1_total": bs,  # a1 is graded once per prompt
                            "a2_correct_count": a2_correct_count,
                            "improvement_count": improvement_count,
                            "regression_count": regression_count,
                            "teacher_grading_a2_correct_count": a2_grading_correct_count,
                            "teacher_grading_total": bs * n,
                            "teacher_mean_reward": sum(teacher_rewards) / len(teacher_rewards),
                            "student2_mean_reward": sum(student2_rewards) / len(student2_rewards),
                        }
                        self._log_self_teach_metrics(self_teach_metrics, bs * n)

                        # Build teacher entries from step 2 generation output
                        teacher_entries = []
                        for idx in range(bs * n):
                            i = idx // n
                            prompt_ids, response_ids = self._extract_real_tokens(f_output, idx)
                            teacher_entries.append(
                                {
                                    "prompt_ids": prompt_ids,
                                    "response_ids": response_ids,
                                    "reward": teacher_rewards[idx],
                                    "uid": f"teacher_{base_uids[i]}",
                                }
                            )

                        # Build student₂ entries from step 4 generation output
                        student2_entries = []
                        for idx in range(bs * n):
                            i = idx // n
                            prompt_ids, response_ids = self._extract_real_tokens(
                                a2_training_output, idx
                            )
                            student2_entries.append(
                                {
                                    "prompt_ids": prompt_ids,
                                    "response_ids": response_ids,
                                    "reward": student2_rewards[idx],
                                    "uid": f"student2_{base_uids[i]}",
                                }
                            )

                        all_entries = teacher_entries + student2_entries
                        batch = self._build_combined_batch(all_entries, pad_token_id)

                    print(
                        f"[SelfTeach step {self.global_steps}] "
                        f"Combined batch: {len(all_entries)} entries "
                        f"({len(teacher_entries)} teacher + {len(student2_entries)} student₂)"
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
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
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
        self_teach_config = SelfTeachConfig(
            enabled=st_cfg_raw.get("enabled", False),
        )
        print(f"[SelfTeach] Config: enabled={self_teach_config.enabled}")

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
