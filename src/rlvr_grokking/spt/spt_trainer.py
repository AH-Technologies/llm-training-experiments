"""Self-Play Teaching (SPT) trainer and task runner.

Subclasses verl's RayPPOTrainer to implement SPT's 3-pass generation
pipeline where a single model alternates between student and teacher roles.

Override strategy: override fit() to replace the sequence generation + reward
computation portion with our 3-pass logic, while keeping all other parts
(old_log_probs, advantages, actor update, validation, checkpointing, logging).
"""

import os
import socket
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils import tensordict_utils as tu
from verl.utils.rollout_skip import RolloutSkip
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from ..rewards.deepscaler_reward import compute_score
from .spt_module import (
    STUDENT_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    build_gen_batch_from_messages,
    build_grpo_batch,
    build_student_turn1_messages,
    build_student_turn2_messages,
    build_teacher_messages,
    compute_teacher_reward,
    decode_responses,
)


@dataclass
class SPTConfig:
    """Configuration for Self-Play Teaching."""
    enabled: bool = False
    n_rollouts: int = 8
    student_system_prompt: str = STUDENT_SYSTEM_PROMPT
    teacher_system_prompt: str = TEACHER_SYSTEM_PROMPT
    temperature: float = 0.6


class SPTRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with Self-Play Teaching.

    Overrides fit() to replace the standard single-pass generation + reward
    with a 3-pass pipeline:
      Pass 1: Student Turn 1 (solve problem)
      Pass 2: Teacher feedback / Student Turn 2 (depending on step type)
      Pass 3: Student Turn 2 / Teacher feedback evaluation

    The GRPO update then applies to either:
      - Student Turn 2 tokens (student step)
      - Teacher feedback tokens (teacher step)
    """

    def __init__(self, *args, spt_config: Optional[SPTConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spt_config = spt_config or SPTConfig()

        if self.spt_config.enabled:
            print(f"[SPT] Initialized. n_rollouts={self.spt_config.n_rollouts}, "
                  f"temperature={self.spt_config.temperature}")

    def _spt_training_step(self, batch: DataProto, step_type: str):
        """Execute the 3-pass SPT generation pipeline.

        Args:
            batch: Original DataProto from dataloader (B prompts)
            step_type: "student" or "teacher"

        Returns:
            (grpo_batch: DataProto, reward_tensor: Tensor, spt_metrics: dict)
            grpo_batch has B*n_rollouts samples, ready for standard GRPO
        """
        n = self.spt_config.n_rollouts
        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        temperature = self.spt_config.temperature

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # 1. Extract questions + ground truths from batch
        # batch.non_tensor_batch["prompt"] contains the original message dicts
        # Each entry is a list like [{"role": "user", "content": "..."}]
        prompts_raw = batch.non_tensor_batch["prompt"]
        questions = []
        for p in prompts_raw:
            if isinstance(p, list):
                # Extract content from first user message
                questions.append(p[0]["content"] if isinstance(p[0], dict) else str(p[0]))
            elif isinstance(p, dict):
                questions.append(p["content"])
            else:
                questions.append(str(p))

        reward_model_data = batch.non_tensor_batch["reward_model"]
        ground_truths = []
        for rm in reward_model_data:
            if isinstance(rm, dict):
                ground_truths.append(str(rm.get("ground_truth", "")))
            else:
                ground_truths.append(str(rm))

        uids = batch.non_tensor_batch["uid"].tolist()
        B = len(questions)

        spt_metrics = {
            "spt/step_type": 0 if step_type == "student" else 1,
        }

        # ---------------------------------------------------------------
        # PASS 1: Student Turn 1 (1 per prompt, B total)
        # ---------------------------------------------------------------
        t0 = time.time()
        turn1_messages = [
            build_student_turn1_messages(q, self.spt_config.student_system_prompt)
            for q in questions
        ]
        turn1_gen_batch = build_gen_batch_from_messages(
            turn1_messages, self.tokenizer, max_prompt_length, temperature
        )
        turn1_output = self.actor_rollout_wg.generate_sequences(turn1_gen_batch)
        turn1_texts = decode_responses(turn1_output, self.tokenizer)
        spt_metrics["spt/generation_time_pass1"] = time.time() - t0

        # Grade Turn 1
        turn1_correct = []
        for text, gt in zip(turn1_texts, ground_truths):
            score = compute_score("math", text, gt)
            turn1_correct.append(score >= 1.0)

        turn1_accuracy = sum(turn1_correct) / len(turn1_correct) if turn1_correct else 0.0
        spt_metrics["spt/turn1_accuracy"] = turn1_accuracy

        if step_type == "student":
            return self._student_step(
                questions, turn1_texts, turn1_correct, ground_truths, uids,
                B, n, max_prompt_length, max_response_length, temperature,
                pad_token_id, spt_metrics,
            )
        else:
            return self._teacher_step(
                questions, turn1_texts, turn1_correct, ground_truths, uids,
                B, n, max_prompt_length, max_response_length, temperature,
                pad_token_id, spt_metrics,
            )

    def _student_step(
        self, questions, turn1_texts, turn1_correct, ground_truths, uids,
        B, n, max_prompt_length, max_response_length, temperature,
        pad_token_id, spt_metrics,
    ):
        """Student step: update problem-solving ability.

        Pass 2: Generate 1 teacher feedback per prompt
        Pass 3: Generate n student Turn 2 responses per prompt (GRPO group)
        """
        # ---------------------------------------------------------------
        # PASS 2: Teacher feedback (1 per prompt, B total)
        # ---------------------------------------------------------------
        t0 = time.time()
        teacher_messages = [
            build_teacher_messages(q, s, gt, self.spt_config.teacher_system_prompt)
            for q, s, gt in zip(questions, turn1_texts, ground_truths)
        ]
        teacher_gen_batch = build_gen_batch_from_messages(
            teacher_messages, self.tokenizer, max_prompt_length, temperature
        )
        teacher_output = self.actor_rollout_wg.generate_sequences(teacher_gen_batch)
        teacher_texts = decode_responses(teacher_output, self.tokenizer)
        spt_metrics["spt/generation_time_pass2"] = time.time() - t0

        # ---------------------------------------------------------------
        # PASS 3: Student Turn 2 (n per prompt, B*n total)
        # ---------------------------------------------------------------
        t0 = time.time()
        turn2_messages = [
            build_student_turn2_messages(q, s, t, self.spt_config.student_system_prompt)
            for q, s, t in zip(questions, turn1_texts, teacher_texts)
        ]
        # Repeat each message n times for n rollouts
        turn2_messages_repeated = [m for m in turn2_messages for _ in range(n)]
        uids_repeated = [u for u in uids for _ in range(n)]

        turn2_gen_batch = build_gen_batch_from_messages(
            turn2_messages_repeated, self.tokenizer, max_prompt_length, temperature
        )
        turn2_output = self.actor_rollout_wg.generate_sequences(turn2_gen_batch)
        turn2_texts = decode_responses(turn2_output, self.tokenizer)
        spt_metrics["spt/generation_time_pass3"] = time.time() - t0

        # Grade Turn 2 -> binary rewards
        ground_truths_repeated = [g for g in ground_truths for _ in range(n)]
        rewards = []
        turn2_correct_count = 0
        for text, gt in zip(turn2_texts, ground_truths_repeated):
            score = compute_score("math", text, gt)
            correct = score >= 1.0
            rewards.append(1.0 if correct else 0.0)
            if correct:
                turn2_correct_count += 1

        turn2_accuracy = turn2_correct_count / len(turn2_texts) if turn2_texts else 0.0
        spt_metrics["spt/turn2_accuracy"] = turn2_accuracy

        # Build GRPO batch from Turn 2 output
        prompt_tokens = [turn2_output.batch["prompts"][i] for i in range(len(turn2_texts))]
        response_tokens = [turn2_output.batch["responses"][i] for i in range(len(turn2_texts))]

        grpo_batch, reward_tensor = build_grpo_batch(
            prompt_token_ids=prompt_tokens,
            response_token_ids=response_tokens,
            rewards=rewards,
            uids=uids_repeated,
            pad_token_id=pad_token_id,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )

        return grpo_batch, reward_tensor, spt_metrics

    def _teacher_step(
        self, questions, turn1_texts, turn1_correct, ground_truths, uids,
        B, n, max_prompt_length, max_response_length, temperature,
        pad_token_id, spt_metrics,
    ):
        """Teacher step: update feedback-giving ability.

        Pass 2: Generate n teacher feedbacks per prompt (GRPO group)
        Pass 3: Generate 1 student Turn 2 per feedback (for evaluation only)
        """
        # ---------------------------------------------------------------
        # PASS 2: Teacher feedback (n per prompt, B*n total)
        # ---------------------------------------------------------------
        t0 = time.time()
        teacher_messages = [
            build_teacher_messages(q, s, gt, self.spt_config.teacher_system_prompt)
            for q, s, gt in zip(questions, turn1_texts, ground_truths)
        ]
        teacher_messages_repeated = [m for m in teacher_messages for _ in range(n)]
        uids_repeated = [u for u in uids for _ in range(n)]

        teacher_gen_batch = build_gen_batch_from_messages(
            teacher_messages_repeated, self.tokenizer, max_prompt_length, temperature
        )
        teacher_output = self.actor_rollout_wg.generate_sequences(teacher_gen_batch)
        teacher_texts = decode_responses(teacher_output, self.tokenizer)
        spt_metrics["spt/generation_time_pass2"] = time.time() - t0

        # ---------------------------------------------------------------
        # PASS 3: Student Turn 2 (1 per feedback, B*n total, for evaluation)
        # ---------------------------------------------------------------
        t0 = time.time()
        turn2_messages = [
            build_student_turn2_messages(q, s, t, self.spt_config.student_system_prompt)
            for q, s, t in zip(
                [q for q in questions for _ in range(n)],
                [s for s in turn1_texts for _ in range(n)],
                teacher_texts,
            )
        ]
        turn2_gen_batch = build_gen_batch_from_messages(
            turn2_messages, self.tokenizer, max_prompt_length, temperature
        )
        turn2_output = self.actor_rollout_wg.generate_sequences(turn2_gen_batch)
        turn2_texts = decode_responses(turn2_output, self.tokenizer)
        spt_metrics["spt/generation_time_pass3"] = time.time() - t0

        # Grade Turn 2 + compute teacher rewards
        ground_truths_repeated = [g for g in ground_truths for _ in range(n)]
        turn1_correct_repeated = [c for c in turn1_correct for _ in range(n)]

        rewards = []
        turn2_correct_count = 0
        positive_transitions = 0
        negative_transitions = 0
        for t1_correct, text, gt in zip(turn1_correct_repeated, turn2_texts, ground_truths_repeated):
            score = compute_score("math", text, gt)
            t2_correct = score >= 1.0
            r = compute_teacher_reward(t1_correct, t2_correct)
            rewards.append(r)
            if t2_correct:
                turn2_correct_count += 1
            if not t1_correct and t2_correct:
                positive_transitions += 1
            if t1_correct and not t2_correct:
                negative_transitions += 1

        spt_metrics["spt/teacher_reward_mean"] = sum(rewards) / len(rewards) if rewards else 0.0
        spt_metrics["spt/teacher_reward_positive"] = positive_transitions
        spt_metrics["spt/teacher_reward_negative"] = negative_transitions
        spt_metrics["spt/turn2_accuracy"] = turn2_correct_count / len(turn2_texts) if turn2_texts else 0.0

        # Build GRPO batch from teacher output (train on teacher feedback tokens)
        prompt_tokens = [teacher_output.batch["prompts"][i] for i in range(len(teacher_texts))]
        response_tokens = [teacher_output.batch["responses"][i] for i in range(len(teacher_texts))]

        grpo_batch, reward_tensor = build_grpo_batch(
            prompt_token_ids=prompt_tokens,
            response_token_ids=response_tokens,
            rewards=rewards,
            uids=uids_repeated,
            pad_token_id=pad_token_id,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )

        return grpo_batch, reward_tensor, spt_metrics

    def _log_spt_metrics(self, spt_metrics: dict):
        """Log SPT metrics to console and wandb."""
        parts = [f"{k.split('/')[-1]}={v:.4f}" if isinstance(v, float) else f"{k.split('/')[-1]}={v}"
                 for k, v in spt_metrics.items()]
        print(f"[SPT step {self.global_steps}] {', '.join(parts)}")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(spt_metrics, step=self.global_steps)
        except ImportError:
            pass

    def fit(self):
        """SPT training loop.

        Overrides the parent fit() to replace the single-pass generation + reward
        with our 3-pass SPT pipeline. Everything else (old_log_probs, advantages,
        actor update, validation, checkpointing, logging) is kept from the parent.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from verl.experimental.dataset.sampler import AbstractCurriculumSampler

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
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

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ===== SPT CUSTOM: 3-pass generation + reward =====
                    step_type = "student" if self.global_steps % 2 == 1 else "teacher"

                    with marked_timer("gen", timing_raw, color="red"):
                        batch, reward_tensor, spt_metrics = self._spt_training_step(batch, step_type)

                    # Log SPT metrics
                    self._log_spt_metrics(spt_metrics)
                    metrics.update(spt_metrics)

                    # ===== FROM PARENT: everything after generation + reward =====

                    # Compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Set temperature for actor update
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                    # Set reward tensor as token_level_scores
                    batch.batch["token_level_scores"] = reward_tensor

                    # Apply KL penalty if configured
                    if self.config.algorithm.use_kl_in_reward:
                        # Need ref log probs first
                        if self.use_reference_policy:
                            with marked_timer(str("RefPolicy"), timing_raw, color="olive"):
                                ref_log_prob = self._compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Operating Mode Selection
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        # Recompute old_log_probs
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

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # Compute reference log_prob (if not already computed for KL)
                    if self.use_reference_policy and not self.config.algorithm.use_kl_in_reward:
                        with marked_timer("RefPolicy", timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute values (if critic enabled)
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # Rollout correction (if applicable)
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # Compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.spt_config.n_rollouts,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update actor (after critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

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

                # Check ESI expiration
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )

                # Save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
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

                # Training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

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

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)


class SPTTaskRunner(TaskRunner):
    """TaskRunner that uses SPTRayPPOTrainer instead of RayPPOTrainer."""

    def run(self, config):
        """Execute SPT-enhanced PPO training workflow."""
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"SPTTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
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

        # Extract SPT config from hydra config
        spt_cfg_raw = config.get("spt", {})
        spt_config = SPTConfig(
            enabled=spt_cfg_raw.get("enabled", False),
            n_rollouts=spt_cfg_raw.get("n_rollouts", 8),
            student_system_prompt=spt_cfg_raw.get("student_system_prompt", STUDENT_SYSTEM_PROMPT),
            teacher_system_prompt=spt_cfg_raw.get("teacher_system_prompt", TEACHER_SYSTEM_PROMPT),
            temperature=spt_cfg_raw.get("temperature", 0.6),
        )
        print(f"[SPT] Config: enabled={spt_config.enabled}, "
              f"n_rollouts={spt_config.n_rollouts}, "
              f"temperature={spt_config.temperature}")

        # Use SPT-enhanced trainer
        trainer = SPTRayPPOTrainer(
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
            spt_config=spt_config,
        )
        trainer.init_workers()
        trainer.fit()
