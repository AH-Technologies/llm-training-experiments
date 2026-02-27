"""BER-enhanced PPO trainer and task runner.

Subclasses verl's RayPPOTrainer to inject Bidirectional Experience Replay
after reward computation and before old_log_probs recomputation.
"""

import os
import socket
from dataclasses import dataclass
from typing import Optional

import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .ber_module import BERCache, classify_and_inject, classify_and_inject_within_step


@dataclass
class BERConfig:
    """Configuration for Bidirectional Experience Replay."""
    enabled: bool = False
    mode: str = "replay"  # "replay" = cross-step buffer, "amplify" = within-step amplification
    correct_cache_path: Optional[str] = None
    max_error_cache_age: int = 500
    buffer_size: int = 32
    injection_fraction: float = 0.1
    stop_grad_injected: bool = False
    recompute_log_probs_injected: bool = False


class BERRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer with Bidirectional Experience Replay.

    Overrides _compute_or_extract_reward to inject BER logic after reward
    computation. This modifies batch tensors in-place (replacing response
    slots for Phase 1/3 groups) and updates reward_tensor accordingly.
    The subsequent old_log_probs recomputation in fit() then naturally
    operates on the modified batch.
    """

    def __init__(self, *args, ber_config: Optional[BERConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ber_config = ber_config or BERConfig()
        self._ber_injected_indices = []  # Track injected indices for ratio reset

        if self.ber_config.enabled:
            if self.config.reward_model.get("launch_reward_fn_async", False):
                raise ValueError(
                    "BER is incompatible with launch_reward_fn_async=True. "
                    "BER hooks into _compute_or_extract_reward which is bypassed in async mode."
                )
            if self.ber_config.stop_grad_injected and self.ber_config.recompute_log_probs_injected:
                raise ValueError(
                    "stop_grad_injected and recompute_log_probs_injected are mutually exclusive. "
                    "stop_grad zeroes gradient for injected slots; recompute_log_probs resets "
                    "their importance ratio to 1."
                )
            if self.ber_config.mode not in ("replay", "amplify"):
                raise ValueError(
                    f"Unknown BER mode '{self.ber_config.mode}'. Must be 'replay' or 'amplify'."
                )
            self.ber_cache = BERCache.from_disk(
                self.ber_config.correct_cache_path,
                buffer_size=self.ber_config.buffer_size,
            )
            print(f"[BER] Initialized. Mode: {self.ber_config.mode}, "
                  f"correct cache: {'loaded' if self.ber_cache.correct_cache else 'None'}, "
                  f"buffer_size={self.ber_config.buffer_size}, "
                  f"injection_fraction={self.ber_config.injection_fraction}")
        else:
            self.ber_cache = BERCache()

    def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False, sum_reward=False):
        """Override to inject BER after reward computation during training.

        BER is only applied during the main training reward call
        (return_dict=False, sum_reward=False). Validation and REMAX baseline
        calls pass through unchanged.
        """
        result = super()._compute_or_extract_reward(batch, reward_fn, return_dict, sum_reward)

        # Only inject BER during main training reward computation
        if not self.ber_config.enabled or return_dict or sum_reward:
            return result

        reward_tensor, reward_extra_infos_dict = result

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if self.ber_config.mode == "amplify":
            batch, reward_tensor, ber_metrics, injected_indices = classify_and_inject_within_step(
                batch=batch,
                reward_tensor=reward_tensor,
                n_rollouts=self.config.actor_rollout_ref.rollout.n,
                ber_cache=self.ber_cache,
                global_step=self.global_steps,
                pad_token_id=pad_token_id,
                injection_fraction=self.ber_config.injection_fraction,
                stop_grad_injected=self.ber_config.stop_grad_injected,
            )
        else:
            batch, reward_tensor, ber_metrics, injected_indices = classify_and_inject(
                batch=batch,
                reward_tensor=reward_tensor,
                n_rollouts=self.config.actor_rollout_ref.rollout.n,
                ber_cache=self.ber_cache,
                global_step=self.global_steps,
                pad_token_id=pad_token_id,
                max_error_cache_age=self.ber_config.max_error_cache_age,
                injection_fraction=self.ber_config.injection_fraction,
                stop_grad_injected=self.ber_config.stop_grad_injected,
            )

        # Store injected indices for ratio reset in _update_actor
        self._ber_injected_indices = injected_indices

        # Log BER metrics
        self._log_ber_metrics(ber_metrics)

        return reward_tensor, reward_extra_infos_dict

    def _update_actor(self, batch):
        """Override to recompute old_log_probs for injected samples.

        When recompute_log_probs_injected is enabled, ensures that
        old_log_probs for injected indices reflect π_θ (current policy)
        rather than stale rollout log_probs. This forces the importance
        ratio to ≈1, converting injected samples to vanilla policy gradient.

        In decoupled mode (default), old_log_probs are already recomputed
        on the modified batch so this is a no-op. In bypass mode, this
        fixes the stale rollout_log_probs for injected tokens.
        """
        if (self.ber_config.recompute_log_probs_injected
                and self._ber_injected_indices):
            rollout_corr = self.config.algorithm.get("rollout_correction", None)
            is_bypass = rollout_corr and rollout_corr.get("bypass_mode", False)
            if is_bypass:
                # In bypass mode, old_log_probs = rollout_log_probs which are
                # wrong for injected tokens. Recompute under current π_θ.
                fresh, _ = self._compute_old_log_prob(batch)
                for idx in self._ber_injected_indices:
                    batch.batch["old_log_probs"][idx] = fresh.batch["old_log_probs"][idx]
                print(f"[BER] Recomputed old_log_probs for {len(self._ber_injected_indices)} "
                      f"injected indices (bypass mode fix)")
        return super()._update_actor(batch)

    def _log_ber_metrics(self, ber_metrics: dict):
        """Log BER metrics to console and wandb."""
        # Console logging
        parts = [f"{k.split('/')[-1]}={v}" for k, v in ber_metrics.items()]
        print(f"[BER step {self.global_steps}] {', '.join(parts)}")

        # Direct wandb logging (merges with trainer's log at same step)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(ber_metrics, step=self.global_steps)
        except ImportError:
            pass


class BERTaskRunner(TaskRunner):
    """TaskRunner that uses BERRayPPOTrainer instead of RayPPOTrainer."""

    def run(self, config):
        """Execute BER-enhanced PPO training workflow."""
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"BERTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
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

        # Extract BER config from hydra config
        ber_cfg_raw = config.get("ber", {})
        ber_config = BERConfig(
            enabled=ber_cfg_raw.get("enabled", False),
            mode=ber_cfg_raw.get("mode", "replay"),
            correct_cache_path=ber_cfg_raw.get("correct_cache_path", None),
            max_error_cache_age=ber_cfg_raw.get("max_error_cache_age", 500),
            buffer_size=ber_cfg_raw.get("buffer_size", 32),
            injection_fraction=ber_cfg_raw.get("injection_fraction", 0.1),
            stop_grad_injected=ber_cfg_raw.get("stop_grad_injected", False),
            recompute_log_probs_injected=ber_cfg_raw.get("recompute_log_probs_injected", False),
        )
        print(f"[BER] Config: enabled={ber_config.enabled}, mode={ber_config.mode}, "
              f"correct_cache={ber_config.correct_cache_path}, "
              f"max_error_cache_age={ber_config.max_error_cache_age}, "
              f"buffer_size={ber_config.buffer_size}, "
              f"injection_fraction={ber_config.injection_fraction}, "
              f"stop_grad_injected={ber_config.stop_grad_injected}, "
              f"recompute_log_probs_injected={ber_config.recompute_log_probs_injected}")

        # Use BER-enhanced trainer instead of standard RayPPOTrainer
        trainer = BERRayPPOTrainer(
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
            ber_config=ber_config,
        )
        trainer.init_workers()
        trainer.fit()
