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

from .ber_module import BERCache, classify_and_inject


@dataclass
class BERConfig:
    """Configuration for Bidirectional Experience Replay."""
    enabled: bool = False
    correct_cache_path: Optional[str] = None
    max_error_cache_age: int = 500
    # Advantage clamping: limits extreme advantages from BER-injected samples
    # to prevent policy collapse from outsized negative gradients.
    adv_clamp_enabled: bool = False
    adv_clamp_min: float = -2.0
    adv_clamp_max: float = 2.0


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

        if self.ber_config.enabled:
            if self.config.reward_model.get("launch_reward_fn_async", False):
                raise ValueError(
                    "BER is incompatible with launch_reward_fn_async=True. "
                    "BER hooks into _compute_or_extract_reward which is bypassed in async mode."
                )
            self.ber_cache = BERCache.from_disk(self.ber_config.correct_cache_path)
            print(f"[BER] Initialized. Correct cache: {'loaded' if self.ber_cache.correct_cache else 'None'}")
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

        batch, reward_tensor, self.ber_cache.error_cache, ber_metrics = classify_and_inject(
            batch=batch,
            reward_tensor=reward_tensor,
            n_rollouts=self.config.actor_rollout_ref.rollout.n,
            correct_cache=self.ber_cache.correct_cache,
            error_cache=self.ber_cache.error_cache,
            global_step=self.global_steps,
            pad_token_id=pad_token_id,
            max_error_cache_age=self.ber_config.max_error_cache_age,
        )

        # Log BER metrics
        self._log_ber_metrics(ber_metrics)

        return reward_tensor, reward_extra_infos_dict

    def _update_actor(self, batch):
        """Override to clamp advantages before the actor update.

        BER injection creates outlier advantages (e.g. -2.645 for a single
        injected negative in an otherwise all-correct group). Clamping prevents
        these from causing destabilising gradient steps.
        """
        if self.ber_config.enabled and self.ber_config.adv_clamp_enabled:
            adv = batch.batch["advantages"]
            clamped = torch.clamp(adv, min=self.ber_config.adv_clamp_min, max=self.ber_config.adv_clamp_max)

            n_clamped_low = (adv < self.ber_config.adv_clamp_min).sum().item()
            n_clamped_high = (adv > self.ber_config.adv_clamp_max).sum().item()

            if n_clamped_low > 0 or n_clamped_high > 0:
                print(
                    f"[BER step {self.global_steps}] Advantage clamping: "
                    f"{n_clamped_low} below {self.ber_config.adv_clamp_min}, "
                    f"{n_clamped_high} above {self.ber_config.adv_clamp_max}. "
                    f"Original range: [{adv.min().item():.3f}, {adv.max().item():.3f}]"
                )

            batch.batch["advantages"] = clamped

            # Log clamping metrics
            clamp_metrics = {
                "ber/adv_clamped_low": n_clamped_low,
                "ber/adv_clamped_high": n_clamped_high,
                "ber/adv_min_pre_clamp": adv.min().item(),
                "ber/adv_max_pre_clamp": adv.max().item(),
            }
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(clamp_metrics, step=self.global_steps)
            except ImportError:
                pass

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
            correct_cache_path=ber_cfg_raw.get("correct_cache_path", None),
            max_error_cache_age=ber_cfg_raw.get("max_error_cache_age", 500),
            adv_clamp_enabled=ber_cfg_raw.get("adv_clamp_enabled", False),
            adv_clamp_min=float(ber_cfg_raw.get("adv_clamp_min", -2.0)),
            adv_clamp_max=float(ber_cfg_raw.get("adv_clamp_max", 2.0)),
        )
        print(f"[BER] Config: enabled={ber_config.enabled}, "
              f"correct_cache={ber_config.correct_cache_path}, "
              f"max_error_cache_age={ber_config.max_error_cache_age}, "
              f"adv_clamp={ber_config.adv_clamp_enabled} "
              f"[{ber_config.adv_clamp_min}, {ber_config.adv_clamp_max}]")

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
