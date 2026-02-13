"""Custom SFT trainer for A* grokking with generation-based evaluation.

Extends verl's FSDPSFTTrainer to add generation-based metrics at configurable
intervals. This measures what matters for grokking: can the model *generate*
correct optimal paths, not just predict next tokens?

Logged metrics (both train_gen/ and gen_eval/ prefixes):
  - optimal_accuracy  (main grokking signal)
  - valid_path_rate
  - format_rate
  - avg_reward
  - avg_gen_tokens

Usage:
    python -m astar_dataset.custom_sft_trainer \
        data.train_files=... \
        data.val_files=... \
        +gen_eval.val_parquet=data/astar_grokking_dataset/astar_val.parquet \
        +gen_eval.train_parquet=data/astar_grokking_dataset/astar_train.parquet \
        +gen_eval.samples=50 \
        +gen_eval.max_tokens=2048 \
        +gen_eval.freq=50 \
        ...
"""

import os
import sys
import time

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_full_state_dict
from verl.utils.tracking import Tracking
from verl.utils.logger import log_with_rank
from torch.distributed.device_mesh import init_device_mesh

# Add src/ to path so astar_dataset is importable
_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class AStarSFTTrainer(FSDPSFTTrainer):
    """SFT trainer with generation-based evaluation for grokking experiments."""

    def __init__(
        self,
        *args,
        val_parquet_path: str | None = None,
        train_parquet_path: str | None = None,
        gen_eval_samples: int = 50,
        gen_eval_max_tokens: int = 2048,
        gen_eval_freq: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.val_parquet_path = val_parquet_path
        self.train_parquet_path = train_parquet_path
        self.gen_eval_samples = gen_eval_samples
        self.gen_eval_max_tokens = gen_eval_max_tokens
        # Default: eval at same freq as checkpoint saves
        self.gen_eval_freq = gen_eval_freq if gen_eval_freq else self.config.trainer.save_freq

    def _run_generation_eval(self, rank, global_step, tracking):
        """Run generation-based evaluation.

        Uses verl's get_fsdp_full_state_dict (works with FSDP1 and FSDP2)
        to gather model weights to rank 0, then loads a temporary model
        for generation and evaluation.
        """
        from astar_dataset.eval import run_generation_eval

        t0 = time.time()

        # Collective op: all ranks participate, rank 0 gets full state dict on CPU
        state_dict = get_fsdp_full_state_dict(
            self.fsdp_model, offload_to_cpu=True, rank0_only=True
        )

        metrics = {}
        if rank == 0:
            # Create a temporary model for generation
            model_config = AutoConfig.from_pretrained(
                copy_to_local(src=self.config.model.partial_pretrain, verbose=False)
            )
            eval_model = AutoModelForCausalLM.from_config(
                model_config, torch_dtype=torch.bfloat16
            )
            eval_model.load_state_dict(state_dict)
            eval_model = eval_model.to("cuda:0")
            eval_model.eval()

            del state_dict

            # Evaluate on val set
            if self.val_parquet_path:
                val_metrics = run_generation_eval(
                    eval_model,
                    self.tokenizer,
                    self.val_parquet_path,
                    max_samples=self.gen_eval_samples,
                    max_new_tokens=self.gen_eval_max_tokens,
                    prefix="gen_eval",
                )
                metrics.update(val_metrics)

            # Evaluate on train subset (tracks memorization for grokking)
            if self.train_parquet_path:
                train_metrics = run_generation_eval(
                    eval_model,
                    self.tokenizer,
                    self.train_parquet_path,
                    max_samples=self.gen_eval_samples,
                    max_new_tokens=self.gen_eval_max_tokens,
                    prefix="gen_train",
                )
                metrics.update(train_metrics)

            del eval_model
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            metrics["gen_eval/total_time_s"] = elapsed

            if metrics:
                tracking.log(data=metrics, step=global_step)

            # Print summary
            val_acc = metrics.get("gen_eval/optimal_accuracy", -1)
            train_acc = metrics.get("gen_train/optimal_accuracy", -1)
            print(
                f"[Step {global_step}] Gen eval: "
                f"val_optimal_acc={val_acc:.3f}, "
                f"train_optimal_acc={train_acc:.3f}, "
                f"time={elapsed:.1f}s"
            )
        else:
            del state_dict

        torch.distributed.barrier()
        return metrics

    def fit(self):
        """Training loop with CE loss validation + generation-based evaluation."""
        import logging

        logger = logging.getLogger(__file__)
        rank = self.device_mesh.get_rank()
        tracking = None

        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps

        log_with_rank(
            f"Total training steps: {self.total_training_steps}, "
            f"Gen eval freq: {self.gen_eval_freq}",
            logger=logger,
            rank=rank,
            log_only_rank_0=True,
        )

        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=rank,
                log_only_rank_0=True,
            )

        start_epoch = global_step // self.steps_per_epoch

        # Run generation eval before any training (step 0 baseline)
        if global_step == 0:
            self._run_generation_eval(rank, 0, tracking)

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(
                    self.device_name
                )
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0
                is_gen_eval_step = (
                    self.gen_eval_freq > 0
                    and global_step % self.gen_eval_freq == 0
                )

                # CE loss validation (cheap, runs at test_freq)
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(
                            val_data,
                            batch_size=self.config.data.micro_batch_size_per_gpu,
                        ).to(self.device_name)
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                # Generation-based evaluation (expensive, runs at gen_eval_freq)
                if is_last_step or is_gen_eval_step:
                    self._run_generation_eval(rank, global_step, tracking)

                # Save checkpoint
                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_astar_sft(config):
    """Set up and run the A* SFT trainer with generation eval."""
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer,
        max_samples=config.data.get("val_max_samples", -1),
    )

    # Extract generation eval config
    gen_eval_cfg = config.get("gen_eval", {})

    trainer = AStarSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        # Generation eval params
        val_parquet_path=gen_eval_cfg.get("val_parquet", None),
        train_parquet_path=gen_eval_cfg.get("train_parquet", None),
        gen_eval_samples=gen_eval_cfg.get("samples", 50),
        gen_eval_max_tokens=gen_eval_cfg.get("max_tokens", 2048),
        gen_eval_freq=gen_eval_cfg.get("freq", None),
    )

    trainer.fit()
    destroy_global_process_group()


if __name__ == "__main__":
    # Point Hydra to verl's config dir so it finds sft_trainer.yaml defaults
    import verl.trainer.config as _verl_config_module

    _verl_config_dir = os.path.dirname(_verl_config_module.__file__)

    @hydra.main(config_path=_verl_config_dir, config_name="sft_trainer", version_base=None)
    def main(config: DictConfig):
        auto_set_device(config)
        run_astar_sft(config)

    main()
