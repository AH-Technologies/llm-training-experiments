"""S1 SFT trainer using HuggingFace Trainer + accelerate FSDP.

Replaces the custom torchtune-based trainer (`sft_trainer.py`) which had
memory-scaling issues at long sequences. Uses the standard HF Trainer loop
with accelerate's FSDP backend, which is well-tested for long-context SFT.

Launched via `accelerate launch` (see `scripts/submit_s1_sft_hf.slurm`):

    accelerate launch --config_file configs/accelerate_fsdp.yaml \\
        -m s1.sft_hf [OPTIONS]
"""

from __future__ import annotations

import argparse
import os

import datasets
import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

IGNORE_INDEX = -100


class BlockingHubPushCallback(TrainerCallback):
    """Upload each checkpoint-N dir to the Hub synchronously on rank 0.

    Each save lands as its own `checkpoint-{step}/` subfolder on `main`, so
    intermediates are preserved. All ranks barrier after the upload; set
    `ddp_timeout` on `TrainingArguments` to exceed the worst-case upload time.
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.api = HfApi()

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            self.api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=self.repo_id,
                path_in_repo=f"checkpoint-{state.global_step}",
                commit_message=f"checkpoint step {state.global_step}",
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--train-file", default="data/s1K/s1k_sft.parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16, help="Global batch size")
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, default=32768)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every-n-epochs", type=int, default=5)
    p.add_argument(
        "--save-final-only",
        action="store_true",
        help=(
            "Skip all periodic saves during training and write a single final "
            "checkpoint to output_dir/ after train() returns. Use for one-shot "
            "SFT runs (pruning sweep cells) where only the final model matters; "
            "saves ~130 GiB per cell vs the default 'steps' strategy and avoids "
            "filling the shared filesystem. Default behavior (steps + final "
            "save_model) is preserved when the flag is omitted — grokking et al."
        ),
    )
    p.add_argument("--wandb-project", default="s1-qwen32b-sft")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tokenize_and_mask(example, tokenizer, seq_length):
    """Tokenise prompt+response, mask prompt tokens in labels."""
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]

    full_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

    # Truncate to seq_length (from the right — drops tail of response if too long)
    if len(full_ids) > seq_length:
        full_ids = full_ids[:seq_length]
        labels = labels[:seq_length]

    # Pad to seq_length so all examples in a batch align
    pad_len = seq_length - len(full_ids)
    attention_mask = [1] * len(full_ids) + [0] * pad_len
    full_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    labels = labels + [IGNORE_INDEX] * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    args = parse_args()

    if args.wandb_run_name:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = datasets.load_dataset("parquet", data_files=args.train_file, split="train")
    ds = ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.sequence_length),
        remove_columns=ds.column_names,
        num_proc=4,
        desc="Tokenising",
    )

    # Derive gradient accumulation from global batch size and world size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, args.batch_size // (world_size * args.per_device_batch_size))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    # Disable HF cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False

    save_steps_per_epoch = len(ds) // (world_size * args.per_device_batch_size * grad_accum)
    save_steps = max(1, save_steps_per_epoch * args.save_every_n_epochs)

    # save_strategy="no" with --save-final-only: no periodic checkpoint-N dirs
    # during training. The single final save to output_dir/ is done explicitly
    # after trainer.train() returns. Prevents the common failure mode where
    # the scheduled save at step N coincides with the final step and leaves
    # two checkpoints on disk (save_total_limit cleanup races with shutdown).
    save_strategy = "no" if args.save_final_only else "steps"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        # AC is handled via fsdp_activation_checkpointing in the accelerate FSDP config
        gradient_checkpointing=False,
        save_strategy=save_strategy,
        save_steps=save_steps,
        # save_only_model: skip optimizer.bin / scheduler.pt / rng_state / pytorch_model_fsdp.bin
        # → ~65 GiB per checkpoint instead of ~418 GiB
        save_only_model=True,
        # save_total_limit=1: keep only the latest checkpoint on local disk. HF Trainer
        # deletes older ones AFTER the new save completes (brief 2-checkpoint overlap).
        # Hub holds the full history.
        save_total_limit=1,
        save_safetensors=True,
        logging_steps=1,
        report_to=["wandb"] if args.wandb_run_name else ["none"],
        run_name=args.wandb_run_name,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        # 2h barrier window — covers rank 0 uploading ~65 GiB per checkpoint
        # while other ranks wait in BlockingHubPushCallback.
        ddp_timeout=7200,
    )

    callbacks = []
    if args.hf_repo:
        is_rank_zero = int(os.environ.get("RANK", "0")) == 0
        if is_rank_zero:
            HfApi().create_repo(repo_id=args.hf_repo, exist_ok=True)
        callbacks.append(BlockingHubPushCallback(args.hf_repo))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # Write a single final copy to output_dir/ top-level in two cases:
    # - --save-final-only: this IS the one and only save.
    # - --hf-repo: needed as the source folder for the final Hub upload below.
    # Default branch (neither flag): the last `checkpoint-N/` subdir from
    # save_strategy="steps" already has the model; skipping the top-level
    # save avoids a ~130 GiB duplicate on shared disk.
    if args.save_final_only or args.hf_repo:
        trainer.save_model()

    if args.hf_repo and trainer.is_world_process_zero():
        HfApi().upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hf_repo,
            path_in_repo="final",
            commit_message="final model",
            ignore_patterns=["checkpoint-*/*"],
        )


if __name__ == "__main__":
    main()
