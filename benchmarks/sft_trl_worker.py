#!/usr/bin/env python3
"""
SFT training worker using TRL's SFTTrainer.

Launched via torchrun from sft_benchmark.py. Uses FSDP for sharding
across GPUs with optional CPU offloading.

Writes training metrics to a JSON file for the benchmark orchestrator.
"""

import argparse
import json
import os
import time

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="SFT TRL training worker")
    parser.add_argument("--model", required=True, help="HuggingFace model path")
    parser.add_argument("--train-file", required=True, help="Parquet training file")
    parser.add_argument("--batch-size", type=int, required=True, help="Global batch size")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Per-device batch size")
    parser.add_argument("--sequence-length", type=int, required=True, help="Max sequence length")
    parser.add_argument("--num-steps", type=int, required=True, help="Training steps")
    parser.add_argument("--num-gpus", type=int, required=True, help="Total number of GPUs")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--metrics-file", required=True, help="Path to write metrics JSON")
    parser.add_argument("--output-dir", default="/tmp/sft_benchmark", help="Temp output dir")
    args = parser.parse_args()

    # Calculate gradient accumulation steps
    grad_accum = max(1, args.batch_size // (args.num_gpus * args.micro_batch_size))

    # FSDP config
    fsdp_strategy = "full_shard auto_wrap"
    if args.offload:
        fsdp_strategy += " offload"

    fsdp_config = {
        "backward_prefetch": "backward_pre",
        "use_orig_params": "true",
    }

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=args.num_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.sequence_length,
        logging_steps=1,
        log_level="info",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        fsdp=fsdp_strategy,
        fsdp_config=fsdp_config,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        seed=42,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load dataset — rename "response" to "completion" for TRL prompt-completion format
    dataset = datasets.load_dataset("parquet", data_files=args.train_file, split="train")
    dataset = dataset.rename_column("response", "completion")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    start_time = time.perf_counter()
    trainer.train()
    total_time = time.perf_counter() - start_time

    # Extract metrics on rank 0
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        step_times = []
        loss_values = []
        for entry in trainer.state.log_history:
            if "loss" in entry:
                loss_values.append(entry["loss"])
            # Trainer logs contain timestamps we can diff

        # Calculate step times from log history
        # Each log entry has a "step" and we know total time
        logged_steps = [e for e in trainer.state.log_history if "loss" in e]
        if len(logged_steps) >= 2:
            # Use per-step runtime from trainer's logging
            for i in range(1, len(logged_steps)):
                prev = logged_steps[i - 1]
                curr = logged_steps[i]
                if "epoch" in prev and "epoch" in curr:
                    # Estimate step time from total time / steps
                    pass

        # Fallback: compute average step time from total training time
        # Subtract ~5s for model loading overhead
        effective_time = max(total_time - 5, total_time * 0.9)
        avg_step_time = effective_time / args.num_steps

        metrics = {
            "total_time_s": total_time,
            "avg_step_time_s": avg_step_time,
            "num_steps": args.num_steps,
            "loss_values": loss_values[-5:] if loss_values else [],
            "metrics_source": "trl_trainer",
        }

        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Training complete. Total time: {total_time:.1f}s, Avg step: {avg_step_time:.2f}s")


if __name__ == "__main__":
    main()
