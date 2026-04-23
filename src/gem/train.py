"""
GEM / CE SFT training script using HuggingFace Accelerate for multi-GPU DDP.

Usage:
    # Single GPU (debug)
    python src/gem/train.py --model Qwen/Qwen2.5-Math-1.5B --dataset data/sft.parquet

    # Multi-GPU via accelerate
    accelerate launch --num_processes=4 src/gem/train.py \
        --model Qwen/Qwen2.5-Math-1.5B --dataset data/sft.parquet --loss gem
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from gem.gem_loss import ce_loss_causal_lm, gem_loss_causal_lm


def parse_args():
    p = argparse.ArgumentParser(description="GEM/CE SFT Training")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--dataset", type=str, required=True, help="Path to parquet with prompt/response columns")
    p.add_argument("--output_dir", type=str, default="checkpoints/gem_sft")
    p.add_argument("--loss", type=str, choices=["gem", "ce"], default="gem")
    p.add_argument("--gem_beta", type=float, default=0.7)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tokenize_dataset(dataset, tokenizer, max_seq_length):
    """Tokenize prompt+response, masking prompt tokens with -100 in labels."""

    def tokenize_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for prompt, response in zip(examples["prompt"], examples["response"]):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            # Add EOS at end of response
            if tokenizer.eos_token_id is not None:
                response_ids = response_ids + [tokenizer.eos_token_id]

            combined = prompt_ids + response_ids

            # Truncate to max_seq_length
            if len(combined) > max_seq_length:
                combined = combined[:max_seq_length]
                prompt_len = min(len(prompt_ids), max_seq_length)
            else:
                prompt_len = len(prompt_ids)

            # Labels: -100 for prompt tokens, real ids for response tokens
            labels = [-100] * prompt_len + combined[prompt_len:]

            # Pad to max_seq_length
            pad_len = max_seq_length - len(combined)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids = combined + [pad_id] * pad_len
            attention_mask = [1] * len(combined) + [0] * pad_len
            labels = labels + [-100] * pad_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    dataset.set_format("torch")
    return dataset


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def main():
    args = parse_args()

    # Init accelerator with wandb if requested
    log_with = "wandb" if args.wandb_project else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        mixed_precision="bf16",
    )

    if args.wandb_project:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"Loss: {args.loss} (beta={args.gem_beta})" if args.loss == "gem" else f"Loss: {args.loss}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Devices: {accelerator.num_processes}")
        print(f"Per-device batch size: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        effective_batch = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
        print(f"Effective batch size: {effective_batch}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load and tokenize dataset
    dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    if accelerator.is_main_process:
        print(f"Dataset size: {len(dataset)} examples")

    dataset = tokenize_dataset(dataset, tokenizer, args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Prepare with accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        print(f"Training steps: {num_training_steps} (warmup: {num_warmup_steps})")
        os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits

                if args.loss == "gem":
                    loss = gem_loss_causal_lm(logits, batch["labels"], beta=args.gem_beta)
                else:
                    loss = ce_loss_causal_lm(logits, batch["labels"])

                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Count response tokens in batch
            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_tokens += n_tokens
            epoch_loss += loss.detach().item()

            # Log every accumulation boundary
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % 10 == 0 and accelerator.is_main_process:
                    elapsed = time.time() - epoch_start
                    tokens_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0

                    log_dict = {
                        "train/loss": loss.detach().item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (step + 1) / len(dataloader),
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/global_step": global_step,
                    }

                    # Always log CE for comparison when using GEM
                    if args.loss == "gem":
                        with torch.no_grad():
                            ce = ce_loss_causal_lm(logits, batch["labels"])
                        log_dict["train/ce_loss"] = ce.item()
                        log_dict["train/gem_loss"] = loss.detach().item()

                    if accelerator.sync_gradients:
                        log_dict["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                    accelerator.log(log_dict, step=global_step)

                    print(
                        f"  step {global_step}/{num_training_steps} | "
                        f"loss={loss.item():.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"tok/s={tokens_per_sec:.0f}"
                    )

        # End of epoch
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.num_epochs} done | avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint: {save_path}")

    # Save final
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved final model: {save_path}")

    if args.wandb_project:
        accelerator.end_training()


if __name__ == "__main__":
    main()
