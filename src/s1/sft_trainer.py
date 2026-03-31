"""
S1 SFT Trainer: Qwen2.5-32B fine-tuning on s1K dataset.

Uses torchtune FSDP2 + Tensor Parallelism for distributed training.
Reuses model loading infrastructure from benchmarks/sft_torchtune_worker.py.

Features beyond the benchmark worker:
- Epoch-based training with per-epoch data shuffling
- Periodic checkpoint saving (HuggingFace format)
- WandB logging
- Configurable LR warmup + cosine decay
- Gradient clipping

Launched via torchrun:
    torchrun --nproc_per_node=4 --nnodes=4 \\
        --rdzv_backend=c10d --rdzv_endpoint=<master>:29500 \\
        -m s1.sft_trainer [OPTIONS]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time

import datasets
import torch
import torch.distributed as dist
from torch.optim import AdamW
from transformers import AutoTokenizer

from benchmarks.sft_torchtune_worker import (
    IGNORE_INDEX,
    get_model_builder,
    setup_distributed,
    _load_from_rank0_broadcast,
)

from torchtune.training import (
    FullModelHFCheckpointer,
    OffloadActivations,
    get_shard_conditions,
    prepare_mha_for_tp,
    set_activation_checkpointing,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torchtune.modules import TransformerSelfAttentionLayer
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self._step <= self.warmup_steps:
                lr = base_lr * self._step / max(1, self.warmup_steps)
            else:
                progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(
    train_file: str,
    tokenizer: AutoTokenizer,
    seq_length: int,
    dp_rank: int,
    dp_size: int,
    epoch: int = 0,
) -> torch.utils.data.DataLoader:
    """Load parquet, tokenize with SFT loss masking, shard across DP ranks."""
    raw = datasets.load_dataset("parquet", data_files=train_file, split="train")

    def tokenize(batch):
        prompts = batch["prompt"]
        responses = batch["response"]

        input_ids_list, labels_list = [], []
        for prompt, response in zip(prompts, responses):
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            r_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
            ids = p_ids + r_ids
            labels = [IGNORE_INDEX] * len(p_ids) + r_ids
            ids = ids[:seq_length]
            labels = labels[:seq_length]
            pad_len = seq_length - len(ids)
            ids += [tokenizer.pad_token_id] * pad_len
            labels += [IGNORE_INDEX] * pad_len
            input_ids_list.append(ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names, desc="Tokenizing")
    tokenized.set_format("torch")

    # Shard across DP ranks
    total = len(tokenized)
    per_rank = total // dp_size
    start = dp_rank * per_rank
    end = start + per_rank
    shard = tokenized.select(range(start, min(end, total)))

    # Seed shuffle with epoch for different ordering each epoch
    generator = torch.Generator().manual_seed(42 + epoch)

    return torch.utils.data.DataLoader(
        shard,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, tokenizer, output_dir: str, step: int, epoch: int, rank: int,
    hf_repo: str | None = None, delete_local: bool = False,
):
    """Save model checkpoint and optionally upload to HF Hub.

    Args:
        hf_repo: HF Hub repo ID (e.g. "alexaau/s1-grokking-qwen32b"). If set,
                 uploads the checkpoint and optionally deletes the local copy.
        delete_local: If True and hf_repo is set, delete local files after upload.
    """
    if rank != 0:
        return

    ckpt_name = f"checkpoint-epoch{epoch}-step{step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Gather full state dict from FSDP2
    from torch.distributed._tensor import DTensor

    state_dict = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            state_dict[name] = param.full_tensor().cpu()
        else:
            state_dict[name] = param.cpu()

    torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
    tokenizer.save_pretrained(ckpt_dir)

    meta = {"step": step, "epoch": epoch}
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[s1] Checkpoint saved: {ckpt_dir}")

    # Upload to HF Hub
    if hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(hf_repo, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=hf_repo,
                path_in_repo=ckpt_name,
                commit_message=f"Checkpoint epoch {epoch} step {step}",
            )
            print(f"[s1] Uploaded to HF Hub: {hf_repo}/{ckpt_name}")

            if delete_local:
                import shutil
                shutil.rmtree(ckpt_dir)
                print(f"[s1] Deleted local checkpoint: {ckpt_dir}")
        except Exception as e:
            print(f"[s1] WARNING: HF upload failed (keeping local): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="S1 SFT Trainer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--train-file", default="data/s1K/s1k_sft.parquet")
    parser.add_argument("--output-dir", default="checkpoints/s1_sft_qwen32b")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch size")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--save-every-n-epochs", type=int, default=5)
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--wandb-project", default="s1-qwen32b-sft")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--hf-repo", default=None, help="HF Hub repo for checkpoint upload (e.g. alexaau/s1-grokking-qwen32b)")
    parser.add_argument("--delete-local-checkpoints", action="store_true", help="Delete local checkpoints after HF upload")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Distributed init ----
    dp_mesh, tp_mesh, device = setup_distributed(args.tp_degree)
    rank = dist.get_rank()
    dp_rank = dp_mesh.get_local_rank()
    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    is_rank0 = (rank == 0)

    if is_rank0:
        print(f"[s1] world={dist.get_world_size()} dp={dp_size} tp={tp_size}")
        print(f"[s1] Config: {vars(args)}")

    torch.manual_seed(args.seed)

    # ---- WandB init (rank 0 only) ----
    wandb_run = None
    if is_rank0:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"s1_sft_{args.model.split('/')[-1]}",
                config=vars(args),
            )
            print(f"[s1] WandB run: {wandb_run.url}")
        except Exception as e:
            print(f"[s1] WandB init failed (continuing without): {e}")

    # ---- Load model ----
    builder_fn, tp_plan_fn = get_model_builder(args.model)

    if is_rank0:
        print(f"[s1] Resolving model cache for {args.model} ...")
    local_model_dir = snapshot_download(args.model, local_files_only=True)

    full_model_sd = {}
    if is_rank0:
        import re as _re
        _model_type_str = "QWEN2" if "qwen" in args.model.lower() else "LLAMA3"
        _shard_files = sorted(
            f for f in os.listdir(local_model_dir)
            if f.endswith(".safetensors") and _re.search(r"\d+-of-\d+", f)
        )
        if _shard_files:
            _nums = _re.findall(r"\d+", _shard_files[-1])
            _ckpt_files = {
                "filename_format": _re.sub(r"\d+", "{}", _shard_files[0], count=2),
                "max_filename": _nums[-1],
            }
        else:
            _ckpt_files = [f for f in os.listdir(local_model_dir) if f.endswith(".safetensors")]

        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=local_model_dir,
            checkpoint_files=_ckpt_files,
            output_dir=args.output_dir,
            model_type=_model_type_str,
        )
        full_model_sd = checkpointer.load_checkpoint()["model"]
        print(f"[s1] Checkpoint loaded ({len(full_model_sd)} tensors)")

    # Build model on meta device
    with torch.device("meta"):
        model = builder_fn()

    # Apply TP
    if tp_size > 1 and tp_plan_fn is not None:
        prepare_mha_for_tp(model, tp_mesh)
        parallelize_module(model, tp_mesh, tp_plan_fn())
        if is_rank0:
            print(f"[s1] TP={tp_size} applied")

    # Apply FSDP2
    fsdp_kwargs = {"reshard_after_forward": True, "mesh": dp_mesh}
    if args.cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=False)

    for n, m in reversed(list(model.named_modules())):
        if get_shard_conditions(n, m):
            fully_shard(m, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # Init RoPE buffers
    with torch.device(device):
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

    # Broadcast weights
    if is_rank0:
        print(f"[s1] Broadcasting weights ...")
    _load_from_rank0_broadcast(model, full_model_sd, device, rank, strict=False, cpu_offload=args.cpu_offload)
    del full_model_sd
    if is_rank0:
        gpu_alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"[s1] Weights loaded. GPU: {gpu_alloc:.1f} GB allocated")

    # Activation checkpointing
    set_activation_checkpointing(model, auto_wrap_policy={TransformerSelfAttentionLayer})

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Calculate training steps ----
    raw_dataset = datasets.load_dataset("parquet", data_files=args.train_file, split="train")
    num_samples = len(raw_dataset)
    del raw_dataset

    samples_per_rank = num_samples // dp_size
    grad_accum_steps = max(1, args.batch_size // (dp_size * args.micro_batch_size))
    steps_per_epoch = samples_per_rank // (args.micro_batch_size * grad_accum_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if is_rank0:
        print(f"[s1] {num_samples} samples, {steps_per_epoch} steps/epoch, "
              f"{total_steps} total steps, {warmup_steps} warmup steps")
        print(f"[s1] grad_accum={grad_accum_steps}, micro_batch={args.micro_batch_size}")

    # ---- Optimizer + scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # ---- Training loop ----
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    act_offload_ctx = OffloadActivations(use_pin_memory=False) if args.cpu_offload else contextlib.nullcontext()

    global_step = 0
    total_start = time.perf_counter()

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        epoch_start = time.perf_counter()
        loader = build_dataset(args.train_file, tokenizer, args.sequence_length, dp_rank, dp_size, epoch=epoch)
        data_iter = iter(loader)

        epoch_loss = 0.0
        epoch_steps = 0

        for step_in_epoch in range(steps_per_epoch):
            step_start = time.perf_counter()
            optimizer.zero_grad()
            accum_loss = torch.tensor(0.0, device=device)

            for _ in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                input_ids = input_ids.expand(args.micro_batch_size, -1)
                labels = labels.expand(args.micro_batch_size, -1)

                with act_offload_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss / grad_accum_steps

                loss.backward()
                accum_loss += loss.detach()

            # Gradient clipping
            from torch.distributed.tensor import DTensor
            norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = p.grad.norm(2.0)
                    if isinstance(grad_norm, DTensor):
                        grad_norm = grad_norm.full_tensor()
                    norms.append(grad_norm)
            if norms:
                total_norm = torch.stack(norms).norm(2.0)
                clip_coef = torch.clamp(args.max_grad_norm / (total_norm + 1e-6), max=1.0)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.detach().mul_(clip_coef)

            optimizer.step()
            scheduler.step()
            global_step += 1

            step_loss = accum_loss.item()
            epoch_loss += step_loss
            epoch_steps += 1
            step_time = time.perf_counter() - step_start

            # Logging
            if is_rank0 and (global_step % args.log_every_n_steps == 0):
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = (args.batch_size * args.sequence_length) / step_time

                log_dict = {
                    "train/loss": step_loss,
                    "train/lr": lr,
                    "train/epoch": epoch + step_in_epoch / steps_per_epoch,
                    "train/global_step": global_step,
                    "train/step_time_ms": step_time * 1000,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/grad_norm": total_norm.item() if norms else 0.0,
                }
                if wandb_run:
                    wandb_run.log(log_dict, step=global_step)

                if global_step % 10 == 0 or global_step == 1:
                    print(f"[s1] epoch={epoch+1}/{args.num_epochs} step={global_step}/{total_steps} "
                          f"loss={step_loss:.4f} lr={lr:.2e} tok/s={tokens_per_sec:.0f} "
                          f"step_time={step_time*1000:.0f}ms")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.perf_counter() - epoch_start
        if is_rank0:
            print(f"[s1] === Epoch {epoch+1}/{args.num_epochs} done. "
                  f"avg_loss={avg_epoch_loss:.4f} time={epoch_time:.0f}s ===")
            if wandb_run:
                wandb_run.log({"train/epoch_loss": avg_epoch_loss, "train/epoch_time_s": epoch_time},
                              step=global_step)

        # Save checkpoint
        if (epoch + 1) % args.save_every_n_epochs == 0 or (epoch + 1) == args.num_epochs:
            dist.barrier()
            save_checkpoint(
                model, tokenizer, args.output_dir, global_step, epoch + 1, rank,
                hf_repo=args.hf_repo, delete_local=args.delete_local_checkpoints,
            )
            dist.barrier()

    # Final summary
    total_time = time.perf_counter() - total_start
    if is_rank0:
        print(f"[s1] Training complete. {total_steps} steps in {total_time:.0f}s "
              f"({total_time/3600:.1f}h)")
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[s1] Peak GPU memory: {peak_mem:.1f} GB")

        if wandb_run:
            wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
