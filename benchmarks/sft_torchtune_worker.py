#!/usr/bin/env python3
"""
SFT training worker using torchtune FSDP2 + Tensor Parallelism.

Unified strategy for all model sizes (0.5B → 180B+):
- FSDP2 shards parameters + gradients + optimizer states across all N GPUs
- TP shards activations within a node (T ≤ gpus_per_node = 4)
- No CPU offload by default; optimizer stays on GPU via FSDP2 sharding

Memory model (matches estimate_memory.py sft_fsdp formula):
  GPU = 16P/N + unit_peak + activations/T + logits/T + overhead

Launched via torchrun from sft_benchmark.py:
  torchrun --nproc_per_node=<gpus> --nnodes=<nodes> \\
      -m benchmarks.sft_torchtune_worker [OPTIONS]

Environment variables set by torchrun / Slurm:
  RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import datasets
import torch
from huggingface_hub import snapshot_download
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

import torchtune.models.llama3_1 as llama3_1_models
import torchtune.models.llama3_2 as llama3_2_models
import torchtune.models.qwen2_5 as qwen2_5_models
from torchtune.training import (
    FullModelHFCheckpointer,
    OffloadActivations,
    get_shard_conditions,
    prepare_mha_for_tp,
    set_activation_checkpointing,
)

from benchmarks.qwen2_5_tp_plan import base_qwen2_5_tp_plan
from torchtune.models.llama3._parallelism import base_llama_tp_plan


# ---------------------------------------------------------------------------
# Model registry — maps HuggingFace model IDs to torchtune builder functions
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, tuple[callable, callable | None]] = {
    # (builder_fn, tp_plan_fn | None)
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B":             (qwen2_5_models.qwen2_5_0_5b,          base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-0.5B-Instruct":    (qwen2_5_models.qwen2_5_0_5b,          base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-1.5B":             (qwen2_5_models.qwen2_5_1_5b_base,      base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-1.5B-Instruct":    (qwen2_5_models.qwen2_5_1_5b_instruct,  base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-3B":               (qwen2_5_models.qwen2_5_3b,             base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-3B-Instruct":      (qwen2_5_models.qwen2_5_3b,             base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-7B":               (qwen2_5_models.qwen2_5_7b_base,        base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-7B-Instruct":      (qwen2_5_models.qwen2_5_7b_instruct,    base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-14B":              (qwen2_5_models.qwen2_5_14b_base,       base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-14B-Instruct":     (qwen2_5_models.qwen2_5_14b_instruct,   base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-32B":              (qwen2_5_models.qwen2_5_32b_base,       base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-32B-Instruct":     (qwen2_5_models.qwen2_5_32b_instruct,   base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-72B":              (qwen2_5_models.qwen2_5_72b_base,       base_qwen2_5_tp_plan),
    "Qwen/Qwen2.5-72B-Instruct":     (qwen2_5_models.qwen2_5_72b_instruct,   base_qwen2_5_tp_plan),
    # Llama 3.2
    "meta-llama/Llama-3.2-1B":           (llama3_2_models.llama3_2_1b,  base_llama_tp_plan),
    "meta-llama/Llama-3.2-1B-Instruct":  (llama3_2_models.llama3_2_1b,  base_llama_tp_plan),
    "meta-llama/Llama-3.2-3B":           (llama3_2_models.llama3_2_3b,  base_llama_tp_plan),
    "meta-llama/Llama-3.2-3B-Instruct":  (llama3_2_models.llama3_2_3b,  base_llama_tp_plan),
    # Llama 3.1
    "meta-llama/Llama-3.1-8B":           (llama3_1_models.llama3_1_8b,  base_llama_tp_plan),
    "meta-llama/Llama-3.1-8B-Instruct":  (llama3_1_models.llama3_1_8b,  base_llama_tp_plan),
    "meta-llama/Llama-3.1-70B":          (llama3_1_models.llama3_1_70b, base_llama_tp_plan),
    "meta-llama/Llama-3.1-70B-Instruct": (llama3_1_models.llama3_1_70b, base_llama_tp_plan),
}


def get_model_builder(model_name: str) -> tuple[callable, callable | None]:
    """Look up torchtune builder for a HuggingFace model name."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    # Fuzzy match on the model name suffix
    for key, val in MODEL_REGISTRY.items():
        if model_name.lower().endswith(key.split("/")[-1].lower()):
            return val
    raise ValueError(
        f"Model '{model_name}' not in MODEL_REGISTRY. "
        f"Available: {list(MODEL_REGISTRY.keys())}"
    )


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed(tp_degree: int) -> tuple[dist.ProcessGroup, dist.ProcessGroup, torch.device]:
    """
    Initialise process groups and build a 2D DeviceMesh (dp x tp).

    Returns (dp_mesh, tp_mesh, local_device).
    """
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    assert world_size % tp_degree == 0, (
        f"world_size ({world_size}) must be divisible by tp_degree ({tp_degree})"
    )
    dp_degree = world_size // tp_degree

    # 2D mesh: first dim = dp, second dim = tp
    mesh = init_device_mesh(
        "cuda",
        (dp_degree, tp_degree),
        mesh_dim_names=("dp", "tp"),
    )
    return mesh["dp"], mesh["tp"], device


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100


def build_dataset(
    train_file: str,
    tokenizer: AutoTokenizer,
    seq_length: int,
    dp_rank: int,
    dp_size: int,
) -> torch.utils.data.DataLoader:
    """Load parquet dataset, tokenize with SFT loss masking, shard across DP ranks."""
    raw = datasets.load_dataset("parquet", data_files=train_file, split="train")

    def tokenize(batch):
        prompts = batch["prompt"]
        responses = batch["response"] if "response" in batch else batch["completion"]

        input_ids_list, labels_list = [], []
        for prompt, response in zip(prompts, responses):
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            r_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
            ids = p_ids + r_ids
            # Mask prompt tokens in labels
            labels = [IGNORE_INDEX] * len(p_ids) + r_ids
            # Truncate
            ids = ids[:seq_length]
            labels = labels[:seq_length]
            # Pad
            pad_len = seq_length - len(ids)
            ids += [tokenizer.pad_token_id] * pad_len
            labels += [IGNORE_INDEX] * pad_len
            input_ids_list.append(ids)
            labels_list.append(labels)
        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names,
                        desc="Tokenizing")
    tokenized.set_format("torch")

    # Shard across DP ranks deterministically
    total = len(tokenized)
    per_rank = total // dp_size
    start = dp_rank * per_rank
    end = start + per_rank
    shard = tokenized.select(range(start, min(end, total)))

    return torch.utils.data.DataLoader(
        shard,
        batch_size=1,  # micro-batch size handled by caller via grad accum
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Custom checkpoint loading: broadcast from rank 0 + local shard
# ---------------------------------------------------------------------------

def _load_from_rank0_broadcast(
    model: torch.nn.Module,
    full_sd: dict,
    device: torch.device,
    rank: int,
    strict: bool = False,
    cpu_offload: bool = False,
) -> None:
    """
    Load a full state dict into an FSDP2+TP model by broadcasting from rank 0.

    Algorithm per parameter:
      1. Rank 0 provides the full tensor; others allocate a receive buffer (on GPU).
      2. dist.broadcast sends it to all ranks via NCCL.
      3. distribute_tensor computes the correct local shard (handles _StridedShard
         for 2D FSDP2+TP).
      4. model.load_state_dict(assign=True) triggers FSDP2's reset_sharded_param.

    For cpu_offload, we must manually move _sharded_param_data to CPU after loading.
    PyTorch's reset_sharded_param has a bug: it only moves to CPU when pin_memory=True,
    but _init_sharded_param moves to CPU for any offload_to_cpu=True. Since params start
    on meta device, _init_sharded_param skips the move, and reset_sharded_param (after
    load_state_dict) also skips it when pin_memory=False. Additionally,
    DTensor.from_local() silently moves CPU tensors back to GPU when the mesh is CUDA.
    """
    from torch.distributed._tensor import DTensor, distribute_tensor

    is_rank0 = (rank == 0)

    # Broadcast available keys once
    sd_keys_list = [sorted(full_sd.keys())] if is_rank0 else [None]
    dist.broadcast_object_list(sd_keys_list, src=0)
    available_keys = set(sd_keys_list[0])

    sharded_sd = {}
    for param_name, param in model.named_parameters():
        if param_name not in available_keys:
            if strict:
                raise KeyError(f"Missing key in checkpoint: {param_name}")
            continue

        # Read global shape and dtype from model metadata (works on meta device)
        if isinstance(param, DTensor):
            global_shape = param.size()   # full unsharded shape
            param_dtype  = param.dtype
            device_mesh  = param.device_mesh
            placements   = param.placements
        else:
            global_shape = param.shape
            param_dtype  = param.dtype
            device_mesh  = None
            placements   = None

        # Rank 0: materialise from state dict; others: allocate receive buffer
        if is_rank0:
            tensor = full_sd[param_name].to(dtype=param_dtype, device=device).contiguous()
        else:
            tensor = torch.empty(global_shape, dtype=param_dtype, device=device)

        # Broadcast the FULL tensor (one param at a time → peak = one param's size)
        dist.broadcast(tensor, src=0)

        # Shard via distribute_tensor (handles _StridedShard for 2D FSDP2+TP).
        # Params stay on GPU here — we'll move to CPU after load_state_dict.
        if device_mesh is not None:
            sharded_tensor = distribute_tensor(tensor, device_mesh, placements)
            del tensor
        else:
            sharded_tensor = tensor

        sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)

    result = model.load_state_dict(sharded_sd, strict=False, assign=True)
    del sharded_sd
    torch.cuda.empty_cache()

    # --- Workaround for FSDP2 bug: reset_sharded_param doesn't move to CPU ---
    # _init_sharded_param checks offload_to_cpu and moves to CPU, but it skips
    # meta tensors. reset_sharded_param (fired by load_state_dict) only moves to
    # CPU when pin_memory=True. On GH200, pin_memory=True would consume GPU HBM
    # (unified memory), so we use pin_memory=False and manually move here.
    if cpu_offload:
        _offload_fsdp_params_to_cpu(model, is_rank0)

    return result


def _offload_fsdp_params_to_cpu(model: torch.nn.Module, verbose: bool = False) -> None:
    """Move all FSDP2-managed _sharded_param_data buffers from GPU to CPU.

    This fixes the gap where reset_sharded_param doesn't offload to CPU
    when pin_memory=False (PyTorch FSDP2 bug).
    """
    moved = 0
    n_modules = 0
    n_with_state = 0
    n_with_group = 0
    for module in model.modules():
        n_modules += 1
        fsdp_state = getattr(module, "_fsdp_state", None)
        if fsdp_state is None:
            continue
        n_with_state += 1
        param_group = getattr(fsdp_state, "_fsdp_param_group", None)
        if param_group is None:
            continue
        n_with_group += 1
        for fsdp_param in param_group.fsdp_params:
            data = fsdp_param._sharded_param_data
            if data is not None and data.device.type != "cpu":
                cpu_data = data.to("cpu")
                fsdp_param._sharded_param_data = cpu_data
                # Also update the DTensor's local tensor to point to CPU data
                if hasattr(fsdp_param.sharded_param, '_local_tensor'):
                    shard_dim = fsdp_param.fsdp_placement.dim
                    length = fsdp_param.sharded_param._local_tensor.size(shard_dim)
                    fsdp_param.sharded_param._local_tensor = cpu_data.view(
                        fsdp_param.padded_sharded_param_size
                    ).narrow(dim=shard_dim, start=0, length=length)
                moved += 1
    torch.cuda.empty_cache()
    if verbose:
        gpu_after = torch.cuda.memory_allocated() / 1024**3
        print(f"[torchtune] Offload: {n_modules} modules, {n_with_state} with _fsdp_state, "
              f"{n_with_group} with param_group, {moved} params moved to CPU, "
              f"GPU now {gpu_after:.1f} GB")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="torchtune FSDP2+TP SFT worker")
    parser.add_argument("--model",            required=True, help="HuggingFace model path")
    parser.add_argument("--train-file",       required=True, help="Parquet training file")
    parser.add_argument("--batch-size",       type=int, required=True, help="Global batch size")
    parser.add_argument("--micro-batch-size", type=int, required=True, help="Per-device batch size")
    parser.add_argument("--sequence-length",  type=int, required=True, help="Max sequence length")
    parser.add_argument("--num-steps",        type=int, required=True, help="Training steps")
    parser.add_argument("--num-gpus",         type=int, required=True, help="Total GPUs (world size)")
    parser.add_argument("--tp-degree",        type=int, default=1,     help="Tensor parallelism degree")
    parser.add_argument("--cpu-offload",      action="store_true",     help="Offload optimizer states to CPU (enables large models on fewer GPUs)")
    parser.add_argument("--metrics-file",     required=True, help="Path to write metrics JSON")
    parser.add_argument("--output-dir",       default="/tmp/sft_torchtune", help="Temp checkpoint dir")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Distributed init
    # -----------------------------------------------------------------------
    dp_mesh, tp_mesh, device = setup_distributed(args.tp_degree)
    rank = dist.get_rank()
    dp_rank = dp_mesh.get_local_rank()
    dp_size = dp_mesh.size()
    tp_size = tp_mesh.size()
    is_rank0 = (rank == 0)

    if is_rank0:
        print(f"[torchtune] world={dist.get_world_size()} dp={dp_size} tp={tp_size}")

    # -----------------------------------------------------------------------
    # Load model weights from HuggingFace checkpoint (rank 0 only)
    # -----------------------------------------------------------------------
    builder_fn, tp_plan_fn = get_model_builder(args.model)

    # FullModelHFCheckpointer needs a local directory path (not a HF model ID).
    # Resolve from HF cache; local_files_only=True avoids network calls.
    if is_rank0:
        print(f"[torchtune] Resolving local cache path for {args.model} ...")
    local_model_dir = snapshot_download(args.model, local_files_only=True)
    if is_rank0:
        print(f"[torchtune] Model cache: {local_model_dir}")

    # Only rank 0 loads the checkpoint — saves ~144 GB RAM on every other rank.
    # Weights are distributed to all ranks later via _load_from_rank0_broadcast.
    full_model_sd = {}
    if is_rank0:
        _model_lower = args.model.lower()
        _model_type_str = "QWEN2" if "qwen" in _model_lower else "LLAMA3"

        import re as _re
        _shard_files = sorted(
            f for f in os.listdir(local_model_dir)
            if f.endswith(".safetensors") and _re.search(r"\d+-of-\d+", f)
        )
        if _shard_files:
            _nums = _re.findall(r"\d+", _shard_files[-1])
            _ckpt_files = {
                "filename_format": _re.sub(r"\d+", "{}", _shard_files[0], count=2),
                "max_filename":    _nums[-1],
            }
        else:
            _ckpt_files = [f for f in os.listdir(local_model_dir) if f.endswith(".safetensors")]

        print(f"[torchtune] checkpoint_files={_ckpt_files}")
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=local_model_dir,
            checkpoint_files=_ckpt_files,
            output_dir=args.output_dir,
            model_type=_model_type_str,
        )
        full_model_sd = checkpointer.load_checkpoint()["model"]
        print(f"[torchtune] Checkpoint loaded on rank 0 ({len(full_model_sd)} tensors)")

    # Build model architecture (no weights yet — meta device for memory efficiency)
    with torch.device("meta"):
        model = builder_fn()

    # -----------------------------------------------------------------------
    # Apply Tensor Parallelism (before FSDP2, while model is on meta device)
    # -----------------------------------------------------------------------
    if tp_size > 1 and tp_plan_fn is not None:
        # Adjust attention heads for GQA under TP
        prepare_mha_for_tp(model, tp_mesh)
        # Apply TP sharding plan
        parallelize_module(model, tp_mesh, tp_plan_fn())
        if is_rank0:
            print(f"[torchtune] TP={tp_size} applied")

    # -----------------------------------------------------------------------
    # Apply FSDP2 (shards params + grads + optimizer states across dp_mesh)
    # -----------------------------------------------------------------------
    if is_rank0 and args.cpu_offload:
        print("[torchtune] CPU offload enabled — optimizer states will be on CPU")

    # Inline shard_model logic so we can pass pin_memory=False to CPUOffloadPolicy.
    # On GH200 unified memory, pinned CPU memory consumes GPU HBM, defeating
    # the purpose of CPU offload.
    fsdp_kwargs = {"reshard_after_forward": True, "mesh": dp_mesh}
    if args.cpu_offload:
        from torch.distributed.fsdp import CPUOffloadPolicy
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=False)

    for n, m in reversed(list(model.named_modules())):
        if get_shard_conditions(n, m):
            fully_shard(m, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # Initialize RoPE buffers on the actual device.
    # They are non-persistent buffers (not in the checkpoint) and stay on meta device
    # after shard_model. Must be initialised before loading weights.
    with torch.device(device):
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()
    if is_rank0:
        print(f"[torchtune] RoPE buffers initialised on {device}")

    # Distribute weights from rank 0 to all ranks.
    # Uses broadcast (not scatter) + local chunking — works correctly for any DTensor
    # dimensionality (1D FSDP2, 1D TP, or 2D FSDP2+TP) without needing DCP.
    # Must happen BEFORE activation checkpointing, which wraps layers in
    # _checkpoint_wrapped_module and changes parameter names.
    if is_rank0:
        print(f"[torchtune] Broadcasting checkpoint to all ranks ...")
    _load_from_rank0_broadcast(
        model, full_model_sd, device, rank,
        strict=False, cpu_offload=args.cpu_offload,
    )
    if is_rank0:
        gpu_alloc = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[torchtune] Weights loaded successfully")
        print(f"[torchtune] GPU memory after load: {gpu_alloc:.1f} GB allocated, {gpu_reserved:.1f} GB reserved")

    # Enable activation (gradient) checkpointing to trade compute for memory.
    # Essential for 72B: storing all 80 layers' activations without checkpointing
    # costs ~30-40 GB and causes OOM. Applied after weight loading because it
    # wraps layers in _checkpoint_wrapped_module, changing parameter names.
    from torchtune.modules import TransformerSelfAttentionLayer
    set_activation_checkpointing(model, auto_wrap_policy={TransformerSelfAttentionLayer})
    if is_rank0:
        print(f"[torchtune] Activation checkpointing enabled")

    # -----------------------------------------------------------------------
    # Tokenizer + dataset
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grad_accum_steps = max(1, args.batch_size // (dp_size * args.micro_batch_size))
    loader = build_dataset(args.train_file, tokenizer, args.sequence_length,
                           dp_rank, dp_size)

    # -----------------------------------------------------------------------
    # Optimizer + scheduler
    # -----------------------------------------------------------------------
    # Do NOT use fused=True: fused AdamW forces all optimizer states (m+v) onto GPU,
    # destroying the benefit of cpu_offload. Without fused, AdamW creates states on
    # the same device as params — with CPUOffloadPolicy, params are on CPU after each
    # backward, so optimizer states (72B×8B/8GPUs = 72 GB) stay on CPU.
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    data_iter = iter(loader)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Activation offloading: moves checkpointed activations to CPU during forward,
    # prefetches them back during backward. ~24% less GPU memory, <1% slower.
    # use_pin_memory=False is critical on GH200 (pinned CPU memory uses GPU HBM).
    act_offload_ctx = OffloadActivations(use_pin_memory=False) if args.cpu_offload else contextlib.nullcontext()
    if is_rank0 and args.cpu_offload:
        print("[torchtune] Activation offloading enabled (pin_memory=False)")

    step_times: list[float] = []
    losses: list[float] = []

    start_total = time.perf_counter()

    for step in range(args.num_steps):
        step_start = time.perf_counter()
        optimizer.zero_grad()

        accum_loss = torch.tensor(0.0, device=device)

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)         # (1, S)
            labels    = batch["labels"].to(device)            # (1, S)

            # Expand to micro_batch_size by repeating (simple but functional)
            input_ids = input_ids.expand(args.micro_batch_size, -1)
            labels    = labels.expand(args.micro_batch_size, -1)

            with act_offload_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)                     # (B, S, V)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        losses.append(accum_loss.item())

        if is_rank0 and (step % 10 == 0 or step == args.num_steps - 1):
            tokens_per_sec = (args.micro_batch_size * args.sequence_length *
                              dp_size * grad_accum_steps) / step_time
            print(f"[torchtune] step {step+1}/{args.num_steps} "
                  f"loss={accum_loss.item():.4f} "
                  f"step_time={step_time*1000:.0f}ms "
                  f"tok/s={tokens_per_sec:.0f}")

    total_time = time.perf_counter() - start_total

    # -----------------------------------------------------------------------
    # Write metrics (rank 0 only)
    # -----------------------------------------------------------------------
    if is_rank0:
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_sec = (args.micro_batch_size * args.sequence_length *
                          dp_size * grad_accum_steps) / avg_step_time
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "total_time_s": total_time,
            "avg_step_time_s": avg_step_time,
            "time_per_step_ms": avg_step_time * 1000,
            "tokens_per_second": tokens_per_sec,
            "samples_per_second": (dp_size * args.micro_batch_size * grad_accum_steps) / avg_step_time,
            "peak_memory_gb": peak_mem_gb,
            "num_steps": args.num_steps,
            "tp_degree": tp_size,
            "dp_degree": dp_size,
            "grad_accum_steps": grad_accum_steps,
            "loss_values": losses[-5:],
            "metrics_source": "torchtune",
        }

        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[torchtune] done. avg_step={avg_step_time*1000:.0f}ms "
              f"tok/s={tokens_per_sec:.0f} peak_mem={peak_mem_gb:.1f}GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
