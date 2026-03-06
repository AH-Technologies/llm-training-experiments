"""
Memory Estimation for LLM Training (SFT & GRPO)

Implements the formulas from MEMORY_ESTIMATION_METHOD.md to estimate peak GPU
and CPU memory across model sizes.

Supports two parallelism backends:
- **Megatron**: tensor parallelism (TP) shards both parameters AND activations
- **FSDP**: shards model states across GPUs, activations NOT sharded

Usage:
    python -m benchmarks.estimate_memory [OPTIONS]

    --num-gpus 4          Total GPUs (default: 4)
    --tp-degree auto      Tensor parallelism degree, or 'auto' (default: auto)
    --micro-batch-size 4  Per-GPU micro-batch size (default: 4)
    --seq-length 4096     Total sequence length (default: 4096)
    --vocab-size 152064   Vocabulary size (default: 152064)
    --alpha 0.15          Overhead factor (default: 0.15)
    --gpu-memory 96       GPU HBM limit in GB (default: 96)
    --cpu-memory 202      CPU memory limit per GPU in GB (default: 202)
    --backend megatron    Backend: 'fsdp' or 'megatron' (default: megatron)
    --output PATH         Output PNG path
    --min-params 0.5      Minimum model size in billions (default: 0.5)
    --max-params 1000     Maximum model size in billions (default: 1000)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelArch:
    """Transformer architecture parameters."""
    P: float   # Total parameters (raw count, not billions)
    h: int     # Hidden dimension
    L: int     # Number of layers
    a: int     # Attention heads
    a_kv: int  # KV heads (GQA)
    d_ff: int  # FFN intermediate size
    V: int     # Vocabulary size

    @property
    def P_billions(self) -> float:
        return self.P / 1e9


# Verified model configs from HuggingFace (Feb 2026)
REFERENCE_MODELS: dict[str, ModelArch] = {
    "Qwen2.5-0.5B":  ModelArch(P=0.49e9,  h=896,   L=24,  a=14,  a_kv=2,  d_ff=4864,   V=151936),
    "Qwen2.5-1.5B":  ModelArch(P=1.54e9,  h=1536,  L=28,  a=12,  a_kv=2,  d_ff=8960,   V=151936),
    "Qwen2.5-3B":    ModelArch(P=3.09e9,  h=2048,  L=36,  a=16,  a_kv=2,  d_ff=11008,  V=151936),
    "Qwen2.5-7B":    ModelArch(P=7.62e9,  h=3584,  L=28,  a=28,  a_kv=4,  d_ff=18944,  V=152064),
    "Qwen2.5-14B":   ModelArch(P=14.77e9, h=5120,  L=48,  a=40,  a_kv=8,  d_ff=13824,  V=152064),
    "Qwen2.5-32B":   ModelArch(P=32.76e9, h=5120,  L=64,  a=40,  a_kv=8,  d_ff=27648,  V=152064),
    "Qwen2.5-72B":   ModelArch(P=72.71e9, h=8192,  L=80,  a=64,  a_kv=8,  d_ff=29568,  V=152064),
    "Llama3-1B":      ModelArch(P=1.24e9,  h=2048,  L=16,  a=32,  a_kv=8,  d_ff=8192,   V=128256),
    "Llama3-3B":      ModelArch(P=3.21e9,  h=3072,  L=28,  a=24,  a_kv=8,  d_ff=8192,   V=128256),
    "Llama3-8B":      ModelArch(P=8.03e9,  h=4096,  L=32,  a=32,  a_kv=8,  d_ff=14336,  V=128256),
    "Llama3-70B":     ModelArch(P=70.55e9, h=8192,  L=80,  a=64,  a_kv=8,  d_ff=28672,  V=128256),
    "Llama3-405B":    ModelArch(P=405.0e9, h=16384, L=126, a=128, a_kv=8,  d_ff=53248,  V=128256),
}


def _round_to_multiple(x: float, multiple: int = 256) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def estimate_arch(P_billions: float, V: int = 152064) -> ModelArch:
    """Estimate architecture from parameter count using scaling heuristics."""
    h = _round_to_multiple(1500 * P_billions ** 0.4, 128)
    L = max(4, round(12 * P_billions ** 0.35))
    d_head = 128 if P_billions >= 3 else 64
    a = max(1, h // d_head)
    a_kv = max(2, min(8, a // 8))
    d_ff = _round_to_multiple(h * 16 / 3, 256)  # SwiGLU: 8h/3 * 2
    P = P_billions * 1e9
    return ModelArch(P=P, h=h, L=L, a=a, a_kv=a_kv, d_ff=d_ff, V=V)


def get_arch(P_billions: float, V: int = 152064) -> ModelArch:
    """Look up reference model or estimate architecture."""
    # Check if close to a known model
    for name, arch in REFERENCE_MODELS.items():
        if abs(arch.P_billions - P_billions) / max(P_billions, 0.1) < 0.15:
            return arch
    return estimate_arch(P_billions, V)


# ---------------------------------------------------------------------------
# Memory estimation building blocks (all return bytes)
# ---------------------------------------------------------------------------

BYTES_BF16 = 2
BYTES_FP32 = 4
CONST_OVERHEAD_BYTES = 0.5 * 1e9  # 500 MB constant framework overhead
GPUS_PER_NODE = 4  # GH200 nodes have 4 GPUs
DATALOADER_WORKER_OVERHEAD_BYTES = 0.5 * 1e9  # ~0.5 GB per dataloader worker
DATALOADER_WORKERS_PER_RANK = 2  # default in our SFTConfig
OS_BASELINE_BYTES = 150 * 1e9  # ~150 GB OS/kernel/runtime baseline (measured ~147 GB in job logs)
DS_COMM_BUFFER_PER_GPU = 4 * 1e9  # ~4 GB per GPU: reduce_bucket + prefetch_bucket, double-buffered


def to_gb(b: float) -> float:
    return b / (1024 ** 3)


def activation_memory(B: int, S: int, h: int, L: int, a: int) -> float:
    """Eq. 3: Activation memory with gradient checkpointing (per-layer recomp peak)."""
    # A_checkpoint = BSh * (2L + 64 + 4aS/h)
    return B * S * h * (2 * L + 64 + 4 * a * S / h) * BYTES_BF16


def logits_memory(B: int, S: int, V: int) -> float:
    """Eq. 4a: Logits tensor in bf16."""
    return 2 * B * S * V


def overhead(model_states: float, activations: float, alpha: float = 0.15) -> float:
    """Eq. 4: Fragmentation + framework overhead."""
    return alpha * (model_states + activations) + CONST_OVERHEAD_BYTES


def fsdp_unit_params(arch: ModelArch) -> int:
    """Largest single FSDP unit ≈ one transformer layer.

    Accounts for GQA (Q proj = h², K/V proj = h*h*a_kv/a each)
    and SwiGLU FFN (gate + up + down = 3 * h * d_ff).
    """
    h, a, a_kv, d_ff = arch.h, arch.a, arch.a_kv, arch.d_ff
    # Attention: Q + K + V projections + output projection
    qkv = h * h + 2 * h * (h * a_kv // a)  # Q is h², K and V are h*(h*a_kv/a)
    attn_out = h * h
    # SwiGLU FFN: gate_proj + up_proj + down_proj
    ffn = 3 * h * d_ff
    # LayerNorms (2 per layer, each h params) — small but included
    ln = 2 * h
    return qkv + attn_out + ffn + ln


# ---------------------------------------------------------------------------
# CPU per-node estimation (all SFT/GRPO modes)
# ---------------------------------------------------------------------------

def cpu_per_node_sft(P: float, N: int, mode: str = "full_offload",
                     gpus_per_node: int = GPUS_PER_NODE) -> dict:
    """Estimate total CPU memory per node for SFT training.

    Accounts for all costs that the per-GPU formula (Eq. 21) misses:
    - Optimizer states (per-GPU shard, summed across GPUs on node)
    - Parameters on CPU (if full offload)
    - Gradient buffers during optimizer step
    - Per-process Python/PyTorch/CUDA overhead
    - Dataloader worker memory
    - Model loading transient peak (all ranks load simultaneously)

    Args:
        P: Total model parameters (raw count)
        N: Total number of GPUs across all nodes
        mode: "no_offload", "optim_offload", or "full_offload"
        gpus_per_node: GPUs per node (default 4)

    Returns:
        Dict with itemized CPU memory breakdown (all in GB).
    """
    gpus_on_node = gpus_per_node
    params_on_node = P * gpus_on_node / N  # total params assigned to this node's GPUs

    P_billions = P / 1e9

    if mode == "no_offload":
        # No offload: CPU only holds process overhead + dataloader
        cpu_optimizer = 0.0
        cpu_params = 0.0
        cpu_grads = 0.0
    elif mode == "optim_offload":
        # Optimizer offload: master weights + momentum + variance on CPU
        cpu_optimizer = params_on_node * 12  # 3 × fp32 per param
        cpu_params = 0.0  # params stay on GPU
        # DeepSpeed CPU Adam casts grads to fp32 for the optimizer step
        cpu_grads = params_on_node * 4  # fp32 grad buffers
    else:  # full_offload
        # Full offload: optimizer states + params + grads all on CPU
        cpu_optimizer = params_on_node * 12  # 3 × fp32 per param
        cpu_params = params_on_node * 2  # bf16 params offloaded
        # DeepSpeed CPU Adam casts grads to fp32 for the optimizer step
        cpu_grads = params_on_node * 4  # fp32 grad buffers

    # DeepSpeed communication buffers: reduce_bucket + prefetch_bucket, double-buffered
    # Fixed-size buckets per GPU (default "auto" = h² elements each, typically ~2 GB combined, ×2 for double-buffering)
    ds_buffers = gpus_on_node * DS_COMM_BUFFER_PER_GPU if mode != "no_offload" else 0.0

    # Per-process overhead: Python runtime, PyTorch allocator, CUDA context
    # Scales slightly with model size (larger models = more metadata, param groups, etc.)
    cpu_process_overhead = (2.0e9 + 0.01e9 * P_billions) * gpus_on_node

    # Dataloader workers: each rank spawns workers
    cpu_dataloader = gpus_on_node * DATALOADER_WORKERS_PER_RANK * DATALOADER_WORKER_OVERHEAD_BYTES

    # OS baseline: kernel, system services, runtime libraries (~147 GB measured in job logs)
    os_baseline = OS_BASELINE_BYTES

    # Model loading transient: with low_cpu_mem_usage=True, HF loads weight
    # shards sequentially but each rank does this independently.
    # DeepSpeed ZeRO-3 init is greedier than FSDP (75% vs 50% of full model per rank)
    loading_fraction = 0.75 if mode != "no_offload" else 0.50
    cpu_loading_peak = gpus_on_node * P * 2 * loading_fraction

    # Steady-state total
    cpu_steady = (cpu_optimizer + cpu_params + cpu_grads + ds_buffers
                  + cpu_process_overhead + cpu_dataloader + os_baseline)

    # Peak = max of steady-state and loading transient + process overhead
    cpu_peak = max(cpu_steady,
                   cpu_loading_peak + cpu_process_overhead + cpu_dataloader + os_baseline)

    return {
        "optimizer_gb": to_gb(cpu_optimizer),
        "params_gb": to_gb(cpu_params),
        "grads_gb": to_gb(cpu_grads),
        "ds_buffers_gb": to_gb(ds_buffers),
        "process_overhead_gb": to_gb(cpu_process_overhead),
        "os_baseline_gb": to_gb(os_baseline),
        "dataloader_gb": to_gb(cpu_dataloader),
        "loading_peak_gb": to_gb(cpu_loading_peak),
        "steady_gb": to_gb(cpu_steady),
        "peak_gb": to_gb(cpu_peak),
    }


def cpu_per_node_grpo(P: float, N: int, gpus_per_node: int = GPUS_PER_NODE) -> dict:
    """Estimate total CPU memory per node for GRPO training.

    GRPO offloads optimizer states + full reference model to CPU.

    Args:
        P: Total model parameters
        N: Total GPUs
        gpus_per_node: GPUs per node
    """
    gpus_on_node = gpus_per_node
    params_on_node = P * gpus_on_node / N
    P_billions = P / 1e9

    # Actor optimizer states (sharded)
    cpu_optimizer = params_on_node * 12

    # Actor gradient buffers (fp32 for CPU Adam)
    cpu_grads = params_on_node * 4

    # Reference model (full, not sharded — each GPU streams from CPU)
    # One copy per node (shared across GPUs on the node)
    # Conservative: full copy per node
    cpu_ref_model = P * 2

    # DeepSpeed communication buffers (fixed-size buckets per GPU)
    ds_buffers = gpus_on_node * DS_COMM_BUFFER_PER_GPU

    # Per-process overhead (model-size-dependent)
    cpu_process_overhead = (2.0e9 + 0.01e9 * P_billions) * gpus_on_node

    cpu_dataloader = gpus_on_node * DATALOADER_WORKERS_PER_RANK * DATALOADER_WORKER_OVERHEAD_BYTES

    # OS baseline
    os_baseline = OS_BASELINE_BYTES

    cpu_steady = (cpu_optimizer + cpu_grads + cpu_ref_model + ds_buffers
                  + cpu_process_overhead + cpu_dataloader + os_baseline)

    return {
        "optimizer_gb": to_gb(cpu_optimizer),
        "grads_gb": to_gb(cpu_grads),
        "ref_model_gb": to_gb(cpu_ref_model),
        "ds_buffers_gb": to_gb(ds_buffers),
        "process_overhead_gb": to_gb(cpu_process_overhead),
        "os_baseline_gb": to_gb(os_baseline),
        "dataloader_gb": to_gb(cpu_dataloader),
        "steady_gb": to_gb(cpu_steady),
        "peak_gb": to_gb(cpu_steady),  # GRPO loads models via Ray, less transient spike
    }


# ---------------------------------------------------------------------------
# SFT estimators
# ---------------------------------------------------------------------------

def sft_no_fsdp(arch: ModelArch, B: int, S: int, alpha: float = 0.15) -> dict:
    """Eq. 5: SFT on single GPU, no sharding."""
    P = arch.P
    states = 16 * P
    act = activation_memory(B, S, arch.h, arch.L, arch.a)
    logits = logits_memory(B, S, arch.V)
    oh = overhead(states, act, alpha)
    gpu_peak = states + act + logits + oh
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": 0.0,
            "cpu_per_node": cpu_per_node_sft(P, 1, "no_offload", gpus_per_node=1)}


def sft_fsdp(arch: ModelArch, B: int, S: int, N: int, T: int = 1,
             alpha: float = 0.15) -> dict:
    """Eq. 8: SFT with FSDP, no offloading. T = tensor parallelism degree."""
    P = arch.P
    states = 16 * P / N
    unit_peak = 2 * fsdp_unit_params(arch) * BYTES_BF16  # Eq. 7
    act = activation_memory(B, S, arch.h, arch.L, arch.a) / T
    logits = logits_memory(B, S, arch.V) / T
    oh = overhead(states, act, alpha)
    gpu_peak = states + act + unit_peak + logits + oh
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": 0.0,
            "cpu_per_node": cpu_per_node_sft(P, N, "no_offload")}


def sft_fsdp_optim_offload(arch: ModelArch, B: int, S: int, N: int, T: int = 1,
                            alpha: float = 0.15) -> dict:
    """Eq. 9: SFT with FSDP + optimizer offload to CPU. T = tensor parallelism degree."""
    P = arch.P
    gpu_states = 4 * P / N  # bf16 weights + bf16 grads
    unit_peak = 2 * fsdp_unit_params(arch) * BYTES_BF16
    act = activation_memory(B, S, arch.h, arch.L, arch.a) / T
    logits = logits_memory(B, S, arch.V) / T
    oh = overhead(gpu_states, act, alpha)
    gpu_peak = gpu_states + act + unit_peak + logits + oh
    cpu_offloaded = 12 * P / N  # master weights + momentum + variance
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": to_gb(cpu_offloaded),
            "cpu_per_node": cpu_per_node_sft(P, N, "optim_offload")}


def sft_fsdp_full_offload(arch: ModelArch, B: int, S: int, N: int, T: int = 1,
                           alpha: float = 0.15) -> dict:
    """Eq. 10: SFT with FSDP + full param & optimizer offload. T = tensor parallelism degree."""
    unit_peak = 2 * fsdp_unit_params(arch) * BYTES_BF16
    act = activation_memory(B, S, arch.h, arch.L, arch.a) / T
    logits = logits_memory(B, S, arch.V) / T
    # grad buffer for current unit
    grad_buf = fsdp_unit_params(arch) * BYTES_BF16
    oh = overhead(unit_peak, act, alpha)
    gpu_peak = unit_peak + act + grad_buf + logits + oh
    cpu_offloaded = 14 * arch.P / N  # 12 (optim) + 2 (params) per shard
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": to_gb(cpu_offloaded),
            "cpu_per_node": cpu_per_node_sft(arch.P, N, "full_offload")}


# ---------------------------------------------------------------------------
# Megatron SFT estimators (TP + PP)
# ---------------------------------------------------------------------------

def cpu_per_node_megatron(P: float, T: int, D: int, N: int,
                          mode: str = "optim_offload",
                          has_ref_model: bool = False,
                          gpus_per_node: int = GPUS_PER_NODE) -> dict:
    """Estimate total CPU memory per node for Megatron-style TP+PP training.

    Unlike FSDP/DeepSpeed, Megatron does not use DS communication buffers.
    Each GPU holds a TP×PP shard of parameters; with distributed optimizer,
    optimizer states are sharded across DP ranks.

    Args:
        P: Total model parameters
        T: Tensor parallelism degree
        D: Pipeline parallelism degree
        N: Total GPUs
        mode: "no_offload", "optim_offload", or "full_offload"
        has_ref_model: True for GRPO (adds ref model on CPU)
        gpus_per_node: GPUs per node
    """
    gpus_on_node = gpus_per_node
    DP = max(1, N // (T * D))
    P_billions = P / 1e9

    # Each GPU holds a TP×PP shard: P / (T * D) params
    # With distributed optimizer, optimizer states are sharded across DP
    params_per_gpu = P / (T * D)

    if mode == "no_offload":
        cpu_optimizer = 0.0
        cpu_params = 0.0
        cpu_grads = 0.0
    elif mode == "optim_offload":
        # Optimizer offloaded: 12 bytes/param, sharded by DP
        cpu_optimizer = gpus_on_node * 12 * params_per_gpu / DP
        cpu_params = 0.0
        cpu_grads = gpus_on_node * 4 * params_per_gpu / DP  # fp32 grad copy for CPU Adam
    else:  # full_offload
        cpu_optimizer = gpus_on_node * 12 * params_per_gpu / DP
        cpu_params = gpus_on_node * 2 * params_per_gpu  # bf16 params offloaded
        cpu_grads = gpus_on_node * 4 * params_per_gpu / DP

    # Reference model (GRPO only) — full copy per node
    cpu_ref_model = P * 2 if has_ref_model else 0.0

    # No DeepSpeed communication buffers for Megatron
    # Megatron uses NCCL all-reduce directly, buffers are smaller
    nccl_buffers = gpus_on_node * 0.5 * 1e9  # ~0.5 GB per GPU for NCCL

    # Per-process overhead (each process only handles its PP stage)
    P_stage_billions = P_billions / D
    cpu_process_overhead = (2.0e9 + 0.01e9 * P_stage_billions) * gpus_on_node

    cpu_dataloader = gpus_on_node * DATALOADER_WORKERS_PER_RANK * DATALOADER_WORKER_OVERHEAD_BYTES

    os_baseline = OS_BASELINE_BYTES

    # Model loading: each rank loads its PP stage
    # With PP, each rank only loads P/D params (~50% overhead for conversion)
    cpu_loading_peak = gpus_on_node * (P / D) * 2 * 0.50

    cpu_steady = (cpu_optimizer + cpu_params + cpu_grads + cpu_ref_model
                  + nccl_buffers + cpu_process_overhead + cpu_dataloader + os_baseline)

    cpu_peak = max(cpu_steady,
                   cpu_loading_peak + cpu_process_overhead + cpu_dataloader + os_baseline)

    result = {
        "optimizer_gb": to_gb(cpu_optimizer),
        "params_gb": to_gb(cpu_params),
        "grads_gb": to_gb(cpu_grads),
        "nccl_buffers_gb": to_gb(nccl_buffers),
        "process_overhead_gb": to_gb(cpu_process_overhead),
        "os_baseline_gb": to_gb(os_baseline),
        "dataloader_gb": to_gb(cpu_dataloader),
        "loading_peak_gb": to_gb(cpu_loading_peak),
        "steady_gb": to_gb(cpu_steady),
        "peak_gb": to_gb(cpu_peak),
    }
    if has_ref_model:
        result["ref_model_gb"] = to_gb(cpu_ref_model)
    return result


def sft_megatron(arch: ModelArch, B: int, S: int, N: int,
                 T: int = 2, D: int = 1,
                 param_offload: bool = True, optimizer_offload: bool = True,
                 alpha: float = 0.15) -> dict:
    """Megatron SFT with TP+PP and optional CPU offloading.

    With Megatron, parameters AND activations are split by TP×PP.
    No FSDP allgather peak — params stay in their TP shard.

    Args:
        arch: Model architecture
        B: Micro-batch size per GPU
        S: Sequence length
        N: Total GPUs
        T: Tensor parallelism degree
        D: Pipeline parallelism degree
        param_offload: Offload params to CPU between compute
        optimizer_offload: Offload optimizer states to CPU
        alpha: Overhead factor
    """
    P = arch.P
    DP = max(1, N // (T * D))

    # Activations split by TP; with PP each stage runs L/D layers
    # Pass L//D so only the layer-count term scales, not the per-layer recompute peak
    act = activation_memory(B, S, arch.h, arch.L // D, arch.a) / T
    logits = logits_memory(B, S, arch.V) / T

    if param_offload and optimizer_offload:
        # Full offload: only current layer params + activations on GPU
        # Megatron streams TP-sharded layer params from CPU
        layer_params = fsdp_unit_params(arch) / T  # one layer's TP shard
        gpu_states = 2 * layer_params * BYTES_BF16  # bf16 params for current layer
        mode = "full_offload"
    elif optimizer_offload:
        # Optimizer offload: bf16 weights + grads on GPU, optimizer on CPU
        gpu_states = 4 * P / (T * D)  # weights + grads, TP×PP sharded
        mode = "optim_offload"
    else:
        # No offload: all model states on GPU
        gpu_states = 16 * P / (T * D)
        mode = "no_offload"

    oh = overhead(gpu_states, act, alpha)
    gpu_peak = gpu_states + act + logits + oh

    cpu_offloaded = 0.0
    if optimizer_offload:
        cpu_offloaded += 12 * P / (T * D) / DP  # optimizer states
    if param_offload:
        cpu_offloaded += 2 * P / (T * D)  # bf16 params

    return {
        "gpu_peak_gb": to_gb(gpu_peak),
        "cpu_offloaded_gb": to_gb(cpu_offloaded),
        "cpu_per_node": cpu_per_node_megatron(P, T, D, N, mode),
    }


# ---------------------------------------------------------------------------
# GRPO estimators (both FSDP and Megatron)
# ---------------------------------------------------------------------------

def grpo_phases(arch: ModelArch, B_micro: int, S: int, N: int, T: int = 1,
                gpu_mem_util: float = 0.7, gpu_total_gb: float = 96.0,
                alpha: float = 0.15) -> dict:
    """
    Estimate per-phase GPU memory for GRPO.

    Returns dict with per-phase GPU memory and overall peak + CPU offloaded.
    Assumes: optimizer offload for actor, ref model offloaded to CPU.
    """
    P = arch.P
    h, L, a, a_kv, V = arch.h, arch.L, arch.a, arch.a_kv, arch.V

    # --- Phase 1: Rollout (vLLM) --- Eq. 12d
    vllm_weights_per_gpu = 2 * P / T
    vllm_overhead = 2 * 1e9  # ~2 GB framework overhead
    # KV cache: what remains after weights + overhead
    available_for_kv = gpu_mem_util * gpu_total_gb * 1e9 - vllm_weights_per_gpu - vllm_overhead
    kv_cache = max(0, available_for_kv)
    rollout_gpu = (vllm_weights_per_gpu + kv_cache + vllm_overhead)
    # Simpler: rollout uses gpu_mem_util fraction + overhead for non-KV
    # But Eq. 12d says: (2P + KV_cache) / T + overhead
    # Since KV fills remaining space, effective usage ≈ gpu_mem_util * total
    rollout_gpu = max(vllm_weights_per_gpu + vllm_overhead,
                      gpu_mem_util * gpu_total_gb * 1e9)

    # --- Phase 2: Reference log-probs (offloaded) --- Eq. 13b / 15b
    unit_peak = 2 * fsdp_unit_params(arch) * BYTES_BF16
    ref_act = B_micro * S * h * BYTES_BF16  # forward-only activations
    ref_logits = logits_memory(B_micro, S, V)
    ref_overhead = 1 * 1e9  # ~1 GB
    ref_gpu = unit_peak + ref_act + ref_logits + ref_overhead

    # --- Phase 3: Actor update (FSDP + optim offload) --- Eq. 15c
    actor_gpu_states = 4 * P / N  # bf16 weights + grads, FSDP-sharded
    actor_act = activation_memory(B_micro, S, h, L, a)
    actor_logits = logits_memory(B_micro, S, V)
    actor_unit_peak = unit_peak
    actor_oh = overhead(actor_gpu_states, actor_act, alpha)
    actor_gpu = actor_gpu_states + actor_act + actor_unit_peak + actor_logits + actor_oh

    # --- Weight sync spike --- Eq. 18
    fsdp_shard = 2 * P / N
    vllm_copy = 2 * P / T
    allgather_buf = 2 * fsdp_unit_params(arch) * BYTES_BF16  # one unit buffer
    weight_sync_gpu = fsdp_shard + vllm_copy + allgather_buf

    # --- Peak (Eq. 19) ---
    gpu_peak = max(rollout_gpu, ref_gpu, actor_gpu, weight_sync_gpu)

    # --- CPU offloaded (Eq. 21c): optimizer states + full ref model ---
    cpu_offloaded = 12 * P / N + 2 * P

    return {
        "rollout_gpu_gb": to_gb(rollout_gpu),
        "ref_gpu_gb": to_gb(ref_gpu),
        "actor_gpu_gb": to_gb(actor_gpu),
        "weight_sync_gpu_gb": to_gb(weight_sync_gpu),
        "gpu_peak_gb": to_gb(gpu_peak),
        "cpu_offloaded_gb": to_gb(cpu_offloaded),
        "cpu_per_node": cpu_per_node_grpo(P, N),
    }


def grpo_megatron_phases(arch: ModelArch, B_micro: int, S: int, N: int,
                         T: int = 2, D: int = 1,
                         gpu_mem_util: float = 0.7, gpu_total_gb: float = 96.0,
                         alpha: float = 0.15) -> dict:
    """Estimate per-phase GPU memory for GRPO with Megatron-style actor.

    Key difference from FSDP: actor training uses TP+PP, so activations
    are split by T×D. Rollout still uses vLLM with TP.

    Args:
        arch: Model architecture
        B_micro: Micro-batch size per GPU
        S: Sequence length
        N: Total GPUs
        T: Tensor parallelism degree
        D: Pipeline parallelism degree
        gpu_mem_util: vLLM GPU memory utilization
        gpu_total_gb: Total GPU memory in GB
        alpha: Overhead factor
    """
    P = arch.P
    h, L, a, a_kv, V = arch.h, arch.L, arch.a, arch.a_kv, arch.V
    DP = max(1, N // (T * D))

    # --- Phase 1: Rollout (vLLM with TP) --- same as FSDP version
    vllm_weights_per_gpu = 2 * P / T
    vllm_overhead = 2 * 1e9
    rollout_gpu = max(vllm_weights_per_gpu + vllm_overhead,
                      gpu_mem_util * gpu_total_gb * 1e9)

    # --- Phase 2: Reference log-probs (offloaded to CPU) ---
    # With Megatron, ref can also use TP sharding for forward pass
    ref_act = B_micro * S * h * BYTES_BF16 / T  # forward-only, TP-sharded
    ref_logits = logits_memory(B_micro, S, V) / T
    ref_overhead = 1 * 1e9
    # Ref model weights streamed from CPU, one TP-sharded layer at a time
    layer_params_tp = fsdp_unit_params(arch) / T
    ref_layer_peak = 2 * layer_params_tp * BYTES_BF16
    ref_gpu = ref_layer_peak + ref_act + ref_logits + ref_overhead

    # --- Phase 3: Actor update (Megatron TP+PP + optimizer offload) ---
    # With Megatron, each GPU holds P/(T*D) params (TP×PP shard)
    # Optimizer offloaded to CPU; GPU has weights + grads
    actor_gpu_states = 4 * P / (T * D)  # bf16 weights + grads
    actor_act = activation_memory(B_micro, S, h, L // D, a) / T
    actor_logits = logits_memory(B_micro, S, V) / T
    actor_oh = overhead(actor_gpu_states, actor_act, alpha)
    actor_gpu = actor_gpu_states + actor_act + actor_logits + actor_oh

    # --- Weight sync (actor → vLLM) ---
    # Megatron actor params are TP-sharded; vLLM also uses same TP degree
    # So weight sync is TP-shard to TP-shard — no allgather needed
    # Both hold P/T params per GPU, sync is in-place or via buffer
    actor_shard = 2 * P / (T * D)
    vllm_shard = 2 * P / T
    sync_buffer = 0.5 * 1e9  # small staging buffer
    weight_sync_gpu = max(actor_shard, vllm_shard) + sync_buffer

    # --- Peak ---
    gpu_peak = max(rollout_gpu, ref_gpu, actor_gpu, weight_sync_gpu)

    # --- CPU offloaded: optimizer states + ref model ---
    cpu_offloaded = 12 * P / (T * D) / DP + 2 * P  # optimizer (sharded by DP) + full ref

    return {
        "rollout_gpu_gb": to_gb(rollout_gpu),
        "ref_gpu_gb": to_gb(ref_gpu),
        "actor_gpu_gb": to_gb(actor_gpu),
        "weight_sync_gpu_gb": to_gb(weight_sync_gpu),
        "gpu_peak_gb": to_gb(gpu_peak),
        "cpu_offloaded_gb": to_gb(cpu_offloaded),
        "cpu_per_node": cpu_per_node_megatron(P, T, D, N, "optim_offload",
                                               has_ref_model=True),
    }


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def fits(gpu_peak_gb: float, cpu_offloaded_gb: float,
         gpu_limit: float = 96.0, cpu_limit: float = 202.0,
         cpu_per_node: dict | None = None, cpu_node_limit: float = 808.0) -> bool:
    if not (gpu_peak_gb <= gpu_limit and cpu_offloaded_gb <= cpu_limit):
        return False
    if cpu_per_node is not None:
        return cpu_per_node["peak_gb"] <= cpu_node_limit
    return True


def min_tp_degree(arch: ModelArch, N: int, B_micro: int, S: int,
                  gpu_mem_util: float = 0.7, gpu_total_gb: float = 96.0,
                  cpu_limit: float = 202.0, alpha: float = 0.15,
                  candidates: list[int] | None = None,
                  backend: str = "megatron",
                  cpu_node_limit: float = 808.0,
                  gpus_per_node: int = GPUS_PER_NODE) -> int | None:
    """Find the minimum TP degree from candidates that makes GRPO fit.

    Returns the smallest T where GPU peak, per-GPU CPU offload, AND
    per-node CPU all fit within limits. Returns None if no candidate works.

    Args:
        backend: "fsdp" or "megatron" — selects which GRPO estimator to use.
        cpu_node_limit: Total CPU memory per node in GB (default: 808).
        gpus_per_node: GPUs per node — TP candidates capped here (default: 4).
    """
    if candidates is None:
        candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node]
    for T in candidates:
        if T > N:
            break
        if backend == "megatron":
            est = grpo_megatron_phases(arch, B_micro, S, N, T,
                                       gpu_mem_util=gpu_mem_util,
                                       gpu_total_gb=gpu_total_gb, alpha=alpha)
        else:
            est = grpo_phases(arch, B_micro, S, N, T,
                              gpu_mem_util=gpu_mem_util,
                              gpu_total_gb=gpu_total_gb, alpha=alpha)
        if fits(est["gpu_peak_gb"], est["cpu_offloaded_gb"],
                gpu_total_gb, cpu_limit,
                cpu_per_node=est.get("cpu_per_node"),
                cpu_node_limit=cpu_node_limit):
            return T
    return None


def min_tp_pp_degree(arch: ModelArch, N: int, B_micro: int, S: int,
                     gpu_total_gb: float = 96.0, cpu_limit: float = 202.0,
                     alpha: float = 0.15,
                     cpu_node_limit: float = 808.0,
                     gpus_per_node: int = GPUS_PER_NODE,
                     mode: str = "grpo") -> tuple[int, int] | None:
    """Find minimum (T, D) that fits for Megatron training.

    Searches TP within-node (T ≤ gpus_per_node) then PP across nodes.
    Prefers lower D (less pipeline bubbles) and lower T.

    Args:
        mode: "sft" or "grpo" — selects which estimator to use.

    Returns (T, D) or None if nothing fits.
    """
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node]
    num_nodes = max(1, N // gpus_per_node)

    # Build PP candidates: powers of 2 up to num_nodes
    pp_candidates = []
    d = 1
    while d <= num_nodes:
        pp_candidates.append(d)
        d *= 2

    # Search: prefer lower D first (less pipeline bubbles), then lower T
    for D in pp_candidates:
        for T in tp_candidates:
            if T * D > N:
                continue
            DP = N // (T * D)
            if DP < 1:
                continue
            if mode == "sft":
                est = sft_megatron(arch, B_micro, S, N, T=T, D=D,
                                   param_offload=False, optimizer_offload=True,
                                   alpha=alpha)
            else:
                est = grpo_megatron_phases(arch, B_micro, S, N, T, D=D,
                                           gpu_total_gb=gpu_total_gb,
                                           alpha=alpha)
            if fits(est["gpu_peak_gb"], est.get("cpu_offloaded_gb", 0.0),
                    gpu_total_gb, cpu_limit,
                    cpu_per_node=est.get("cpu_per_node"),
                    cpu_node_limit=cpu_node_limit):
                return T, D
    return None


def binding_constraint(gpu_peak_gb: float, cpu_offloaded_gb: float,
                       gpu_limit: float = 96.0, cpu_limit: float = 202.0) -> str:
    gpu_ok = gpu_peak_gb <= gpu_limit
    cpu_ok = cpu_offloaded_gb <= cpu_limit
    if gpu_ok and cpu_ok:
        return "fits"
    if not gpu_ok and not cpu_ok:
        return "both"
    return "gpu" if not gpu_ok else "cpu"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_COLORS = {
    "gpu": "#2563eb",
    "cpu": "#ea580c",
    "gpu_limit": "#dc2626",
    "cpu_limit": "#7c3aed",
    "rollout": "#2563eb",
    "ref": "#16a34a",
    "actor": "#dc2626",
    "weight_sync": "#ea580c",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.15, linestyle="--")


def _add_limits(ax: plt.Axes, gpu_limit: float, cpu_limit: float,
                xlim: tuple[float, float]) -> None:
    """Draw horizontal limit lines and shaded forbidden regions.

    gpu_limit: per-GPU HBM limit (GB)
    cpu_limit: per-node CPU RAM limit (GB)
    """
    ax.axhline(gpu_limit, color=PLOT_COLORS["gpu_limit"], linestyle="--",
               linewidth=1.2, label=f"GPU limit ({gpu_limit:.0f} GB/gpu)")
    ax.axhline(cpu_limit, color=PLOT_COLORS["cpu_limit"], linestyle="--",
               linewidth=1.2, label=f"CPU limit ({cpu_limit:.0f} GB/node)")
    upper = max(gpu_limit, cpu_limit) * 1.5
    ax.fill_between(xlim, gpu_limit, upper,
                    color=PLOT_COLORS["gpu_limit"], alpha=0.06)
    ax.fill_between(xlim, cpu_limit, upper,
                    color=PLOT_COLORS["cpu_limit"], alpha=0.06)


def _annotate_crossing(ax: plt.Axes, x_vals: np.ndarray, y_vals: np.ndarray,
                       limit: float, label: str, color: str) -> None:
    """Annotate where a line crosses a limit."""
    for i in range(len(y_vals) - 1):
        if y_vals[i] <= limit < y_vals[i + 1]:
            # Linear interpolation for crossing point
            frac = (limit - y_vals[i]) / (y_vals[i + 1] - y_vals[i])
            x_cross = x_vals[i] * (x_vals[i + 1] / x_vals[i]) ** frac
            ax.annotate(f"{label}\n({x_cross:.0f}B)",
                        xy=(x_cross, limit),
                        xytext=(15, 15), textcoords="offset points",
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            break


def _mark_reference_models(ax: plt.Axes, estimator, ref_models: dict[str, ModelArch],
                           key: str, **kwargs) -> None:
    """Plot dots for verified reference models."""
    for name, arch in ref_models.items():
        result = estimator(arch, **kwargs)
        ax.plot(arch.P_billions, result[key], "o", color="gray",
                markersize=4, zorder=5, alpha=0.7)


def generate_plots(
    N: int = 4,
    T: int | str = "auto",
    B_micro: int = 4,
    S: int = 4096,
    V: int = 152064,
    alpha: float = 0.15,
    gpu_limit: float = 96.0,
    cpu_limit: float = 202.0,
    cpu_node_limit: float = 808.0,
    min_params: float = 0.5,
    max_params: float = 1000.0,
    output_path: Path | None = None,
    backend: str = "megatron",
) -> tuple[Path, list[dict]]:
    """Generate SFT and GRPO memory estimation plots (two separate figures).

    SFT figure: for FSDP (3 panels: no offload, optim offload, full offload)
                for Megatron (2 panels: optim offload, full offload)
    GRPO figure (2 panels): peak memory, per-phase breakdown

    All plots show GPU peak (per-GPU) and CPU per-node (total for the node).

    Returns (sft_plot_path, all_data).
    """
    use_megatron = (backend == "megatron")
    auto_tp = (T == "auto")

    def _resolve_tp_pp(arch: ModelArch) -> tuple[int, int]:
        """Return (T, D) for this arch. Tries GRPO (most demanding) first, then best-effort."""
        if auto_tp and use_megatron:
            # Try GRPO fit first (most demanding — works for all panels)
            res = min_tp_pp_degree(arch, N, B_micro, S,
                                   gpu_total_gb=gpu_limit, cpu_limit=cpu_limit,
                                   alpha=alpha, cpu_node_limit=cpu_node_limit,
                                   gpus_per_node=GPUS_PER_NODE, mode="grpo")
            if res is not None:
                return res
            # GRPO doesn't fit — try SFT (covers SFT panels)
            res = min_tp_pp_degree(arch, N, B_micro, S,
                                   gpu_total_gb=gpu_limit, cpu_limit=cpu_limit,
                                   alpha=alpha, cpu_node_limit=cpu_node_limit,
                                   gpus_per_node=GPUS_PER_NODE, mode="sft")
            if res is not None:
                return res
            # Nothing fits — pick max TP, max PP to minimize per-GPU memory
            num_nodes = max(1, N // GPUS_PER_NODE)
            best_T = min(4, N)
            best_D = 1
            d = 1
            while d <= num_nodes:
                if best_T * d <= N:
                    best_D = d
                d *= 2
            return best_T, best_D
        if auto_tp:
            t = min_tp_degree(arch, N, B_micro, S,
                              gpu_total_gb=gpu_limit, cpu_limit=cpu_limit, alpha=alpha,
                              backend=backend, cpu_node_limit=cpu_node_limit)
            return (t if t is not None else min(4, N)), 1
        return (T if isinstance(T, int) else 1), 1

    # Model sizes: log-spaced sweep
    P_values = np.geomspace(min_params, max_params, 200)
    xlim = (min_params * 0.8, max_params * 1.2)

    # Compute estimates for each model size
    all_data: list[dict] = []
    sft_no_off_gpu, sft_no_off_cpu_node = [], []
    sft_optim_gpu, sft_optim_cpu_node = [], []
    sft_full_gpu, sft_full_cpu_node = [], []
    grpo_gpu, grpo_cpu_node = [], []
    grpo_rollout, grpo_ref, grpo_actor, grpo_wsync = [], [], [], []

    for P_b in P_values:
        arch = get_arch(P_b, V)

        selected_T, selected_D = _resolve_tp_pp(arch)

        if use_megatron:
            sft_no = sft_megatron(arch, B_micro, S, N, T=selected_T, D=selected_D,
                                  param_offload=False, optimizer_offload=False, alpha=alpha)
            sft_opt = sft_megatron(arch, B_micro, S, N, T=selected_T, D=selected_D,
                                   param_offload=False, optimizer_offload=True, alpha=alpha)
            sft_ful = sft_megatron(arch, B_micro, S, N, T=selected_T, D=selected_D,
                                   param_offload=True, optimizer_offload=True, alpha=alpha)
            grpo_est = grpo_megatron_phases(arch, B_micro, S, N, selected_T, D=selected_D,
                                            gpu_total_gb=gpu_limit, alpha=alpha)
        else:
            sft_no = sft_fsdp(arch, B_micro, S, N, T=selected_T, alpha=alpha)
            sft_opt = sft_fsdp_optim_offload(arch, B_micro, S, N, T=selected_T, alpha=alpha)
            sft_ful = sft_fsdp_full_offload(arch, B_micro, S, N, T=selected_T, alpha=alpha)
            grpo_est = grpo_phases(arch, B_micro, S, N, selected_T,
                                   gpu_total_gb=gpu_limit, alpha=alpha)

        sft_no_off_gpu.append(sft_no["gpu_peak_gb"])
        sft_no_off_cpu_node.append(sft_no["cpu_per_node"]["peak_gb"])
        sft_optim_gpu.append(sft_opt["gpu_peak_gb"])
        sft_optim_cpu_node.append(sft_opt["cpu_per_node"]["peak_gb"])
        sft_full_gpu.append(sft_ful["gpu_peak_gb"])
        sft_full_cpu_node.append(sft_ful["cpu_per_node"]["peak_gb"])
        grpo_gpu.append(grpo_est["gpu_peak_gb"])
        grpo_cpu_node.append(grpo_est["cpu_per_node"]["peak_gb"])
        grpo_rollout.append(grpo_est["rollout_gpu_gb"])
        grpo_ref.append(grpo_est["ref_gpu_gb"])
        grpo_actor.append(grpo_est["actor_gpu_gb"])
        grpo_wsync.append(grpo_est["weight_sync_gpu_gb"])

        sft_label = "sft_megatron" if use_megatron else "sft_fsdp"
        row = {
            "model_size_B": float(f"{P_b:.2f}"),
            "arch_h": arch.h,
            "arch_L": arch.L,
            "tp_degree": selected_T,
            "pp_degree": selected_D,
            "backend": backend,
            f"{sft_label}_no_offload": sft_no,
            f"{sft_label}_optim_offload": sft_opt,
            f"{sft_label}_full_offload": sft_ful,
            "grpo": grpo_est,
        }
        all_data.append(row)

    # Convert to arrays
    P_arr = np.array(P_values)
    sft_no_off_gpu = np.array(sft_no_off_gpu)
    sft_no_off_cpu_node = np.array(sft_no_off_cpu_node)
    sft_optim_gpu = np.array(sft_optim_gpu)
    sft_optim_cpu_node = np.array(sft_optim_cpu_node)
    sft_full_gpu = np.array(sft_full_gpu)
    sft_full_cpu_node = np.array(sft_full_cpu_node)
    grpo_gpu_arr = np.array(grpo_gpu)
    grpo_cpu_node_arr = np.array(grpo_cpu_node)

    num_nodes = max(1, N // GPUS_PER_NODE)
    tp_label = "auto" if auto_tp else str(T)
    backend_label = "Megatron TP+PP" if use_megatron else "FSDP"
    node_str = f"{num_nodes} node{'s' if num_nodes > 1 else ''}"
    config_note = (f"TP=auto, PP=auto, $B_{{\\mathrm{{micro}}}}$={B_micro}, $S$={S}, "
                   f"bf16, gradient checkpointing, {GPUS_PER_NODE} GPUs/node")
    y_max = max(gpu_limit, cpu_node_limit) * 1.5

    # =====================================================================
    # Figure 1: SFT (2 panels — optimizer offload, full offload)
    # =====================================================================
    fig_sft, axes_sft = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_sft.suptitle(f"SFT Peak Memory Estimation — {node_str} ({N} GPUs)",
                     fontsize=13, y=1.0)

    sft_panels = [
        ("(a) SFT with optimizer offload", sft_optim_gpu, sft_optim_cpu_node),
        ("(b) SFT with full offload", sft_full_gpu, sft_full_cpu_node),
    ]

    def _sft_ref_estimate(arch, ref_T, mode, ref_D=1):
        """Get SFT estimate for a reference model, dispatching to correct backend."""
        if use_megatron:
            if mode == "no_offload":
                return sft_megatron(arch, B_micro, S, N, T=ref_T, D=ref_D,
                                    param_offload=False, optimizer_offload=False, alpha=alpha)
            elif mode == "optim_offload":
                return sft_megatron(arch, B_micro, S, N, T=ref_T, D=ref_D,
                                    param_offload=False, optimizer_offload=True, alpha=alpha)
            else:
                return sft_megatron(arch, B_micro, S, N, T=ref_T, D=ref_D,
                                    param_offload=True, optimizer_offload=True, alpha=alpha)
        else:
            if mode == "no_offload":
                return sft_fsdp(arch, B_micro, S, N, T=ref_T, alpha=alpha)
            elif mode == "optim_offload":
                return sft_fsdp_optim_offload(arch, B_micro, S, N, T=ref_T, alpha=alpha)
            else:
                return sft_fsdp_full_offload(arch, B_micro, S, N, T=ref_T, alpha=alpha)

    sft_modes = ["optim_offload", "full_offload"]
    for ax, (title, gpu_data, cpu_node_data), mode in zip(axes_sft, sft_panels, sft_modes):
        _style_ax(ax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.plot(P_arr, gpu_data, color=PLOT_COLORS["gpu"], linewidth=2, label="GPU peak (per GPU)")
        ax.plot(P_arr, cpu_node_data, color=PLOT_COLORS["cpu"], linewidth=2, label="CPU total (per node)")
        _add_limits(ax, gpu_limit, cpu_node_limit, xlim)
        _annotate_crossing(ax, P_arr, gpu_data, gpu_limit, "GPU full", PLOT_COLORS["gpu_limit"])
        _annotate_crossing(ax, P_arr, cpu_node_data, cpu_node_limit, "CPU full", PLOT_COLORS["cpu_limit"])
        # Reference model dots
        for name, arch in REFERENCE_MODELS.items():
            ref_T, ref_D = _resolve_tp_pp(arch) if auto_tp else ((T if isinstance(T, int) else 1), 1)
            r = _sft_ref_estimate(arch, ref_T, mode, ref_D=ref_D)
            ax.plot(arch.P_billions, r["gpu_peak_gb"], "o", color=PLOT_COLORS["gpu"],
                    markersize=3.5, zorder=5)
            ax.plot(arch.P_billions, r["cpu_per_node"]["peak_gb"], "o", color=PLOT_COLORS["cpu"],
                    markersize=3.5, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters)")
        ax.set_ylabel("Memory (GB)")
        ax.set_xlim(xlim)
        ax.set_ylim(0, y_max)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=7, loc="upper left")

    fig_sft.text(0.5, -0.01, config_note,
                 ha="center", fontsize=7, color="gray")
    fig_sft.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save SFT figure
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "memory_estimation_sft.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_sft.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig_sft)
    print(f"SFT plot saved to: {output_path}")

    # =====================================================================
    # Figure 2: GRPO (2 panels — peak memory, per-phase breakdown)
    # =====================================================================
    grpo_path = output_path.with_name(
        output_path.stem.replace("sft", "grpo") + output_path.suffix
    )
    # Fallback if stem doesn't contain "sft"
    if grpo_path == output_path:
        grpo_path = output_path.with_name("memory_estimation_grpo.png")

    fig_grpo, axes_grpo = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_grpo.suptitle(f"GRPO Peak Memory Estimation — {node_str} ({N} GPUs)",
                      fontsize=13, y=1.0)

    # Panel (a): GRPO peak
    ax = axes_grpo[0]
    _style_ax(ax)
    ax.set_title("(a) GRPO peak memory (max across all phases)",
                 fontsize=10, fontweight="bold")
    ax.plot(P_arr, grpo_gpu_arr, color=PLOT_COLORS["gpu"], linewidth=2,
            label="GPU peak (per GPU)")
    ax.plot(P_arr, grpo_cpu_node_arr, color=PLOT_COLORS["cpu"], linewidth=2,
            label="CPU total (per node)")
    _add_limits(ax, gpu_limit, cpu_node_limit, xlim)
    _annotate_crossing(ax, P_arr, grpo_gpu_arr, gpu_limit, "GPU full", PLOT_COLORS["gpu_limit"])
    _annotate_crossing(ax, P_arr, grpo_cpu_node_arr, cpu_node_limit, "CPU full", PLOT_COLORS["cpu_limit"])
    for name, arch in REFERENCE_MODELS.items():
        ref_T, ref_D = _resolve_tp_pp(arch) if auto_tp else ((T if isinstance(T, int) else 1), 1)
        if use_megatron:
            r = grpo_megatron_phases(arch, B_micro, S, N, ref_T, D=ref_D,
                                     gpu_total_gb=gpu_limit, alpha=alpha)
        else:
            r = grpo_phases(arch, B_micro, S, N, ref_T,
                            gpu_total_gb=gpu_limit, alpha=alpha)
        ax.plot(arch.P_billions, r["gpu_peak_gb"], "o", color=PLOT_COLORS["gpu"],
                markersize=3.5, zorder=5)
        ax.plot(arch.P_billions, r["cpu_per_node"]["peak_gb"], "o", color=PLOT_COLORS["cpu"],
                markersize=3.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (billions of parameters)")
    ax.set_ylabel("Memory (GB)")
    ax.set_xlim(xlim)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    # Panel (b): GRPO phase breakdown
    ax = axes_grpo[1]
    _style_ax(ax)
    ax.set_title("(b) GRPO per-phase GPU breakdown", fontsize=10, fontweight="bold")
    ax.plot(P_arr, grpo_rollout, color=PLOT_COLORS["rollout"], linewidth=1.8,
            label="Rollout (inference engine, TP-sharded)")
    ax.plot(P_arr, grpo_ref, color=PLOT_COLORS["ref"], linewidth=1.8,
            label="Reference model log-probs")
    ax.plot(P_arr, grpo_actor, color=PLOT_COLORS["actor"], linewidth=1.8,
            label="Actor update (training step)")
    ax.plot(P_arr, grpo_wsync, color=PLOT_COLORS["weight_sync"], linewidth=1.8,
            linestyle="-.", label="Weight synchronisation")
    ax.axhline(gpu_limit, color=PLOT_COLORS["gpu_limit"], linestyle="--",
               linewidth=1.2, label=f"GPU limit ({gpu_limit:.0f} GB)")
    ax.fill_between(xlim, gpu_limit, gpu_limit * 3,
                    color=PLOT_COLORS["gpu_limit"], alpha=0.06)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (billions of parameters)")
    ax.set_ylabel("GPU memory per phase (GB)")
    ax.set_xlim(xlim)
    ax.set_ylim(0, gpu_limit * 3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    fig_grpo.text(0.5, -0.01,
                  config_note + ", optimizer + reference model offloaded to CPU",
                  ha="center", fontsize=7, color="gray")
    fig_grpo.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig_grpo.savefig(str(grpo_path), dpi=150, bbox_inches="tight")
    plt.close(fig_grpo)
    print(f"GRPO plot saved to: {grpo_path}")

    return output_path, all_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU/CPU memory for SFT & GRPO training across model sizes."
    )
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs (default: 4)")
    parser.add_argument("--tp-degree", type=str, default="auto",
                        help="Tensor parallelism degree, or 'auto' (default: auto)")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Per-GPU micro-batch size (default: 4)")
    parser.add_argument("--seq-length", type=int, default=4096,
                        help="Total sequence length (default: 4096)")
    parser.add_argument("--vocab-size", type=int, default=152064,
                        help="Vocabulary size (default: 152064)")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Overhead/fragmentation factor (default: 0.15)")
    parser.add_argument("--gpu-memory", type=float, default=96.0,
                        help="GPU HBM limit in GB (default: 96)")
    parser.add_argument("--cpu-memory", type=float, default=202.0,
                        help="CPU memory limit per GPU in GB (default: 202)")
    parser.add_argument("--gpus-per-node", type=int, default=4,
                        help="GPUs per node (default: 4)")
    parser.add_argument("--node-memory", type=float, default=808.0,
                        help="Total CPU memory per node in GB (default: 808)")
    parser.add_argument("--backend", type=str, default="megatron",
                        choices=["megatron", "fsdp"],
                        help="Training backend: 'megatron' (TP+PP) or 'fsdp' (default: megatron)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: benchmarks/results/memory_estimation_sft.png)")
    parser.add_argument("--min-params", type=float, default=0.5,
                        help="Minimum model size in billions (default: 0.5)")
    parser.add_argument("--max-params", type=float, default=1000.0,
                        help="Maximum model size in billions (default: 1000)")

    args = parser.parse_args()

    # Parse TP degree: "auto" or integer
    tp_degree = "auto" if args.tp_degree == "auto" else int(args.tp_degree)
    use_megatron = (args.backend == "megatron")

    output_path, all_data = generate_plots(
        N=args.num_gpus,
        T=tp_degree,
        B_micro=args.micro_batch_size,
        S=args.seq_length,
        V=args.vocab_size,
        alpha=args.alpha,
        gpu_limit=args.gpu_memory,
        cpu_limit=args.cpu_memory,
        cpu_node_limit=args.node_memory,
        min_params=args.min_params,
        max_params=args.max_params,
        output_path=args.output,
        backend=args.backend,
    )

    # Helper to resolve TP+PP for a given arch
    auto_tp = (tp_degree == "auto")

    def _resolve_tp_pp(arch):
        """Return (T, D) for this arch. Tries GRPO (most demanding) first, then best-effort."""
        if auto_tp and use_megatron:
            # Try GRPO fit first (most demanding — works for all panels)
            res = min_tp_pp_degree(arch, args.num_gpus, args.micro_batch_size,
                                   args.seq_length, gpu_total_gb=args.gpu_memory,
                                   cpu_limit=args.cpu_memory, alpha=args.alpha,
                                   cpu_node_limit=args.node_memory,
                                   gpus_per_node=args.gpus_per_node, mode="grpo")
            if res is not None:
                return res
            # GRPO doesn't fit — try SFT (covers SFT panels)
            res = min_tp_pp_degree(arch, args.num_gpus, args.micro_batch_size,
                                   args.seq_length, gpu_total_gb=args.gpu_memory,
                                   cpu_limit=args.cpu_memory, alpha=args.alpha,
                                   cpu_node_limit=args.node_memory,
                                   gpus_per_node=args.gpus_per_node, mode="sft")
            if res is not None:
                return res
            # Nothing fits — pick max TP, max PP to minimize per-GPU memory
            num_nodes = max(1, args.num_gpus // args.gpus_per_node)
            best_T = min(4, args.num_gpus)
            best_D = 1
            d = 1
            while d <= num_nodes:
                if best_T * d <= args.num_gpus:
                    best_D = d
                d *= 2
            return best_T, best_D
        if auto_tp:
            t = min_tp_degree(arch, args.num_gpus, args.micro_batch_size,
                              args.seq_length, gpu_total_gb=args.gpu_memory,
                              cpu_limit=args.cpu_memory, alpha=args.alpha,
                              backend=args.backend,
                              cpu_node_limit=args.node_memory)
            return (t if t is not None else min(4, args.num_gpus)), 1
        return (tp_degree if isinstance(tp_degree, int) else 1), 1

    def _sft_estimate(arch, ref_T, param_offload=True, optimizer_offload=True, ref_D=1):
        if use_megatron:
            return sft_megatron(arch, args.micro_batch_size, args.seq_length,
                                args.num_gpus, T=ref_T, D=ref_D,
                                param_offload=param_offload,
                                optimizer_offload=optimizer_offload,
                                alpha=args.alpha)
        else:
            if param_offload and optimizer_offload:
                return sft_fsdp_full_offload(arch, args.micro_batch_size, args.seq_length,
                                             args.num_gpus, T=ref_T, alpha=args.alpha)
            elif optimizer_offload:
                return sft_fsdp_optim_offload(arch, args.micro_batch_size, args.seq_length,
                                              args.num_gpus, T=ref_T, alpha=args.alpha)
            else:
                return sft_fsdp(arch, args.micro_batch_size, args.seq_length,
                                args.num_gpus, T=ref_T, alpha=args.alpha)

    def _grpo_estimate(arch, ref_T, ref_D=1):
        if use_megatron:
            return grpo_megatron_phases(arch, args.micro_batch_size, args.seq_length,
                                        args.num_gpus, ref_T, D=ref_D,
                                        gpu_total_gb=args.gpu_memory, alpha=args.alpha)
        else:
            return grpo_phases(arch, args.micro_batch_size, args.seq_length,
                               args.num_gpus, ref_T,
                               gpu_total_gb=args.gpu_memory, alpha=args.alpha)

    # Write JSON alongside the plot
    json_path = output_path.with_suffix(".json")
    json_output = {
        "config": {
            "num_gpus": args.num_gpus,
            "tp_degree": args.tp_degree,
            "micro_batch_size": args.micro_batch_size,
            "seq_length": args.seq_length,
            "vocab_size": args.vocab_size,
            "alpha": args.alpha,
            "gpu_memory_gb": args.gpu_memory,
            "cpu_memory_gb": args.cpu_memory,
            "backend": args.backend,
        },
        "reference_models": {
            name: (lambda _t_d=_resolve_tp_pp(arch): {
                "P_billions": arch.P_billions,
                "h": arch.h, "L": arch.L, "a": arch.a,
                "a_kv": arch.a_kv, "d_ff": arch.d_ff, "V": arch.V,
                "tp_degree": _t_d[0],
                "pp_degree": _t_d[1],
                "sft_no_offload": _sft_estimate(arch, _t_d[0],
                                                param_offload=False, optimizer_offload=False,
                                                ref_D=_t_d[1]),
                "sft_optim_offload": _sft_estimate(arch, _t_d[0],
                                                   param_offload=False, optimizer_offload=True,
                                                   ref_D=_t_d[1]),
                "sft_full_offload": _sft_estimate(arch, _t_d[0],
                                                  param_offload=True, optimizer_offload=True,
                                                  ref_D=_t_d[1]),
                "grpo": _grpo_estimate(arch, _t_d[0], ref_D=_t_d[1]),
            })()
            for name, arch in REFERENCE_MODELS.items()
        },
        "sweep": all_data,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to: {json_path}")

    # Print summary for reference models
    num_nodes = args.num_gpus // args.gpus_per_node
    backend_label = "TP+PP" if use_megatron else "FSDP"
    print(f"\nConfiguration: {backend_label}, {args.num_gpus} GPUs, {num_nodes} node(s), "
          f"{args.gpus_per_node} GPUs/node, {args.node_memory:.0f} GB CPU/node")

    # GPU memory table
    print(f"\n{'Model':<18} {'TP':>3} {'PP':>3} {'SFT NoOff GPU':>14} {'SFT OptOff GPU':>15} "
          f"{'SFT FullOff GPU':>16} {'GRPO GPU':>10} {'GRPO CPU/gpu':>12} {'GPU Fit?':>8}")
    print("-" * 116)
    for name, arch in sorted(REFERENCE_MODELS.items(), key=lambda x: x[1].P):
        ref_T, ref_D = _resolve_tp_pp(arch)
        sft_no = _sft_estimate(arch, ref_T, param_offload=False, optimizer_offload=False, ref_D=ref_D)
        sft_opt = _sft_estimate(arch, ref_T, param_offload=False, optimizer_offload=True, ref_D=ref_D)
        sft_ful = _sft_estimate(arch, ref_T, param_offload=True, optimizer_offload=True, ref_D=ref_D)
        grpo = _grpo_estimate(arch, ref_T, ref_D=ref_D)
        gpu_ok = (sft_ful["gpu_peak_gb"] <= args.gpu_memory)
        print(f"{name:<18} {ref_T:>3} {ref_D:>3} {sft_no['gpu_peak_gb']:>11.1f} GB "
              f"{sft_opt['gpu_peak_gb']:>12.1f} GB {sft_ful['gpu_peak_gb']:>13.1f} GB "
              f"{grpo['gpu_peak_gb']:>7.1f} GB {grpo['cpu_offloaded_gb']:>9.1f} GB "
              f"{'yes' if gpu_ok else 'NO':>8}")

    # CPU per-node memory table
    print(f"\n{'='*120}")
    print(f"CPU MEMORY PER NODE (total, not per-GPU) — {args.node_memory:.0f} GB limit")
    print(f"{'='*120}")
    buf_label = "nccl_buf" if use_megatron else "ds_buf"
    print(f"{'Model':<18} {'SFT FullOff':>12} {'Breakdown (optim + params + grads + ' + buf_label + ' + proc + os + dl)':>62} "
          f"{'Load Peak':>10} {'Fits?':>6}")
    print("-" * 130)
    for name, arch in sorted(REFERENCE_MODELS.items(), key=lambda x: x[1].P):
        ref_T, ref_D = _resolve_tp_pp(arch)
        sft_ful = _sft_estimate(arch, ref_T, param_offload=True, optimizer_offload=True, ref_D=ref_D)
        cpn = sft_ful["cpu_per_node"]
        node_ok = cpn["peak_gb"] <= args.node_memory
        buf_key = "nccl_buffers_gb" if use_megatron else "ds_buffers_gb"
        print(f"{name:<18} {cpn['steady_gb']:>9.0f} GB "
              f"  ({cpn['optimizer_gb']:>6.0f} + {cpn['params_gb']:>5.0f} + "
              f"{cpn['grads_gb']:>5.0f} + {cpn.get(buf_key, 0):>5.0f} + "
              f"{cpn['process_overhead_gb']:>4.0f} + {cpn['os_baseline_gb']:>4.0f} + "
              f"{cpn['dataloader_gb']:>3.0f})"
              f" {cpn['loading_peak_gb']:>9.0f} GB"
              f" {'yes' if node_ok else 'NO':>6}")

    # GRPO CPU per node
    buf_header = "nccl" if use_megatron else "ds_buf"
    print(f"\n{'Model':<18} {'GRPO CPU/node':>14} {'Breakdown (optim + grads + ref + ' + buf_header + ' + proc + os + dl)':>62} {'Fits?':>6}")
    print("-" * 112)
    for name, arch in sorted(REFERENCE_MODELS.items(), key=lambda x: x[1].P):
        ref_T, ref_D = _resolve_tp_pp(arch)
        grpo = _grpo_estimate(arch, ref_T, ref_D=ref_D)
        cpn = grpo["cpu_per_node"]
        node_ok = cpn["peak_gb"] <= args.node_memory
        buf_key = "nccl_buffers_gb" if use_megatron else "ds_buffers_gb"
        print(f"{name:<18} {cpn['steady_gb']:>11.0f} GB "
              f"  ({cpn['optimizer_gb']:>6.0f} + {cpn['grads_gb']:>5.0f} + "
              f"{cpn.get('ref_model_gb', 0):>5.0f} + {cpn.get(buf_key, 0):>5.0f} + "
              f"{cpn['process_overhead_gb']:>4.0f} + {cpn['os_baseline_gb']:>4.0f} + "
              f"{cpn['dataloader_gb']:>3.0f})"
              f" {'yes' if node_ok else 'NO':>6}")


if __name__ == "__main__":
    main()