"""
Memory Estimation for LLM Training (SFT & GRPO)

Implements the formulas from MEMORY_ESTIMATION_METHOD.md to estimate peak GPU
and CPU memory across model sizes, and generates a 4-panel diagnostic plot.

Usage:
    python -m benchmarks.estimate_memory [OPTIONS]

    --num-gpus 4          FSDP world size (default: 4)
    --tp-degree 1         Tensor parallelism for vLLM rollout (default: 1)
    --micro-batch-size 4  Per-GPU micro-batch size (default: 4)
    --seq-length 4096     Total sequence length (default: 4096)
    --vocab-size 152064   Vocabulary size (default: 152064)
    --alpha 0.15          Overhead factor (default: 0.15)
    --gpu-memory 96       GPU HBM limit in GB (default: 96)
    --cpu-memory 202      CPU memory limit per GPU in GB (default: 202)
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


def fsdp_unit_params(h: int) -> int:
    """Largest single FSDP unit ≈ one transformer layer."""
    return 12 * h * h


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
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": 0.0}


def sft_fsdp(arch: ModelArch, B: int, S: int, N: int, alpha: float = 0.15) -> dict:
    """Eq. 8: SFT with FSDP, no offloading."""
    P = arch.P
    states = 16 * P / N
    unit_peak = 2 * fsdp_unit_params(arch.h) * BYTES_BF16  # Eq. 7
    act = activation_memory(B, S, arch.h, arch.L, arch.a)
    logits = logits_memory(B, S, arch.V)
    oh = overhead(states, act, alpha)
    gpu_peak = states + act + unit_peak + logits + oh
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": 0.0}


def sft_fsdp_optim_offload(arch: ModelArch, B: int, S: int, N: int,
                            alpha: float = 0.15) -> dict:
    """Eq. 9: SFT with FSDP + optimizer offload to CPU."""
    P = arch.P
    gpu_states = 4 * P / N  # bf16 weights + bf16 grads
    unit_peak = 2 * fsdp_unit_params(arch.h) * BYTES_BF16
    act = activation_memory(B, S, arch.h, arch.L, arch.a)
    logits = logits_memory(B, S, arch.V)
    oh = overhead(gpu_states, act, alpha)
    gpu_peak = gpu_states + act + unit_peak + logits + oh
    cpu_offloaded = 12 * P / N  # master weights + momentum + variance
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": to_gb(cpu_offloaded)}


def sft_fsdp_full_offload(arch: ModelArch, B: int, S: int, N: int,
                           alpha: float = 0.15) -> dict:
    """Eq. 10: SFT with FSDP + full param & optimizer offload."""
    unit_peak = 2 * fsdp_unit_params(arch.h) * BYTES_BF16
    act = activation_memory(B, S, arch.h, arch.L, arch.a)
    logits = logits_memory(B, S, arch.V)
    # grad buffer for current unit
    grad_buf = fsdp_unit_params(arch.h) * BYTES_BF16
    oh = overhead(unit_peak, act, alpha)
    gpu_peak = unit_peak + act + grad_buf + logits + oh
    cpu_offloaded = 14 * arch.P / N  # 12 (optim) + 2 (params) per shard
    return {"gpu_peak_gb": to_gb(gpu_peak), "cpu_offloaded_gb": to_gb(cpu_offloaded)}


# ---------------------------------------------------------------------------
# GRPO estimators
# ---------------------------------------------------------------------------

def grpo_phases(arch: ModelArch, B_micro: int, S: int, N: int, T: int = 1,
                gpu_mem_util: float = 0.7, gpu_total_gb: float = 96.0,
                alpha: float = 0.15) -> dict:
    """
    Estimate per-phase GPU memory for GRPO (verl).

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
    unit_peak = 2 * fsdp_unit_params(h) * BYTES_BF16
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
    allgather_buf = 2 * fsdp_unit_params(h) * BYTES_BF16  # one unit buffer
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
    }


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def fits(gpu_peak_gb: float, cpu_offloaded_gb: float,
         gpu_limit: float = 96.0, cpu_limit: float = 202.0) -> bool:
    return gpu_peak_gb <= gpu_limit and cpu_offloaded_gb <= cpu_limit


def min_tp_degree(arch: ModelArch, N: int, B_micro: int, S: int,
                  gpu_mem_util: float = 0.7, gpu_total_gb: float = 96.0,
                  cpu_limit: float = 202.0, alpha: float = 0.15,
                  candidates: list[int] | None = None) -> int | None:
    """Find the minimum TP degree from candidates that makes GRPO fit.

    Returns the smallest T where both GPU peak and CPU offload fit within
    limits, or None if no candidate works.
    """
    if candidates is None:
        candidates = [1, 2, 4, 8]
    for T in candidates:
        if T > N:
            break  # TP degree can't exceed FSDP world size
        est = grpo_phases(arch, B_micro, S, N, T,
                          gpu_mem_util=gpu_mem_util,
                          gpu_total_gb=gpu_total_gb, alpha=alpha)
        if fits(est["gpu_peak_gb"], est["cpu_offloaded_gb"],
                gpu_total_gb, cpu_limit):
            return T
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
    """Draw horizontal limit lines and shaded forbidden regions."""
    ax.axhline(gpu_limit, color=PLOT_COLORS["gpu_limit"], linestyle="--",
               linewidth=1.2, label=f"GPU limit ({gpu_limit:.0f} GB)")
    ax.axhline(cpu_limit, color=PLOT_COLORS["cpu_limit"], linestyle="--",
               linewidth=1.2, label=f"CPU limit ({cpu_limit:.0f} GB)")
    ax.fill_between(xlim, gpu_limit, max(gpu_limit, cpu_limit) * 1.5,
                    color=PLOT_COLORS["gpu_limit"], alpha=0.06)
    ax.fill_between(xlim, cpu_limit, max(gpu_limit, cpu_limit) * 1.5,
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
    min_params: float = 0.5,
    max_params: float = 1000.0,
    output_path: Path | None = None,
) -> tuple[Path, list[dict]]:
    """Generate the 4-panel memory estimation plot and return (path, data)."""
    auto_tp = (T == "auto")

    # Model sizes: log-spaced sweep
    P_values = np.geomspace(min_params, max_params, 200)
    xlim = (min_params * 0.8, max_params * 1.2)

    # Compute estimates for each model size
    all_data: list[dict] = []
    sft_optim_gpu, sft_optim_cpu = [], []
    sft_full_gpu, sft_full_cpu = [], []
    grpo_gpu, grpo_cpu = [], []
    grpo_rollout, grpo_ref, grpo_actor, grpo_wsync = [], [], [], []

    for P_b in P_values:
        arch = get_arch(P_b, V)

        sft_opt = sft_fsdp_optim_offload(arch, B_micro, S, N, alpha)
        sft_ful = sft_fsdp_full_offload(arch, B_micro, S, N, alpha)

        if auto_tp:
            selected_T = min_tp_degree(arch, N, B_micro, S,
                                       gpu_total_gb=gpu_limit,
                                       cpu_limit=cpu_limit, alpha=alpha)
        else:
            selected_T = T

        if selected_T is not None:
            grpo_est = grpo_phases(arch, B_micro, S, N, selected_T,
                                   gpu_total_gb=gpu_limit, alpha=alpha)
        else:
            # No TP degree fits — use largest candidate for reporting
            selected_T = min(8, N)
            grpo_est = grpo_phases(arch, B_micro, S, N, selected_T,
                                   gpu_total_gb=gpu_limit, alpha=alpha)

        sft_optim_gpu.append(sft_opt["gpu_peak_gb"])
        sft_optim_cpu.append(sft_opt["cpu_offloaded_gb"])
        sft_full_gpu.append(sft_ful["gpu_peak_gb"])
        sft_full_cpu.append(sft_ful["cpu_offloaded_gb"])
        grpo_gpu.append(grpo_est["gpu_peak_gb"])
        grpo_cpu.append(grpo_est["cpu_offloaded_gb"])
        grpo_rollout.append(grpo_est["rollout_gpu_gb"])
        grpo_ref.append(grpo_est["ref_gpu_gb"])
        grpo_actor.append(grpo_est["actor_gpu_gb"])
        grpo_wsync.append(grpo_est["weight_sync_gpu_gb"])

        row = {
            "model_size_B": float(f"{P_b:.2f}"),
            "arch_h": arch.h,
            "arch_L": arch.L,
            "tp_degree": selected_T,
            "sft_fsdp_optim_offload": sft_opt,
            "sft_fsdp_full_offload": sft_ful,
            "grpo": grpo_est,
        }
        all_data.append(row)

    # Convert to arrays
    P_arr = np.array(P_values)
    sft_optim_gpu = np.array(sft_optim_gpu)
    sft_optim_cpu = np.array(sft_optim_cpu)
    sft_full_gpu = np.array(sft_full_gpu)
    sft_full_cpu = np.array(sft_full_cpu)
    grpo_gpu_arr = np.array(grpo_gpu)
    grpo_cpu_arr = np.array(grpo_cpu)

    # --- Create figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tp_label = "auto" if auto_tp else str(T)
    subtitle = (f"N={N} GPUs, TP={tp_label}, B_micro={B_micro}, S={S}, "
                f"V={V:,}, α={alpha}")
    fig.suptitle("Memory Estimation: SFT & GRPO Training\n"
                 f"({subtitle})", fontsize=13, y=0.98)

    # --- Panel (a): SFT FSDP + optimizer offload ---
    ax = axes[0, 0]
    _style_ax(ax)
    ax.set_title("(a) SFT — FSDP + Optimizer Offload", fontsize=10, fontweight="bold")
    ax.plot(P_arr, sft_optim_gpu, color=PLOT_COLORS["gpu"], linewidth=2,
            label="GPU peak")
    ax.plot(P_arr, sft_optim_cpu, color=PLOT_COLORS["cpu"], linewidth=2,
            label="CPU offloaded")
    _add_limits(ax, gpu_limit, cpu_limit, xlim)
    _annotate_crossing(ax, P_arr, sft_optim_gpu, gpu_limit, "GPU full", PLOT_COLORS["gpu_limit"])
    _annotate_crossing(ax, P_arr, sft_optim_cpu, cpu_limit, "CPU full", PLOT_COLORS["cpu_limit"])
    # Reference model dots
    for name, arch in REFERENCE_MODELS.items():
        r = sft_fsdp_optim_offload(arch, B_micro, S, N, alpha)
        ax.plot(arch.P_billions, r["gpu_peak_gb"], "o", color=PLOT_COLORS["gpu"],
                markersize=3.5, zorder=5)
        ax.plot(arch.P_billions, r["cpu_offloaded_gb"], "o", color=PLOT_COLORS["cpu"],
                markersize=3.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Memory (GB)")
    ax.set_xlim(xlim)
    y_max = max(gpu_limit, cpu_limit) * 2.5  # Cap at ~500 GB for readability
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    # --- Panel (b): SFT FSDP + full offload ---
    ax = axes[0, 1]
    _style_ax(ax)
    ax.set_title("(b) SFT — FSDP + Full Offload", fontsize=10, fontweight="bold")
    ax.plot(P_arr, sft_full_gpu, color=PLOT_COLORS["gpu"], linewidth=2,
            label="GPU peak")
    ax.plot(P_arr, sft_full_cpu, color=PLOT_COLORS["cpu"], linewidth=2,
            label="CPU offloaded")
    _add_limits(ax, gpu_limit, cpu_limit, xlim)
    _annotate_crossing(ax, P_arr, sft_full_gpu, gpu_limit, "GPU full", PLOT_COLORS["gpu_limit"])
    _annotate_crossing(ax, P_arr, sft_full_cpu, cpu_limit, "CPU full", PLOT_COLORS["cpu_limit"])
    for name, arch in REFERENCE_MODELS.items():
        r = sft_fsdp_full_offload(arch, B_micro, S, N, alpha)
        ax.plot(arch.P_billions, r["gpu_peak_gb"], "o", color=PLOT_COLORS["gpu"],
                markersize=3.5, zorder=5)
        ax.plot(arch.P_billions, r["cpu_offloaded_gb"], "o", color=PLOT_COLORS["cpu"],
                markersize=3.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Memory (GB)")
    ax.set_xlim(xlim)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    # --- Panel (c): GRPO peak ---
    ax = axes[1, 0]
    _style_ax(ax)
    ax.set_title("(c) GRPO — FSDP + Optim Offload, Ref Offloaded", fontsize=10,
                 fontweight="bold")
    ax.plot(P_arr, grpo_gpu_arr, color=PLOT_COLORS["gpu"], linewidth=2,
            label="GPU peak (max of phases)")
    ax.plot(P_arr, grpo_cpu_arr, color=PLOT_COLORS["cpu"], linewidth=2,
            label="CPU offloaded")
    _add_limits(ax, gpu_limit, cpu_limit, xlim)
    _annotate_crossing(ax, P_arr, grpo_gpu_arr, gpu_limit, "GPU full", PLOT_COLORS["gpu_limit"])
    _annotate_crossing(ax, P_arr, grpo_cpu_arr, cpu_limit, "CPU full", PLOT_COLORS["cpu_limit"])
    for name, arch in REFERENCE_MODELS.items():
        ref_T = (min_tp_degree(arch, N, B_micro, S, gpu_total_gb=gpu_limit,
                               cpu_limit=cpu_limit, alpha=alpha)
                 if auto_tp else T)
        if ref_T is None:
            ref_T = min(8, N)
        r = grpo_phases(arch, B_micro, S, N, ref_T, gpu_total_gb=gpu_limit, alpha=alpha)
        ax.plot(arch.P_billions, r["gpu_peak_gb"], "o", color=PLOT_COLORS["gpu"],
                markersize=3.5, zorder=5)
        ax.plot(arch.P_billions, r["cpu_offloaded_gb"], "o", color=PLOT_COLORS["cpu"],
                markersize=3.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Memory (GB)")
    ax.set_xlim(xlim)
    ax.set_ylim(0, y_max)  # Same cap as SFT panels for visual consistency
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    # --- Panel (d): GRPO phase breakdown ---
    ax = axes[1, 1]
    _style_ax(ax)
    ax.set_title("(d) GRPO — Per-Phase GPU Breakdown", fontsize=10, fontweight="bold")
    ax.plot(P_arr, grpo_rollout, color=PLOT_COLORS["rollout"], linewidth=1.8,
            label="Rollout (vLLM)")
    ax.plot(P_arr, grpo_ref, color=PLOT_COLORS["ref"], linewidth=1.8,
            label="Ref log-probs")
    ax.plot(P_arr, grpo_actor, color=PLOT_COLORS["actor"], linewidth=1.8,
            label="Actor update")
    ax.plot(P_arr, grpo_wsync, color=PLOT_COLORS["weight_sync"], linewidth=1.8,
            linestyle="-.", label="Weight sync")
    ax.axhline(gpu_limit, color=PLOT_COLORS["gpu_limit"], linestyle="--",
               linewidth=1.2, label=f"GPU limit ({gpu_limit:.0f} GB)")
    ax.fill_between(xlim, gpu_limit, gpu_limit * 3,
                    color=PLOT_COLORS["gpu_limit"], alpha=0.06)
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("GPU Memory per Phase (GB)")
    ax.set_xlim(xlim)
    ax.set_ylim(0, gpu_limit * 3)  # Focus on GPU-relevant range
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    if output_path is None:
        output_path = Path(__file__).parent / "results" / "memory_estimation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

    return output_path, all_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU/CPU memory for SFT & GRPO training across model sizes."
    )
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs / FSDP world size (default: 4)")
    parser.add_argument("--tp-degree", type=str, default="auto",
                        help="Tensor parallelism degree for vLLM rollout, or 'auto' "
                             "to select minimum viable T per model (default: auto)")
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
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: benchmarks/results/memory_estimation.png)")
    parser.add_argument("--min-params", type=float, default=0.5,
                        help="Minimum model size in billions (default: 0.5)")
    parser.add_argument("--max-params", type=float, default=1000.0,
                        help="Maximum model size in billions (default: 1000)")

    args = parser.parse_args()

    # Parse TP degree: "auto" or integer
    tp_degree = "auto" if args.tp_degree == "auto" else int(args.tp_degree)

    output_path, all_data = generate_plots(
        N=args.num_gpus,
        T=tp_degree,
        B_micro=args.micro_batch_size,
        S=args.seq_length,
        V=args.vocab_size,
        alpha=args.alpha,
        gpu_limit=args.gpu_memory,
        cpu_limit=args.cpu_memory,
        min_params=args.min_params,
        max_params=args.max_params,
        output_path=args.output,
    )

    # Helper to resolve TP for a given arch
    auto_tp = (tp_degree == "auto")

    def _resolve_tp(arch):
        if auto_tp:
            t = min_tp_degree(arch, args.num_gpus, args.micro_batch_size,
                              args.seq_length, gpu_total_gb=args.gpu_memory,
                              cpu_limit=args.cpu_memory, alpha=args.alpha)
            return t if t is not None else min(8, args.num_gpus)
        return tp_degree

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
        },
        "reference_models": {
            name: {
                "P_billions": arch.P_billions,
                "h": arch.h, "L": arch.L, "a": arch.a,
                "a_kv": arch.a_kv, "d_ff": arch.d_ff, "V": arch.V,
                "tp_degree": _resolve_tp(arch),
                "sft_optim_offload": sft_fsdp_optim_offload(
                    arch, args.micro_batch_size, args.seq_length,
                    args.num_gpus, args.alpha),
                "sft_full_offload": sft_fsdp_full_offload(
                    arch, args.micro_batch_size, args.seq_length,
                    args.num_gpus, args.alpha),
                "grpo": grpo_phases(
                    arch, args.micro_batch_size, args.seq_length,
                    args.num_gpus, _resolve_tp(arch),
                    gpu_total_gb=args.gpu_memory, alpha=args.alpha),
            }
            for name, arch in REFERENCE_MODELS.items()
        },
        "sweep": all_data,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON saved to: {json_path}")

    # Print summary for reference models
    print(f"\n{'Model':<18} {'TP':>3} {'SFT Opt.Off GPU':>16} {'SFT Opt.Off CPU':>16} "
          f"{'GRPO GPU':>12} {'GRPO CPU':>12} {'Fits?':>6}")
    print("-" * 87)
    for name, arch in sorted(REFERENCE_MODELS.items(), key=lambda x: x[1].P):
        ref_T = _resolve_tp(arch)
        sft = sft_fsdp_optim_offload(arch, args.micro_batch_size, args.seq_length,
                                      args.num_gpus, args.alpha)
        grpo = grpo_phases(arch, args.micro_batch_size, args.seq_length,
                           args.num_gpus, ref_T,
                           gpu_total_gb=args.gpu_memory, alpha=args.alpha)
        grpo_ok = fits(grpo["gpu_peak_gb"], grpo["cpu_offloaded_gb"],
                       args.gpu_memory, args.cpu_memory)
        print(f"{name:<18} {ref_T:>3} {sft['gpu_peak_gb']:>13.1f} GB {sft['cpu_offloaded_gb']:>13.1f} GB "
              f"{grpo['gpu_peak_gb']:>9.1f} GB {grpo['cpu_offloaded_gb']:>9.1f} GB "
              f"{'yes' if grpo_ok else 'NO':>6}")


if __name__ == "__main__":
    main()
