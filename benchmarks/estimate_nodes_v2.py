"""
Node Estimation for LLM Training (SFT & GRPO)

Uses the memory formulas from estimate_memory.py to find the minimum number
of GPUs/nodes required to train a given model size, and plots the result.

Training uses tensor parallelism (TP) within nodes and pipeline parallelism
(PP) across nodes, with optimizer states offloaded to CPU.

Usage:
    python -m benchmarks.estimate_nodes [OPTIONS]

    --micro-batch-size 4  Per-GPU micro-batch size (default: 4)
    --seq-length 4096     Total sequence length (default: 4096)
    --vocab-size 152064   Vocabulary size (default: 152064)
    --alpha 0.15          Overhead factor (default: 0.15)
    --gpu-memory 96       GPU HBM limit in GB (default: 96)
    --node-memory 808     Total CPU memory per node in GB (default: 808)
    --gpus-per-node 4     GPUs per node (default: 4)
    --max-nodes 64        Maximum nodes to search (default: 64)
    --output PATH         Output PNG path
    --min-params 0.5      Minimum model size in billions (default: 0.5)
    --max-params 1000     Maximum model size in billions (default: 1000)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from benchmarks.estimate_memory import (
    GPUS_PER_NODE,
    REFERENCE_MODELS,
    ModelArch,
    fits,
    get_arch,
    grpo_megatron_phases,
    min_tp_degree,
    min_tp_pp_degree,
    sft_megatron,
)


def _gpu_candidates(gpus_per_node: int, max_nodes: int) -> list[int]:
    """Return candidate GPU counts: 1, then multiples of gpus_per_node."""
    candidates = [1] if gpus_per_node > 1 else []
    max_gpus = max_nodes * gpus_per_node
    candidates += list(range(gpus_per_node, max_gpus + 1, gpus_per_node))
    return candidates


def min_gpus_sft(arch: ModelArch, B_micro: int, S: int,
                 alpha: float = 0.15, gpu_limit: float = 96.0,
                 cpu_node_limit: float = 808.0,
                 gpus_per_node: int = GPUS_PER_NODE,
                 max_nodes: int = 64) -> tuple[int, int, int] | None:
    """Find minimum GPUs for SFT training with TP+PP.

    Uses TP to shard both parameters and activations.
    Optimizer offloaded to CPU, params optionally offloaded.
    PP (pipeline parallelism) across nodes for large models.

    Searches N (total GPUs), T (TP degree), and D (PP degree) jointly.
    Returns (N, T, D) for the minimum feasible configuration, or None.
    """
    tp_candidates = [t for t in [1, 2, 4, 8] if t <= gpus_per_node]

    for N in _gpu_candidates(gpus_per_node, max_nodes):
        num_nodes = max(1, N // gpus_per_node)
        # PP candidates: powers of 2 up to num_nodes
        pp_candidates = []
        d = 1
        while d <= num_nodes:
            pp_candidates.append(d)
            d *= 2

        # Prefer lower D (less pipeline bubbles), then lower T
        for D in pp_candidates:
            for T in tp_candidates:
                if T * D > N:
                    continue
                DP = N // (T * D)
                if DP < 1:
                    continue
                # Try full offload first, then optim-only offload
                for param_off, optim_off in [(True, True), (False, True)]:
                    est = sft_megatron(arch, B_micro, S, N, T=T, D=D,
                                       param_offload=param_off,
                                       optimizer_offload=optim_off,
                                       alpha=alpha)
                    if fits(est["gpu_peak_gb"], est.get("cpu_offloaded_gb", 0.0),
                            gpu_limit=gpu_limit,
                            cpu_limit=gpu_limit * 100,
                            cpu_per_node=est["cpu_per_node"],
                            cpu_node_limit=cpu_node_limit):
                        return N, T, D
    return None


def min_gpus_grpo(arch: ModelArch, B_micro: int, S: int,
                  alpha: float = 0.15, gpu_limit: float = 96.0,
                  cpu_node_limit: float = 808.0,
                  gpus_per_node: int = GPUS_PER_NODE,
                  max_nodes: int = 64) -> tuple[int, int, int] | None:
    """Find minimum GPUs for GRPO training with TP+PP actor.

    Searches N (total GPUs), T (TP degree), and D (PP degree) jointly.
    Returns (N, T, D) for the minimum feasible configuration, or None.
    """
    for N in _gpu_candidates(gpus_per_node, max_nodes):
        res = min_tp_pp_degree(arch, N, B_micro, S,
                               gpu_total_gb=gpu_limit,
                               cpu_limit=gpu_limit * 100,
                               alpha=alpha,
                               cpu_node_limit=cpu_node_limit,
                               gpus_per_node=gpus_per_node,
                               mode="grpo")
        if res is not None:
            return N, res[0], res[1]
    return None


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.15, linestyle="--")


PLOT_COLORS = {
    "sft": "#2563eb",
    "grpo": "#16a34a",
}


def generate_plots(
    B_micro: int = 4,
    S: int = 4096,
    V: int = 152064,
    alpha: float = 0.15,
    gpu_limit: float = 96.0,
    cpu_node_limit: float = 808.0,
    gpus_per_node: int = GPUS_PER_NODE,
    max_nodes: int = 64,
    min_params: float = 0.5,
    max_params: float = 1000.0,
    output_path: Path | None = None,
) -> Path:
    """Generate plot: model size (x) vs minimum GPUs/nodes (y).

    SFT line = TP + PP + CPU offload (auto TP/PP degree).
    GRPO line = TP+PP actor + TP-sharded inference rollout.
    """
    P_values = np.geomspace(min_params, max_params, 200)

    # result values: GPU count (int) or None; store TP+PP degree alongside
    sft_gpus:  list[int | None] = []
    sft_tp:    list[int | None] = []
    sft_pp:    list[int | None] = []
    grpo_gpus: list[int | None] = []
    grpo_tp:   list[int | None] = []
    grpo_pp:   list[int | None] = []

    for P_b in P_values:
        arch = get_arch(P_b, V)

        res = min_gpus_sft(arch, B_micro, S, alpha=alpha,
                           gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                           gpus_per_node=gpus_per_node, max_nodes=max_nodes)
        if res is not None:
            sft_gpus.append(res[0])
            sft_tp.append(res[1])
            sft_pp.append(res[2])
        else:
            sft_gpus.append(None)
            sft_tp.append(None)
            sft_pp.append(None)

        res_grpo = min_gpus_grpo(arch, B_micro, S, alpha=alpha,
                                 gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                                 gpus_per_node=gpus_per_node, max_nodes=max_nodes)
        if res_grpo is not None:
            grpo_gpus.append(res_grpo[0])
            grpo_tp.append(res_grpo[1])
            grpo_pp.append(res_grpo[2])
        else:
            grpo_gpus.append(None)
            grpo_tp.append(None)
            grpo_pp.append(None)

    subtitle = (f"$B_{{\\mathrm{{micro}}}}$={B_micro}, $S$={S}, bf16, gradient checkpointing, "
                f"{gpus_per_node} GPUs/node, 96 GB HBM, ~808 GB CPU/node")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    fig.suptitle("Minimum Nodes Required for Training on Olivia",
                 fontsize=13, y=0.98)

    P_arr = np.array(P_values)
    max_gpus = max_nodes * gpus_per_node

    all_series = {
        "sft":  (sft_gpus,  "SFT (TP + PP, optimizer offload)"),
        "grpo": (grpo_gpus, "GRPO (TP + PP, optimizer + ref offload)"),
    }

    for key, (gpu_list, label) in all_series.items():
        color = PLOT_COLORS[key]
        gpu_vals  = np.array([v if v is not None else np.nan for v in gpu_list])
        node_vals = gpu_vals / gpus_per_node

        valid = ~np.isnan(gpu_vals)
        if not valid.any():
            continue

        ax.step(P_arr[valid], node_vals[valid], color=color, linewidth=2,
                label=label, where="post")

        # "beyond limit" marker
        for P_b, v in zip(P_values, gpu_list):
            if v is None:
                ax.annotate("▶", xy=(P_b, max_nodes), fontsize=8,
                            color=color, ha="center", va="center")
                break

        # Reference model dots + name labels
        for name, arch in REFERENCE_MODELS.items():
            if key == "sft":
                res = min_gpus_sft(arch, B_micro, S, alpha=alpha,
                                   gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                                   gpus_per_node=gpus_per_node, max_nodes=max_nodes)
                n = res[0] if res is not None else None
                t = res[1] if res is not None else None
                d = res[2] if res is not None else None
            else:
                res_g = min_gpus_grpo(arch, B_micro, S, alpha=alpha,
                                      gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                                      gpus_per_node=gpus_per_node, max_nodes=max_nodes)
                n = res_g[0] if res_g is not None else None
                t = res_g[1] if res_g is not None else None
                d = res_g[2] if res_g is not None else None
            if n is not None:
                ax.plot(arch.P_billions, n / gpus_per_node, "o", color=color,
                        markersize=5, zorder=5)

    # Style
    _style_ax(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model size (billions of parameters)")
    ax.set_ylabel(f"Minimum number of nodes ({gpus_per_node} GPUs each)")
    ax.set_xlim(min_params * 0.8, max_params * 1.2)
    ax.set_ylim(0.2, max_nodes * 1.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))

    # Secondary axis: GPU count
    ax_r = ax.secondary_yaxis(
        "right",
        functions=(lambda x: x * gpus_per_node, lambda x: x / gpus_per_node),
    )
    ax_r.set_ylabel("Number of GPUs")

    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=7, color="gray")

    if output_path is None:
        output_path = Path(__file__).parent / "results" / "node_estimation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

    # Summary table
    print(f"\n{'Model':<18} {'Params':>8} {'SFT (TP+PP, offload)':>24} {'GRPO':>24}")
    print("-" * 80)
    for name, arch in sorted(REFERENCE_MODELS.items(), key=lambda x: x[1].P):
        res = min_gpus_sft(arch, B_micro, S, alpha=alpha,
                           gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                           gpus_per_node=gpus_per_node, max_nodes=max_nodes)
        if res is not None:
            n, t, d = res
            nodes = max(1, n // gpus_per_node)
            pp_tag = f" PP={d}" if d > 1 else ""
            sft_str = f"{nodes}N/{n}G TP={t}{pp_tag}"
        else:
            sft_str = f">{max_nodes}N"

        res_grpo = min_gpus_grpo(arch, B_micro, S, alpha=alpha,
                                 gpu_limit=gpu_limit, cpu_node_limit=cpu_node_limit,
                                 gpus_per_node=gpus_per_node, max_nodes=max_nodes)
        if res_grpo is not None:
            n_g, t_g, d_g = res_grpo
            nodes_g = max(1, n_g // gpus_per_node)
            pp_tag_g = f" PP={d_g}" if d_g > 1 else ""
            grpo_str = f"{nodes_g}N/{n_g}G TP={t_g}{pp_tag_g}"
        else:
            grpo_str = f">{max_nodes}N"

        print(f"{name:<18} {arch.P_billions:>6.1f}B {sft_str:>24} {grpo_str:>24}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate minimum GPUs/nodes for SFT & GRPO training across model sizes."
    )
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Per-GPU micro-batch size (default: 4)")
    parser.add_argument("--seq-length", type=int, default=4096,
                        help="Sequence length (default: 4096)")
    parser.add_argument("--vocab-size", type=int, default=152064,
                        help="Vocabulary size (default: 152064)")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Overhead/fragmentation factor (default: 0.15)")
    parser.add_argument("--gpu-memory", type=float, default=96.0,
                        help="GPU HBM limit in GB (default: 96)")
    parser.add_argument("--node-memory", type=float, default=808.0,
                        help="Total CPU memory per node in GB (default: 808)")
    parser.add_argument("--gpus-per-node", type=int, default=4,
                        help="GPUs per node (default: 4)")
    parser.add_argument("--max-nodes", type=int, default=64,
                        help="Maximum nodes to search up to (default: 64)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: benchmarks/results/node_estimation.png)")
    parser.add_argument("--min-params", type=float, default=0.5,
                        help="Minimum model size in billions (default: 0.5)")
    parser.add_argument("--max-params", type=float, default=1000.0,
                        help="Maximum model size in billions (default: 1000)")

    args = parser.parse_args()
    generate_plots(
        B_micro=args.micro_batch_size,
        S=args.seq_length,
        V=args.vocab_size,
        alpha=args.alpha,
        gpu_limit=args.gpu_memory,
        cpu_node_limit=args.node_memory,
        gpus_per_node=args.gpus_per_node,
        max_nodes=args.max_nodes,
        min_params=args.min_params,
        max_params=args.max_params,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()