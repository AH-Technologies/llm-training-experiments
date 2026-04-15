#!/usr/bin/env python3
"""Plot reasoning head evolution across training checkpoints.

Reproduces the analysis from Thinking Sparks (Park et al., 2025) Figure 2:
  (A) Cohort stacked area chart — tracks when heads first emerge and persist
  (C) Heatmap — head change from base to final checkpoint

Reads head_importance.pt files produced by identify_heads.py for the base model
and each checkpoint, then classifies heads as "active" in a circuit using a
top-k threshold.

Usage:
  python -m reasoning_head_analysis.plot_head_evolution \
      --base reasoning_head_analysis/results/Qwen_Qwen2.5-Math-1.5B/head_importance.pt \
      --checkpoints \
          100:reasoning_head_analysis/results/global_step_100/head_importance.pt \
          200:reasoning_head_analysis/results/global_step_200/head_importance.pt \
          400:reasoning_head_analysis/results/global_step_400/head_importance.pt \
      --output_dir reasoning_head_analysis/results/evolution

  # Or auto-discover from a results directory:
  python -m reasoning_head_analysis.plot_head_evolution \
      --base reasoning_head_analysis/results/Qwen_Qwen2.5-Math-1.5B/head_importance.pt \
      --results_dir reasoning_head_analysis/results \
      --output_dir reasoning_head_analysis/results/evolution
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


def load_active_heads(importance_path, top_k=20):
    """Load head importance and return the set of active (layer, head) tuples."""
    data = torch.load(importance_path, map_location="cpu", weights_only=True)
    head_scores = data["head_scores"] if isinstance(data, dict) else data
    n_layers, n_heads = head_scores.shape

    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    active = set()
    for idx in sorted_idx[:top_k]:
        l, h = idx.item() // n_heads, idx.item() % n_heads
        active.add((l, h))

    return active, head_scores, n_layers, n_heads


def auto_discover_checkpoints(results_dir):
    """Find head_importance.pt files with step numbers in their path."""
    checkpoints = {}
    for root, dirs, files in os.walk(results_dir):
        if "head_importance.pt" in files:
            # Look for step number in path
            match = re.search(r"(?:global_)?step_?(\d+)", root)
            if match:
                step = int(match.group(1))
                checkpoints[step] = os.path.join(root, "head_importance.pt")
    return checkpoints


def plot_cohort_chart(base_heads, checkpoint_data, output_path):
    """Stacked area chart showing head cohorts over training.

    Reproduces Figure 2(A) from Thinking Sparks.
    """
    steps = sorted(checkpoint_data.keys())
    if not steps:
        print("No checkpoint data to plot cohort chart")
        return

    # Track cohorts: which step each head was first seen
    # A head is "newly activated" if it's active in the checkpoint but not in the base
    head_birth = {}  # (l, h) -> step when first seen
    maintained_at_step = {}  # step -> count of base heads still active
    total_new_at_step = {}  # step -> count of all non-base heads active

    for step in steps:
        active = checkpoint_data[step]["active"]
        new_heads = active - base_heads
        maintained = active & base_heads

        maintained_at_step[step] = len(maintained)
        total_new_at_step[step] = len(new_heads)

        for head in new_heads:
            if head not in head_birth:
                head_birth[head] = step

    # Build cohort counts per step
    # For each checkpoint, count how many heads from each birth cohort are still active
    birth_steps = sorted(set(head_birth.values()))
    cohort_counts = {bs: [] for bs in birth_steps}

    for step in steps:
        active = checkpoint_data[step]["active"]
        new_heads = active - base_heads
        for bs in birth_steps:
            count = sum(1 for h in new_heads if head_birth.get(h) == bs)
            cohort_counts[bs].append(count)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Stacked area for cohorts
    bottoms = np.zeros(len(steps))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(birth_steps) - 1, 1)) for i in range(len(birth_steps))]

    for i, bs in enumerate(birth_steps):
        values = np.array(cohort_counts[bs])
        ax.fill_between(steps, bottoms, bottoms + values, alpha=0.7,
                        color=colors[i], label=f"Born at step {bs}")
        bottoms += values

    # Blue line: total newly activated
    new_counts = [total_new_at_step[s] for s in steps]
    ax.plot(steps, new_counts, "b-", linewidth=2, label="Newly Activated Nodes")

    # Red dashed line: maintained base heads
    maint_counts = [maintained_at_step[s] for s in steps]
    ax.plot(steps, maint_counts, "r--", linewidth=2, label="Maintained Nodes")

    ax.set_xlabel("Checkpoint Step", fontsize=13)
    ax.set_ylabel("Number of Attention Head Nodes", fontsize=13)
    ax.set_title("Emerging Attention Heads During Training", fontsize=14)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", fontsize=9, ncol=2,
              title="Node Groups", title_fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved cohort chart to {output_path}")


def plot_head_change_heatmap(base_heads, final_heads, base_scores, final_scores,
                             n_layers, n_heads, output_path):
    """Sparse heatmap showing head activation change from base to final checkpoint.

    Reproduces Figure 2(C) from Thinking Sparks.
    Red = base model head (lost or fading), Blue = newly emerged head.
    Black border = active in final checkpoint.
    """
    # Build change matrix: only show heads active in either base or final
    change = np.full((n_heads, n_layers), np.nan)

    all_relevant = base_heads | final_heads
    for (l, h) in all_relevant:
        in_base = (l, h) in base_heads
        in_final = (l, h) in final_heads

        if in_base and not in_final:
            # Lost head: red (negative)
            change[h, l] = -1.0
        elif not in_base and in_final:
            # New head: blue (positive)
            change[h, l] = 1.0
        elif in_base and in_final:
            # Retained: slight change based on relative importance shift
            base_norm = base_scores[l, h] / base_scores.max() if base_scores.max() > 0 else 0
            final_norm = final_scores[l, h] / final_scores.max() if final_scores.max() > 0 else 0
            change[h, l] = float(final_norm - base_norm)

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.45), max(5, n_heads * 0.4)))

    # Custom colormap: red for negative (base), white for neutral, blue for positive (new)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for l in range(n_layers):
        for h in range(n_heads):
            val = change[h, l]
            if np.isnan(val):
                continue

            color = plt.cm.RdBu(norm(val))
            rect = mpatches.FancyBboxPatch(
                (l - 0.4, h - 0.4), 0.8, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="none",
            )
            ax.add_patch(rect)

            # Black border if active in final checkpoint
            if (l, h) in final_heads:
                border = mpatches.FancyBboxPatch(
                    (l - 0.4, h - 0.4), 0.8, 0.8,
                    boxstyle="round,pad=0.05",
                    facecolor="none", edgecolor="black", linewidth=1.5,
                )
                ax.add_patch(border)

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.5, n_heads - 0.5)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 15)))
    ax.set_yticks(range(n_heads))
    ax.set_xlabel("Layer IDs", fontsize=13)
    ax.set_ylabel("Head IDs", fontsize=13)
    ax.set_title("Attention Head Change from Base to Final Checkpoint", fontsize=14)
    ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Base ← → New", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def plot_head_activation_over_time(base_heads, checkpoint_data, n_layers, n_heads, output_path):
    """Heatmap showing per-head activation frequency across all checkpoints.

    Similar to Figure 2(C) extended: rows are heads (L.H), columns are steps.
    Red = base head fading, Blue = new head appearing.
    """
    steps = sorted(checkpoint_data.keys())
    if not steps:
        return

    # Collect all heads that are ever active
    all_active = set(base_heads)
    for step in steps:
        all_active |= checkpoint_data[step]["active"]

    # Sort by layer then head
    all_active = sorted(all_active)
    head_to_idx = {h: i for i, h in enumerate(all_active)}
    n_active = len(all_active)

    # Build matrix: rows = heads, cols = steps
    matrix = np.zeros((n_active, len(steps)))
    for j, step in enumerate(steps):
        active = checkpoint_data[step]["active"]
        for head in all_active:
            i = head_to_idx[head]
            in_base = head in base_heads
            in_ckpt = head in active
            if in_base and in_ckpt:
                matrix[i, j] = -0.5  # retained base (light red)
            elif in_base and not in_ckpt:
                matrix[i, j] = -1.0  # lost base (dark red)
            elif not in_base and in_ckpt:
                matrix[i, j] = 1.0   # new (blue)
            # else: 0 (not active)

    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 1.2), max(6, n_active * 0.3)))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45)
    ax.set_yticks(range(n_active))
    ax.set_yticklabels([f"L{l}H{h}" for l, h in all_active], fontsize=7)
    ax.set_xlabel("Checkpoint Step", fontsize=13)
    ax.set_ylabel("Attention Head", fontsize=13)
    ax.set_title("Head Activation Across Checkpoints", fontsize=14)

    # Mark heads active in final checkpoint with a dot
    final_active = checkpoint_data[steps[-1]]["active"]
    for head in all_active:
        if head in final_active:
            i = head_to_idx[head]
            ax.plot(len(steps) - 1, i, "k.", markersize=4)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Base ← → New")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved activation timeline to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot reasoning head evolution across training checkpoints")
    parser.add_argument("--base", type=str, required=True,
                        help="Path to head_importance.pt for the base model")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=[],
                        help="step:path pairs, e.g. 100:results/step100/head_importance.pt")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Auto-discover checkpoints from this directory")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top heads to consider 'active' (default: 20)")
    parser.add_argument("--output_dir", type=str, default="reasoning_head_analysis/results/evolution")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model heads from {args.base}")
    base_heads, base_scores, n_layers, n_heads = load_active_heads(args.base, top_k=args.top_k)
    print(f"  Architecture: {n_layers} layers x {n_heads} heads")
    print(f"  Active heads (top {args.top_k}): {sorted(base_heads)}")

    # Load checkpoints
    ckpt_paths = {}

    # From --checkpoints args
    for spec in args.checkpoints:
        step_str, path = spec.split(":", 1)
        ckpt_paths[int(step_str)] = path

    # From --results_dir auto-discovery
    if args.results_dir:
        discovered = auto_discover_checkpoints(args.results_dir)
        for step, path in discovered.items():
            if step not in ckpt_paths:
                ckpt_paths[step] = path

    if not ckpt_paths:
        print("ERROR: No checkpoints found. Use --checkpoints or --results_dir")
        return

    print(f"\nLoading {len(ckpt_paths)} checkpoints:")
    checkpoint_data = {}
    for step in sorted(ckpt_paths.keys()):
        path = ckpt_paths[step]
        print(f"  Step {step}: {path}")
        active, scores, _, _ = load_active_heads(path, top_k=args.top_k)
        new = active - base_heads
        lost = base_heads - active
        retained = active & base_heads
        print(f"    Active: {len(active)}, New: {len(new)}, Retained: {len(retained)}, Lost base: {len(lost)}")
        checkpoint_data[step] = {"active": active, "scores": scores}

    # Get final checkpoint data
    final_step = max(checkpoint_data.keys())
    final_heads = checkpoint_data[final_step]["active"]
    final_scores = checkpoint_data[final_step]["scores"]

    # Plot 1: Cohort stacked area chart (Figure 2A)
    print("\nPlotting cohort chart...")
    plot_cohort_chart(base_heads, checkpoint_data,
                      os.path.join(args.output_dir, "cohort_chart.png"))

    # Plot 2: Base-to-final heatmap (Figure 2C)
    print("Plotting base-to-final heatmap...")
    plot_head_change_heatmap(base_heads, final_heads, base_scores, final_scores,
                             n_layers, n_heads,
                             os.path.join(args.output_dir, "head_change_heatmap.png"))

    # Plot 3: Full activation timeline
    print("Plotting activation timeline...")
    plot_head_activation_over_time(base_heads, checkpoint_data, n_layers, n_heads,
                                   os.path.join(args.output_dir, "activation_timeline.png"))

    # Save summary
    summary = {
        "base_heads": sorted(list(base_heads)),
        "final_heads": sorted(list(final_heads)),
        "new_in_final": sorted(list(final_heads - base_heads)),
        "lost_from_base": sorted(list(base_heads - final_heads)),
        "retained": sorted(list(base_heads & final_heads)),
        "steps": sorted(list(checkpoint_data.keys())),
        "top_k": args.top_k,
        "per_step": {
            str(step): {
                "n_active": len(d["active"]),
                "n_new": len(d["active"] - base_heads),
                "n_retained": len(d["active"] & base_heads),
                "new_heads": sorted(list(d["active"] - base_heads)),
            }
            for step, d in checkpoint_data.items()
        },
    }
    summary_path = os.path.join(args.output_dir, "evolution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
