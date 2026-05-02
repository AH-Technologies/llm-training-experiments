#!/usr/bin/env python3
"""Publication-quality cross-method comparison plots for systematic ablation.

Generates:
  1. Incremental ablation overlay — all methods on one plot (zeroing only)
  2. Grouped bar chart — accuracy at key ablation sizes (top-1, 5, 10, 20) + controls
  3. Head location scatter — where in the model each method places its top heads
  4. Individual head impact — delta from baseline when zeroing each method's top-10 heads
  5. Response length analysis — how ablation changes generation length

Usage:
  srun ... python scripts/reasoning_head_analysis/compare_methods_plots.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

RESULTS_BASE = "results/reasoning_head_analysis/ablation/systematic"
IDENT_BASE = "results/reasoning_head_analysis/identification/systematic_base"

METHODS = {
    "eap_ig":       {"label": "EAP-IG",       "color": "#2166ac", "marker": "o"},
    "neurosurgery": {"label": "Neurosurgery",  "color": "#d6604d", "marker": "s"},
    "retrieval":    {"label": "Retrieval",     "color": "#1b7837", "marker": "^"},
}

METHOD_ORDER = ["eap_ig", "neurosurgery", "retrieval"]


def load_csv(method, benchmark):
    path = os.path.join(RESULTS_BASE, method, benchmark, "all_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_stats(method, benchmark):
    path = os.path.join(RESULTS_BASE, method, benchmark, "stats.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_random_stats(benchmark):
    return load_stats("random_control", benchmark)


def load_top_heads(method, k=20):
    path = os.path.join(IDENT_BASE, method, "aggregated", "head_importance.pt")
    if not os.path.exists(path):
        return []
    d = torch.load(path, weights_only=False)
    return d["active_heads"][:k]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - spread), min(1, centre + spread)


def get_incr_data(stats, scale=0.0):
    """Extract incremental curve data from stats dict."""
    conds = stats["conditions"]
    points = []
    for name, data in conds.items():
        m = re.match(r"incr_top(\d+)_s(.+)", name)
        if m and abs(float(m.group(2)) - scale) < 1e-6:
            k = int(m.group(1))
            acc = data["pass_at_1_avg"] * 100
            lo = data["wilson_95_lo"] * 100
            hi = data["wilson_95_hi"] * 100
            points.append((k, acc, lo, hi))
    points.sort()
    return points


def get_random_band(rand_stats, k_heads=10):
    """Get mean and range of random controls."""
    if rand_stats is None:
        return None
    conds = rand_stats["conditions"]
    accs = []
    for name, data in conds.items():
        if re.match(rf"rand\d+_k{k_heads}_s0\.0", name):
            accs.append(data["pass_at_1_avg"] * 100)
    if not accs:
        return None
    return {"mean": np.mean(accs), "min": np.min(accs), "max": np.max(accs),
            "std": np.std(accs), "n": len(accs)}


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Incremental ablation overlay
# ═══════════════════════════════════════════════════════════════════════

def plot_incremental_overlay(benchmark, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Load baseline
    bl_acc = None
    for method in METHOD_ORDER:
        stats = load_stats(method, benchmark)
        if stats and "baseline" in stats["conditions"]:
            bl_acc = stats["conditions"]["baseline"]["pass_at_1_avg"] * 100
            break

    # Random control band
    rand_stats = load_random_stats(benchmark)
    rand10 = get_random_band(rand_stats, 10)
    if rand10:
        ax.fill_between([0.5, 20.5], rand10["min"], rand10["max"],
                        alpha=0.10, color="grey", label=f"Random k=10 range (n={rand10['n']})")
        ax.axhline(rand10["mean"], color="grey", ls=":", alpha=0.5)

    # Method curves
    for method in METHOD_ORDER:
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue
        points = get_incr_data(stats, scale=0.0)
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        lo = [p[2] for p in points]
        hi = [p[3] for p in points]
        ax.plot(xs, ys, marker=info["marker"], markersize=6, linewidth=2,
                label=info["label"], color=info["color"], zorder=3)
        ax.fill_between(xs, lo, hi, alpha=0.12, color=info["color"])

    # Baseline
    if bl_acc is not None:
        ax.axhline(bl_acc, color="black", ls="--", linewidth=1.5, alpha=0.7, label="Baseline (unmodified)")

    ax.set_xlabel("Number of top heads zeroed", fontsize=12)
    ax.set_ylabel("pass@1 (%)", fontsize=12)
    ax.set_title(f"Incremental Head Ablation — Method Comparison ({benchmark.upper()})", fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, f"incremental_overlay_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Grouped bar chart at key ablation sizes + controls
# ═══════════════════════════════════════════════════════════════════════

def plot_grouped_bars(benchmark, output_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Conditions to show
    groups = [
        ("top-1", "incr_top1_s0.0"),
        ("top-5", "incr_top5_s0.0"),
        ("top-10", "incr_top10_s0.0"),
        ("top-20", "incr_top20_s0.0"),
        ("bottom-10", "bottom10_s0.0"),
        ("bottom-20", "bottom20_s0.0"),
    ]

    x = np.arange(len(groups))
    width = 0.22
    offsets = [-width, 0, width]

    bl_acc = None
    for i, method in enumerate(METHOD_ORDER):
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue
        if bl_acc is None:
            bl_acc = stats["conditions"]["baseline"]["pass_at_1_avg"] * 100

        accs = []
        errs_lo = []
        errs_hi = []
        for label, cond in groups:
            if cond in stats["conditions"]:
                d = stats["conditions"][cond]
                acc = d["pass_at_1_avg"] * 100
                lo = d["wilson_95_lo"] * 100
                hi = d["wilson_95_hi"] * 100
                accs.append(acc)
                errs_lo.append(max(0, acc - lo))
                errs_hi.append(max(0, hi - acc))
            else:
                accs.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        bars = ax.bar(x + offsets[i], accs, width, label=info["label"],
                      color=info["color"], alpha=0.85, zorder=3)
        ax.errorbar(x + offsets[i], accs, yerr=[errs_lo, errs_hi],
                    fmt="none", ecolor="black", capsize=3, capthick=1, zorder=4)

    # Random control (average of k=10 zeroed)
    rand_stats = load_random_stats(benchmark)
    rand10 = get_random_band(rand_stats, 10)
    rand20 = get_random_band(rand_stats, 20)

    if bl_acc is not None:
        ax.axhline(bl_acc, color="black", ls="--", linewidth=1.5, alpha=0.7, label="Baseline")

    if rand10:
        # Draw random bands at the corresponding x positions (top-10 = index 2)
        ax.axhspan(rand10["min"], rand10["max"], xmin=0, xmax=1,
                   alpha=0.06, color="grey")
        ax.axhline(rand10["mean"], color="grey", ls=":", alpha=0.5, label=f"Random avg (k=10: {rand10['mean']:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_ylabel("pass@1 (%)", fontsize=12)
    ax.set_title(f"Ablation Impact by Method and Condition ({benchmark.upper()})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, f"grouped_bars_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Head location scatter — layer × head position
# ═══════════════════════════════════════════════════════════════════════

def plot_head_locations(output_dir, k=20):
    fig, ax = plt.subplots(figsize=(10, 6))

    y_offsets = {"eap_ig": -0.15, "neurosurgery": 0, "retrieval": 0.15}
    sizes = {"eap_ig": 60, "neurosurgery": 50, "retrieval": 50}

    for method in METHOD_ORDER:
        info = METHODS[method]
        heads = load_top_heads(method, k)
        if not heads:
            continue

        layers = [h[0] for h in heads]
        head_ids = [h[1] for h in heads]
        # Size proportional to rank (top-1 biggest)
        ss = [sizes[method] * (1.2 - 0.5 * i / len(heads)) for i in range(len(heads))]

        ax.scatter([h + y_offsets[method] for h in head_ids], layers,
                   s=ss, c=info["color"], marker=info["marker"],
                   label=info["label"], alpha=0.7, edgecolors="white", linewidth=0.5,
                   zorder=3)

        # Label top-5 with rank
        for rank, (layer, head) in enumerate(heads[:5]):
            ax.annotate(f"#{rank+1}", (head + y_offsets[method], layer),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 6), textcoords="offset points",
                        color=info["color"], fontweight="bold")

    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Top-20 Head Locations by Method", fontsize=13, fontweight="bold")
    ax.set_xticks(range(12))
    ax.set_yticks(range(28))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 27.5)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.15)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(output_dir, "head_locations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Individual head impact — delta from baseline
# ═══════════════════════════════════════════════════════════════════════

def plot_individual_deltas(benchmark, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, method in enumerate(METHOD_ORDER):
        ax = axes[ax_idx]
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue

        bl_acc = stats["conditions"]["baseline"]["pass_at_1_avg"] * 100

        # Find individual head conditions (zeroing only)
        indiv = []
        for name, data in stats["conditions"].items():
            m = re.match(r"indiv_L(\d+)H(\d+)_s0\.0", name)
            if m:
                layer, head = int(m.group(1)), int(m.group(2))
                acc = data["pass_at_1_avg"] * 100
                delta = acc - bl_acc
                p_val = data.get("paired_wilcoxon_p")
                sig = p_val is not None and p_val < 0.05
                indiv.append((f"L{layer}H{head}", delta, sig))

        if not indiv:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(info["label"], fontsize=12, fontweight="bold")
            continue

        # Sort by delta
        indiv.sort(key=lambda x: x[1])
        labels = [x[0] for x in indiv]
        deltas = [x[1] for x in indiv]
        sigs = [x[2] for x in indiv]

        colors = [info["color"] if s else "#cccccc" for s in sigs]
        bars = ax.barh(range(len(labels)), deltas, color=colors, alpha=0.8, edgecolor="white")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("$\\Delta$ pass@1 (pp)", fontsize=11)
        ax.set_title(info["label"], fontsize=12, fontweight="bold", color=info["color"])
        ax.grid(axis="x", alpha=0.2)

        # Add significance legend
        ax.text(0.98, 0.02, "Colored = p<0.05", fontsize=8, ha="right", va="bottom",
                transform=ax.transAxes, color="grey")

    fig.suptitle(f"Individual Head Zeroing — Change from Baseline ({benchmark.upper()})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f"individual_deltas_{benchmark}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: Response length vs accuracy scatter
# ═══════════════════════════════════════════════════════════════════════

def plot_length_vs_accuracy(benchmark, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for method in METHOD_ORDER:
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue

        # Collect all zeroing conditions
        for name, data in stats["conditions"].items():
            if name == "baseline":
                continue
            if not name.endswith("_s0.0"):
                continue

            acc = data["pass_at_1_avg"] * 100
            length = data["mean_length_tokens"]

            # Determine marker size by type
            if name.startswith("incr_"):
                m = re.match(r"incr_top(\d+)_", name)
                k = int(m.group(1)) if m else 10
                alpha = 0.5
                size = 20 + k * 2
            elif name.startswith("indiv_"):
                alpha = 0.3
                size = 25
            elif name.startswith("bottom"):
                alpha = 0.4
                size = 35
            else:
                continue

            ax.scatter(length, acc, s=size, c=info["color"], marker=info["marker"],
                       alpha=alpha, zorder=3)

    # Baseline
    for method in METHOD_ORDER:
        stats = load_stats(method, benchmark)
        if stats and "baseline" in stats["conditions"]:
            bl = stats["conditions"]["baseline"]
            ax.scatter(bl["mean_length_tokens"], bl["pass_at_1_avg"] * 100,
                       s=120, c="black", marker="*", zorder=5, label="Baseline")
            break

    # Random control
    rand_stats = load_random_stats(benchmark)
    if rand_stats:
        for name, data in rand_stats["conditions"].items():
            if name.startswith("rand") and name.endswith("_s0.0"):
                ax.scatter(data["mean_length_tokens"], data["pass_at_1_avg"] * 100,
                           s=25, c="grey", marker="x", alpha=0.4, zorder=2)

    # Legend
    handles = [plt.scatter([], [], c=METHODS[m]["color"], marker=METHODS[m]["marker"], s=40, label=METHODS[m]["label"])
               for m in METHOD_ORDER]
    handles.append(plt.scatter([], [], c="grey", marker="x", s=25, label="Random", alpha=0.5))
    handles.append(plt.scatter([], [], c="black", marker="*", s=80, label="Baseline"))
    ax.legend(handles=handles, fontsize=9, loc="best")

    ax.set_xlabel("Mean response length (tokens)", fontsize=12)
    ax.set_ylabel("pass@1 (%)", fontsize=12)
    ax.set_title(f"Accuracy vs Response Length Under Ablation ({benchmark.upper()})",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(output_dir, f"length_vs_accuracy_{benchmark}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot 6: Summary dashboard — key numbers in a clean visual
# ═══════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(benchmark, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    bl_acc = None
    for method in METHOD_ORDER:
        stats = load_stats(method, benchmark)
        if stats and "baseline" in stats["conditions"]:
            bl_acc = stats["conditions"]["baseline"]["pass_at_1_avg"] * 100
            break

    rand_stats = load_random_stats(benchmark)
    rand10 = get_random_band(rand_stats, 10)

    # Panel 1: Incremental zeroing at key K values
    ax = axes[0]
    key_ks = [1, 2, 3, 5, 10, 20]
    for method in METHOD_ORDER:
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue
        accs = []
        for k in key_ks:
            cond = f"incr_top{k}_s0.0"
            if cond in stats["conditions"]:
                accs.append(stats["conditions"][cond]["pass_at_1_avg"] * 100)
            else:
                accs.append(np.nan)
        ax.plot(key_ks, accs, marker=info["marker"], linewidth=2, markersize=7,
                color=info["color"], label=info["label"])

    if bl_acc is not None:
        ax.axhline(bl_acc, color="black", ls="--", linewidth=1, alpha=0.6, label="Baseline")
    if rand10:
        ax.axhspan(rand10["min"], rand10["max"], alpha=0.08, color="grey")
    ax.set_xlabel("Heads zeroed (K)")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Incremental Zeroing", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # Panel 2: Top-K zeroed vs Bottom-K zeroed
    ax = axes[1]
    conditions = [("top-10\nzeroed", "incr_top10_s0.0"), ("top-20\nzeroed", "incr_top20_s0.0"),
                  ("bottom-10\nzeroed", "bottom10_s0.0"), ("bottom-20\nzeroed", "bottom20_s0.0")]
    x = np.arange(len(conditions))
    width = 0.25
    offsets = [-width, 0, width]

    for i, method in enumerate(METHOD_ORDER):
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue
        accs = []
        for label, cond in conditions:
            if cond in stats["conditions"]:
                accs.append(stats["conditions"][cond]["pass_at_1_avg"] * 100)
            else:
                accs.append(0)
        ax.bar(x + offsets[i], accs, width, color=info["color"], alpha=0.85, label=info["label"])

    if bl_acc is not None:
        ax.axhline(bl_acc, color="black", ls="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=9)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Top vs Bottom Ablation", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # Panel 3: Effect of scaling (top-10 at different scales)
    ax = axes[2]
    scales = [0.0, 0.5, 1.5, 2.0]
    for method in METHOD_ORDER:
        info = METHODS[method]
        stats = load_stats(method, benchmark)
        if stats is None:
            continue
        accs = []
        for s in scales:
            cond = f"incr_top10_s{s}"
            if cond in stats["conditions"]:
                accs.append(stats["conditions"][cond]["pass_at_1_avg"] * 100)
            else:
                accs.append(np.nan)
        ax.plot(scales, accs, marker=info["marker"], linewidth=2, markersize=7,
                color=info["color"], label=info["label"])

    if bl_acc is not None:
        ax.axhline(bl_acc, color="black", ls="--", linewidth=1, alpha=0.6, label="Baseline")
    ax.axvline(1.0, color="grey", ls=":", alpha=0.3)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Scale Response (top-10)", fontweight="bold")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle(f"Systematic Ablation Summary — {benchmark.upper()}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f"summary_dashboard_{benchmark}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    output_dir = os.path.join(RESULTS_BASE, "cross_method")
    os.makedirs(output_dir, exist_ok=True)

    benchmarks = ["math500", "amc"]

    print("=" * 60)
    print("  Cross-Method Comparison Plots")
    print("=" * 60)

    # Head locations (benchmark-independent)
    print("\nPlot: Head locations...")
    plot_head_locations(output_dir)

    for bench in benchmarks:
        print(f"\n--- {bench.upper()} ---")

        print("Plot: Incremental overlay...")
        plot_incremental_overlay(bench, output_dir)

        print("Plot: Grouped bars...")
        plot_grouped_bars(bench, output_dir)

        print("Plot: Individual deltas...")
        plot_individual_deltas(bench, output_dir)

        print("Plot: Length vs accuracy...")
        plot_length_vs_accuracy(bench, output_dir)

        print("Plot: Summary dashboard...")
        plot_summary_dashboard(bench, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
