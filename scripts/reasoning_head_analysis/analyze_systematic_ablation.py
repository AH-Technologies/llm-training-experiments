#!/usr/bin/env python3
"""Cross-method comparison of systematic ablation results.

Generates:
  1. Method comparison plot: overlay incremental curves for all methods
  2. Summary table: accuracy at key points vs baseline vs random
  3. Statistical tests: paired Wilcoxon vs baseline at each step

Usage:
  python scripts/reasoning_head_analysis/analyze_systematic_ablation.py
  python scripts/reasoning_head_analysis/analyze_systematic_ablation.py --results_dir <custom_dir>
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from reasoning_head_analysis.ablate_heads_v2 import (
    cohens_d,
    paired_wilcoxon,
    per_problem_passk,
    wilson_ci,
)

RESULTS_BASE = "results/reasoning_head_analysis/ablation/systematic"

METHODS = {
    "eap_ig": {"label": "EAP-IG", "color": "steelblue", "marker": "o"},
    "neurosurgery": {"label": "Neurosurgery", "color": "darkorange", "marker": "s"},
    "retrieval": {"label": "Retrieval", "color": "seagreen", "marker": "^"},
    "random_control": {"label": "Random", "color": "crimson", "marker": "x"},
}

SEP = "=" * 70


def load_method_results(results_dir, method, benchmark):
    """Load all_results.csv for a method/benchmark combo."""
    path = os.path.join(results_dir, method, benchmark, "all_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def parse_incr_condition(name):
    """Parse incr_topK_sS → (K, scale) or None."""
    m = re.match(r"incr_top(\d+)_s(.+)", name)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def get_incremental_curve(df, scale=0.0):
    """Extract (n_heads, accuracy, ci_lo, ci_hi) for incremental conditions at given scale."""
    points = []
    for cond in df["condition"].unique():
        k, s = parse_incr_condition(cond)
        if k is not None and abs(s - scale) < 1e-6:
            sub = df[df["condition"] == cond]
            n_correct = int(sub["correct"].sum())
            n_total = len(sub)
            acc = n_correct / n_total * 100
            lo, hi = wilson_ci(n_correct, n_total)
            points.append((k, acc, lo * 100, hi * 100))
    points.sort()
    return points


def get_baseline_acc(df):
    bl = df[df["condition"] == "baseline"]
    if len(bl) == 0:
        return 0, 0, 0
    k = int(bl["correct"].sum())
    n = len(bl)
    lo, hi = wilson_ci(k, n)
    return k / n * 100, lo * 100, hi * 100


def get_random_acc(df, k_heads=10):
    """Get mean random-control accuracy for sets of size k_heads."""
    rand_conds = [c for c in df["condition"].unique()
                  if re.match(rf"rand\d+_k{k_heads}_s0\.0", c)]
    if not rand_conds:
        return None, None, None
    sub = df[df["condition"].isin(rand_conds)]
    k = int(sub["correct"].sum())
    n = len(sub)
    lo, hi = wilson_ci(k, n)
    return k / n * 100, lo * 100, hi * 100


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Method comparison (incremental curves overlaid)
# ═══════════════════════════════════════════════════════════════════════

def plot_method_comparison(results_dir, benchmark, output_dir, scale=0.0):
    """Overlay incremental ablation curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for method, info in METHODS.items():
        df = load_method_results(results_dir, method, benchmark)
        if df is None:
            continue
        points = get_incremental_curve(df, scale=scale)
        if not points:
            # For random_control, show random k=10 and k=20 as horizontal lines
            if method == "random_control":
                for k_h, ls in [(10, "--"), (20, ":")]:
                    acc, lo, hi = get_random_acc(df, k_h)
                    if acc is not None:
                        ax.axhline(acc, color=info["color"], ls=ls, alpha=0.7,
                                   label=f"{info['label']} k={k_h}")
                        ax.fill_between(ax.get_xlim(), lo, hi,
                                        alpha=0.08, color=info["color"])
                        has_data = True
            continue
        has_data = True
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        lo = [p[2] for p in points]
        hi = [p[3] for p in points]
        ax.plot(xs, ys, marker=info["marker"], markersize=5,
                label=info["label"], color=info["color"])
        ax.fill_between(xs, lo, hi, alpha=0.15, color=info["color"])

    if not has_data:
        plt.close(fig)
        return

    # Baseline from any available method
    for method in METHODS:
        df = load_method_results(results_dir, method, benchmark)
        if df is not None:
            bl_acc, bl_lo, bl_hi = get_baseline_acc(df)
            ax.axhline(bl_acc, color="black", ls="--", alpha=0.7, label="baseline")
            break

    scale_label = "ablation (scale=0)" if scale == 0.0 else f"scale={scale}"
    ax.set_xlabel("Number of heads removed")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{benchmark}: Incremental {scale_label} — Method Comparison")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"method_comparison_{benchmark}_s{scale}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════

def print_summary_table(results_dir, benchmark, scale=0.0):
    """Print accuracy at key incremental steps vs baseline vs random."""
    print(f"\n{SEP}")
    print(f"  SUMMARY: {benchmark} (scale={scale})")
    print(SEP)

    key_steps = [1, 5, 10, 20]

    # Header
    header = f"  {'Method':<15s}  {'Baseline':>8s}"
    for k in key_steps:
        header += f"  {'top-'+str(k):>8s}"
    header += f"  {'rand-10':>8s}  {'rand-20':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for method, info in METHODS.items():
        df = load_method_results(results_dir, method, benchmark)
        if df is None:
            continue

        bl_acc, _, _ = get_baseline_acc(df)
        row = f"  {info['label']:<15s}  {bl_acc:7.1f}%"

        for k in key_steps:
            cond = f"incr_top{k}_s{scale}"
            sub = df[df["condition"] == cond]
            if len(sub) > 0:
                acc = sub["correct"].mean() * 100
                row += f"  {acc:7.1f}%"
            else:
                row += f"  {'—':>8s}"

        for k_h in [10, 20]:
            acc, _, _ = get_random_acc(df, k_h)
            if acc is not None:
                row += f"  {acc:7.1f}%"
            else:
                row += f"  {'—':>8s}"

        print(row)


# ═══════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════

def print_statistical_tests(results_dir, benchmark, scale=0.0):
    """Paired Wilcoxon vs baseline at each incremental step."""
    print(f"\n{SEP}")
    print(f"  STATISTICAL TESTS: {benchmark} (scale={scale})")
    print(SEP)

    key_steps = [1, 5, 10, 20]

    for method, info in METHODS.items():
        df = load_method_results(results_dir, method, benchmark)
        if df is None:
            continue

        bl = df[df["condition"] == "baseline"]
        if len(bl) == 0:
            continue
        bl_passk = per_problem_passk(bl)

        print(f"\n  {info['label']}:")
        print(f"  {'Step':>6s}  {'acc%':>6s}  {'delta':>7s}  {'Wilcoxon p':>10s}  {'Cohen d':>8s}  {'sig':>5s}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*5}")

        for k in key_steps:
            cond = f"incr_top{k}_s{scale}"
            sub = df[df["condition"] == cond]
            if len(sub) == 0:
                continue
            acc = sub["correct"].mean() * 100
            bl_acc = bl["correct"].mean() * 100
            delta = acc - bl_acc
            passk = per_problem_passk(sub)
            if len(passk) == len(bl_passk):
                stat, p_val = paired_wilcoxon(bl_passk, passk)
                d = cohens_d(bl_passk, passk)
                sig = "***" if p_val and p_val < 0.001 else \
                      "**" if p_val and p_val < 0.01 else \
                      "*" if p_val and p_val < 0.05 else ""
                p_str = f"{p_val:.2e}" if p_val is not None else "—"
                d_str = f"{d:.3f}" if d is not None else "—"
            else:
                p_str, d_str, sig = "—", "—", ""
            print(f"  top-{k:>2d}  {acc:5.1f}%  {delta:+6.1f}%  {p_str:>10s}  {d_str:>8s}  {sig:>5s}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-method comparison of systematic ablation results")
    parser.add_argument("--results_dir", default=RESULTS_BASE,
                        help="Base results directory")
    parser.add_argument("--benchmarks", default="math500|amc",
                        help="Pipe-separated benchmark list")
    parser.add_argument("--scales", default="0.0",
                        help="Scales to analyze (pipe-separated)")
    args = parser.parse_args()

    benchmarks = [b for b in re.split(r"[,|;\s]+", args.benchmarks.strip()) if b]
    scales = [float(s) for s in re.split(r"[,|;\s]+", args.scales.strip()) if s]

    output_dir = os.path.join(args.results_dir, "cross_method")
    os.makedirs(output_dir, exist_ok=True)

    print(SEP)
    print("  SYSTEMATIC ABLATION — CROSS-METHOD ANALYSIS")
    print(SEP)

    # Check what's available
    available = {}
    for method in METHODS:
        for bench in benchmarks:
            df = load_method_results(args.results_dir, method, bench)
            if df is not None:
                available.setdefault(bench, []).append(method)
                n_conds = df["condition"].nunique()
                print(f"  {method}/{bench}: {n_conds} conditions, {len(df)} rows")

    if not available:
        print("\n  No results found. Run ablation jobs first.")
        sys.exit(1)

    for bench in benchmarks:
        if bench not in available:
            print(f"\n  {bench}: no results available, skipping")
            continue

        for scale in scales:
            plot_method_comparison(args.results_dir, bench, output_dir, scale=scale)
            print_summary_table(args.results_dir, bench, scale=scale)
            print_statistical_tests(args.results_dir, bench, scale=scale)

    # Save combined stats JSON
    combined = {}
    for bench in benchmarks:
        for method in METHODS:
            stats_path = os.path.join(args.results_dir, method, bench, "stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    combined.setdefault(bench, {})[method] = json.load(f)

    if combined:
        out_path = os.path.join(output_dir, "combined_stats.json")
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\n  Saved combined stats to {out_path}")

    print(f"\n{SEP}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Plots saved to: {output_dir}")
    print(SEP)


if __name__ == "__main__":
    main()
