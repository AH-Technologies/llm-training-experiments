#!/usr/bin/env python3
"""Systematic ablation study across head identification methods.

Tests causal importance of each method's heads via:
  - Incremental removal curves (top-1 through top-N)
  - Individual head scaling
  - Random controls
  - Bottom-K controls

Reuses vLLM infrastructure from ablate_heads_v2_vllm.py.

Usage:
  python -m reasoning_head_analysis.ablate_systematic \
      --importance_path .../aggregated/head_importance.pt \
      --output_dir results/.../ablation/systematic/eap_ig \
      --do_incremental --do_individual --do_bottom
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from reasoning_head_analysis.ablate_heads_v2 import (
    BENCHMARKS,
    MATH_SYSTEM_PROMPT,
    _BOXED_RE,
    build_prompt,
    cohens_d,
    grade,
    load_amc,
    load_math500,
    paired_wilcoxon,
    parse_scales,
    per_problem_passk,
    wilson_ci,
)
from reasoning_head_analysis.ablate_heads_v2_vllm import (
    generate_samples_vllm,
    modify_weights_for_heads,
    restore_weights,
    run_condition_vllm,
    save_original_weights,
)
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Condition builder
# ═══════════════════════════════════════════════════════════════════════

def build_systematic_conditions(ranked_heads, all_heads, scales,
                                max_incr=20, top_individual=10,
                                n_random=10, do_incremental=False,
                                do_individual=False, do_random=False,
                                do_bottom=False, seed=42):
    """Build conditions for systematic ablation study.

    Returns list of (name, heads, scale, kind) tuples.
    """
    conds = [("baseline", [], 1.0, "baseline")]

    n_avail = len(ranked_heads)

    if do_incremental:
        top_n = min(max_incr, n_avail)
        for k in range(1, top_n + 1):
            heads_k = ranked_heads[:k]
            for s in scales:
                conds.append((f"incr_top{k}_s{s}", heads_k, s, "incremental"))

    if do_individual:
        top_n = min(top_individual, n_avail)
        for i in range(top_n):
            l, h = ranked_heads[i]
            for s in scales:
                conds.append((f"indiv_L{l}H{h}_s{s}", [(l, h)], s, "individual"))

    if do_random:
        rng = random.Random(seed)
        avail = list(all_heads)
        # 10 random sets of 10 heads, scale=0.0
        for r in range(n_random):
            draw = list(avail)
            rng.shuffle(draw)
            conds.append((f"rand{r}_k10_s0.0", draw[:10], 0.0, "random"))
        # 10 random sets of 20 heads, scale=0.0
        for r in range(n_random):
            draw = list(avail)
            rng.shuffle(draw)
            conds.append((f"rand{r}_k20_s0.0", draw[:20], 0.0, "random"))

    if do_bottom:
        # Bottom-K heads (least important by score)
        reversed_heads = list(reversed(ranked_heads))
        for k in [10, 20]:
            if k <= len(reversed_heads):
                conds.append((f"bottom{k}_s0.0", reversed_heads[:k], 0.0, "bottom"))

    return conds


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_incremental_curve(all_df, output, benchmark):
    """x = number of heads removed, y = pass@1. One line per scale. CI bands."""
    incr = all_df[all_df["condition"].str.startswith("incr_top")].copy()
    if incr.empty:
        return

    # Parse K and scale from condition name
    def parse_incr(name):
        m = re.match(r"incr_top(\d+)_s(.+)", name)
        if m:
            return int(m.group(1)), float(m.group(2))
        return None, None

    incr["n_heads_removed"] = incr["condition"].apply(lambda c: parse_incr(c)[0])
    incr["scale_val"] = incr["condition"].apply(lambda c: parse_incr(c)[1])
    incr = incr.dropna(subset=["n_heads_removed"])

    # Get baseline accuracy
    bl = all_df[all_df["condition"] == "baseline"]
    bl_k = int(bl["correct"].sum())
    bl_n = len(bl)
    bl_acc = bl_k / bl_n * 100 if bl_n > 0 else 0
    bl_lo, bl_hi = wilson_ci(bl_k, bl_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ci, (scale, grp) in enumerate(sorted(incr.groupby("scale_val"))):
        agg = []
        for k_val, sub in grp.groupby("n_heads_removed"):
            k = int(sub["correct"].sum())
            n = len(sub)
            lo, hi = wilson_ci(k, n)
            agg.append((k_val, k / n * 100, lo * 100, hi * 100))
        agg.sort()
        xs = [a[0] for a in agg]
        ys = [a[1] for a in agg]
        lo = [a[2] for a in agg]
        hi = [a[3] for a in agg]
        label = f"scale={scale}"
        if scale == 0.0:
            label += " (ablate)"
        ax.plot(xs, ys, marker="o", markersize=4, label=label, color=colors[ci % 10])
        ax.fill_between(xs, lo, hi, alpha=0.15, color=colors[ci % 10])

    ax.axhline(bl_acc, color="black", ls="--", alpha=0.7, label="baseline")
    ax.fill_between(ax.get_xlim(), bl_lo * 100, bl_hi * 100, alpha=0.1, color="black")
    ax.set_xlabel("Number of heads intervened")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{benchmark}: Incremental Head Ablation/Scaling")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {output}")


def plot_individual_head_impact(all_df, output, benchmark):
    """Grouped bar chart: each head on x-axis, bars for each scale, y = accuracy."""
    indiv = all_df[all_df["condition"].str.startswith("indiv_")].copy()
    if indiv.empty:
        return

    def parse_indiv(name):
        m = re.match(r"indiv_L(\d+)H(\d+)_s(.+)", name)
        if m:
            return (int(m.group(1)), int(m.group(2))), float(m.group(3))
        return None, None

    indiv["head"] = indiv["condition"].apply(lambda c: parse_indiv(c)[0])
    indiv["scale_val"] = indiv["condition"].apply(lambda c: parse_indiv(c)[1])
    indiv = indiv.dropna(subset=["head"])

    heads = sorted(indiv["head"].unique())
    scales = sorted(indiv["scale_val"].unique())

    # Get baseline
    bl = all_df[all_df["condition"] == "baseline"]
    bl_acc = bl["correct"].mean() * 100 if len(bl) > 0 else 0

    fig, ax = plt.subplots(figsize=(max(10, len(heads) * 1.2), 6))
    x = np.arange(len(heads))
    width = 0.8 / max(len(scales), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(scales), 1)))

    for si, scale in enumerate(scales):
        accs = []
        errs_lo = []
        errs_hi = []
        for head in heads:
            sub = indiv[(indiv["head"] == head) & (indiv["scale_val"] == scale)]
            if len(sub) == 0:
                accs.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
                continue
            k = int(sub["correct"].sum())
            n = len(sub)
            acc = k / n * 100
            lo, hi = wilson_ci(k, n)
            accs.append(acc)
            errs_lo.append(acc - lo * 100)
            errs_hi.append(hi * 100 - acc)
        offset = (si - (len(scales) - 1) / 2) * width
        label = f"scale={scale}"
        if scale == 0.0:
            label += " (ablate)"
        ax.bar(x + offset, accs, width, yerr=[errs_lo, errs_hi],
               capsize=2, label=label, color=colors[si], alpha=0.85,
               edgecolor="black", linewidth=0.5)

    ax.axhline(bl_acc, color="black", ls="--", alpha=0.7, label="baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{h[0]}H{h[1]}" for h in heads], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{benchmark}: Individual Head Impact")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {output}")


def plot_scale_response(all_df, output, benchmark):
    """For top-10 and top-20, plot accuracy vs scale factor."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Get baseline
    bl = all_df[all_df["condition"] == "baseline"]
    bl_k = int(bl["correct"].sum())
    bl_n = len(bl)
    bl_acc = bl_k / bl_n * 100 if bl_n > 0 else 0

    for k_val, color, marker in [(10, "darkorange", "o"), (20, "steelblue", "s")]:
        # Filter incremental conditions for this k
        prefix = f"incr_top{k_val}_s"
        sub = all_df[all_df["condition"].str.startswith(prefix)].copy()
        if sub.empty:
            continue

        def get_scale(name):
            m = re.match(rf"incr_top{k_val}_s(.+)", name)
            return float(m.group(1)) if m else None

        sub["scale_val"] = sub["condition"].apply(get_scale)
        sub = sub.dropna(subset=["scale_val"])

        agg = []
        for scale, grp in sub.groupby("scale_val"):
            k = int(grp["correct"].sum())
            n = len(grp)
            lo, hi = wilson_ci(k, n)
            agg.append((scale, k / n * 100, lo * 100, hi * 100))
        agg.sort()
        if not agg:
            continue
        xs = [a[0] for a in agg]
        ys = [a[1] for a in agg]
        lo = [a[2] for a in agg]
        hi = [a[3] for a in agg]
        ax.plot(xs, ys, marker=marker, label=f"top-{k_val} heads", color=color)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=color)

    ax.axhline(bl_acc, color="black", ls="--", alpha=0.7, label="baseline")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.5, label="no intervention")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{benchmark}: Scale Response Curve")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved {output}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Systematic ablation study for head identification methods")
    parser.add_argument("--importance_path", default=None,
                        help="Path to aggregated head_importance.pt (not needed for --do_random only)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", default="math500|amc",
                        help="Pipe-separated benchmark list")
    parser.add_argument("--math500_path", default="data/math500.parquet")
    parser.add_argument("--math500_n", type=int, default=500,
                        help="Number of MATH-500 problems to use")
    parser.add_argument("--amc_n", type=int, default=40,
                        help="Number of AMC problems to use")
    parser.add_argument("--scales", default="0.0|0.5|1.5|2.0")
    parser.add_argument("--max_incr", type=int, default=20,
                        help="Incremental: remove top-1 through top-N")
    parser.add_argument("--top_individual", type=int, default=10,
                        help="Individual: test top-N heads")
    parser.add_argument("--n_random", type=int, default=10,
                        help="Random: N random draws")
    parser.add_argument("--do_incremental", action="store_true")
    parser.add_argument("--do_individual", action="store_true")
    parser.add_argument("--do_random", action="store_true")
    parser.add_argument("--do_bottom", action="store_true")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Head selection ────────────────────────────────────────────────
    if args.importance_path:
        logger.info(f"Loading head importance from {args.importance_path}")
        data = torch.load(args.importance_path, map_location="cpu", weights_only=False)
        head_scores = data["head_scores"]
        n_layers, n_heads_total = head_scores.shape
        flat = head_scores.flatten()
        sorted_idx = torch.argsort(flat, descending=True)
        ranked_heads = [(i.item() // n_heads_total, i.item() % n_heads_total)
                        for i in sorted_idx]
        all_heads = ranked_heads.copy()
        logger.info(f"Loaded {n_layers}x{n_heads_total} head scores; "
                    f"top-5: {ranked_heads[:5]}")
    elif args.do_random:
        # For random-only mode, generate all possible heads
        # Qwen2.5-1.5B: 28 layers, 12 heads
        n_layers, n_heads_total = 28, 12
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_total)]
        ranked_heads = all_heads  # no ranking needed for random
        logger.info(f"Random-only mode: {len(all_heads)} heads")
    else:
        parser.error("--importance_path required unless --do_random only")

    scales = parse_scales(args.scales)

    bench_list = [b for b in re.split(r"[,|;\s]+", args.benchmarks.strip()) if b]
    bench_n = {"math500": args.math500_n, "amc": args.amc_n}

    # ─── Build conditions ──────────────────────────────────────────────
    conditions = build_systematic_conditions(
        ranked_heads, all_heads, scales,
        max_incr=args.max_incr,
        top_individual=args.top_individual,
        n_random=args.n_random,
        do_incremental=args.do_incremental,
        do_individual=args.do_individual,
        do_random=args.do_random,
        do_bottom=args.do_bottom,
        seed=args.seed,
    )
    logger.info(f"{len(conditions)} conditions; benchmarks: {bench_list}")

    # ─── vLLM model ───────────────────────────────────────────────────
    llm, tokenizer = None, None
    if not args.analyze_only:
        logger.info(f"Loading vLLM model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=args.seed,
        )
        tokenizer = llm.get_tokenizer()
        n_saved = save_original_weights(llm)
        logger.info(f"Saved original o_proj weights for {n_saved} layers")

    # ─── Run each benchmark ───────────────────────────────────────────
    for bench in bench_list:
        n_prob = bench_n.get(bench, 500)
        logger.info(f"\n{'='*60} Benchmark: {bench} (n={n_prob}) {'='*60}")

        if bench == "math500":
            problems = load_math500(args.math500_path, n_prob)
        elif bench == "amc":
            problems = load_amc(n_prob)
        else:
            raise ValueError(f"Unknown benchmark: {bench}")
        logger.info(f"Loaded {len(problems)} problems")

        bench_dir = os.path.join(args.output_dir, bench)
        os.makedirs(bench_dir, exist_ok=True)

        if not args.analyze_only:
            my_conditions = [(i, c) for i, c in enumerate(conditions)
                             if i % args.n_shards == args.shard_idx]
            logger.info(f"Shard {args.shard_idx}/{args.n_shards}: "
                        f"{len(my_conditions)} of {len(conditions)} conditions")

            for ci, (name, heads, scale, kind) in my_conditions:
                out_path = os.path.join(bench_dir, f"results_{name}.csv")
                if os.path.exists(out_path):
                    logger.info(f"  {name}: already done, skipping")
                    continue
                full_path = os.path.join(
                    bench_dir, f"full_responses_shard{args.shard_idx}.jsonl")
                df = run_condition_vllm(
                    llm, tokenizer, problems, bench, name, heads, scale,
                    args.n_samples, args.max_new_tokens, args.temperature,
                    top_p=args.top_p,
                    full_responses_path=full_path,
                )
                df["condition_kind"] = kind
                df.to_csv(out_path, index=False)

            restore_weights(llm)

        # ─── Per-benchmark aggregation ────────────────────────────────
        all_dfs = []
        for name, _, _, _ in conditions:
            p = os.path.join(bench_dir, f"results_{name}.csv")
            if os.path.exists(p):
                all_dfs.append(pd.read_csv(p))
        if not all_dfs:
            logger.warning(f"No result CSVs in {bench_dir}; skipping analysis.")
            continue

        all_df = pd.concat(all_dfs, ignore_index=True)
        have = set(all_df["condition"].unique())
        planned = {c[0] for c in conditions}
        missing = planned - have
        if missing:
            logger.warning(f"[{bench}] Missing {len(missing)} condition(s): "
                           f"{sorted(missing)[:5]}...")
        all_df.to_csv(os.path.join(bench_dir, "all_results.csv"), index=False)

        # Stats
        baseline_df = all_df[all_df["condition"] == "baseline"]
        baseline_passk = per_problem_passk(baseline_df) if len(baseline_df) > 0 else np.array([])
        stats_out = {}
        for cond, sub in all_df.groupby("condition"):
            k = int(sub["correct"].sum())
            n = len(sub)
            lo, hi = wilson_ci(k, n)
            passk = per_problem_passk(sub)
            if cond != "baseline" and len(passk) == len(baseline_passk) and len(baseline_passk) > 0:
                stat, p_val = paired_wilcoxon(baseline_passk, passk)
                d = cohens_d(baseline_passk, passk)
            else:
                stat, p_val, d = None, None, None
            stats_out[cond] = {
                "pass_at_1_avg": float(k / n) if n > 0 else 0,
                "n_gens": int(n),
                "wilson_95_lo": float(lo), "wilson_95_hi": float(hi),
                "pass_at_k": float(passk.mean()) if len(passk) > 0 else 0,
                "paired_wilcoxon_stat": stat, "paired_wilcoxon_p": p_val,
                "cohens_d_vs_baseline": d,
                "mean_length_tokens": float(sub["length_tokens"].mean()),
            }

        with open(os.path.join(bench_dir, "stats.json"), "w") as f:
            json.dump({
                "model": args.model, "benchmark": bench,
                "n_problems": len(problems), "n_samples": args.n_samples,
                "scales": scales,
                "conditions": stats_out,
                "importance_path": args.importance_path,
            }, f, indent=2, default=str)

        # Plots
        plot_incremental_curve(all_df,
                               os.path.join(bench_dir, "incremental_curve.png"),
                               bench)
        plot_individual_head_impact(all_df,
                                    os.path.join(bench_dir, "individual_head_impact.png"),
                                    bench)
        plot_scale_response(all_df,
                            os.path.join(bench_dir, "scale_response.png"),
                            bench)

    logger.info(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
