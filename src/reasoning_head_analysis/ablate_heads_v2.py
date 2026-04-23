#!/usr/bin/env python3
"""Thesis-quality head-ablation + scaling study.

Improvements over ablate_heads.py:
  - Full benchmark sweeps (MATH-500 500 problems, AIME'24 30, AMC 40)
  - n_samples sampling at training temperature → pass@1 averaging + pass@k
  - Wilson 95% CIs, paired Wilcoxon, Cohen's d vs baseline
  - Richer controls: 10× random-K (not 3), optional individual-head scan,
    support for applying one model's heads to another (cross-model control)
  - Length + per-category + per-difficulty breakdowns
  - Per-sample DataFrame saved so analysis can be rerun without recomputing

Conditions run:
  - baseline          : no intervention
  - topk_ablate_0.0   : the K identified reasoning heads zeroed
  - topk_scale_{s}    : same K heads, scaled by factor s (for each s in --scales)
  - random_N_ablate   : N=0..n_random-1 random-K ablations (controls)
  - random_scale_{s}  : one random-K per non-baseline scale (for the sweep plot)
  - indiv_L{l}H{h}    : single-head ablation for each of the K heads

Outputs:
  - results_<condition>.csv  per-sample rows (incremental)
  - all_results.csv          concatenated
  - stats.json               per-condition acc + CI + Wilcoxon + Cohen's d
  - bar_conditions.png       bar chart w/ Wilson CIs
  - scale_sweep.png          accuracy vs scale, top-K vs random, CI band
  - length_by_difficulty.png response length by (condition × difficulty)
  - category_breakdown.png   accuracy by (condition × MATH category)

Usage:
  python -m reasoning_head_analysis.ablate_heads_v2 \\
      --model <path> \\
      --importance_path .../head_importance.pt \\
      --use_active_heads --exclude_layer0 \\
      --benchmark math500 --num_problems 500 --n_samples 4 \\
      --output_dir reasoning_head_analysis/results/ablate_v2_pi13_step700
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, wilcoxon
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
PROJECT_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(_SRC_DIR))

from rlvr_grokking.rewards.verl_reward import compute_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Benchmark loaders
# ═══════════════════════════════════════════════════════════════════════

def _extract_user(prompt):
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        prompt = json.loads(prompt)
    return next(m for m in prompt if m["role"] == "user")["content"]


def load_math500(path, n=None):
    df = pd.read_parquet(path)
    if n:
        df = df.head(n)
    probs = []
    for i, row in df.iterrows():
        q = _extract_user(row["prompt"])
        gt = row["reward_model"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        if hasattr(gt, "tolist"):
            gt = gt.tolist()
        ans = gt["ground_truth"] if isinstance(gt, dict) else str(gt)
        extra = row.get("extra_info", {}) if "extra_info" in row else {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        # MATH-500 parquet stores "subject" (algebra, number theory, etc.) and
        # "level" (1-5 difficulty).
        cat = extra.get("subject", "") if isinstance(extra, dict) else ""
        diff = str(extra.get("level", "")) if isinstance(extra, dict) else ""
        probs.append({"idx": int(i), "question": q, "answer": ans,
                      "category": cat, "difficulty": str(diff)})
    return probs


def load_aime24(n=None):
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    probs = []
    for i, row in enumerate(ds):
        if n and i >= n:
            break
        probs.append({"idx": i, "question": row["problem"],
                      "answer": str(row["answer"]),
                      "category": "aime", "difficulty": "hard"})
    return probs


def load_amc(n=None):
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    probs = []
    for i, row in enumerate(ds):
        if n and i >= n:
            break
        probs.append({"idx": i, "question": row["problem"],
                      "answer": str(row["answer"]),
                      "category": "amc", "difficulty": "medium"})
    return probs


BENCHMARKS = {"math500": load_math500, "aime24": load_aime24, "amc": load_amc}


# ═══════════════════════════════════════════════════════════════════════
# Prompt formatting + grading per benchmark
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(tokenizer, question, benchmark):
    msgs = [{"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def grade(response, answer, benchmark):
    """Return 1 if correct else 0."""
    try:
        return int(compute_score("math_dapo", response, answer) > 0)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════
# Head intervention hooks
# ═══════════════════════════════════════════════════════════════════════

def register_scaling_hooks(model, heads, scale=0.0):
    """Multiply specified attention heads' output by scale via o_proj pre-hooks.
    scale=0 ablates, scale=1 is no-op, scale>1 amplifies."""
    if not heads or scale == 1.0:
        return []
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.model.layers[0].self_attn.o_proj.in_features // n_heads)
    layer_heads = {}
    for layer, head in heads:
        layer_heads.setdefault(layer, []).append(head)
    handles = []
    for layer_idx, h_indices in layer_heads.items():
        def make_hook(hs, s):
            def hook(_, args):
                inp = args[0]
                inp = inp.view(inp.shape[0], inp.shape[1], n_heads, head_dim)
                for h in hs:
                    if s == 0.0:
                        inp[:, :, h, :] = 0.0
                    else:
                        inp[:, :, h, :] = inp[:, :, h, :] * s
                return (inp.view(inp.shape[0], inp.shape[1], -1),) + args[1:]
            return hook
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handles.append(o_proj.register_forward_pre_hook(make_hook(h_indices, scale)))
    return handles


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, n_samples, max_new_tokens,
                     temperature, batch_size, top_p=0.95):
    """Return list of lists: results[prompt_idx][sample_idx] = response str."""
    results = [[None] * n_samples for _ in prompts]
    # Flatten (prompt_idx, sample_idx) pairs
    flat = [(pi, si) for pi in range(len(prompts)) for si in range(n_samples)]
    for i in range(0, len(flat), batch_size):
        chunk = flat[i:i + batch_size]
        batch_prompts = [prompts[pi] for pi, _ in chunk]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        for j, (pi, si) in enumerate(chunk):
            resp = tokenizer.decode(outputs[j, input_len:], skip_special_tokens=True)
            results[pi][si] = resp
        if (i // batch_size) % 10 == 0:
            logger.info(f"    generated {min(i + batch_size, len(flat))}/{len(flat)}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Condition runner
# ═══════════════════════════════════════════════════════════════════════

def run_condition(model, tokenizer, problems, benchmark, name, heads, scale,
                  n_samples, max_new_tokens, temperature, batch_size, top_p=0.95,
                  full_responses_path=None):
    logger.info(f"=== {name}  (n_heads={len(heads) if heads else 0}, scale={scale}) ===")
    t0 = time.time()
    handles = register_scaling_hooks(model, heads or [], scale)
    try:
        prompts = [build_prompt(tokenizer, p["question"], benchmark) for p in problems]
        samples = generate_samples(model, tokenizer, prompts, n_samples,
                                   max_new_tokens, temperature, batch_size,
                                   top_p=top_p)
    finally:
        for h in handles:
            h.remove()
    rows = []
    full_records = []  # full responses go to a JSONL side-file; CSV stays compact
    for pi, (p, resp_list) in enumerate(zip(problems, samples)):
        for si, resp in enumerate(resp_list):
            correct = grade(resp, p["answer"], benchmark)
            length = len(tokenizer(resp)["input_ids"])
            boxed_match = _BOXED_RE.search(resp)
            boxed = boxed_match.group(1) if boxed_match else ""
            rows.append({
                "condition": name, "benchmark": benchmark,
                "problem_idx": p["idx"], "category": p["category"],
                "difficulty": p["difficulty"], "sample_idx": si,
                "correct": correct, "length_tokens": length,
                "has_boxed": bool(boxed_match),
                "boxed_content": boxed[:100],
                "response_head": resp[:800],
                "response_tail": resp[-800:] if len(resp) > 800 else "",
                "scale": scale, "n_heads": len(heads) if heads else 0,
            })
            full_records.append({
                "condition": name,
                "problem_idx": int(p["idx"]),
                "sample_idx": si,
                "correct": correct,
                "length_tokens": length,
                "response": resp,
            })

    # Append full responses to a per-run JSONL file (gzip-compatible).
    if full_responses_path is not None:
        with open(full_responses_path, "a") as f:
            for r in full_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    pass1 = df["correct"].mean()
    pass_k = df.groupby("problem_idx")["correct"].max().mean()
    logger.info(f"  pass@1={pass1*100:.1f}%   pass@{n_samples}={pass_k*100:.1f}%   "
                f"mean_len={df['length_tokens'].mean():.0f}   t={time.time() - t0:.0f}s")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z ** 2 / (4 * n * n)) ** 0.5) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def paired_wilcoxon(baseline, treat):
    """Inputs: 1D arrays of per-problem pass@k (0..1). Returns (stat, p)."""
    if len(baseline) != len(treat):
        return (None, None)
    diffs = np.asarray(treat, float) - np.asarray(baseline, float)
    if (diffs == 0).all():
        return (0.0, 1.0)
    try:
        r = wilcoxon(diffs)
        return (float(r.statistic), float(r.pvalue))
    except Exception:
        return (None, None)


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = (((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) ** 0.5) or 1e-9
    return float((b.mean() - a.mean()) / pooled)


def per_problem_passk(df):
    """Return per-problem pass@k values (max correct across samples), ordered by problem_idx."""
    return df.groupby("problem_idx")["correct"].max().sort_index().values


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_bar_with_cis(df, baseline_name, output, benchmark):
    summary = []
    for cond, sub in df.groupby("condition"):
        k = int(sub["correct"].sum())
        n = len(sub)
        lo, hi = wilson_ci(k, n)
        summary.append({"condition": cond, "acc": k / n, "lo": lo, "hi": hi})
    s = pd.DataFrame(summary).sort_values("acc", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(10, len(s) * 0.5), 6))
    xs = range(len(s))
    accs = s["acc"].values * 100
    err = np.vstack([(s["acc"] - s["lo"]) * 100, (s["hi"] - s["acc"]) * 100])
    colors = ["steelblue" if c == baseline_name
              else "crimson" if c.startswith("random")
              else "seagreen" if c.startswith("indiv")
              else "darkorange" for c in s["condition"]]
    ax.bar(xs, accs, yerr=err, capsize=3, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(s["condition"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{benchmark}: conditions with Wilson 95% CIs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def plot_scale_sweep(df, output, benchmark):
    sub = df[df["condition_kind"].isin(["topk", "random"])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for kind, label, color in [("topk", "reasoning heads", "darkorange"),
                                ("random", "random heads (mean of N)", "crimson")]:
        grp = sub[sub["condition_kind"] == kind]
        agg = []
        for scale, ss in grp.groupby("scale"):
            k = int(ss["correct"].sum()); n = len(ss)
            lo, hi = wilson_ci(k, n)
            agg.append((scale, k / n * 100, lo * 100, hi * 100))
        agg.sort()
        if not agg:
            continue
        xs = [a[0] for a in agg]; ys = [a[1] for a in agg]
        lo = [a[2] for a in agg]; hi = [a[3] for a in agg]
        ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=color)
    ax.axvline(1.0, color="gray", ls=":", label="no intervention")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{benchmark}: head scaling sweep (95% CI band)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def plot_length_by_difficulty(df, output, benchmark):
    if df["difficulty"].fillna("").eq("").all():
        return
    sub = df.copy()
    sub["difficulty"] = sub["difficulty"].replace({"": "unknown"}).fillna("unknown")
    focus = sub[sub["condition"].isin(["baseline", "topk_ablate_0.0",
                                        "topk_scale_2.0", "topk_scale_4.0"])]
    if focus.empty:
        return
    pivot = focus.groupby(["difficulty", "condition"])["length_tokens"].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean response length (tokens)")
    ax.set_title(f"{benchmark}: response length by difficulty × condition")
    plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def plot_cross_benchmark(root_dir, bench_list):
    """Grouped bar chart: pass@1 for each (condition, benchmark) with 95% CIs.
    Only plots conditions that appear in ALL benchmarks."""
    bench_dfs = {}
    for b in bench_list:
        p = os.path.join(root_dir, b, "all_results.csv")
        if os.path.exists(p):
            bench_dfs[b] = pd.read_csv(p)
    if len(bench_dfs) < 2:
        return
    common = set.intersection(*[set(df["condition"].unique()) for df in bench_dfs.values()])
    # Sort by the first benchmark's baseline accuracy for visual consistency
    def cond_order(c):
        prio = 0 if c == "baseline" else 1 if c.startswith("topk_ablate") else \
               2 if c.startswith("topk_scale") else 3 if c.startswith("random_") else 4
        return (prio, c)
    conds = sorted(common, key=cond_order)
    bars = {b: [] for b in bench_dfs}
    lo_err = {b: [] for b in bench_dfs}
    hi_err = {b: [] for b in bench_dfs}
    for cond in conds:
        for b, df in bench_dfs.items():
            sub = df[df["condition"] == cond]
            k, n = int(sub["correct"].sum()), len(sub)
            acc = k / n
            lo, hi = wilson_ci(k, n)
            bars[b].append(acc * 100)
            lo_err[b].append((acc - lo) * 100)
            hi_err[b].append((hi - acc) * 100)
    fig, ax = plt.subplots(figsize=(max(12, len(conds) * 0.6), 6))
    x = np.arange(len(conds))
    width = 0.8 / len(bench_dfs)
    colors = {"math500": "steelblue", "aime24": "darkorange", "amc": "seagreen"}
    for i, b in enumerate(bench_dfs):
        ax.bar(x + (i - (len(bench_dfs) - 1) / 2) * width, bars[b], width,
               yerr=[lo_err[b], hi_err[b]], capsize=3, label=b,
               color=colors.get(b, f"C{i}"), alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Condition × benchmark comparison (Wilson 95% CIs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(root_dir, "cross_benchmark.png"), dpi=150)
    plt.close(fig)


def plot_category_breakdown(df, output, benchmark):
    if df["category"].fillna("").eq("").all():
        return
    focus = df[df["condition"].isin(["baseline", "topk_ablate_0.0"])].copy()
    focus["category"] = focus["category"].replace({"": "unknown"}).fillna("unknown")
    agg = focus.groupby(["category", "condition"])["correct"].mean().unstack()
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    (agg * 100).plot(kind="bar", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{benchmark}: accuracy by category (baseline vs top-K ablated)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Condition builder
# ═══════════════════════════════════════════════════════════════════════

def build_conditions(ranked_heads, all_heads, top_k, scales,
                     n_random, do_individual, seed=42):
    """Return list of (name, heads, scale, kind) tuples."""
    conds = [("baseline", [], 1.0, "baseline")]
    top_heads = ranked_heads[:top_k]
    conds.append((f"topk_ablate_0.0", top_heads, 0.0, "topk"))
    for s in scales:
        if s in (0.0, 1.0):
            continue
        conds.append((f"topk_scale_{s}", top_heads, s, "topk"))

    rng = random.Random(seed)
    avail = [h for h in all_heads if h not in set(top_heads)]
    # Multiple random-K draws at scale=0 for a proper null distribution
    for r in range(n_random):
        heads_r = list(avail)
        rng.shuffle(heads_r)
        conds.append((f"random_{r}_ablate_0.0", heads_r[:top_k], 0.0, "random"))
    # One random-K at each non-trivial scale so the sweep has a control curve
    for s in scales:
        if s in (0.0, 1.0):
            continue
        heads_r = list(avail)
        rng.shuffle(heads_r)
        conds.append((f"random_scale_{s}", heads_r[:top_k], s, "random"))

    if do_individual:
        for (l, h) in top_heads:
            conds.append((f"indiv_L{l}H{h}_0.0", [(l, h)], 0.0, "individual"))
    return conds


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_heads(spec):
    out = []
    for tok in re.split(r"[,|;\s]+", spec.strip()):
        if not tok:
            continue
        l, h = tok.split(".")
        out.append((int(l), int(h)))
    return out


def parse_scales(spec):
    vals = [float(s) for s in re.split(r"[,|;\s]+", spec.strip()) if s]
    return sorted(set(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--importance_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark", default="math500",
                        choices=list(BENCHMARKS.keys()),
                        help="Legacy single-benchmark flag; ignored if --benchmarks is set.")
    parser.add_argument("--benchmarks", default=None,
                        help="Pipe- or comma-separated list, e.g. "
                             "'math500|aime24|amc'. Each benchmark writes to "
                             "$output_dir/<bench>/ and gets its own plots.")
    parser.add_argument("--math500_path", default="data/math500.parquet")
    parser.add_argument("--num_problems", type=int, default=500,
                        help="Default problem count; per-benchmark overrides available.")
    parser.add_argument("--aime24_n", type=int, default=30)
    parser.add_argument("--amc_n", type=int, default=40)
    parser.add_argument("--math500_n", type=int, default=None,
                        help="Override --num_problems for math500 specifically.")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling cutoff (paper uses 0.95). Setting "
                             "to 1.0 disables nucleus sampling — but then the "
                             "model can sample garbage low-prob tokens and loop.")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Long enough for genuine CoT on nearly all "
                             "MATH-500 problems; overthinking conditions may "
                             "still run close to this cap.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="A100 40GB: fits 1.5B model + 8k tokens × 8 batch "
                             "comfortably (KV cache ≈ 7GB + 3GB weights). "
                             "Lower if you OOM.")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--scales", default="0.0|0.25|0.5|0.75|1.5|2.0|3.0|4.0")
    parser.add_argument("--n_random", type=int, default=10)
    parser.add_argument("--use_active_heads", action="store_true")
    parser.add_argument("--exclude_layer0", action="store_true")
    parser.add_argument("--heads", default=None,
                        help="Explicit heads 'L.H|L.H|...' — overrides ranking/active")
    parser.add_argument("--skip_individual", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Condition-sharding for parallelism: run only conditions "
                             "where (i mod n_shards) equals shard_idx. Launch N sbatch jobs, "
                             "each with a different shard_idx, to parallelise across GPUs.")
    parser.add_argument("--n_shards", type=int, default=1,
                        help="Total number of shards (see --shard_idx).")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip generation; just compute stats + plots from existing "
                             "results_*.csv files in --output_dir. Use after all shards "
                             "have run.")
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Head selection ────────────────────────────────────────────────
    logger.info(f"Loading head importance from {args.importance_path}")
    data = torch.load(args.importance_path, map_location="cpu", weights_only=True)
    head_scores = data["head_scores"]
    n_layers, n_heads = head_scores.shape
    flat = head_scores.flatten()
    sorted_idx = torch.argsort(flat, descending=True)
    ranked_heads = [(i.item() // n_heads, i.item() % n_heads) for i in sorted_idx]
    all_heads = ranked_heads.copy()

    if args.heads:
        ranked_heads = parse_heads(args.heads)
        logger.info(f"Explicit heads ({len(ranked_heads)}): {ranked_heads}")
    elif args.use_active_heads:
        active = data.get("active_heads") or []
        if not active:
            raise ValueError("No 'active_heads' in .pt — rerun identify_heads.py")
        s = {tuple(h) for h in active}
        ranked_heads = [h for h in ranked_heads if h in s]
        logger.info(f"Filtered to {len(ranked_heads)} active heads")
    if args.exclude_layer0 and not args.heads:
        ranked_heads = [(l, h) for (l, h) in ranked_heads if l != 0]
        logger.info(f"Excluded layer-0; {len(ranked_heads)} candidates remain")

    top_k = min(args.top_k, len(ranked_heads))
    logger.info(f"Top-{top_k}: {ranked_heads[:top_k]}")

    scales = parse_scales(args.scales)

    # Resolve benchmark list + per-benchmark problem counts.
    bench_list = (re.split(r"[,|;\s]+", args.benchmarks.strip())
                  if args.benchmarks else [args.benchmark])
    bench_list = [b for b in bench_list if b]
    bench_n = {
        "math500": args.math500_n if args.math500_n is not None else args.num_problems,
        "aime24": args.aime24_n,
        "amc": args.amc_n,
    }

    # ─── Model (skipped in analyze_only mode) ──────────────────────────
    model, tokenizer = None, None
    if not args.analyze_only:
        logger.info(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        ).eval()

    # ─── Conditions (same list for all benchmarks) ─────────────────────
    conditions = build_conditions(
        ranked_heads, all_heads, top_k, scales,
        n_random=args.n_random, do_individual=not args.skip_individual,
        seed=args.seed,
    )
    logger.info(f"{len(conditions)} conditions planned; benchmarks: {bench_list}")

    # ─── Run each benchmark into its own subdir ─────────────────────────
    for bench in bench_list:
        n_prob = bench_n[bench]
        logger.info(f"\n━━━ Benchmark: {bench}  (n_problems={n_prob}) ━━━")
        if bench == "math500":
            problems = load_math500(args.math500_path, n_prob)
        elif bench == "aime24":
            problems = load_aime24(n_prob)
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
            for _, (name, heads, scale, kind) in my_conditions:
                out_path = os.path.join(bench_dir, f"results_{name}.csv")
                if os.path.exists(out_path):
                    logger.info(f"  {name}: already done, skipping")
                    continue
                full_path = os.path.join(
                    bench_dir, f"full_responses_shard{args.shard_idx}.jsonl")
                df = run_condition(model, tokenizer, problems, bench, name,
                                   heads, scale, args.n_samples, args.max_new_tokens,
                                   args.temperature, args.batch_size, top_p=args.top_p,
                                   full_responses_path=full_path)
                df["condition_kind"] = kind
                df.to_csv(out_path, index=False)

        # Per-benchmark aggregation
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

        # Per-benchmark stats
        baseline_df = all_df[all_df["condition"] == "baseline"]
        baseline_passk = per_problem_passk(baseline_df)
        stats_out = {}
        for cond, sub in all_df.groupby("condition"):
            k = int(sub["correct"].sum())
            n = len(sub)
            lo, hi = wilson_ci(k, n)
            passk = per_problem_passk(sub)
            if cond != "baseline" and len(passk) == len(baseline_passk):
                stat, p_val = paired_wilcoxon(baseline_passk, passk)
                d = cohens_d(baseline_passk, passk)
            else:
                stat, p_val, d = None, None, None
            stats_out[cond] = {
                "pass_at_1_avg": float(k / n), "n_gens": int(n),
                "wilson_95_lo": float(lo), "wilson_95_hi": float(hi),
                "pass_at_k": float(passk.mean()),
                "paired_wilcoxon_stat": stat, "paired_wilcoxon_p": p_val,
                "cohens_d_vs_baseline": d,
                "mean_length_tokens": float(sub["length_tokens"].mean()),
            }
        with open(os.path.join(bench_dir, "stats.json"), "w") as f:
            json.dump({
                "model": args.model, "benchmark": bench,
                "n_problems": len(problems), "n_samples": args.n_samples,
                "top_k": top_k, "heads_used": ranked_heads[:top_k],
                "scales": scales, "n_random_controls": args.n_random,
                "conditions": stats_out,
            }, f, indent=2, default=str)

        plot_bar_with_cis(all_df, "baseline",
                          os.path.join(bench_dir, "bar_conditions.png"), bench)
        plot_scale_sweep(all_df,
                         os.path.join(bench_dir, "scale_sweep.png"), bench)
        plot_length_by_difficulty(all_df,
                                  os.path.join(bench_dir, "length_by_difficulty.png"),
                                  bench)
        plot_category_breakdown(all_df,
                                os.path.join(bench_dir, "category_breakdown.png"),
                                bench)

    # ─── Cross-benchmark comparison plot ────────────────────────────────
    plot_cross_benchmark(args.output_dir, bench_list)

    logger.info(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
