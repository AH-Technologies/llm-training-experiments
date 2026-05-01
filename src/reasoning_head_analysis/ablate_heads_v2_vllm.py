#!/usr/bin/env python3
"""vLLM-accelerated head-ablation + scaling study.

Drop-in replacement for ablate_heads_v2.py using vLLM for ~10-50x faster
generation. Instead of register_forward_pre_hook (unsupported by vLLM),
we directly modify o_proj.weight columns to ablate/scale heads.

See ablate_heads_v2.py for full documentation of conditions + outputs.
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time

# vLLM 0.12+ runs models in a separate process; apply_model sends functions
# via pickle, which requires this env var.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import numpy as np
import pandas as pd
import torch

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
PROJECT_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(_SRC_DIR))

# Reuse everything we can from the original script
from reasoning_head_analysis.ablate_heads_v2 import (
    BENCHMARKS,
    MATH_SYSTEM_PROMPT,
    _BOXED_RE,
    build_conditions,
    build_prompt,
    cohens_d,
    grade,
    load_aime24,
    load_amc,
    load_math500,
    paired_wilcoxon,
    parse_heads,
    parse_scales,
    per_problem_passk,
    plot_bar_with_cis,
    plot_category_breakdown,
    plot_cross_benchmark,
    plot_length_by_difficulty,
    plot_scale_sweep,
    wilson_ci,
)

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Weight-based head intervention (replaces hook-based approach)
#
# vLLM 0.12+ runs the model in a separate worker process, so we use
# llm.apply_model() to execute weight modifications inside that process.
# Original weights are stored worker-side in model._oproj_originals.
# ═══════════════════════════════════════════════════════════════════════

def save_original_weights(llm):
    """Clone all o_proj weights inside the worker process."""
    def _save(model):
        originals = {}
        for i, layer in enumerate(model.model.layers):
            originals[i] = layer.self_attn.o_proj.weight.data.clone()
        model._oproj_originals = originals
        return len(originals)
    results = llm.apply_model(_save)
    return results[0]  # number of layers saved


def restore_weights(llm):
    """Restore o_proj weights to originals inside the worker process."""
    def _restore(model):
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.o_proj.weight.data.copy_(model._oproj_originals[i])
    llm.apply_model(_restore)


def modify_weights_for_heads(llm, heads, scale):
    """Modify o_proj weights to ablate/scale specific heads.

    o_proj.weight has shape (hidden_size, hidden_size).
    It maps from the concatenated head outputs to hidden_size.
    Head h's contribution comes from COLUMNS [h*head_dim : (h+1)*head_dim].
    Scaling these columns by `scale` scales that head's output.
    """
    if not heads or scale == 1.0:
        return

    # Group heads by layer (must be done before sending to worker)
    layer_heads = {}
    for layer_idx, head_idx in heads:
        layer_heads.setdefault(layer_idx, []).append(head_idx)

    def _modify(model, layer_heads=layer_heads, scale=scale):
        config = model.config
        n_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // n_heads
        for layer_idx, h_indices in layer_heads.items():
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            # Restore from original first, then scale
            o_proj.weight.data.copy_(model._oproj_originals[layer_idx])
            for h in h_indices:
                col_start = h * head_dim
                col_end = (h + 1) * head_dim
                o_proj.weight.data[:, col_start:col_end] *= scale

    llm.apply_model(_modify)


def modify_weights_per_head(llm, head_scales):
    """Apply different scales to different heads.

    head_scales: dict mapping (layer_idx, head_idx) -> scale
    """
    if not head_scales:
        return

    # Group by layer
    layer_head_scales = {}
    for (layer_idx, head_idx), scale in head_scales.items():
        layer_head_scales.setdefault(layer_idx, []).append((head_idx, scale))

    def _modify_multi(model, layer_head_scales=layer_head_scales):
        config = model.config
        hidden_size = config.hidden_size
        head_dim = hidden_size // config.num_attention_heads
        for layer_idx, hs_list in layer_head_scales.items():
            o_proj = model.model.layers[layer_idx].self_attn.o_proj
            # Restore from original first
            o_proj.weight.data.copy_(model._oproj_originals[layer_idx])
            for h, s in hs_list:
                col_start = h * head_dim
                col_end = (h + 1) * head_dim
                o_proj.weight.data[:, col_start:col_end] *= s

    llm.apply_model(_modify_multi)


# ═══════════════════════════════════════════════════════════════════════
# vLLM Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_samples_vllm(llm, prompts, n_samples, max_tokens,
                          temperature, top_p=0.95):
    """Generate n_samples per prompt using vLLM. Returns list of lists."""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_samples,
    )
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        samples = [o.text for o in output.outputs]
        results.append(samples)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Condition runner (vLLM version)
# ═══════════════════════════════════════════════════════════════════════

def run_condition_vllm(llm, tokenizer, problems, benchmark, name, heads, scale,
                       n_samples, max_tokens, temperature, top_p=0.95,
                       full_responses_path=None):
    logger.info(f"=== {name}  (n_heads={len(heads) if heads else 0}, scale={scale}) ===")
    t0 = time.time()

    # Apply weight modification (originals stored worker-side)
    restore_weights(llm)
    modify_weights_for_heads(llm, heads or [], scale)

    prompts = [build_prompt(tokenizer, p["question"], benchmark) for p in problems]
    samples = generate_samples_vllm(llm, prompts, n_samples, max_tokens,
                                    temperature, top_p=top_p)

    rows = []
    full_records = []
    for pi, (p, resp_list) in enumerate(zip(problems, samples)):
        for si, resp in enumerate(resp_list):
            correct = grade(resp, p["answer"], benchmark)
            length = len(tokenizer.encode(resp))
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
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--importance_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmark", default="math500",
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--benchmarks", default=None,
                        help="Pipe- or comma-separated list, e.g. 'math500|aime24|amc'")
    parser.add_argument("--math500_path", default="data/math500.parquet")
    parser.add_argument("--num_problems", type=int, default=500)
    parser.add_argument("--aime24_n", type=int, default=30)
    parser.add_argument("--amc_n", type=int, default=40)
    parser.add_argument("--math500_n", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens per generation (Qwen2.5-Math-1.5B has 4k context)")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--scales", default="0.0|0.25|0.5|0.75|1.5|2.0|3.0|4.0")
    parser.add_argument("--n_random", type=int, default=10)
    parser.add_argument("--use_active_heads", action="store_true")
    parser.add_argument("--exclude_layer0", action="store_true")
    parser.add_argument("--heads", default=None,
                        help="Explicit heads 'L.H|L.H|...' — overrides ranking/active")
    parser.add_argument("--skip_individual", action="store_true")
    parser.add_argument("--extra_conditions", default=None,
                        help="JSON list of custom conditions with per-head scales. "
                             'E.g. \'[{"name":"my_cond","head_scales":{"15.7":0.0,"10.5":1.5}}]\'')
    parser.add_argument("--extra_conditions_file", default=None,
                        help="Path to JSON file with extra conditions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--analyze_only", action="store_true")
    # vLLM-specific args
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM max model sequence length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory for vLLM")
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Head selection ────────────────────────────────────────────────
    logger.info(f"Loading head importance from {args.importance_path}")
    data = torch.load(args.importance_path, map_location="cpu", weights_only=False)
    head_scores = data["head_scores"]
    n_layers, n_heads_total = head_scores.shape
    flat = head_scores.flatten()
    sorted_idx = torch.argsort(flat, descending=True)
    ranked_heads = [(i.item() // n_heads_total, i.item() % n_heads_total)
                    for i in sorted_idx]
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

    bench_list = (re.split(r"[,|;\s]+", args.benchmarks.strip())
                  if args.benchmarks else [args.benchmark])
    bench_list = [b for b in bench_list if b]
    bench_n = {
        "math500": args.math500_n if args.math500_n is not None else args.num_problems,
        "aime24": args.aime24_n,
        "amc": args.amc_n,
    }

    # ─── vLLM model (skipped in analyze_only mode) ────────────────────
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
        # Save original weights inside the worker for restoration between conditions
        n_saved = save_original_weights(llm)
        logger.info(f"Saved original o_proj weights for {n_saved} layers")

    # ─── Conditions ───────────────────────────────────────────────────
    conditions = build_conditions(
        ranked_heads, all_heads, top_k, scales,
        n_random=args.n_random, do_individual=not args.skip_individual,
        seed=args.seed,
    )

    # Parse extra conditions with per-head scales
    extra_conditions = []
    extra_json = None
    if args.extra_conditions:
        extra_json = json.loads(args.extra_conditions)
    elif args.extra_conditions_file:
        with open(args.extra_conditions_file) as f:
            extra_json = json.load(f)
    if extra_json:
        for ec in extra_json:
            name = ec["name"]
            head_scales = {}
            for head_str, scale in ec["head_scales"].items():
                parts = head_str.split(".")
                head_scales[(int(parts[0]), int(parts[1]))] = float(scale)
            extra_conditions.append((name, head_scales))
        logger.info(f"Added {len(extra_conditions)} extra conditions with per-head scales")

    logger.info(f"{len(conditions) + len(extra_conditions)} conditions planned; benchmarks: {bench_list}")

    # ─── Run each benchmark ──────────────────────────────────────────
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
                df = run_condition_vllm(
                    llm, tokenizer, problems, bench, name, heads, scale,
                    args.n_samples, args.max_new_tokens, args.temperature,
                    top_p=args.top_p,
                    full_responses_path=full_path,
                )
                df["condition_kind"] = kind
                df.to_csv(out_path, index=False)

            # Run extra conditions (per-head scales), sharded across GPUs
            if extra_conditions:
                ec_shard = [ec for i, ec in enumerate(extra_conditions)
                            if i % args.n_shards == args.shard_idx]
                logger.info(f"Extra conditions: {len(ec_shard)}/{len(extra_conditions)} on shard {args.shard_idx}")
            if extra_conditions and ec_shard:
                for ec_name, head_scales in ec_shard:
                    out_path = os.path.join(bench_dir, f"results_{ec_name}.csv")
                    if os.path.exists(out_path):
                        logger.info(f"  {ec_name}: already done, skipping")
                        continue
                    full_path = os.path.join(
                        bench_dir, f"full_responses_shard{args.shard_idx}.jsonl")
                    logger.info(f"=== {ec_name}  (per-head scales: {len(head_scales)} heads) ===")
                    t0 = time.time()
                    restore_weights(llm)
                    modify_weights_per_head(llm, head_scales)

                    prompts = [build_prompt(tokenizer, p["question"], bench)
                               for p in problems]
                    samples = generate_samples_vllm(
                        llm, prompts, args.n_samples, args.max_new_tokens,
                        args.temperature, top_p=args.top_p)

                    rows = []
                    full_records = []
                    for pi, (p, resp_list) in enumerate(zip(problems, samples)):
                        for si, resp in enumerate(resp_list):
                            correct = grade(resp, p["answer"], bench)
                            length = len(tokenizer.encode(resp))
                            boxed_match = _BOXED_RE.search(resp)
                            boxed = boxed_match.group(1) if boxed_match else ""
                            rows.append({
                                "condition": ec_name, "benchmark": bench,
                                "problem_idx": p["idx"], "category": p["category"],
                                "difficulty": p["difficulty"], "sample_idx": si,
                                "correct": correct, "length_tokens": length,
                                "has_boxed": bool(boxed_match),
                                "boxed_content": boxed[:100],
                                "response_head": resp[:800],
                                "response_tail": resp[-800:] if len(resp) > 800 else "",
                                "scale": -1, "n_heads": len(head_scales),
                            })
                            full_records.append({
                                "condition": ec_name,
                                "problem_idx": int(p["idx"]),
                                "sample_idx": si,
                                "correct": correct,
                                "length_tokens": length,
                                "response": resp,
                            })
                    if full_path:
                        with open(full_path, "a") as f:
                            for r in full_records:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    df = pd.DataFrame(rows)
                    df["condition_kind"] = "extra"
                    df.to_csv(out_path, index=False)
                    pass1 = df["correct"].mean()
                    logger.info(f"  pass@1={pass1*100:.1f}%  "
                                f"mean_len={df['length_tokens'].mean():.0f}  "
                                f"t={time.time()-t0:.0f}s")

            # Restore weights after all conditions for this benchmark
            restore_weights(llm)

        # ─── Per-benchmark aggregation ────────────────────────────────
        all_dfs = []
        for name, _, _, _ in conditions:
            p = os.path.join(bench_dir, f"results_{name}.csv")
            if os.path.exists(p):
                all_dfs.append(pd.read_csv(p))
        # Include extra conditions in aggregation
        for ec_name, _ in extra_conditions:
            p = os.path.join(bench_dir, f"results_{ec_name}.csv")
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

    # ─── Cross-benchmark comparison plot ──────────────────────────────
    plot_cross_benchmark(args.output_dir, bench_list)

    logger.info(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
