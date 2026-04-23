#!/usr/bin/env python3
"""Incremental head ablation evaluation on MATH-500.

Loads the EAP-IG head importance ranking, then progressively ablates heads
from most to least important while measuring accuracy on MATH-500.

Supports running individual conditions via --condition for multi-GPU parallelism,
or --condition all to run everything sequentially on one GPU.

Conditions:
  topk_curve   — ablate top-k heads one by one (k=0..max_k)
  random_curve — ablate random-k heads (same schedule, 3 seeds)
  bar_chart    — baseline vs top-10 vs 3× random-10
  plot         — merge JSONs from the above and produce plots
  all          — run everything sequentially

Usage:
  # Parallel on 3 GPUs:
  CUDA_VISIBLE_DEVICES=0 python -m ... --condition topk_curve &
  CUDA_VISIBLE_DEVICES=1 python -m ... --condition random_curve &
  CUDA_VISIBLE_DEVICES=2 python -m ... --condition bar_chart &
  wait
  python -m ... --condition plot
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.rlvr_grokking.rewards.verl_reward import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_math500(data_path: str, num_problems: int = 0):
    """Load MATH-500 problems and ground truths from parquet.

    Args:
        num_problems: if > 0, limit to first N problems

    Returns: list of (prompt_messages, ground_truth) tuples
    """
    df = pd.read_parquet(data_path)
    if num_problems > 0:
        df = df.head(num_problems)

    problems = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        gt = row["reward_model"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        if hasattr(gt, "tolist"):
            gt = gt.tolist()
        ground_truth = gt.get("ground_truth", gt) if isinstance(gt, dict) else str(gt)

        problems.append((prompt, ground_truth))
    return problems


def format_prompts(tokenizer, problems):
    """Format all problems into prompt strings for generation."""
    prompts = []
    for prompt_msgs, _ in problems:
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_str)
    return prompts


# ═══════════════════════════════════════════════════════════════════════
# Ablation hooks
# ═══════════════════════════════════════════════════════════════════════

def register_ablation_hooks(model, heads_to_ablate):
    """Register forward pre-hooks on o_proj to zero out specified heads.

    Args:
        model: HuggingFace model
        heads_to_ablate: list of (layer, head) tuples

    Returns:
        list of hook handles (call .remove() to clean up)
    """
    if not heads_to_ablate:
        return []

    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.model.layers[0].self_attn.o_proj.in_features // n_heads
    )

    # Group heads by layer
    layer_heads = {}
    for layer, head in heads_to_ablate:
        layer_heads.setdefault(layer, []).append(head)

    handles = []
    for layer_idx, head_indices in layer_heads.items():
        def make_hook(h_indices):
            def hook_fn(module, args):
                inp = args[0]  # (batch, seq_len, n_heads * head_dim)
                inp = inp.view(inp.shape[0], inp.shape[1], n_heads, head_dim)
                for h in h_indices:
                    inp[:, :, h, :] = 0.0
                return (inp.view(inp.shape[0], inp.shape[1], -1),) + args[1:]
            return hook_fn

        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(make_hook(head_indices))
        handles.append(handle)

    return handles


# ═══════════════════════════════════════════════════════════════════════
# Generation + scoring
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_and_score(model, tokenizer, prompts, ground_truths,
                       max_new_tokens=2048, batch_size=4):
    """Generate responses and compute accuracy.

    Returns: (accuracy_pct, num_correct, num_total)
    """
    correct = 0
    total = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_gts = ground_truths[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        for j, (output_ids, gt) in enumerate(zip(outputs, batch_gts)):
            # Decode only the generated part
            input_len = inputs["input_ids"][j].shape[0]
            response_ids = output_ids[input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            score = compute_score("math_dapo", response_text, gt)
            if score > 0:
                correct += 1
            total += 1

        if (i // batch_size) % 5 == 0:
            logger.info(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} "
                       f"(running acc: {correct}/{total} = {100*correct/max(total,1):.1f}%)")

    accuracy = 100.0 * correct / max(total, 1)
    return accuracy, correct, total


# ═══════════════════════════════════════════════════════════════════════
# Condition runners
# ═══════════════════════════════════════════════════════════════════════

def run_topk_curve(model, tokenizer, prompts, ground_truths, ranked_heads,
                   max_k=10, max_new_tokens=2048, batch_size=4):
    """Ablate top-k heads for k=0..max_k, one at a time."""
    results = {}
    schedule = list(range(0, max_k + 1))

    for k in schedule:
        heads_to_ablate = ranked_heads[:k]
        handles = register_ablation_hooks(model, heads_to_ablate)
        acc, correct, total = generate_and_score(
            model, tokenizer, prompts, ground_truths,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        for h in handles:
            h.remove()
        results[k] = acc
        logger.info(f"  [top-k] k={k}: {acc:.1f}% ({correct}/{total})")

        if acc == 0.0 and k > 0:
            logger.info(f"  [top-k] Accuracy hit 0 at k={k}, stopping")
            break

    return results


def run_random_curve(model, tokenizer, prompts, ground_truths, all_heads,
                     max_k=10, seeds=(42, 123, 456),
                     max_new_tokens=2048, batch_size=4):
    """Ablate random-k heads for k=0..max_k, averaged over seeds."""
    results = {}
    schedule = list(range(0, max_k + 1))

    for k in schedule:
        accs = []
        for seed in seeds:
            rng = random.Random(seed)
            random_heads = rng.sample(all_heads, k) if k > 0 else []
            handles = register_ablation_hooks(model, random_heads)
            acc, _, _ = generate_and_score(
                model, tokenizer, prompts, ground_truths,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
            )
            for h in handles:
                h.remove()
            accs.append(acc)
            logger.info(f"  [random-k] k={k}, seed={seed}: {acc:.1f}%")
        results[k] = {"mean": float(np.mean(accs)), "per_seed": accs}
        logger.info(f"  [random-k] k={k} mean: {np.mean(accs):.1f}%")

    return results


def run_bar_chart(model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
                  max_new_tokens=2048, batch_size=4):
    """Baseline vs top-10 vs 3× random-10."""
    results = {}

    # Baseline
    logger.info("Bar chart: baseline")
    acc, _, _ = generate_and_score(model, tokenizer, prompts, ground_truths,
                                   max_new_tokens=max_new_tokens, batch_size=batch_size)
    results["baseline"] = acc
    logger.info(f"  Baseline: {acc:.1f}%")

    # Top-10
    logger.info("Bar chart: top-10 ablated")
    handles = register_ablation_hooks(model, ranked_heads[:10])
    acc, _, _ = generate_and_score(model, tokenizer, prompts, ground_truths,
                                   max_new_tokens=max_new_tokens, batch_size=batch_size)
    for h in handles:
        h.remove()
    results["top10"] = acc
    logger.info(f"  Top-10: {acc:.1f}%")

    # 3× random-10
    random_accs = []
    for seed in [42, 123, 456]:
        rng = random.Random(seed)
        random_heads = rng.sample(all_heads, 10)
        logger.info(f"Bar chart: random-10 (seed={seed})")
        handles = register_ablation_hooks(model, random_heads)
        acc, _, _ = generate_and_score(model, tokenizer, prompts, ground_truths,
                                       max_new_tokens=max_new_tokens, batch_size=batch_size)
        for h in handles:
            h.remove()
        random_accs.append(acc)
        logger.info(f"  Random-10 seed={seed}: {acc:.1f}%")

    results["random10_seeds"] = random_accs
    results["random10_mean"] = float(np.mean(random_accs))
    results["random10_std"] = float(np.std(random_accs))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting (from merged JSONs)
# ═══════════════════════════════════════════════════════════════════════

def plot_ablation_curve(topk_results, random_results, output_path):
    """Plot accuracy vs number of ablated heads."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Top-k curve
    ks_top = sorted(topk_results.keys())
    accs_top = [topk_results[k] for k in ks_top]
    ax.plot(ks_top, accs_top, "o-", color="tab:red", linewidth=2, markersize=6,
            label="Top-k EAP-IG heads ablated")

    # Random-k curve
    if random_results:
        ks_rand = sorted(random_results.keys())
        accs_rand_mean = [random_results[k]["mean"] for k in ks_rand]
        ax.plot(ks_rand, accs_rand_mean, "s--", color="tab:blue", linewidth=2, markersize=6,
                label="Random-k heads ablated (avg over 3 seeds)")

        accs_rand_per_seed = [random_results[k]["per_seed"] for k in ks_rand]
        accs_rand_std = [np.std(s) for s in accs_rand_per_seed]
        accs_rand_mean_arr = np.array(accs_rand_mean)
        accs_rand_std_arr = np.array(accs_rand_std)
        ax.fill_between(ks_rand,
                         accs_rand_mean_arr - accs_rand_std_arr,
                         accs_rand_mean_arr + accs_rand_std_arr,
                         alpha=0.2, color="tab:blue")

    ax.set_xlabel("Number of heads ablated", fontsize=13)
    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=13)
    ax.set_title("Incremental Head Ablation: EAP-IG Top-k vs Random-k", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved ablation curve to {output_path}")


def plot_bar_chart(bar_results, output_path):
    """Plot bar chart: baseline vs top-10 vs random-10."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Baseline\n(no ablation)", "Top-10\nEAP-IG ablated"]
    values = [bar_results["baseline"], bar_results["top10"]]
    colors = ["tab:green", "tab:red"]

    for i, (seed, acc) in enumerate(zip([42, 123, 456], bar_results["random10_seeds"])):
        labels.append(f"Random-10\n(seed {seed})")
        values.append(acc)
        colors.append("tab:blue")

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=13)
    ax.set_title("Ablation Comparison: EAP-IG Top-10 vs Random-10", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0, top=max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved bar chart to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Model + data loading helper
# ═══════════════════════════════════════════════════════════════════════

def load_model_and_data(args):
    """Load importance ranking, data, model, tokenizer. Returns everything needed."""
    # Head importance
    logger.info(f"Loading head importance from {args.importance_path}")
    importance_data = torch.load(args.importance_path, map_location="cpu", weights_only=True)
    if isinstance(importance_data, dict):
        importance_matrix = importance_data.get("importance", importance_data.get("importance_matrix"))
    else:
        importance_matrix = importance_data

    n_layers, n_heads = importance_matrix.shape
    logger.info(f"Importance matrix: {n_layers} layers × {n_heads} heads")

    flat_importance = importance_matrix.flatten()
    sorted_indices = torch.argsort(flat_importance, descending=True)
    ranked_heads = [(idx.item() // n_heads, idx.item() % n_heads) for idx in sorted_indices]
    logger.info(f"Top-10 heads: {ranked_heads[:10]}")

    # Data
    logger.info(f"Loading data from {args.data_path} (n={args.num_problems})")
    problems = load_math500(args.data_path, num_problems=args.num_problems)
    logger.info(f"Loaded {len(problems)} problems")

    # Model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"  # required for correct batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Model loaded on {model.device}")

    prompts = format_prompts(tokenizer, problems)
    ground_truths = [gt for _, gt in problems]

    return model, tokenizer, prompts, ground_truths, ranked_heads


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Incremental head ablation eval")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--importance_path", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(PROJECT_DIR, "data/math500.parquet"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--max_k", type=int, default=10,
                        help="Max heads to ablate in incremental curve")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--condition", type=str, default="all",
                        choices=["topk_curve", "random_curve", "bar_chart", "plot", "all"],
                        help="Which condition to run (for multi-GPU parallelism)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Plot-only mode: merge JSONs and produce figures ──
    if args.condition == "plot":
        logger.info("Plot mode: merging results and generating figures")
        merged = {}
        for name in ["topk_curve", "random_curve", "bar_chart"]:
            path = os.path.join(args.output_dir, f"{name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    merged[name] = json.load(f)
                logger.info(f"  Loaded {path}")

        # Parse topk results (keys are strings from JSON)
        topk = {}
        if "topk_curve" in merged:
            for k, v in merged["topk_curve"].items():
                topk[int(k)] = v

        random_ctrl = {}
        if "random_curve" in merged:
            for k, v in merged["random_curve"].items():
                random_ctrl[int(k)] = v

        if topk:
            plot_ablation_curve(topk, random_ctrl,
                                os.path.join(args.output_dir, "ablation_curve.png"))

        if "bar_chart" in merged:
            plot_bar_chart(merged["bar_chart"],
                           os.path.join(args.output_dir, "ablation_barplot.png"))

        # Save merged results
        all_results = {"topk_curve": merged.get("topk_curve", {}),
                       "random_curve": merged.get("random_curve", {}),
                       "bar_chart": merged.get("bar_chart", {})}
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("Done plotting.")
        return

    # ── Load model + data (needed for all non-plot conditions) ──
    model, tokenizer, prompts, ground_truths, ranked_heads = load_model_and_data(args)
    all_heads = ranked_heads.copy()

    t0 = time.time()

    if args.condition in ("topk_curve", "all"):
        logger.info("=" * 60)
        logger.info(f"Running top-k ablation curve (k=0..{args.max_k})")
        logger.info("=" * 60)
        topk_results = run_topk_curve(
            model, tokenizer, prompts, ground_truths, ranked_heads,
            max_k=args.max_k, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        out_path = os.path.join(args.output_dir, "topk_curve.json")
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in topk_results.items()}, f, indent=2)
        logger.info(f"Saved top-k results to {out_path}")

    if args.condition in ("random_curve", "all"):
        logger.info("=" * 60)
        logger.info(f"Running random-k control curve (k=0..{args.max_k}, 3 seeds)")
        logger.info("=" * 60)
        random_results = run_random_curve(
            model, tokenizer, prompts, ground_truths, all_heads,
            max_k=args.max_k, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        out_path = os.path.join(args.output_dir, "random_curve.json")
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in random_results.items()}, f, indent=2)
        logger.info(f"Saved random-k results to {out_path}")

    if args.condition in ("bar_chart", "all"):
        logger.info("=" * 60)
        logger.info("Running bar chart comparison")
        logger.info("=" * 60)
        bar_results = run_bar_chart(
            model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        out_path = os.path.join(args.output_dir, "bar_chart.json")
        with open(out_path, "w") as f:
            json.dump(bar_results, f, indent=2)
        logger.info(f"Saved bar chart results to {out_path}")

    elapsed = time.time() - t0
    logger.info(f"Condition '{args.condition}' took {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # If running all, also produce plots directly
    if args.condition == "all":
        topk_parsed = {int(k): v for k, v in topk_results.items()}
        random_parsed = {int(k): v for k, v in random_results.items()}
        plot_ablation_curve(topk_parsed, random_parsed,
                            os.path.join(args.output_dir, "ablation_curve.png"))
        plot_bar_chart(bar_results, os.path.join(args.output_dir, "ablation_barplot.png"))

        all_res = {"topk_curve": {str(k): v for k, v in topk_results.items()},
                   "random_curve": {str(k): v for k, v in random_results.items()},
                   "bar_chart": bar_results}
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(all_res, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()
