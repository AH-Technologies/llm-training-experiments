#!/usr/bin/env python3
"""Ablation study: validate reasoning heads by measuring accuracy impact.

Loads head importance from identify_heads.py, then runs three experiments:
1. Baseline accuracy (no ablation)
2. Top-10 reasoning heads ablated
3. 10 random heads ablated (3 seeds)
4. Incremental top-k ablation curve until accuracy hits 0

Produces:
- ablation_barplot.png   — bar chart comparing baseline / top-10 / random-10
- ablation_curve.png     — accuracy vs number of heads ablated (top-k vs random)
- ablation_results.json  — all numerical results

Requires: transformers, datasets, pandas

Usage:
  python -m reasoning_head_analysis.ablate_heads \
      --importance_path reasoning_head_analysis/results/Qwen_Qwen2.5-1.5B-Instruct/head_importance.pt \
      --model Qwen/Qwen2.5-1.5B-Instruct
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

PROJECT_DIR = os.environ.get(
    "PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.rlvr_grokking.rewards.verl_reward import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_math500(data_path, num_problems=0):
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
    """Zero out the output of specified attention heads via o_proj pre-hooks."""
    if not heads_to_ablate:
        return []

    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.model.layers[0].self_attn.o_proj.in_features // n_heads
    )

    layer_heads = {}
    for layer, head in heads_to_ablate:
        layer_heads.setdefault(layer, []).append(head)

    handles = []
    for layer_idx, head_indices in layer_heads.items():
        def make_hook(h_indices):
            def hook_fn(module, args):
                inp = args[0]
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        for j, (output_ids, gt) in enumerate(zip(outputs, batch_gts)):
            input_len = inputs["input_ids"][j].shape[0]
            response_ids = output_ids[input_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            score = compute_score("math_dapo", response_text, gt)
            if score > 0:
                correct += 1
            total += 1

        if (i // batch_size) % 5 == 0:
            logger.info(f"  {min(i + batch_size, len(prompts))}/{len(prompts)} "
                       f"(running acc: {correct}/{total} = {100*correct/max(total,1):.1f}%)")

    accuracy = 100.0 * correct / max(total, 1)
    return accuracy, correct, total


# ═══════════════════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════════════════

def run_bar_chart(model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
                  max_new_tokens=2048, batch_size=4):
    """Baseline vs top-10 reasoning heads vs 3x random-10."""
    results = {}

    # Baseline
    logger.info("=== Baseline (no ablation) ===")
    acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
    results["baseline"] = acc
    logger.info(f"Baseline: {acc:.1f}% ({c}/{t})")

    # Top-10 reasoning heads
    logger.info("=== Top-10 reasoning heads ablated ===")
    heads = ranked_heads[:10]
    logger.info(f"Ablating: {heads}")
    handles = register_ablation_hooks(model, heads)
    acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
    for h in handles:
        h.remove()
    results["top10"] = acc
    logger.info(f"Top-10 ablated: {acc:.1f}% ({c}/{t})")

    # 3x random-10
    random_accs = []
    for seed in [42, 123, 456]:
        rng = random.Random(seed)
        random_heads = rng.sample(all_heads, 10)
        logger.info(f"=== Random-10 (seed={seed}) ===")
        logger.info(f"Ablating: {random_heads}")
        handles = register_ablation_hooks(model, random_heads)
        acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                        max_new_tokens=max_new_tokens, batch_size=batch_size)
        for h in handles:
            h.remove()
        random_accs.append(acc)
        logger.info(f"Random-10 seed={seed}: {acc:.1f}% ({c}/{t})")

    results["random10_seeds"] = random_accs
    results["random10_mean"] = float(np.mean(random_accs))
    results["random10_std"] = float(np.std(random_accs))
    return results


def run_topk_curve(model, tokenizer, prompts, ground_truths, ranked_heads,
                   max_new_tokens=2048, batch_size=4):
    """Ablate top-k heads for k=0,1,2,... until accuracy hits 0."""
    results = {}
    max_possible = len(ranked_heads)

    for k in range(0, max_possible + 1):
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
            logger.info(f"  Accuracy hit 0 at k={k}, stopping")
            break

    return results


def run_random_curve(model, tokenizer, prompts, ground_truths, all_heads,
                     max_k, seeds=(42, 123, 456),
                     max_new_tokens=2048, batch_size=4):
    """Ablate random-k heads for same k values, averaged over seeds."""
    results = {}
    for k in range(0, max_k + 1):
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
        results[k] = {"mean": float(np.mean(accs)), "per_seed": accs}
        logger.info(f"  [random-k] k={k}: mean={np.mean(accs):.1f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_bar_chart(bar_results, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Baseline\n(no ablation)", "Top-10\nreasoning heads"]
    values = [bar_results["baseline"], bar_results["top10"]]
    colors = ["tab:green", "tab:red"]

    for seed, acc in zip([42, 123, 456], bar_results["random10_seeds"]):
        labels.append(f"Random-10\n(seed {seed})")
        values.append(acc)
        colors.append("tab:blue")

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=13)
    ax.set_title("Head Ablation: Reasoning Heads vs Random Heads", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0, top=max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved bar chart to {output_path}")


def plot_ablation_curve(topk_results, random_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Top-k curve
    ks_top = sorted(topk_results.keys())
    accs_top = [topk_results[k] for k in ks_top]
    ax.plot(ks_top, accs_top, "o-", color="tab:red", linewidth=2, markersize=6,
            label="Top-k reasoning heads ablated")

    # Random-k curve (if available for same range)
    if random_results:
        ks_rand = sorted(random_results.keys())
        accs_mean = [random_results[k]["mean"] for k in ks_rand]
        accs_std = [np.std(random_results[k]["per_seed"]) for k in ks_rand]
        ax.plot(ks_rand, accs_mean, "s--", color="tab:blue", linewidth=2, markersize=6,
                label="Random-k heads ablated (avg 3 seeds)")
        ax.fill_between(ks_rand,
                         np.array(accs_mean) - np.array(accs_std),
                         np.array(accs_mean) + np.array(accs_std),
                         alpha=0.2, color="tab:blue")

    ax.set_xlabel("Number of heads ablated", fontsize=13)
    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=13)
    ax.set_title("Incremental Head Ablation: Reasoning Heads vs Random", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ablation curve to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation study for reasoning heads")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--importance_path", type=str, required=True,
                        help="Path to head_importance.pt from identify_heads.py")
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_DIR, "data/math500.parquet"))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    safe_model = args.model.replace("/", "_")
    output_dir = args.output_dir or os.path.join("reasoning_head_analysis", "results", safe_model, "ablation")
    os.makedirs(output_dir, exist_ok=True)

    # Load head importance
    logger.info(f"Loading head importance from {args.importance_path}")
    data = torch.load(args.importance_path, map_location="cpu", weights_only=True)
    head_scores = data["head_scores"] if isinstance(data, dict) else data

    n_layers, n_heads = head_scores.shape
    logger.info(f"Importance matrix: {n_layers} layers x {n_heads} heads")

    flat = head_scores.flatten()
    sorted_indices = torch.argsort(flat, descending=True)
    ranked_heads = [(idx.item() // n_heads, idx.item() % n_heads) for idx in sorted_indices]
    all_heads = ranked_heads.copy()

    logger.info(f"Top-10 reasoning heads: {ranked_heads[:10]}")

    # Load model + data
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Loading {args.num_problems} problems from {args.data_path}")
    problems = load_math500(args.data_path, num_problems=args.num_problems)
    prompts = format_prompts(tokenizer, problems)
    ground_truths = [gt for _, gt in problems]

    t0 = time.time()

    # Experiment 1: Bar chart (baseline vs top-10 vs 3x random-10)
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Bar chart comparison")
    logger.info("=" * 60)
    bar_results = run_bar_chart(
        model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    # Experiment 2: Incremental top-k ablation curve
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Incremental top-k ablation curve")
    logger.info("=" * 60)
    topk_results = run_topk_curve(
        model, tokenizer, prompts, ground_truths, ranked_heads,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    # Experiment 3: Random-k control curve (same k range as top-k)
    max_k_reached = max(topk_results.keys())
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 3: Random-k control curve (k=0..{max_k_reached}, 3 seeds)")
    logger.info("=" * 60)
    random_results = run_random_curve(
        model, tokenizer, prompts, ground_truths, all_heads,
        max_k=max_k_reached, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    elapsed = time.time() - t0
    logger.info(f"All experiments completed in {elapsed / 60:.1f} min")

    # Save results
    all_results = {
        "bar_chart": bar_results,
        "topk_curve": {str(k): v for k, v in topk_results.items()},
        "random_curve": {str(k): v for k, v in random_results.items()},
        "config": {
            "model": args.model,
            "importance_path": args.importance_path,
            "num_problems": args.num_problems,
            "elapsed_minutes": elapsed / 60,
        },
    }
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Plots
    plot_bar_chart(bar_results, os.path.join(output_dir, "ablation_barplot.png"))
    plot_ablation_curve(topk_results, random_results, os.path.join(output_dir, "ablation_curve.png"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
