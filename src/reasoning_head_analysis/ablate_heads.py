#!/usr/bin/env python3
"""Head scaling study: validate reasoning heads by adjusting their output.

Loads head importance from identify_heads.py, then runs experiments that
multiply the output of identified heads by various scale factors (0 = ablate,
0.5 = dampen, 2.0 = amplify, ...) and measures MATH-500 accuracy.

Experiments:
1. Baseline accuracy (scale=1, unmodified).
2. Scale sweep: top-k reasoning heads and random-k control at each scale.
3. Incremental top-k ablation curve at scale=0 (how many heads to kill
   accuracy).
4. Random-k control curve at scale=0.

Produces:
- scale_sweep.png        — accuracy vs scale, top-k vs random-k
- ablation_barplot.png   — bar chart at scale=0 (baseline / top-k / random-k)
- ablation_curve.png     — accuracy vs k heads ablated (scale=0)
- ablation_results.json  — all numerical results

Usage:
  python -m reasoning_head_analysis.ablate_heads \
      --importance_path .../head_importance.pt \
      --model Qwen/Qwen2.5-Math-1.5B \
      --scales "0.0,0.5,2.0,4.0" \
      --top_k 10
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

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
PROJECT_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(_SRC_DIR))

from rlvr_grokking.rewards.verl_reward import compute_score

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

def register_scaling_hooks(model, heads, scale=0.0):
    """Multiply the output of specified attention heads by `scale` via o_proj
    pre-hooks. scale=0 → ablation, scale=1 → no-op, scale>1 → amplification,
    0<scale<1 → dampening."""
    if not heads or scale == 1.0:
        return []

    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.model.layers[0].self_attn.o_proj.in_features // n_heads
    )

    layer_heads = {}
    for layer, head in heads:
        layer_heads.setdefault(layer, []).append(head)

    handles = []
    for layer_idx, head_indices in layer_heads.items():
        def make_hook(h_indices, s):
            def hook_fn(module, args):
                inp = args[0]
                inp = inp.view(inp.shape[0], inp.shape[1], n_heads, head_dim)
                for h in h_indices:
                    if s == 0.0:
                        inp[:, :, h, :] = 0.0
                    else:
                        inp[:, :, h, :] = inp[:, :, h, :] * s
                return (inp.view(inp.shape[0], inp.shape[1], -1),) + args[1:]
            return hook_fn

        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(make_hook(head_indices, scale))
        handles.append(handle)

    return handles


# Back-compat alias: old name zeroes heads.
def register_ablation_hooks(model, heads_to_ablate):
    return register_scaling_hooks(model, heads_to_ablate, scale=0.0)


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
                  max_new_tokens=2048, batch_size=4, top_k=10):
    """Baseline vs top-K reasoning heads vs 3x random-K (all at scale=0)."""
    results = {}

    logger.info("=== Baseline (no ablation) ===")
    acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
    results["baseline"] = acc
    logger.info(f"Baseline: {acc:.1f}% ({c}/{t})")

    logger.info(f"=== Top-{top_k} reasoning heads ablated ===")
    heads = ranked_heads[:top_k]
    logger.info(f"Ablating: {heads}")
    handles = register_scaling_hooks(model, heads, scale=0.0)
    acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
    for h in handles:
        h.remove()
    results["top_ablated"] = acc
    logger.info(f"Top-{top_k} ablated: {acc:.1f}% ({c}/{t})")

    random_accs = []
    for seed in [42, 123, 456]:
        rng = random.Random(seed)
        random_heads = rng.sample(all_heads, top_k)
        logger.info(f"=== Random-{top_k} (seed={seed}) ===")
        handles = register_scaling_hooks(model, random_heads, scale=0.0)
        acc, c, t = generate_and_score(model, tokenizer, prompts, ground_truths,
                                        max_new_tokens=max_new_tokens, batch_size=batch_size)
        for h in handles:
            h.remove()
        random_accs.append(acc)
        logger.info(f"Random-{top_k} seed={seed}: {acc:.1f}% ({c}/{t})")

    results["random_seeds"] = random_accs
    results["random_mean"] = float(np.mean(random_accs))
    results["random_std"] = float(np.std(random_accs))
    results["top_k"] = top_k
    return results


def run_scale_sweep(model, tokenizer, prompts, ground_truths,
                    ranked_heads, all_heads, scales, top_k,
                    max_new_tokens=2048, batch_size=4,
                    baseline_acc=None, random_seeds=(42, 123, 456)):
    """For each scale in `scales`, measure accuracy with top-k heads scaled
    vs random-k heads scaled (averaged over seeds). scale=1 is the baseline
    (we reuse baseline_acc if provided to avoid an extra pass)."""
    heads_top = ranked_heads[:top_k]
    per_scale = {}

    for scale in scales:
        logger.info(f"=== Scale sweep: scale={scale} ===")

        if scale == 1.0 and baseline_acc is not None:
            top_acc = baseline_acc
        else:
            logger.info(f"  top-{top_k} @ scale={scale}: {heads_top}")
            handles = register_scaling_hooks(model, heads_top, scale=scale)
            top_acc, c, t = generate_and_score(
                model, tokenizer, prompts, ground_truths,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
            )
            for h in handles:
                h.remove()
            logger.info(f"  top-{top_k} @ scale={scale}: {top_acc:.1f}% ({c}/{t})")

        rand_accs = []
        for seed in random_seeds:
            rng = random.Random(seed)
            rand_heads = rng.sample(all_heads, top_k)
            handles = register_scaling_hooks(model, rand_heads, scale=scale)
            ra, _, _ = generate_and_score(
                model, tokenizer, prompts, ground_truths,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
            )
            for h in handles:
                h.remove()
            rand_accs.append(ra)
            logger.info(f"  random-{top_k} seed={seed} @ scale={scale}: {ra:.1f}%")

        per_scale[scale] = {
            "top_k_acc": top_acc,
            "random_seeds": rand_accs,
            "random_mean": float(np.mean(rand_accs)),
            "random_std": float(np.std(rand_accs)),
        }

    return per_scale


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

    k = bar_results.get("top_k", 10)
    labels = ["Baseline\n(no ablation)", f"Top-{k}\nreasoning heads"]
    values = [bar_results["baseline"], bar_results["top_ablated"]]
    colors = ["tab:green", "tab:red"]

    for seed, acc in zip([42, 123, 456], bar_results["random_seeds"]):
        labels.append(f"Random-{k}\n(seed {seed})")
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


def plot_scale_sweep(per_scale, baseline_acc, top_k, output_path):
    """Accuracy vs scale factor. Top-k reasoning heads (line) vs random-k
    control (line with std band). Dashed horizontal: baseline."""
    scales = sorted(per_scale.keys())
    top_accs = [per_scale[s]["top_k_acc"] for s in scales]
    rand_means = [per_scale[s]["random_mean"] for s in scales]
    rand_stds = [per_scale[s]["random_std"] for s in scales]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(baseline_acc, ls="--", color="tab:green", alpha=0.7,
               label=f"Baseline ({baseline_acc:.1f}%)")
    ax.plot(scales, top_accs, "o-", color="tab:red", linewidth=2, markersize=8,
            label=f"Top-{top_k} reasoning heads scaled")
    ax.plot(scales, rand_means, "s--", color="tab:blue", linewidth=2, markersize=7,
            label=f"Random-{top_k} heads scaled (mean of 3 seeds)")
    ax.fill_between(scales,
                    np.array(rand_means) - np.array(rand_stds),
                    np.array(rand_means) + np.array(rand_stds),
                    alpha=0.15, color="tab:blue")

    ax.set_xlabel("Scale factor applied to head output", fontsize=13)
    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=13)
    ax.set_title(f"Head Activation Scaling: top-{top_k} reasoning heads vs random",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved scale sweep to {output_path}")


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
    parser.add_argument("--top_k", type=int, default=10,
                        help="How many top reasoning heads to scale/ablate")
    parser.add_argument("--use_active_heads", action="store_true",
                        help="Restrict ablation candidates to the 'active_heads' set "
                             "(paper's threshold+prune circuit) saved in the .pt file. "
                             "Within that set, rank by head_score and take top_k.")
    parser.add_argument("--exclude_layer0", action="store_true",
                        help="Exclude layer-0 heads from the candidate set "
                             "(usually dominated by trivial input-projection heads).")
    parser.add_argument("--heads", type=str, default=None,
                        help="Explicit head list as 'L.H,L.H,...' (e.g. '11.8,15.7,19.6'). "
                             "Overrides score ranking and filters; uses these heads in order. "
                             "Useful for cross-model studies where you want to apply the heads "
                             "found in one checkpoint to a different model.")
    parser.add_argument("--scales", type=str, default="0.0,0.25,0.5,1.5,2.0,4.0",
                        help="Comma-separated scale factors for the sweep")
    args = parser.parse_args()
    import re as _re_scales
    scales = [float(s) for s in _re_scales.split(r"[,|;\s]+", args.scales.strip()) if s]

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

    if args.heads:
        # Explicit head list overrides everything else.
        # Accept ',' '|' ';' or whitespace as separators — sbatch --export
        # consumes plain commas as variable separators so use '|' over SLURM.
        import re as _re
        parsed = []
        for token in _re.split(r"[,|;\s]+", args.heads.strip()):
            if not token:
                continue
            l, h = token.split(".")
            parsed.append((int(l), int(h)))
        ranked_heads = parsed
        logger.info(f"Using explicit heads ({len(ranked_heads)}): {ranked_heads}")
    elif args.use_active_heads:
        active = data.get("active_heads") if isinstance(data, dict) else None
        if not active:
            raise ValueError("--use_active_heads set but no 'active_heads' key in .pt file. "
                             "Re-run identify_heads.py to produce one.")
        active_set = {tuple(h) for h in active}
        ranked_heads = [h for h in ranked_heads if h in active_set]
        logger.info(f"Filtered to {len(ranked_heads)} active (circuit) heads")

    if args.exclude_layer0 and not args.heads:
        ranked_heads = [(l, h) for (l, h) in ranked_heads if l != 0]
        logger.info(f"Excluded layer-0 heads; {len(ranked_heads)} candidates remain")

    logger.info(f"Top-10 candidates: {ranked_heads[:10]}")

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

    # Experiment 1: Bar chart at scale=0 (baseline vs top-k vs 3x random-k)
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 1: Bar chart (top-{args.top_k} ablated, scale=0)")
    logger.info("=" * 60)
    bar_results = run_bar_chart(
        model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    # Experiment 2: Scale sweep (top-k and random-k at each scale)
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 2: Scale sweep — scales={scales}, top-{args.top_k} vs random")
    logger.info("=" * 60)
    scale_results = run_scale_sweep(
        model, tokenizer, prompts, ground_truths, ranked_heads, all_heads,
        scales=scales, top_k=args.top_k,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        baseline_acc=bar_results["baseline"],
    )

    # Experiment 3: Incremental top-k ablation curve (scale=0 only)
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Incremental top-k ablation curve (scale=0)")
    logger.info("=" * 60)
    topk_results = run_topk_curve(
        model, tokenizer, prompts, ground_truths, ranked_heads,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
    )

    # Experiment 4: Random-k control curve (scale=0, same k range)
    max_k_reached = max(topk_results.keys())
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 4: Random-k control curve (k=0..{max_k_reached}, 3 seeds)")
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
        "scale_sweep": {str(s): v for s, v in scale_results.items()},
        "topk_curve": {str(k): v for k, v in topk_results.items()},
        "random_curve": {str(k): v for k, v in random_results.items()},
        "config": {
            "model": args.model,
            "importance_path": args.importance_path,
            "num_problems": args.num_problems,
            "top_k": args.top_k,
            "scales": scales,
            "elapsed_minutes": elapsed / 60,
        },
    }
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Plots
    plot_bar_chart(bar_results, os.path.join(output_dir, "ablation_barplot.png"))
    plot_scale_sweep(scale_results, bar_results["baseline"], args.top_k,
                     os.path.join(output_dir, "scale_sweep.png"))
    plot_ablation_curve(topk_results, random_results, os.path.join(output_dir, "ablation_curve.png"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
