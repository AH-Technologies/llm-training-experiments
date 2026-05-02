#!/usr/bin/env python3
"""Visualize attention patterns of the top-K EAP-IG heads from Phase 1.

Reads the head_importance_qwen3.pt output from analyze_heads_qwen3.py and:
1. Generates math responses (greedy)
2. Extracts attention patterns for the top-K heads
3. Produces:
   - Per-head attention heatmap grid (4x5 for top-20)
   - Aggregated top-K attention heatmap (A_bar_eapig)
   - Comparison with WAAD/FAI-classified H_loc/H_glob heads

Usage:
  python scripts/attention_sparks_thinking/visualize_top_heads.py \
      --model_name Qwen/Qwen3-4B-Base \
      --importance_path attention_sparks_thinking/logs/head_importance_qwen3.pt
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from attention_sparks_thinking.head_classifier import classify_heads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Helpers (reused from visualize_rhythm.py)
# ═══════════════════════════════════════════════════════════════════════

def _clean_token(tok: str) -> str:
    """Clean a token string for display on axes (ASCII-safe)."""
    tok = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")
    tok = tok.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
    tok = tok.encode("ascii", errors="replace").decode("ascii")
    if len(tok) > 12:
        tok = tok[:10] + ".."
    return tok


def load_short_problems(data_path: str, num: int = 3) -> list[dict]:
    """Load math problems likely to produce short responses."""
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        alt = os.path.join(PROJECT_DIR, "attention_based_rewards/data/dapo_math_17k.parquet")
        df = pd.read_parquet(alt)

    problems = []
    for _, row in df.iterrows():
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        user_msg = next((m["content"] for m in prompt_msgs if m["role"] == "user"), "")
        if 20 < len(user_msg) < 300:
            problems.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "user_text": user_msg,
            })
        if len(problems) >= num * 3:
            break

    selected = problems[::3][:num]
    if len(selected) < num:
        selected = problems[:num]
    return selected


def generate_response(model, tokenizer, messages: list[dict], max_new_tokens: int = 512):
    """Generate a greedy response. Returns (full_ids, response_start)."""
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(m["content"] + "\n\n")
            elif m["role"] == "user":
                parts.append("Problem: " + m["content"] + "\n\nSolution:\n")
        prompt_text = "".join(parts)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
        )

    return output_ids, prompt_len


# ═══════════════════════════════════════════════════════════════════════
# Attention extraction for specific heads via hooks
# ═══════════════════════════════════════════════════════════════════════

def extract_head_attention(
    model, input_ids: torch.Tensor,
    target_heads: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """Extract attention patterns for specific (layer, head) pairs.

    Uses forward hooks on self_attn with output_attentions=True.
    Returns dict[(layer, head)] -> (seq_len, seq_len) attention matrix.
    """
    # Group by layer for efficient hooking
    heads_by_layer: dict[int, list[int]] = {}
    for l, h in target_heads:
        heads_by_layer.setdefault(l, []).append(h)

    results: dict[tuple[int, int], torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx, head_indices):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1][0]  # (n_heads, seq_len, seq_len)
                for h in head_indices:
                    results[(layer_idx, h)] = attn_weights[h].detach().cpu()
        return hook_fn

    for layer_idx, head_indices in heads_by_layer.items():
        attn_module = model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(make_hook(layer_idx, head_indices))
        handles.append(handle)

    with torch.no_grad():
        model(input_ids, output_attentions=True)

    for handle in handles:
        handle.remove()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_head_grid(
    head_attns: dict[tuple[int, int], np.ndarray],
    top_heads: list[tuple[int, int, float]],
    response_start: int,
    problem_text: str,
    save_path: str,
    ncols: int = 5,
):
    """Plot a grid of attention heatmaps for top-K heads (response portion only)."""
    k = len(top_heads)
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    fig.suptitle(
        f"Top-{k} EAP-IG Heads — Attention Patterns (response tokens)\n"
        f"{problem_text[:100]}{'...' if len(problem_text) > 100 else ''}",
        fontsize=11, y=0.98,
    )

    if nrows == 1:
        axes = axes.reshape(1, -1)

    for rank, (layer, head, score) in enumerate(top_heads):
        r, c = divmod(rank, ncols)
        ax = axes[r, c]

        key = (layer, head)
        if key in head_attns:
            attn = head_attns[key]
            # Show response-to-response portion
            attn_resp = attn[response_start:, response_start:]
            im = ax.imshow(
                attn_resp, aspect="auto", origin="upper", cmap="hot",
                norm=mcolors.PowerNorm(gamma=0.4),
            )
            ax.set_title(f"L{layer}H{head} (#{rank+1}, s={score:.4f})", fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"L{layer}H{head} (missing)", fontsize=8)

        ax.set_xlabel("src", fontsize=7)
        ax.set_ylabel("tgt", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for rank in range(k, nrows * ncols):
        r, c = divmod(rank, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved head grid to {save_path}")


def plot_aggregated_attention(
    head_attns: dict[tuple[int, int], np.ndarray],
    top_heads: list[tuple[int, int, float]],
    response_start: int,
    problem_text: str,
    save_path: str,
):
    """Plot aggregated attention from all top-K heads (averaged)."""
    attn_sum = None
    count = 0
    for layer, head, _ in top_heads:
        key = (layer, head)
        if key in head_attns:
            if attn_sum is None:
                attn_sum = head_attns[key].astype(np.float64)
            else:
                attn_sum += head_attns[key].astype(np.float64)
            count += 1

    if count == 0:
        logger.warning("No attention data to aggregate")
        return

    A_bar = attn_sum / count
    A_bar_resp = A_bar[response_start:, response_start:]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        A_bar_resp, aspect="auto", origin="upper", cmap="hot",
        norm=mcolors.PowerNorm(gamma=0.4),
    )
    ax.set_title(
        f"Aggregated Attention: Top-{len(top_heads)} EAP-IG Heads (A_bar_eapig)\n"
        f"{problem_text[:100]}{'...' if len(problem_text) > 100 else ''}",
        fontsize=11,
    )
    ax.set_xlabel("Source token (attended to)", fontsize=10)
    ax.set_ylabel("Target token (attending from)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved aggregated attention to {save_path}")


def plot_head_comparison(
    eapig_heads: list[tuple[int, int, float]],
    H_loc: list[tuple[int, int]],
    H_glob: list[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    save_path: str,
):
    """Compare EAP-IG top heads with WAAD/FAI-classified H_loc/H_glob.

    Produces a (n_layers, n_heads) grid colored by head membership:
      - Red: EAP-IG top-K only
      - Blue: H_loc only
      - Green: H_glob only
      - Purple: EAP-IG + H_loc overlap
      - Orange: EAP-IG + H_glob overlap
    """
    eapig_set = {(l, h) for l, h, _ in eapig_heads}
    loc_set = set(H_loc)
    glob_set = set(H_glob)

    # Create color-coded grid
    grid = np.zeros((n_layers, n_heads, 3), dtype=np.float32)

    for l in range(n_layers):
        for h in range(n_heads):
            in_eapig = (l, h) in eapig_set
            in_loc = (l, h) in loc_set
            in_glob = (l, h) in glob_set

            if in_eapig and in_loc:
                grid[l, h] = [0.6, 0.2, 0.8]   # purple
            elif in_eapig and in_glob:
                grid[l, h] = [1.0, 0.6, 0.0]   # orange
            elif in_eapig:
                grid[l, h] = [0.9, 0.2, 0.2]   # red
            elif in_loc:
                grid[l, h] = [0.2, 0.4, 0.9]   # blue
            elif in_glob:
                grid[l, h] = [0.2, 0.8, 0.4]   # green
            else:
                grid[l, h] = [0.95, 0.95, 0.95] # light gray

    fig, ax = plt.subplots(figsize=(max(14, n_heads * 0.45), max(10, n_layers * 0.3)))
    ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Head index", fontsize=12)
    ax.set_ylabel("Layer index", fontsize=12)
    ax.set_title(
        f"Head Classification Comparison\n"
        f"Red=EAP-IG only | Blue=H_loc | Green=H_glob | "
        f"Purple=EAP-IG+H_loc | Orange=EAP-IG+H_glob",
        fontsize=10,
    )

    # Add grid lines
    for l in range(n_layers):
        ax.axhline(y=l - 0.5, color="white", linewidth=0.3)
    for h in range(n_heads):
        ax.axvline(x=h - 0.5, color="white", linewidth=0.3)

    ax.set_xticks(range(0, n_heads, max(1, n_heads // 16)))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 18)))

    # Overlap stats
    eapig_in_loc = eapig_set & loc_set
    eapig_in_glob = eapig_set & glob_set
    eapig_in_either = eapig_set & (loc_set | glob_set)
    stats_text = (
        f"EAP-IG top-{len(eapig_heads)}: "
        f"{len(eapig_in_loc)} in H_loc, "
        f"{len(eapig_in_glob)} in H_glob, "
        f"{len(eapig_in_either)} in either, "
        f"{len(eapig_heads) - len(eapig_in_either)} exclusive"
    )
    ax.text(
        0.5, -0.08, stats_text, ha="center", va="top",
        transform=ax.transAxes, fontsize=9, style="italic",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison to {save_path}")

    # Print overlap stats
    print("\n" + "=" * 70)
    print("HEAD CLASSIFICATION COMPARISON")
    print("=" * 70)
    print(f"EAP-IG top-{len(eapig_heads)}: {sorted((l,h) for l,h,_ in eapig_heads)}")
    print(f"  Overlap with H_loc ({len(H_loc)} heads): {sorted(eapig_in_loc)}")
    print(f"  Overlap with H_glob ({len(H_glob)} heads): {sorted(eapig_in_glob)}")
    print(f"  Overlap with either: {len(eapig_in_either)}/{len(eapig_heads)}")
    print(f"  Exclusive to EAP-IG: {len(eapig_heads) - len(eapig_in_either)}")

    # Jaccard similarity
    if eapig_set | loc_set:
        j_loc = len(eapig_set & loc_set) / len(eapig_set | loc_set)
        print(f"  Jaccard(EAP-IG, H_loc) = {j_loc:.3f}")
    if eapig_set | glob_set:
        j_glob = len(eapig_set & glob_set) / len(eapig_set | glob_set)
        print(f"  Jaccard(EAP-IG, H_glob) = {j_glob:.3f}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualize top EAP-IG heads")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--importance_path", type=str, required=True,
                        help="Path to head_importance_qwen3.pt from Phase 1")
    parser.add_argument("--num_problems", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_class_prompts", type=int, default=50,
                        help="Number of prompts for WAAD/FAI head classification")
    parser.add_argument("--head_quantile", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "attention_sparks_thinking/logs"))
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(PROJECT_DIR, "attention_sparks_thinking/data/dapo_math_17k.parquet"))
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load Phase 1 results ──
    logger.info(f"Loading importance data from {args.importance_path}")
    data = torch.load(args.importance_path, map_location="cpu", weights_only=False)
    importance = data["importance"]
    top_heads_raw = data["top_heads"]  # list of (layer, head, score)
    n_layers = data["n_layers"]
    n_heads = data["n_heads"]
    top_heads = top_heads_raw[:args.top_k]

    print(f"\nTop-{args.top_k} heads from Phase 1:")
    for rank, (layer, head, score) in enumerate(top_heads):
        print(f"  #{rank+1:2d}: Layer {layer:2d}, Head {head:2d}  (score = {score:.6f})")
    print()

    # ── Load model ──
    logger.info(f"Loading model: {args.model_name} (eager attention for attn weights)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    logger.info(f"GPU mem after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # ── Load problems ──
    problems = load_short_problems(args.data_path, num=args.num_problems)
    logger.info(f"Selected {len(problems)} problems")

    # ── WAAD/FAI head classification for comparison ──
    logger.info("Running WAAD/FAI head classification for comparison...")
    try:
        df = pd.read_parquet(args.data_path)
    except FileNotFoundError:
        df = pd.read_parquet(os.path.join(PROJECT_DIR, "attention_based_rewards/data/dapo_math_17k.parquet"))

    class_prompts = [p["user_text"] for p in problems]
    for _, row in df.head(args.num_class_prompts + 10).iterrows():
        msgs = row["prompt"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        text = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if text and text not in class_prompts:
            class_prompts.append(text)
        if len(class_prompts) >= args.num_class_prompts:
            break

    H_loc, H_glob, _ = classify_heads(
        model, tokenizer, class_prompts[:args.num_class_prompts],
        head_quantile=args.head_quantile,
    )
    logger.info(f"H_loc: {len(H_loc)} heads, H_glob: {len(H_glob)} heads")

    # ── Comparison plot ──
    plot_head_comparison(
        top_heads, H_loc, H_glob, n_layers, n_heads,
        save_path=os.path.join(args.output_dir, "head_comparison_eapig_vs_waad.png"),
    )

    # ── Process each problem ──
    target_head_pairs = [(l, h) for l, h, _ in top_heads]

    for idx, problem in enumerate(problems):
        print("=" * 70)
        print(f"PROBLEM {idx}: {problem['user_text'][:100]}...")
        print("=" * 70)

        # Generate response
        logger.info(f"Generating response for problem {idx}...")
        full_ids, response_start = generate_response(
            model, tokenizer, problem["messages"], max_new_tokens=args.max_new_tokens,
        )
        seq_len = full_ids.shape[1]
        num_response = seq_len - response_start

        if num_response < 10:
            logger.warning(f"Problem {idx}: response too short ({num_response} tokens), skipping")
            continue

        response_ids = full_ids[0, response_start:].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(f"Response ({num_response} tokens):")
        print(response_text[:400])
        if len(response_text) > 400:
            print(f"... [{len(response_text)} chars total]")
        print()

        # Extract attention for top-K heads
        logger.info(f"Extracting attention patterns for {len(target_head_pairs)} heads...")
        head_attns = extract_head_attention(model, full_ids, target_head_pairs)
        head_attns_np = {k: v.float().numpy() for k, v in head_attns.items()}

        # Plot 1: Per-head grid
        plot_head_grid(
            head_attns_np, top_heads, response_start,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"top_heads_grid_{idx}.png"),
        )

        # Plot 2: Aggregated attention
        plot_aggregated_attention(
            head_attns_np, top_heads, response_start,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"top_heads_aggregated_{idx}.png"),
        )

        del head_attns, head_attns_np
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DONE — all plots saved to:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
