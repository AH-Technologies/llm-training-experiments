#!/usr/bin/env python3
"""Visualize the attention rhythm pipeline end-to-end.

Validates correctness of head classification, WAAD, FAI, and gamma by
producing stacked diagnostic plots for real model-generated math responses.

Produces per problem:
  - rhythm_viz_{i}.png           Full-length 4-subplot rhythm figure
  - rhythm_viz_{i}_zoom.png      100-token zoomed window with every token labeled
  - rhythm_viz_{i}_attn_matrices.png       Full A_bar_loc / A_bar_glob heatmaps
  - rhythm_viz_{i}_attn_matrices_zoom.png  Zoomed heatmaps with token labels

Usage (on GPU node):
  python scripts/attention_sparks_thinking/visualize_rhythm.py \
      --model_name Qwen/Qwen2.5-Math-1.5B \
      --num_problems 3
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

from attention_sparks_thinking.attention_rhythm import (
    compute_fai_vectorized as compute_fai,
    compute_gamma,
    compute_waad,
)
from attention_sparks_thinking.head_classifier import (
    aggregate_attention_hooks,
    classify_heads,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _clean_token(tok: str) -> str:
    """Clean a token string for display on axes (ASCII-safe)."""
    tok = tok.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")
    tok = tok.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
    # Strip to ASCII to avoid font rendering issues
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
    """Generate a greedy response and return (full_ids, response_start)."""
    # Try chat template first; fall back to raw text for base models
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Base model without chat template — just concatenate messages
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
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    return output_ids, prompt_len


def compute_token_entropy(model, input_ids: torch.Tensor, response_start: int) -> torch.Tensor:
    """Compute per-token entropy of the next-token distribution."""
    with torch.no_grad():
        logits = model(input_ids).logits

    logits_resp = logits[0, response_start - 1 : -1, :]
    log_probs = torch.log_softmax(logits_resp.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.cpu()


# ═══════════════════════════════════════════════════════════════════════
# Plot: Full-length 4-subplot rhythm figure
# ═══════════════════════════════════════════════════════════════════════

def plot_rhythm(tokens, waad, fai, entropy, gamma, problem_text, save_path):
    """Full-length rhythm figure. Labels every 10th token."""
    n = len(tokens)
    x = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(max(14, n * 0.12), 16), sharex=True)
    fig.suptitle(
        f"Attention Rhythm Diagnostic\n{problem_text[:120]}{'...' if len(problem_text) > 120 else ''}",
        fontsize=11, y=0.98,
    )

    axes[0].plot(x, waad, color="#2563eb", linewidth=0.9, alpha=0.9)
    axes[0].fill_between(x, waad, alpha=0.15, color="#2563eb")
    axes[0].set_ylabel("WAAD", fontsize=10)
    axes[0].set_title("Windowed Average Attention Distance (should show sawtooth pattern)", fontsize=9, loc="left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, fai, color="#dc2626", linewidth=0.9, alpha=0.9)
    axes[1].fill_between(x, fai, alpha=0.15, color="#dc2626")
    axes[1].set_ylabel("FAI", fontsize=10)
    axes[1].set_title("Future Attention Influence (should show sparse spikes at anchors)", fontsize=9, loc="left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, entropy, color="#7c3aed", linewidth=0.9, alpha=0.9)
    axes[2].fill_between(x, entropy, alpha=0.15, color="#7c3aed")
    axes[2].set_ylabel("Entropy (nats)", fontsize=10)
    axes[2].set_title("Next-token entropy (high entropy should correlate with WAAD peaks)", fontsize=9, loc="left")
    axes[2].grid(True, alpha=0.3)

    colors = ["#059669" if g >= 1.45 else "#f59e0b" if g > 1.0 else "#d1d5db" for g in gamma]
    axes[3].bar(x, gamma, width=1.0, color=colors, edgecolor="none")
    axes[3].set_ylabel("Gamma", fontsize=10)
    axes[3].set_title("Per-token gamma (green=1.5, amber=1.25, gray=1.0)", fontsize=9, loc="left")
    axes[3].set_ylim(0.9, max(1.6, gamma.max() + 0.1))
    axes[3].axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    axes[3].grid(True, alpha=0.3)

    label_positions = list(range(0, n, 10))
    label_texts = [f"{p}: {_clean_token(tokens[p])}" for p in label_positions if p < n]
    axes[3].set_xticks(label_positions[:len(label_texts)])
    axes[3].set_xticklabels(label_texts, rotation=90, fontsize=6, ha="center")
    axes[3].set_xlabel("Token position", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot: Zoomed 100-token window with vertical trace lines
# ═══════════════════════════════════════════════════════════════════════

def plot_rhythm_zoom(tokens, waad, fai, entropy, gamma, problem_text, save_path, window_size=100):
    """Zoomed view of a 100-token window from the middle of the response.

    Every token is labeled. Vertical dashed lines at gamma > 1.0 span all subplots.
    """
    n = len(tokens)
    if n <= window_size:
        start = 0
        end = n
    else:
        start = max(0, (n - window_size) // 2)
        end = start + window_size

    w_tokens = tokens[start:end]
    w_waad = waad[start:end]
    w_fai = fai[start:end]
    w_entropy = entropy[start:end]
    w_gamma = gamma[start:end]
    w_n = len(w_tokens)
    x = np.arange(w_n)

    fig, axes = plt.subplots(4, 1, figsize=(max(20, w_n * 0.25), 20), sharex=True)
    fig.suptitle(
        f"Zoomed Rhythm (tokens {start}–{end - 1})\n"
        f"{problem_text[:120]}{'...' if len(problem_text) > 120 else ''}",
        fontsize=11,
    )

    # Draw vertical trace lines for high-gamma positions across all subplots
    high_gamma_pos = np.where(w_gamma > 1.0)[0]
    for ax in axes:
        for pos in high_gamma_pos:
            color = "#059669" if w_gamma[pos] >= 1.45 else "#d97706"
            ax.axvline(x=pos, color=color, linewidth=0.5, linestyle="--", alpha=0.35)

    # Subplot 1: WAAD
    axes[0].plot(x, w_waad, color="#2563eb", linewidth=1.2, alpha=0.9, marker=".", markersize=2)
    axes[0].fill_between(x, w_waad, alpha=0.12, color="#2563eb")
    axes[0].set_ylabel("WAAD", fontsize=10)
    axes[0].set_title("WAAD — sawtooth peaks at reasoning step boundaries", fontsize=9, loc="left")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: FAI
    axes[1].plot(x, w_fai, color="#dc2626", linewidth=1.2, alpha=0.9, marker=".", markersize=2)
    axes[1].fill_between(x, w_fai, alpha=0.12, color="#dc2626")
    axes[1].set_ylabel("FAI", fontsize=10)
    axes[1].set_title("FAI — sparse spikes at anchor tokens", fontsize=9, loc="left")
    axes[1].grid(True, alpha=0.3)

    # Subplot 3: Entropy
    axes[2].plot(x, w_entropy, color="#7c3aed", linewidth=1.2, alpha=0.9, marker=".", markersize=2)
    axes[2].fill_between(x, w_entropy, alpha=0.12, color="#7c3aed")
    axes[2].set_ylabel("Entropy (nats)", fontsize=10)
    axes[2].set_title("Token entropy — high values = model uncertainty", fontsize=9, loc="left")
    axes[2].grid(True, alpha=0.3)

    # Subplot 4: Gamma bars
    colors = ["#059669" if g >= 1.45 else "#f59e0b" if g > 1.0 else "#d1d5db" for g in w_gamma]
    axes[3].bar(x, w_gamma, width=0.8, color=colors, edgecolor="none")
    axes[3].set_ylabel("Gamma", fontsize=10)
    axes[3].set_title("Gamma (green=anchor 1.5, amber=dominated/intro 1.25)", fontsize=9, loc="left")
    axes[3].set_ylim(0.9, max(1.6, w_gamma.max() + 0.1))
    axes[3].axhline(y=1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    axes[3].grid(True, alpha=0.3)

    # X-axis labels: show every 10th token position
    step = max(1, w_n // 10)
    shown_ticks = list(range(0, w_n, step))
    shown_labels = [str(start + i) for i in shown_ticks]
    axes[3].set_xticks(shown_ticks)
    axes[3].set_xticklabels(shown_labels, fontsize=7)
    axes[3].set_xlabel("Token position (response-relative)", fontsize=10)

    fig.subplots_adjust(bottom=0.06, top=0.93, left=0.05, right=0.98, hspace=0.15)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved zoomed plot to {save_path}")

    return start, end


# ═══════════════════════════════════════════════════════════════════════
# Plot: Attention matrix heatmaps (full length)
# ═══════════════════════════════════════════════════════════════════════

def plot_attn_matrices(A_bar_loc_np, A_bar_glob_np, response_start, problem_text, save_path):
    """Side-by-side heatmaps of A_bar_loc and A_bar_glob (full sequence)."""
    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle(
        f"Aggregated Attention Matrices (full sequence, response starts at {response_start})\n"
        f"{problem_text[:100]}{'...' if len(problem_text) > 100 else ''}",
        fontsize=11, y=0.98,
    )

    # Use only the response portion for cleaner viz
    loc_resp = A_bar_loc_np[response_start:, response_start:]
    glob_resp = A_bar_glob_np[response_start:, response_start:]

    im_l = ax_l.imshow(loc_resp, aspect="auto", origin="upper", cmap="hot",
                        norm=mcolors.PowerNorm(gamma=0.4))
    ax_l.set_title("A_bar_loc (local heads)\nExpect: near-diagonal block/sawtooth structure", fontsize=9)
    ax_l.set_xlabel("Source token (attended to)", fontsize=9)
    ax_l.set_ylabel("Target token (attending from)", fontsize=9)
    fig.colorbar(im_l, ax=ax_l, shrink=0.8)

    im_g = ax_g.imshow(glob_resp, aspect="auto", origin="upper", cmap="hot",
                        norm=mcolors.PowerNorm(gamma=0.4))
    ax_g.set_title("A_bar_glob (global heads)\nExpect: vertical stripes at anchor positions", fontsize=9)
    ax_g.set_xlabel("Source token (attended to)", fontsize=9)
    ax_g.set_ylabel("Target token (attending from)", fontsize=9)
    fig.colorbar(im_g, ax=ax_g, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved attention matrices to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Plot: Attention matrix heatmaps (zoomed window with token labels)
# ═══════════════════════════════════════════════════════════════════════

def plot_attn_matrices_zoom(A_bar_loc_np, A_bar_glob_np, response_start, tokens,
                            zoom_start, zoom_end, problem_text, save_path):
    """Zoomed heatmaps with token labels on both axes."""
    # The attention matrices are full-sequence (seq_len x seq_len).
    # Zoom window is in response-token coordinates (0-indexed from response_start).
    abs_start = response_start + zoom_start
    abs_end = response_start + zoom_end
    w = zoom_end - zoom_start

    loc_crop = A_bar_loc_np[abs_start:abs_end, abs_start:abs_end]
    glob_crop = A_bar_glob_np[abs_start:abs_end, abs_start:abs_end]

    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(max(16, w * 0.18), max(12, w * 0.14)))
    fig.suptitle(
        f"Attention Matrices (zoomed: response tokens {zoom_start}–{zoom_end - 1})\n"
        f"{problem_text[:100]}{'...' if len(problem_text) > 100 else ''}",
        fontsize=11,
    )

    im_l = ax_l.imshow(loc_crop, aspect="auto", origin="upper", cmap="hot",
                        norm=mcolors.PowerNorm(gamma=0.4))
    ax_l.set_title("A_bar_loc (local)\nNear-diagonal = correct", fontsize=9)
    fig.colorbar(im_l, ax=ax_l, shrink=0.7)

    im_g = ax_g.imshow(glob_crop, aspect="auto", origin="upper", cmap="hot",
                        norm=mcolors.PowerNorm(gamma=0.4))
    ax_g.set_title("A_bar_glob (global)\nVertical stripes = anchor tokens", fontsize=9)
    fig.colorbar(im_g, ax=ax_g, shrink=0.7)

    # Numeric tick labels (every 10th position)
    step = max(1, w // 10)
    shown = list(range(0, w, step))
    shown_labels = [str(zoom_start + i) for i in shown]

    for ax in (ax_l, ax_g):
        ax.set_xticks(shown)
        ax.set_xticklabels(shown_labels, fontsize=7)
        ax.set_yticks(shown)
        ax.set_yticklabels(shown_labels, fontsize=7)
        ax.set_xlabel("Source position (response-relative)", fontsize=8)
        ax.set_ylabel("Target position (response-relative)", fontsize=8)

    fig.subplots_adjust(bottom=0.18, top=0.90, left=0.10, right=0.95, wspace=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches=None)
    plt.close(fig)
    logger.info(f"Saved zoomed attention matrices to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualize attention rhythm pipeline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--num_problems", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_class_prompts", type=int, default=50)
    parser.add_argument("--head_quantile", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "attention_sparks_thinking/logs"))
    parser.add_argument("--zoom_window", type=int, default=100,
                        help="Number of tokens in the zoomed window")
    parser.add_argument("--waad_W", type=int, default=10)
    parser.add_argument("--fai_H_lo", type=int, default=10)
    parser.add_argument("--fai_H_hi", type=int, default=50)
    parser.add_argument("--quantile_q", type=float, default=0.4)
    parser.add_argument("--gamma_amp", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--neighborhood_k", type=int, default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load model ──
    logger.info(f"Loading model: {args.model_name} (eager attention)")
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
    logger.info(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # ── Load problems ──
    data_path = os.path.join(PROJECT_DIR, "attention_sparks_thinking/data/dapo_math_17k.parquet")
    problems = load_short_problems(data_path, num=args.num_problems)
    logger.info(f"Selected {len(problems)} problems")

    # ── Head classification ──
    logger.info(f"Running head classification with {args.num_class_prompts} prompts...")
    class_prompts = [p["user_text"] for p in problems]
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        df = pd.read_parquet(os.path.join(PROJECT_DIR, "attention_based_rewards/data/dapo_math_17k.parquet"))
    for _, row in df.head(args.num_class_prompts + 10).iterrows():
        msgs = row["prompt"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        text = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if text and text not in class_prompts:
            class_prompts.append(text)
        if len(class_prompts) >= args.num_class_prompts:
            break

    H_loc, H_glob, d_matrix = classify_heads(
        model, tokenizer, class_prompts[:args.num_class_prompts],
        head_quantile=args.head_quantile,
    )

    # Build reverse mapping from (layer, head) -> d value
    n_layers = model.config.num_hidden_layers
    sampled_layers = []
    lo = n_layers // 3
    hi = 2 * n_layers // 3
    n_sample = min(5, hi - lo + 1)
    if n_sample <= 1:
        sampled_layers = [(lo + hi) // 2]
    else:
        step = (hi - lo) / (n_sample - 1)
        sampled_layers = [lo + round(i * step) for i in range(n_sample)]
    layer_to_si = {l: si for si, l in enumerate(sampled_layers)}

    print("\n" + "=" * 70)
    print("HEAD CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"\nH_loc ({len(H_loc)} local heads):")
    for l, h in sorted(H_loc):
        si = layer_to_si.get(l, None)
        d_val = f"d={d_matrix[si, h].item():.2f}" if si is not None else "d=?"
        print(f"  Layer {l:2d}, Head {h:2d}  ({d_val})")
    print(f"\nH_glob ({len(H_glob)} global heads):")
    for l, h in sorted(H_glob):
        si = layer_to_si.get(l, None)
        d_val = f"d={d_matrix[si, h].item():.2f}" if si is not None else "d=?"
        print(f"  Layer {l:2d}, Head {h:2d}  ({d_val})")
    print(f"\nH_loc layers: {sorted(set(l for l, h in H_loc))}")
    print(f"H_glob layers: {sorted(set(l for l, h in H_glob))}")
    print()

    # ── Process each problem ──
    for idx, problem in enumerate(problems):
        print("=" * 70)
        print(f"PROBLEM {idx}: {problem['user_text'][:100]}...")
        print("=" * 70)

        # Generate response
        logger.info(f"Generating response for problem {idx}...")
        full_ids, response_start = generate_response(
            model, tokenizer, problem["messages"], max_new_tokens=args.max_new_tokens
        )
        seq_len = full_ids.shape[1]
        num_response = seq_len - response_start

        if num_response < 5:
            logger.warning(f"Problem {idx}: response too short ({num_response} tokens), skipping")
            continue

        response_ids = full_ids[0, response_start:].tolist()
        response_tokens = [tokenizer.decode([tid]) for tid in response_ids]

        print(f"\nResponse ({num_response} tokens):")
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(response_text[:500])
        if len(response_text) > 500:
            print(f"... [{len(response_text)} chars total]")
        print()

        # ── Aggregate attention maps ──
        logger.info(f"Extracting attention maps for problem {idx}...")
        attn_mask = torch.ones_like(full_ids)
        A_bar_loc, A_bar_glob = aggregate_attention_hooks(
            model, full_ids, attn_mask, H_loc, H_glob
        )
        A_bar_loc_np = A_bar_loc.cpu().float().numpy()
        A_bar_glob_np = A_bar_glob.cpu().float().numpy()

        # ── Compute metrics ──
        waad = compute_waad(A_bar_loc, response_start, W=args.waad_W)
        waad_np = waad.cpu().float().numpy()

        fai = compute_fai(A_bar_glob, response_start, H_lo=args.fai_H_lo, H_hi=args.fai_H_hi)
        fai_np = fai.cpu().float().numpy()

        logger.info(f"Computing token entropy for problem {idx}...")
        entropy = compute_token_entropy(model, full_ids, response_start)
        entropy_np = entropy.float().numpy()

        gamma_tensor, stats = compute_gamma(
            waad, fai,
            q=args.quantile_q,
            gamma_amp=args.gamma_amp,
            alpha=args.alpha,
            k=args.neighborhood_k,
        )
        gamma_np = gamma_tensor.cpu().float().numpy()

        # Trim to shortest
        min_len = min(len(waad_np), len(fai_np), len(entropy_np), len(gamma_np), len(response_tokens))
        waad_np = waad_np[:min_len]
        fai_np = fai_np[:min_len]
        entropy_np = entropy_np[:min_len]
        gamma_np = gamma_np[:min_len]
        response_tokens = response_tokens[:min_len]

        # ── Print full diagnostics ──
        frac_amplified = (gamma_np > 1.0).mean()
        print(f"Gamma stats:")
        print(f"  mean={gamma_np.mean():.4f}, std={gamma_np.std():.4f}")
        print(f"  tokens with gamma > 1.0: {(gamma_np > 1.0).sum()}/{min_len} ({frac_amplified:.1%})")
        print(f"  n_regular_anchor={stats.get('n_regular_anchor', 0)}")
        print(f"  n_dominated_anchor={stats.get('n_dominated_anchor', 0)}")
        print(f"  n_intro={stats.get('n_intro', 0)}")

        top_indices = np.argsort(gamma_np)[::-1][:10]
        print(f"\nTop 10 tokens by gamma:")
        for rank, ti in enumerate(top_indices):
            print(f"  {rank+1:2d}. pos={ti:4d}: {repr(response_tokens[ti]):30s} "
                  f"-> gamma={gamma_np[ti]:.3f}")

        if min_len > 10:
            corr = np.corrcoef(waad_np, entropy_np)[0, 1]
            if not np.isnan(corr):
                print(f"\nWAAD-entropy Pearson correlation: {corr:.3f}")
        print()

        # ── Figure 1: Full-length rhythm ──
        plot_rhythm(
            tokens=response_tokens, waad=waad_np, fai=fai_np,
            entropy=entropy_np, gamma=gamma_np,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"rhythm_viz_{idx}.png"),
        )

        # ── Figure 2: Zoomed rhythm with per-token labels ──
        zoom_start, zoom_end = plot_rhythm_zoom(
            tokens=response_tokens, waad=waad_np, fai=fai_np,
            entropy=entropy_np, gamma=gamma_np,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"rhythm_viz_{idx}_zoom.png"),
            window_size=args.zoom_window,
        )

        # Print zoomed-window diagnostics
        z_s = zoom_start
        z_e = zoom_end
        z_gamma = gamma_np[z_s:z_e]
        z_waad = waad_np[z_s:z_e]
        z_fai = fai_np[z_s:z_e]
        z_tokens = response_tokens[z_s:z_e]
        z_top = np.argsort(z_gamma)[::-1][:10]

        print(f"ZOOMED WINDOW (tokens {z_s}–{z_e - 1}):")
        print(f"  Top 10 high-gamma tokens in window:")
        for rank, ti in enumerate(z_top):
            abs_pos = z_s + ti
            print(f"  {rank+1:2d}. pos={abs_pos:4d}: {repr(z_tokens[ti]):25s} "
                  f"-> gamma={z_gamma[ti]:.2f}, waad={z_waad[ti]:.2f}, fai={z_fai[ti]:.4f}")
        print()

        # ── Figure 3: Full attention matrix heatmaps ──
        plot_attn_matrices(
            A_bar_loc_np, A_bar_glob_np, response_start,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"rhythm_viz_{idx}_attn_matrices.png"),
        )

        # ── Figure 4: Zoomed attention matrix heatmaps ──
        plot_attn_matrices_zoom(
            A_bar_loc_np, A_bar_glob_np, response_start, response_tokens,
            zoom_start, zoom_end,
            problem_text=problem["user_text"],
            save_path=os.path.join(args.output_dir, f"rhythm_viz_{idx}_attn_matrices_zoom.png"),
        )

    print("\n" + "=" * 70)
    print("DONE — all plots saved to:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
