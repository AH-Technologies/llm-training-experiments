#!/usr/bin/env python3
"""EAP-IG-style head discovery for Qwen3-4B-Base via activation patching.

Implements the "Thinking Sparks!" (Park et al., 2025) circuit analysis:
1. For each math problem, create clean and word-scrambled corrupted inputs
2. Measure each head's causal importance using activation patching
3. Aggregate importance scores across problems
4. Output ranked heads and importance heatmap

Two methods:
  --method attribution  (default, fast) Gradient-based approximation of
                        activation patching. This IS what EAP-IG computes.
                        ~3 passes per problem.
  --method patching     (exact, slow) Full activation patching — one forward
                        pass per head per problem. O(n_layers * n_heads) passes.

Output:
  - head_importance_qwen3.pt               Importance matrix + top-K list
  - head_importance_qwen3_heatmap.png      Heatmap visualization

Usage:
  python scripts/attention_sparks_thinking/analyze_heads_qwen.py3 \
      --model_name Qwen/Qwen3-4B-Base --num_problems 100
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


# ═══════════════════════════════════════════════════════════════════════
# Corruption: word scrambling (standard EAP-IG practice)
# ═══════════════════════════════════════════════════════════════════════

def scramble_question(text: str, seed: int = 42) -> str:
    """Randomly shuffle words in the question to destroy semantics."""
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_problems(data_path: str, num: int = 100) -> list[str]:
    """Load math problem texts from parquet."""
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
        if 20 < len(user_msg) < 500:
            problems.append(user_msg)
        if len(problems) >= num:
            break

    return problems


def format_prompt(tokenizer, question: str) -> str:
    """Format a question into a full prompt string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return SYSTEM_PROMPT + "\n\nProblem: " + question + "\n\nSolution:\n"


# ═══════════════════════════════════════════════════════════════════════
# Method 1: Attribution patching (gradient-based, fast)
#
# importance(l,h) ≈ Σ_pos (clean_act - corrupt_act) · grad_act
# where grad_act = d(loss)/d(pre_oproj_act) evaluated at clean input
# and loss = -log p(next_token) at the last position
#
# This is the standard EAP (Edge Attribution Patching) approximation.
# ═══════════════════════════════════════════════════════════════════════

def attribution_patching_importance(
    model, clean_ids: torch.Tensor, corrupt_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-head importance via attribution patching.

    Returns: (n_layers, n_heads) importance tensor
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    # head_dim may differ from hidden_size // n_heads (e.g. Qwen3-4B: 128 vs 80)
    head_dim = getattr(model.config, "head_dim", None) or model.model.layers[0].self_attn.o_proj.in_features // n_heads

    # ── Step 1: Forward on clean input WITH gradients on pre-o_proj ──
    # Capture raw o_proj inputs (no .view() — avoids graph disconnection issues)
    clean_raw = {}  # layer_idx -> (batch, seq_len, attn_hidden) in-graph tensor
    handles = []

    def make_capture_hook(layer_idx):
        def hook_fn(module, args):
            clean_raw[layer_idx] = args[0]  # keep as-is in computation graph
        return hook_fn

    for layer_idx in range(n_layers):
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(make_capture_hook(layer_idx))
        handles.append(handle)

    output = model(clean_ids)
    logits_last = output.logits[0, -1]
    target_token = logits_last.argmax()
    loss = -F.log_softmax(logits_last, dim=-1)[target_token]

    for handle in handles:
        handle.remove()

    # Compute gradients for all layers at once
    grad_targets = [clean_raw[i] for i in range(n_layers)]
    grads = torch.autograd.grad(loss, grad_targets, allow_unused=True)

    # ── Step 2: Forward on corrupt input (no grad needed) ──
    corrupt_raw = {}
    handles2 = []

    def make_capture_hook_nograd(layer_idx):
        def hook_fn(module, args):
            corrupt_raw[layer_idx] = args[0].detach()
        return hook_fn

    for layer_idx in range(n_layers):
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(make_capture_hook_nograd(layer_idx))
        handles2.append(handle)

    with torch.no_grad():
        model(corrupt_ids)

    for handle in handles2:
        handle.remove()

    # ── Step 3: Compute attribution scores ──
    device = clean_ids.device
    importance = torch.zeros(n_layers, n_heads, device=device)
    n_missing = 0

    for layer_idx in range(n_layers):
        grad_act = grads[layer_idx]
        if grad_act is None:
            n_missing += 1
            continue

        # Reshape to per-head: (batch, seq, attn_hidden) -> (batch, seq, n_heads, head_dim)
        clean_act = clean_raw[layer_idx].detach().view(-1, clean_ids.shape[1], n_heads, head_dim)
        corrupt_act = corrupt_raw[layer_idx].view(-1, clean_ids.shape[1], n_heads, head_dim)
        grad_heads = grad_act.view(-1, clean_ids.shape[1], n_heads, head_dim)

        # Attribution = |(clean - corrupt) · grad|, summed over seq and head_dim
        attr = ((clean_act - corrupt_act) * grad_heads).sum(dim=(0, 1, 3)).abs()
        importance[layer_idx] = attr

    if n_missing > 0:
        logger.warning(f"No gradient for {n_missing}/{n_layers} layers")

    model.zero_grad()
    return importance


# ═══════════════════════════════════════════════════════════════════════
# Method 2: Exact activation patching (slow but precise)
# ═══════════════════════════════════════════════════════════════════════

def cache_pre_oproj(model, input_ids: torch.Tensor):
    """Cache pre-o_proj activations for every layer. Returns dict + logits."""
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or model.model.layers[0].self_attn.o_proj.in_features // n_heads

    cached = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, args):
            inp = args[0].detach().clone()
            batch, seq_len, _ = inp.shape
            cached[layer_idx] = inp.view(batch, seq_len, n_heads, head_dim)
        return hook_fn

    for layer_idx in range(model.config.num_hidden_layers):
        o_proj = model.model.layers[layer_idx].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(make_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        output = model(input_ids)

    for handle in handles:
        handle.remove()

    return cached, output.logits


def patched_forward(
    model, input_ids: torch.Tensor,
    corrupt_pre_oproj: dict[int, torch.Tensor],
    target_layer: int, target_head: int,
) -> torch.Tensor:
    """Forward pass replacing one head's pre-o_proj activations with corrupted."""
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or model.model.layers[0].self_attn.o_proj.in_features // n_heads
    attn_hidden = n_heads * head_dim  # may differ from model hidden_size

    def o_proj_pre_hook(module, args):
        inp = args[0]  # (batch, seq_len, hidden_size)
        batch, seq_len, _ = inp.shape
        inp_heads = inp.view(batch, seq_len, n_heads, head_dim).clone()
        inp_heads[:, :, target_head, :] = corrupt_pre_oproj[target_layer][:, :, target_head, :]
        return (inp_heads.reshape(batch, seq_len, attn_hidden),)

    o_proj = model.model.layers[target_layer].self_attn.o_proj
    handle = o_proj.register_forward_pre_hook(o_proj_pre_hook)

    with torch.no_grad():
        output = model(input_ids)

    handle.remove()
    return output.logits


def compute_kl_divergence(clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
    """KL(clean || patched) at the last token position."""
    kl = F.kl_div(
        F.log_softmax(patched_logits[0, -1].float(), dim=-1),
        F.softmax(clean_logits[0, -1].float(), dim=-1),
        reduction="sum",
        log_target=False,
    )
    return kl.item()


def exact_patching_importance(
    model, clean_ids: torch.Tensor, corrupt_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-head importance via exact activation patching.

    For each head: replace its pre-o_proj activation from clean with corrupted,
    run forward, and measure KL divergence.

    Returns: (n_layers, n_heads) importance tensor
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    device = clean_ids.device

    # Cache activations
    _, clean_logits = cache_pre_oproj(model, clean_ids)
    corrupt_cache, _ = cache_pre_oproj(model, corrupt_ids)

    importance = torch.zeros(n_layers, n_heads, device=device)

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            patched_logits = patched_forward(
                model, clean_ids, corrupt_cache, layer_idx, head_idx,
            )
            kl = compute_kl_divergence(clean_logits, patched_logits)
            importance[layer_idx, head_idx] = kl

    del corrupt_cache, clean_logits
    return importance


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_importance_heatmap(
    importance: torch.Tensor, top_k: int, save_path: str,
    model_name: str = "Qwen3-4B-Base",
):
    """Plot head importance heatmap with top-K heads labeled."""
    n_layers, n_heads = importance.shape
    imp_np = importance.cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(max(14, n_heads * 0.5), max(10, n_layers * 0.35)))
    im = ax.imshow(
        imp_np, aspect="auto", origin="lower", cmap="YlOrRd",
        norm=mcolors.PowerNorm(gamma=0.5),
    )
    ax.set_xlabel("Head index", fontsize=12)
    ax.set_ylabel("Layer index", fontsize=12)
    ax.set_title(
        f"Head Importance (activation patching)\n{model_name}", fontsize=13,
    )
    fig.colorbar(im, ax=ax, label="Importance score", shrink=0.8)

    # Label top-K heads
    flat = importance.flatten()
    top_indices = flat.argsort(descending=True)[:top_k]
    for rank, idx in enumerate(top_indices.tolist()):
        layer, head = divmod(idx, n_heads)
        ax.plot(head, layer, "k*", markersize=8)
        ax.annotate(
            f"#{rank+1}", (head, layer),
            textcoords="offset points", xytext=(4, 4),
            fontsize=6, fontweight="bold", color="black",
        )

    ax.set_xticks(range(0, n_heads, max(1, n_heads // 16)))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 18)))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved importance heatmap to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EAP-IG-style head discovery")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--prefix_tokens", type=int, default=50,
                        help="Greedy-generated response prefix tokens for context")
    parser.add_argument("--method", choices=["attribution", "patching"], default="attribution",
                        help="'attribution' (fast, gradient-based) or 'patching' (exact, slow)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "attention_sparks_thinking/logs"))
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(PROJECT_DIR, "attention_sparks_thinking/data/dapo_math_17k.parquet"))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load model ──
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or model.model.layers[0].self_attn.o_proj.in_features // n_heads
    logger.info(f"Model: {n_layers} layers, {n_heads} heads, head_dim={head_dim}")
    logger.info(f"Method: {args.method}")
    logger.info(f"GPU mem after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    # ── Load problems ──
    problems = load_problems(args.data_path, num=args.num_problems)
    logger.info(f"Loaded {len(problems)} problems")

    # ── Main loop ──
    importance = torch.zeros(n_layers, n_heads, device=device)
    n_valid = 0

    for prob_idx, question in enumerate(problems):
        t0 = time.time()

        # Create clean and corrupted prompts
        clean_prompt = format_prompt(tokenizer, question)
        corrupted_question = scramble_question(question, seed=args.seed + prob_idx)
        corrupt_prompt = format_prompt(tokenizer, corrupted_question)

        # Tokenize prompts
        clean_prompt_ids = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(device)
        corrupt_prompt_ids = tokenizer(corrupt_prompt, return_tensors="pt").input_ids.to(device)

        # Generate short response prefix (greedy) for reasoning context
        with torch.no_grad():
            clean_with_prefix = model.generate(
                clean_prompt_ids,
                max_new_tokens=args.prefix_tokens,
                do_sample=False,
            )
            corrupt_with_prefix = model.generate(
                corrupt_prompt_ids,
                max_new_tokens=args.prefix_tokens,
                do_sample=False,
            )

        # Truncate to same length
        min_len = min(clean_with_prefix.shape[1], corrupt_with_prefix.shape[1])
        clean_ids = clean_with_prefix[:, :min_len]
        corrupt_ids = corrupt_with_prefix[:, :min_len]

        try:
            if args.method == "attribution":
                prob_importance = attribution_patching_importance(model, clean_ids, corrupt_ids)
            else:
                prob_importance = exact_patching_importance(model, clean_ids, corrupt_ids)
            importance += prob_importance
            n_valid += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM on problem {prob_idx}, skipping (seq_len={min_len})")
            torch.cuda.empty_cache()
            model.zero_grad()
            continue

        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        if (prob_idx + 1) % 10 == 0 or prob_idx == 0:
            logger.info(
                f"Problem {prob_idx+1}/{len(problems)} done in {elapsed:.1f}s "
                f"(seq_len={min_len})"
            )

    if n_valid == 0:
        logger.error("No valid problems processed!")
        return

    importance /= n_valid
    logger.info(f"Processed {n_valid}/{len(problems)} problems")

    # ── Top-K heads ──
    flat = importance.flatten()
    top_indices = flat.argsort(descending=True)[:args.top_k]
    top_heads = []
    print("\n" + "=" * 70)
    print(f"TOP-{args.top_k} HEADS BY ACTIVATION PATCHING IMPORTANCE")
    print(f"Method: {args.method}, Problems: {n_valid}")
    print("=" * 70)
    for rank, idx in enumerate(top_indices.tolist()):
        layer, head = divmod(idx, n_heads)
        val = importance[layer, head].item()
        top_heads.append((layer, head, val))
        print(f"  #{rank+1:2d}: Layer {layer:2d}, Head {head:2d}  (score = {val:.6f})")
    print()

    # Layer distribution
    layer_counts = {}
    for layer, head, val in top_heads:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    print("Layer distribution of top-20:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer:2d}: {layer_counts[layer]} heads")
    print()

    # ── Save results ──
    save_data = {
        "importance": importance.cpu(),
        "top_heads": top_heads,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "model_name": args.model_name,
        "num_problems": n_valid,
        "method": args.method,
        "args": vars(args),
    }
    pt_path = os.path.join(args.output_dir, "head_importance_qwen3.pt")
    torch.save(save_data, pt_path)
    logger.info(f"Saved importance data to {pt_path}")

    # ── Plot heatmap ──
    plot_importance_heatmap(
        importance, top_k=args.top_k,
        save_path=os.path.join(args.output_dir, "head_importance_qwen3_heatmap.png"),
        model_name=args.model_name,
    )

    print("=" * 70)
    print("DONE")
    print(f"Results: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
