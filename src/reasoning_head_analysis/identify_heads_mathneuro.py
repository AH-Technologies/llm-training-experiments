#!/usr/bin/env python3
"""Identify task-specific attention heads using neurosurgery (forward-pass-only).

Based on: "Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities
Using Only Forward Passes" (Christ et al., 2024, arXiv:2410.16930)

Method:
1. Compute Wanda-style importance (|W| * ||activation||_2) on positive and negative data
2. Select top-K% important parameters for each domain
3. Set difference: positive-important AND NOT negative-important = task-specific
4. Aggregate task-specific parameter counts to attention head level

Supports multiple contrasts via --contrast:
  math_vs_reading   — GSM8K vs RACE (original MathNeuro paper)
  reasoning_vs_easy  — ARC-Challenge vs ARC-Easy (reasoning difficulty)
  cot_vs_direct      — GSM8K full CoT vs GSM8K final-answer-only

Output is compatible with the ablation pipeline (head_importance.pt with head_scores
and active_heads keys).

Usage:
  # Original MathNeuro (default)
  python -m reasoning_head_analysis.identify_heads_mathneuro \
      --model Qwen/Qwen2.5-Math-1.5B-Instruct --contrast math_vs_reading

  # Reasoning vs factual contrast
  python -m reasoning_head_analysis.identify_heads_mathneuro \
      --model /path/to/checkpoint --contrast reasoning_vs_easy

  # CoT vs direct answer contrast
  python -m reasoning_head_analysis.identify_heads_mathneuro \
      --model /path/to/checkpoint --contrast cot_vs_direct
"""
import argparse
import logging
import os
import random
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Dataset preparation
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# ── Contrast presets ─────────────────────────────────────────────────
# Each contrast defines a (positive, negative) dataset pair.
# "positive" = the capability we want to isolate.
# "negative" = the baseline capability to subtract out.
CONTRASTS = {
    "math_vs_reading": {
        "description": "Math reasoning vs reading comprehension (original MathNeuro)",
        "positive": "gsm8k",
        "negative": "race",
    },
    "reasoning_vs_easy": {
        "description": "Hard reasoning vs easy factual recall (ARC-Challenge vs ARC-Easy)",
        "positive": "arc_challenge",
        "negative": "arc_easy",
    },
    "cot_vs_direct": {
        "description": "Chain-of-thought vs direct answer (same questions, different answer style)",
        "positive": "gsm8k",
        "negative": "gsm8k_direct",
    },
    "math_cot_vs_direct": {
        "description": "MATH CoT vs direct answer (competition math, different answer style)",
        "positive": "math_cot",
        "negative": "math_direct",
    },
}


def format_chat(question, answer=None):
    """Format as Qwen2.5 chat template."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if answer:
        prompt += answer + "<|im_end|>"
    return prompt


def _tokenize_texts(texts, tokenizer, max_length, min_tokens=16):
    """Tokenize a list of strings, filtering out very short ones."""
    samples = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        return_tensors="pt")
        if enc["input_ids"].shape[1] >= min_tokens:
            samples.append(enc)
    return samples


def _load_gsm8k(n_samples):
    """Load GSM8K with full chain-of-thought answers."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:  # over-sample in case some are filtered
        if len(texts) >= n_samples:
            break
        texts.append(format_chat(ds[i]["question"], ds[i]["answer"]))
    return texts


def _load_gsm8k_direct(n_samples):
    """Load GSM8K with only the final numeric answer (no reasoning trace)."""
    import re
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:
        if len(texts) >= n_samples:
            break
        answer = ds[i]["answer"]
        # GSM8K answers end with "#### <number>"
        match = re.search(r"####\s*(.+)", answer)
        final_answer = match.group(1).strip() if match else answer.split("\n")[-1]
        texts.append(format_chat(ds[i]["question"], f"The answer is {final_answer}."))
    return texts


def _load_race(n_samples):
    """Load RACE reading comprehension (non-math QA)."""
    ds = load_dataset("ehovy/race", "all", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:
        if len(texts) >= n_samples:
            break
        texts.append(format_chat(
            f"Read the passage and answer the question.\n\n"
            f"Passage: {ds[i]['article']}\n\nQuestion: {ds[i]['question']}",
            f"The answer is {ds[i]['answer']}."
        ))
    return texts


def _load_arc(n_samples, config_name):
    """Load ARC (AI2 Reasoning Challenge) — 'ARC-Challenge' or 'ARC-Easy'."""
    ds = load_dataset("allenai/ai2_arc", config_name, split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:
        if len(texts) >= n_samples:
            break
        item = ds[i]
        question = item["question"]
        choices = item["choices"]
        # Format choices
        choice_text = "\n".join(
            f"  {label}) {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        answer_key = item["answerKey"]
        # Find the answer text
        try:
            answer_idx = choices["label"].index(answer_key)
            answer_text = choices["text"][answer_idx]
        except ValueError:
            answer_text = answer_key
        texts.append(format_chat(
            f"{question}\n{choice_text}",
            f"The answer is {answer_key}) {answer_text}."
        ))
    return texts


def _load_math_cot(n_samples):
    """Load MATH with full chain-of-thought solutions."""
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:
        if len(texts) >= n_samples:
            break
        texts.append(format_chat(ds[i]["problem"], ds[i]["solution"]))
    return texts


def _load_math_direct(n_samples):
    """Load MATH with only the boxed final answer (no reasoning trace)."""
    import re
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    texts = []
    for i in indices[:n_samples * 2]:
        if len(texts) >= n_samples:
            break
        solution = ds[i]["solution"]
        match = re.search(r"\\boxed\{([^}]+)\}", solution)
        final_answer = match.group(1) if match else solution.split("\n")[-1]
        texts.append(format_chat(ds[i]["problem"], f"The answer is {final_answer}."))
    return texts


# Registry of dataset loaders
DATASET_LOADERS = {
    "gsm8k": _load_gsm8k,
    "gsm8k_direct": _load_gsm8k_direct,
    "race": _load_race,
    "arc_challenge": lambda n: _load_arc(n, "ARC-Challenge"),
    "arc_easy": lambda n: _load_arc(n, "ARC-Easy"),
    "math_cot": _load_math_cot,
    "math_direct": _load_math_direct,
}


def load_data(dataset_name, n_samples, tokenizer, max_length):
    """Load and tokenize data for a named dataset."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_LOADERS.keys())}")
    logger.info(f"Loading {dataset_name} ({n_samples} samples)...")
    texts = DATASET_LOADERS[dataset_name](n_samples)
    samples = _tokenize_texts(texts, tokenizer, max_length)
    logger.info(f"  Loaded {len(samples)} {dataset_name} samples")
    return samples


# ═══════════════════════════════════════════════════════════════════════
# Wanda-style importance scoring
# ═══════════════════════════════════════════════════════════════════════

def compute_importance(model, samples, device):
    """Compute Wanda-style importance for all Linear layers.

    For each Linear layer with weight W [out, in] and input activation X:
        score(i, j) = |W_ij| * ||X_j||_2

    Scores are accumulated across all samples.

    Returns:
        dict: layer_name -> importance tensor [out_features, in_features]
    """
    importance = {}
    activations = {}
    hooks = []

    # Register hooks on all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    activations[layer_name] = input[0].detach()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    t_start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)

            model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute and accumulate importance
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in activations:
                    act = activations[name]
                    if act.dim() == 3:
                        act = act.reshape(-1, act.shape[-1])  # [N_tokens, d_in]
                    # Per-feature L2 norm across all tokens
                    feat_norm = act.float().norm(p=2, dim=0)  # [d_in]
                    # Wanda score: |W| * ||X_j||_2
                    score = module.weight.data.float().abs() * feat_norm.unsqueeze(0)

                    if name not in importance:
                        importance[name] = score.cpu()
                    else:
                        importance[name] += score.cpu()

            activations.clear()

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                elapsed_s = time.time() - t_start
                rate = (i + 1) / max(elapsed_s, 1e-6)
                eta = (len(samples) - i - 1) / max(rate, 1e-6)
                logger.info(f"  [{device}] Processed {i+1}/{len(samples)} samples "
                            f"({elapsed_s:.0f}s elapsed, ~{eta:.0f}s remaining)")

    for h in hooks:
        h.remove()

    return importance


# ═══════════════════════════════════════════════════════════════════════
# Multi-GPU worker
# ═══════════════════════════════════════════════════════════════════════

def worker(rank, n_gpus, model_name, positive_samples, negative_samples,
           result_dict):
    """Process a shard of positive and negative samples on one GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Shard the data
    def shard(data, rank, n_gpus):
        size = len(data) // n_gpus
        start = rank * size
        end = start + size if rank < n_gpus - 1 else len(data)
        return data[start:end]

    pos_shard = shard(positive_samples, rank, n_gpus)
    neg_shard = shard(negative_samples, rank, n_gpus)
    logger.info(f"[GPU {rank}] Positive: {len(pos_shard)}, Negative: {len(neg_shard)}")

    # Load model
    logger.info(f"[GPU {rank}] Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True,
    ).to(device)
    logger.info(f"[GPU {rank}] Model loaded in {time.time() - t0:.1f}s")

    # Compute importance for positive data
    logger.info(f"[GPU {rank}] Computing positive importance...")
    pos_imp = compute_importance(model, pos_shard, device)

    # Compute importance for negative data
    logger.info(f"[GPU {rank}] Computing negative importance...")
    neg_imp = compute_importance(model, neg_shard, device)

    result_dict[rank] = {"positive": pos_imp, "negative": neg_imp}
    logger.info(f"[GPU {rank}] Done.")


# ═══════════════════════════════════════════════════════════════════════
# Head-level aggregation
# ═══════════════════════════════════════════════════════════════════════

def top_k_mask(tensor, keep_ratio):
    """Create a binary mask keeping the top keep_ratio fraction of params."""
    flat = tensor.flatten()
    k = max(1, int(flat.numel() * keep_ratio))
    threshold = flat.abs().topk(k).values[-1]
    return flat.abs() >= threshold


def aggregate_to_heads(math_importance, nonmath_importance, config, keep_ratio):
    """Aggregate parameter-level importance to attention head scores.

    For each head (l, h), count the number of math-specific parameters in its
    q_proj, k_proj, v_proj, and o_proj projections. A parameter is math-specific
    if it is in the top-K% for math but NOT in the top-K% for non-math.

    Also computes a continuous score: sum of (math_imp - nonmath_imp) for
    math-specific parameters in each head.

    Returns:
        head_scores: Tensor[n_layers, n_heads] — continuous importance
        head_counts: Tensor[n_layers, n_heads] — math-specific param counts
    """
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = config.hidden_size // n_heads
    heads_per_kv = n_heads // n_kv_heads

    head_scores = torch.zeros(n_layers, n_heads)
    head_counts = torch.zeros(n_layers, n_heads)

    # Also track total math-specific params across all layers for logging
    total_math_specific = 0
    total_params = 0

    for layer in range(n_layers):
        prefix = f"model.layers.{layer}.self_attn"

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"{prefix}.{proj}"
            if key not in math_importance or key not in nonmath_importance:
                continue

            math_imp = math_importance[key]
            nonmath_imp = nonmath_importance[key]

            # Binary masks: top-K% for each domain
            math_mask = top_k_mask(math_imp, keep_ratio)
            nonmath_mask = top_k_mask(nonmath_imp, keep_ratio)

            # Set difference: math-specific = math AND NOT nonmath
            math_specific = math_mask & ~nonmath_mask

            # Reshape back to weight shape
            math_specific = math_specific.view(math_imp.shape)
            math_diff = (math_imp - nonmath_imp) * math_specific.float()

            total_math_specific += math_specific.sum().item()
            total_params += math_imp.numel()

            # Aggregate to head level based on projection type
            if proj == "o_proj":
                # o_proj: [hidden_size, n_heads * head_dim]
                # Columns [h*head_dim:(h+1)*head_dim] belong to head h
                for h in range(n_heads):
                    cols = slice(h * head_dim, (h + 1) * head_dim)
                    head_counts[layer, h] += math_specific[:, cols].sum().item()
                    head_scores[layer, h] += math_diff[:, cols].abs().sum().item()
            elif proj == "q_proj":
                # q_proj: [n_heads * head_dim, hidden_size]
                # Rows [h*head_dim:(h+1)*head_dim] belong to head h
                for h in range(n_heads):
                    rows = slice(h * head_dim, (h + 1) * head_dim)
                    head_counts[layer, h] += math_specific[rows, :].sum().item()
                    head_scores[layer, h] += math_diff[rows, :].abs().sum().item()
            elif proj in ("k_proj", "v_proj"):
                # k/v_proj: [n_kv_heads * head_dim, hidden_size]
                # With GQA, distribute KV head importance equally
                for kv_h in range(n_kv_heads):
                    rows = slice(kv_h * head_dim, (kv_h + 1) * head_dim)
                    count = math_specific[rows, :].sum().item()
                    score = math_diff[rows, :].abs().sum().item()
                    for h in range(kv_h * heads_per_kv, (kv_h + 1) * heads_per_kv):
                        head_counts[layer, h] += count / heads_per_kv
                        head_scores[layer, h] += score / heads_per_kv

    pct = total_math_specific / max(total_params, 1) * 100
    logger.info(f"Math-specific parameters (attn only): {total_math_specific:,.0f} / "
                f"{total_params:,.0f} ({pct:.2f}%)")

    return head_scores, head_counts


def select_active_heads(head_scores, top_k=None, threshold_pct=None):
    """Select active heads by top-K or by percentile threshold.

    Default: top-K=8 (matching EAP-IG ablation pipeline default).
    """
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    n_heads_total = flat.numel()

    if top_k is not None:
        k = min(top_k, n_heads_total)
    elif threshold_pct is not None:
        # Select heads above the threshold_pct percentile
        threshold = torch.quantile(flat.float(), 1.0 - threshold_pct / 100)
        k = (flat >= threshold).sum().item()
    else:
        k = 20

    n_cols = head_scores.shape[1]
    active = []
    for i in range(k):
        idx = sorted_idx[i].item()
        l, h = idx // n_cols, idx % n_cols
        active.append((l, h))

    return sorted(active)


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap(head_scores, output_path, title="MathNeuro: Math-Specific Head Importance"):
    """Save head importance heatmap."""
    n_layers, n_heads = head_scores.shape
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6), max(6, n_layers * 0.3)))
        sns.heatmap(
            head_scores.numpy(), ax=ax,
            xticklabels=[f"H{h}" for h in range(n_heads)],
            yticklabels=[f"L{l}" for l in range(n_layers)],
            cmap="Reds",
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6), max(6, n_layers * 0.3)))
        im = ax.imshow(head_scores.numpy(), cmap="Reds", aspect="auto")
        ax.set_xticks(range(n_heads), [f"H{h}" for h in range(n_heads)])
        ax.set_yticks(range(n_layers), [f"L{l}" for l in range(n_layers)])
        fig.colorbar(im)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Identify task-specific attention heads via neurosurgery")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--contrast", type=str, default="math_vs_reading",
                        choices=list(CONTRASTS.keys()),
                        help="Predefined contrast: " + ", ".join(
                            f"'{k}' ({v['description']})" for k, v in CONTRASTS.items()))
    parser.add_argument("--positive_dataset", type=str, default=None,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Override positive dataset (default: from --contrast)")
    parser.add_argument("--negative_dataset", type=str, default=None,
                        choices=list(DATASET_LOADERS.keys()),
                        help="Override negative dataset (default: from --contrast)")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples per domain")
    parser.add_argument("--keep_ratio", type=float, default=0.05,
                        help="Top-K%% of parameters to consider important (0.05 = 5%%)")
    parser.add_argument("--top_k_heads", type=int, default=None,
                        help="Number of top heads to mark as active (default: 8)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for each sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda' (default: auto-detect)")
    args = parser.parse_args()

    # Resolve contrast → dataset names
    contrast = CONTRASTS[args.contrast]
    pos_dataset = args.positive_dataset or contrast["positive"]
    neg_dataset = args.negative_dataset or contrast["negative"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device setup
    use_cpu = (args.device == "cpu") if args.device else not torch.cuda.is_available()
    n_gpus = 0 if use_cpu else torch.cuda.device_count()
    device_str = "cpu" if use_cpu else f"{n_gpus} GPU(s)"

    safe_model = args.model.replace("/", "_").replace(".", "_")
    output_dir = args.output_dir or os.path.join(
        "results", "reasoning_head_analysis", "identification", "mathneuro", safe_model)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device_str}")
    logger.info(f"Contrast: {args.contrast} — {contrast['description']}")
    logger.info(f"  Positive: {pos_dataset}, Negative: {neg_dataset}")
    logger.info(f"Config: {args.n_samples} samples/domain, keep_ratio={args.keep_ratio}, "
                f"max_length={args.max_length}")
    logger.info(f"Output: {output_dir}")

    # Load tokenizer for data prep
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prepare datasets
    positive_samples = load_data(pos_dataset, args.n_samples, tokenizer, args.max_length)
    negative_samples = load_data(neg_dataset, args.n_samples, tokenizer, args.max_length)

    # Run importance computation
    logger.info(f"Computing importance scores ({device_str})...")
    t0 = time.time()

    if use_cpu or n_gpus <= 1:
        device = "cpu" if use_cpu else "cuda:0"
        logger.info(f"Loading model on {device}...")
        dtype = torch.float32 if use_cpu else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)

        logger.info("Computing positive importance...")
        pos_imp = compute_importance(model, positive_samples, device)
        logger.info("Computing negative importance...")
        neg_imp = compute_importance(model, negative_samples, device)
        del model
    else:
        # Multi-GPU: shard data across GPUs
        result_dict = mp.Manager().dict()
        mp.spawn(
            worker,
            args=(n_gpus, args.model, positive_samples, negative_samples, result_dict),
            nprocs=n_gpus,
            join=True,
        )

        # Aggregate across GPUs
        logger.info("Aggregating across GPUs...")
        pos_imp = {}
        neg_imp = {}
        for rank in range(n_gpus):
            for key, val in result_dict[rank]["positive"].items():
                if key not in pos_imp:
                    pos_imp[key] = val
                else:
                    pos_imp[key] += val
            for key, val in result_dict[rank]["negative"].items():
                if key not in neg_imp:
                    neg_imp[key] = val
                else:
                    neg_imp[key] += val

    elapsed = time.time() - t0
    logger.info(f"Importance computation completed in {elapsed / 60:.1f} min")

    # Get model config for head dimensions
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Aggregate to head level
    logger.info("Aggregating to attention head level...")
    head_scores, head_counts = aggregate_to_heads(
        pos_imp, neg_imp, config, args.keep_ratio)

    n_layers, n_heads = head_scores.shape

    # Select active heads
    active_heads = select_active_heads(
        head_scores, top_k=args.top_k_heads)
    logger.info(f"Active heads ({len(active_heads)}): {active_heads}")

    # Rank and display top heads
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    logger.info(f"Top 20 task-specific heads ({args.contrast}):")
    for rank, idx in enumerate(sorted_idx[:20]):
        l, h = idx.item() // n_heads, idx.item() % n_heads
        score = flat[idx].item()
        count = head_counts[l, h].item()
        logger.info(f"  #{rank+1}: L{l}H{h} = {score:.4f} ({count:.0f} task-specific params)")

    # Save results (compatible with ablation pipeline)
    importance_path = os.path.join(output_dir, "head_importance.pt")
    torch.save({
        "head_scores": head_scores,
        "head_counts": head_counts,
        "active_heads": active_heads,
        "config": {
            "model": args.model,
            "method": "neurosurgery",
            "contrast": args.contrast,
            "positive_dataset": pos_dataset,
            "negative_dataset": neg_dataset,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "keep_ratio": args.keep_ratio,
            "max_length": args.max_length,
            "device": device_str,
            "elapsed_minutes": elapsed / 60,
        },
    }, importance_path)
    logger.info(f"Saved head importance to {importance_path}")

    # Save parameter-level data for deeper analysis
    param_path = os.path.join(output_dir, "parameter_importance.pt")
    torch.save({
        "positive": {k: v.half() for k, v in pos_imp.items()},
        "negative": {k: v.half() for k, v in neg_imp.items()},
    }, param_path)
    logger.info(f"Saved parameter-level importance to {param_path}")

    # Heatmaps
    contrast_label = args.contrast.replace("_", " ").title()
    plot_heatmap(head_scores, os.path.join(output_dir, "head_importance_heatmap.png"),
                 title=f"Neurosurgery: Head Importance ({contrast_label})")
    plot_heatmap(head_counts,
                 os.path.join(output_dir, "head_counts_heatmap.png"),
                 title=f"Neurosurgery: Task-Specific Param Count ({contrast_label})")

    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")
    logger.info("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
