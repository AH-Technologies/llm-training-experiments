#!/usr/bin/env python3
"""Analyze systematic head identification results across methods, seeds, and datasets.

Loads results from 18 runs (3 methods × 2 datasets × 3 seeds) and assesses:
1. Per-method summary (top-20 heads with scores)
2. Seed stability (same dataset, different seeds)
3. Dataset stability (same seed, different datasets)
4. Per-method consensus (frequency across all 6 runs)
5. Cross-method consensus (overlap of robust heads)
6. Token attention profile (GPU, requires --skip-attention to skip)
7. Visualizations (rank heatmap, frequency bar chart, seed variance)

Usage:
  # Full analysis (needs GPU):
  python scripts/reasoning_head_analysis/analyze_systematic_identification.py

  # Stats only (no GPU):
  python scripts/reasoning_head_analysis/analyze_systematic_identification.py --skip-attention
"""
import argparse
import itertools
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = "results/reasoning_head_analysis/identification/systematic_base"
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

METHODS = {
    "eap_ig": {
        "datasets": ["aime", "math"],
        "label": "EAP-IG",
    },
    "neurosurgery": {
        "datasets": ["gsm8k", "math"],
        "label": "Neurosurgery",
    },
    "retrieval": {
        "datasets": ["wikitext", "pg19"],
        "label": "Retrieval",
    },
}

SEEDS = [42, 123, 456]
TOP_K = 20
SEP = "=" * 70


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def top_k(scores, k):
    """Return top-k heads as list of (layer, head) sorted by score descending."""
    flat = scores.flatten()
    n_h = scores.shape[1]
    idx = flat.argsort(descending=True)[:k]
    return [(ii.item() // n_h, ii.item() % n_h) for ii in idx]


def top_k_set(scores, k):
    return set(top_k(scores, k))


def head_rank_map(scores):
    """Return dict mapping (layer, head) -> rank (0-indexed)."""
    flat = scores.flatten()
    n_h = scores.shape[1]
    idx = flat.argsort(descending=True)
    rank = {}
    for r, ii in enumerate(idx):
        l, h = ii.item() // n_h, ii.item() % n_h
        rank[(l, h)] = r
    return rank


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def fmt_head(lh):
    return f"L{lh[0]}H{lh[1]}"


# ═══════════════════════════════════════════════════════════════════════
# Load results
# ═══════════════════════════════════════════════════════════════════════

def load_all_results():
    """Load all available head_importance.pt files.

    Returns:
        results: dict[method][dataset_seed] = {head_scores, config, ...}
        missing: list of (method, dataset, seed) tuples not found
    """
    results = defaultdict(dict)
    missing = []

    for method, info in METHODS.items():
        for dataset in info["datasets"]:
            for seed in SEEDS:
                key = f"{dataset}_seed{seed}"
                path = os.path.join(BASE_DIR, method, key, "head_importance.pt")
                if os.path.exists(path):
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    results[method][key] = data
                else:
                    missing.append((method, dataset, seed))

    return dict(results), missing


# ═══════════════════════════════════════════════════════════════════════
# Analysis 1: Per-Method Summary
# ═══════════════════════════════════════════════════════════════════════

def analysis_per_method_summary(results):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 1: PER-METHOD SUMMARY")
    print(f"{'#' * 70}")

    for method, info in METHODS.items():
        runs = results.get(method, {})
        if not runs:
            print(f"\n  {info['label']}: no results available")
            continue

        print(f"\n{SEP}")
        print(f"  {info['label']} ({len(runs)} runs)")
        print(SEP)

        for key in sorted(runs.keys()):
            data = runs[key]
            scores = data["head_scores"]
            t20 = top_k(scores, TOP_K)
            cfg = data.get("config", {})
            print(f"\n  {key}:")
            print(f"    method={cfg.get('method','?')}, seed={cfg.get('seed','?')}")
            for rank, (l, h) in enumerate(t20):
                score = scores[l, h].item()
                print(f"    #{rank+1:2d}: L{l:2d}H{h:2d} = {score:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# Analysis 2: Seed Stability
# ═══════════════════════════════════════════════════════════════════════

def analysis_seed_stability(results):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 2: SEED STABILITY (same dataset, different seeds)")
    print(f"{'#' * 70}")

    for method, info in METHODS.items():
        runs = results.get(method, {})
        if not runs:
            continue

        for dataset in info["datasets"]:
            seed_runs = {}
            for seed in SEEDS:
                key = f"{dataset}_seed{seed}"
                if key in runs:
                    seed_runs[seed] = runs[key]

            if len(seed_runs) < 2:
                print(f"\n  {info['label']} / {dataset}: <2 seeds available, skipping")
                continue

            print(f"\n{SEP}")
            print(f"  {info['label']} / {dataset} ({len(seed_runs)} seeds)")
            print(SEP)

            seed_list = sorted(seed_runs.keys())
            top20_sets = {s: top_k_set(seed_runs[s]["head_scores"], TOP_K) for s in seed_list}

            # Pairwise Jaccard and Spearman
            for s1, s2 in itertools.combinations(seed_list, 2):
                j = jaccard(top20_sets[s1], top20_sets[s2])
                overlap = top20_sets[s1] & top20_sets[s2]
                sc1 = seed_runs[s1]["head_scores"].flatten().numpy()
                sc2 = seed_runs[s2]["head_scores"].flatten().numpy()
                rho, p = spearmanr(sc1, sc2)
                print(f"  seed {s1} vs {s2}:")
                print(f"    Top-{TOP_K} Jaccard = {j:.3f} (overlap: {len(overlap)}/{TOP_K})")
                print(f"    Spearman rho = {rho:.4f} (p={p:.2e})")

            # Seed-stable heads (in top-20 for ALL seeds)
            if len(seed_list) >= 2:
                stable = set.intersection(*top20_sets.values())
                print(f"  Seed-stable heads (top-{TOP_K} in all {len(seed_list)} seeds): "
                      f"{len(stable)}")
                if stable:
                    for h in sorted(stable):
                        ranks = []
                        for s in seed_list:
                            rm = head_rank_map(seed_runs[s]["head_scores"])
                            ranks.append(f"s{s}:#{rm[h]+1}")
                        print(f"    {fmt_head(h)}: {', '.join(ranks)}")


# ═══════════════════════════════════════════════════════════════════════
# Analysis 3: Dataset Stability
# ═══════════════════════════════════════════════════════════════════════

def analysis_dataset_stability(results):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 3: DATASET STABILITY (same seed, different datasets)")
    print(f"{'#' * 70}")

    for method, info in METHODS.items():
        runs = results.get(method, {})
        datasets = info["datasets"]
        if len(datasets) < 2 or not runs:
            continue

        for seed in SEEDS:
            keys = [f"{d}_seed{seed}" for d in datasets]
            available = [k for k in keys if k in runs]
            if len(available) < 2:
                print(f"\n  {info['label']} / seed {seed}: <2 datasets available, skipping")
                continue

            print(f"\n{SEP}")
            print(f"  {info['label']} / seed {seed}")
            print(SEP)

            d1, d2 = datasets
            k1, k2 = f"{d1}_seed{seed}", f"{d2}_seed{seed}"

            s1 = runs[k1]["head_scores"]
            s2 = runs[k2]["head_scores"]
            t1 = top_k_set(s1, TOP_K)
            t2 = top_k_set(s2, TOP_K)

            overlap = t1 & t2
            j = jaccard(t1, t2)
            rho, p = spearmanr(s1.flatten().numpy(), s2.flatten().numpy())

            print(f"  {d1} vs {d2}:")
            print(f"    Top-{TOP_K} overlap: {len(overlap)}/{TOP_K}, Jaccard = {j:.3f}")
            print(f"    Spearman rho = {rho:.4f} (p={p:.2e})")

            if overlap:
                print(f"    Dataset-robust heads:")
                for h in sorted(overlap):
                    r1 = head_rank_map(s1)
                    r2 = head_rank_map(s2)
                    print(f"      {fmt_head(h)}: {d1}=#{r1[h]+1}, {d2}=#{r2[h]+1}")


# ═══════════════════════════════════════════════════════════════════════
# Analysis 4: Per-Method Consensus
# ═══════════════════════════════════════════════════════════════════════

def analysis_per_method_consensus(results):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 4: PER-METHOD CONSENSUS (frequency across all runs)")
    print(f"{'#' * 70}")

    method_robust = {}  # method -> set of heads appearing >=4/6

    for method, info in METHODS.items():
        runs = results.get(method, {})
        if not runs:
            continue

        print(f"\n{SEP}")
        print(f"  {info['label']} ({len(runs)} runs)")
        print(SEP)

        freq = Counter()
        for key, data in runs.items():
            for h in top_k_set(data["head_scores"], TOP_K):
                freq[h] += 1

        # Sort by frequency then by layer/head
        by_freq = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

        n_runs = len(runs)
        highly_robust = []
        robust = []

        print(f"\n  Head frequency table (top-{TOP_K} across {n_runs} runs):")
        print(f"  {'Head':>8s}  {'Count':>5s}  {'Pct':>5s}  Status")
        print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*20}")

        for (l, h), count in by_freq:
            pct = count / n_runs * 100
            status = ""
            if count == n_runs:
                status = "HIGHLY ROBUST (all runs)"
                highly_robust.append((l, h))
            elif count >= 4:
                status = f"ROBUST (>= 4/{n_runs})"
                robust.append((l, h))
            print(f"  L{l:2d}H{h:2d}     {count:3d}   {pct:5.1f}%  {status}")

        method_robust[method] = set(highly_robust + robust)
        print(f"\n  Summary: {len(highly_robust)} highly robust, "
              f"{len(robust)} robust, "
              f"{len(method_robust[method])} total (>= 4/{n_runs})")

    return method_robust


# ═══════════════════════════════════════════════════════════════════════
# Analysis 5: Cross-Method Consensus
# ═══════════════════════════════════════════════════════════════════════

def analysis_cross_method_consensus(results, method_robust):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 5: CROSS-METHOD CONSENSUS")
    print(f"{'#' * 70}")

    available_methods = [m for m in METHODS if m in method_robust and method_robust[m]]
    if len(available_methods) < 2:
        print("\n  <2 methods with robust heads, skipping")
        return

    # Show robust heads per method
    for method in available_methods:
        label = METHODS[method]["label"]
        heads = sorted(method_robust[method])
        print(f"\n  {label} robust heads ({len(heads)}): "
              f"{', '.join(fmt_head(h) for h in heads)}")

    # Pairwise overlap
    print(f"\n  Pairwise overlap of robust heads:")
    for m1, m2 in itertools.combinations(available_methods, 2):
        l1, l2 = METHODS[m1]["label"], METHODS[m2]["label"]
        s1, s2 = method_robust[m1], method_robust[m2]
        overlap = s1 & s2
        j = jaccard(s1, s2)
        print(f"    {l1} vs {l2}: {len(overlap)} shared, Jaccard={j:.3f}")
        if overlap:
            print(f"      {', '.join(fmt_head(h) for h in sorted(overlap))}")

    # Universal (in all methods)
    if len(available_methods) >= 2:
        universal = set.intersection(*(method_robust[m] for m in available_methods))
        print(f"\n  Universal reasoning heads (robust in ALL {len(available_methods)} methods): "
              f"{len(universal)}")
        if universal:
            for h in sorted(universal):
                which = [METHODS[m]["label"] for m in available_methods if h in method_robust[m]]
                print(f"    {fmt_head(h)}: {', '.join(which)}")

    # Average pairwise Spearman between methods (average head_scores across runs)
    print(f"\n  Average pairwise Spearman (method-averaged scores):")
    method_avg_scores = {}
    for method in available_methods:
        runs = results[method]
        all_scores = torch.stack([d["head_scores"] for d in runs.values()])
        method_avg_scores[method] = all_scores.mean(dim=0)

    for m1, m2 in itertools.combinations(available_methods, 2):
        l1, l2 = METHODS[m1]["label"], METHODS[m2]["label"]
        s1 = method_avg_scores[m1].flatten().numpy()
        s2 = method_avg_scores[m2].flatten().numpy()
        rho, p = spearmanr(s1, s2)
        print(f"    {l1} vs {l2}: rho={rho:.4f} (p={p:.2e})")


# ═══════════════════════════════════════════════════════════════════════
# Analysis 6: Token Attention Profile (GPU)
# ═══════════════════════════════════════════════════════════════════════

TOKEN_CATEGORIES = {
    "numbers": re.compile(r"^\d[\d,\.]*$"),
    "math_ops": re.compile(r"^[+\-*/=\^()><≤≥×÷%]$"),
    "reasoning": re.compile(
        r"^(step|first|then|therefore|so|because|let|we|since|thus|hence|next|now|if|need|means|implies|gives|note|recall|know|consider|suppose|assume|want|find|get|have|use|compute|calculate|simplify|substitute|solve|apply|notice|observe|check|verify|confirm)$",
        re.IGNORECASE,
    ),
    "self_correction": re.compile(
        r"^(wait|actually|but|however|alternatively|hmm|oops|no|wrong|instead|correction|sorry|hold|retry|reconsider|scratch|redo|err|mistake)$",
        re.IGNORECASE,
    ),
    "answer_format": re.compile(
        r"^(boxed|answer|final|therefore|result|solution|total|output|equals|\\boxed)$",
        re.IGNORECASE,
    ),
}


def classify_token(token_str):
    """Classify a decoded token string into a category."""
    t = token_str.strip()
    if not t:
        return "other"
    for cat, pattern in TOKEN_CATEGORIES.items():
        if pattern.match(t):
            return cat
    return "other"


def run_token_attention_profile(results, method_robust, n_prompts=50, model_name=None):
    """Analysis 6: For each method's robust heads, profile what token types they attend to."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 6: TOKEN ATTENTION PROFILE")
    print(f"{'#' * 70}")

    available_methods = [m for m in METHODS if m in method_robust and method_robust[m]]
    if not available_methods:
        print("\n  No robust heads found, skipping attention profile")
        return

    # Collect all heads we need to analyze
    all_heads = set()
    for m in available_methods:
        all_heads |= method_robust[m]
    print(f"\n  Analyzing {len(all_heads)} unique heads across {len(available_methods)} methods")

    # Load model
    if model_name is None:
        model_name = "Qwen/Qwen2.5-Math-1.5B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Loading model {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    n_attn_heads = model.config.num_attention_heads
    gqa_group = n_attn_heads // n_kv_heads  # heads per KV group

    # Load GSM8K prompts
    print(f"  Loading {n_prompts} GSM8K prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = []
    for i, ex in enumerate(ds):
        if i >= n_prompts:
            break
        # Format as chat
        question = ex["question"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Track per-head category counts
    categories = list(TOKEN_CATEGORIES.keys()) + ["question", "other"]
    head_cat_counts = {h: Counter() for h in all_heads}
    head_total = {h: 0 for h in all_heads}

    print(f"  Running prefill on {len(prompts)} prompts...")
    for pi, prompt in enumerate(prompts):
        if (pi + 1) % 10 == 0:
            print(f"    {pi+1}/{len(prompts)}...")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]

        # Decode each token for classification
        token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

        # Find question boundary (tokens from user message)
        # Heuristic: tokens before "assistant" marker are question tokens
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        # Find where the assistant generation prompt starts
        assistant_markers = ["<|im_start|>assistant", "<|assistant|>"]
        question_end_pos = seq_len  # default: all question
        for marker in assistant_markers:
            marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
            # Search for this subsequence
            for start in range(seq_len - len(marker_tokens) + 1):
                if input_ids[start:start+len(marker_tokens)].tolist() == marker_tokens:
                    question_end_pos = start
                    break

        # Classify tokens
        token_cats = []
        for ti, ts in enumerate(token_strs):
            if ti < question_end_pos:
                cat = classify_token(ts)
                if cat == "other":
                    cat = "question"  # tokens in question portion default to "question"
            else:
                cat = classify_token(ts)
            token_cats.append(cat)

        with torch.no_grad():
            out = model(**inputs, output_attentions=True, use_cache=False)

        # out.attentions: tuple of n_layers, each [batch, n_heads, seq_len, seq_len]
        attentions = out.attentions

        for (layer, head) in all_heads:
            if layer >= len(attentions):
                continue
            # attentions[layer]: [1, n_heads, seq_len, seq_len]
            attn = attentions[layer][0, head, :, :]  # [seq_len, seq_len]

            # For each query position, find the argmax attended-to position
            argmax_pos = attn.argmax(dim=-1)  # [seq_len]

            for qi in range(seq_len):
                attended_pos = argmax_pos[qi].item()
                if attended_pos < len(token_cats):
                    head_cat_counts[(layer, head)][token_cats[attended_pos]] += 1
                    head_total[(layer, head)] += 1

        # Free attention tensors
        del attentions, out
        torch.cuda.empty_cache()

    # Print results per method
    for method in available_methods:
        label = METHODS[method]["label"]
        heads = sorted(method_robust[method])
        if not heads:
            continue

        print(f"\n{SEP}")
        print(f"  {label} — Robust Head Attention Profile ({len(heads)} heads)")
        print(SEP)

        # Header
        cat_header = "  " + f"{'Head':>8s}"
        for cat in categories:
            cat_header += f"  {cat[:10]:>10s}"
        print(cat_header)
        print("  " + "-" * (8 + 12 * len(categories)))

        # Accumulate for method average
        method_avg = Counter()
        method_total = 0

        for h in heads:
            total = head_total[h]
            if total == 0:
                continue
            row = f"  {fmt_head(h):>8s}"
            for cat in categories:
                frac = head_cat_counts[h][cat] / total
                row += f"  {frac:10.3f}"
                method_avg[cat] += head_cat_counts[h][cat]
            method_total += total
            print(row)

        # Method average
        if method_total > 0:
            print("  " + "-" * (8 + 12 * len(categories)))
            row = f"  {'Average':>8s}"
            for cat in categories:
                frac = method_avg[cat] / method_total
                row += f"  {frac:10.3f}"
            print(row)

    # Save bar chart per method
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(available_methods), figsize=(6 * len(available_methods), 5),
                             squeeze=False)

    for mi, method in enumerate(available_methods):
        ax = axes[0, mi]
        label = METHODS[method]["label"]
        heads = sorted(method_robust[method])

        # Average distribution for this method's robust heads
        total = sum(head_total[h] for h in heads)
        if total == 0:
            continue
        fracs = []
        for cat in categories:
            fracs.append(sum(head_cat_counts[h][cat] for h in heads) / total)

        bars = ax.bar(range(len(categories)), fracs, color="steelblue")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c[:8] for c in categories], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction of top-attended tokens")
        ax.set_title(f"{label}\n({len(heads)} robust heads)")
        ax.set_ylim(0, max(fracs) * 1.2 if fracs else 1)

    fig.suptitle("Token Attention Profile of Robust Heads", fontsize=13)
    fig.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "token_attention_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved token attention profile to {path}")


# ═══════════════════════════════════════════════════════════════════════
# Analysis 7: Visualizations
# ═══════════════════════════════════════════════════════════════════════

def analysis_visualizations(results, method_robust):
    print(f"\n\n{'#' * 70}")
    print("  ANALYSIS 7: VISUALIZATIONS")
    print(f"{'#' * 70}")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # --- 7a: Rank heatmap per method ---
    for method, info in METHODS.items():
        runs = results.get(method, {})
        if not runs:
            continue

        # Collect all heads that appear in any top-20
        all_top = set()
        for data in runs.values():
            all_top |= top_k_set(data["head_scores"], TOP_K)

        if not all_top:
            continue

        heads_sorted = sorted(all_top)
        run_keys = sorted(runs.keys())

        # Build rank matrix: rows=heads, cols=runs
        rank_matrix = np.full((len(heads_sorted), len(run_keys)), np.nan)
        for ci, key in enumerate(run_keys):
            rm = head_rank_map(runs[key]["head_scores"])
            for ri, h in enumerate(heads_sorted):
                rank_matrix[ri, ci] = rm.get(h, float("nan"))

        fig, ax = plt.subplots(figsize=(max(8, len(run_keys) * 1.5),
                                        max(6, len(heads_sorted) * 0.3)))

        # Only show ranks up to 2*TOP_K for visibility, clip higher ranks
        display_matrix = np.clip(rank_matrix, 0, 2 * TOP_K)
        im = ax.imshow(display_matrix, cmap="RdYlGn_r", aspect="auto",
                        vmin=0, vmax=2 * TOP_K)
        ax.set_xticks(range(len(run_keys)))
        ax.set_xticklabels([k.replace("_seed", "\ns") for k in run_keys],
                           fontsize=7, rotation=0)
        ax.set_yticks(range(len(heads_sorted)))
        ax.set_yticklabels([fmt_head(h) for h in heads_sorted], fontsize=7)
        ax.set_xlabel("Run")
        ax.set_ylabel("Head")
        ax.set_title(f"{info['label']} — Head Rank Across Runs (green=high rank)")
        fig.colorbar(im, ax=ax, label="Rank (lower=more important)")
        fig.tight_layout()
        path = os.path.join(ANALYSIS_DIR, f"rank_heatmap_{method}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    # --- 7b: Per-head frequency bar chart across all 18 runs ---
    freq = Counter()
    total_runs = 0
    for method in METHODS:
        runs = results.get(method, {})
        for data in runs.values():
            for h in top_k_set(data["head_scores"], TOP_K):
                freq[h] += 1
            total_runs += 1

    if freq:
        # Show heads that appear in >= 2 runs
        heads_to_show = sorted([h for h, c in freq.items() if c >= 2],
                               key=lambda h: (-freq[h], h))
        if heads_to_show:
            fig, ax = plt.subplots(figsize=(max(10, len(heads_to_show) * 0.4), 5))
            counts = [freq[h] for h in heads_to_show]
            colors = ["#d62728" if c == total_runs else
                      "#ff7f0e" if c >= total_runs * 0.67 else
                      "#2ca02c" if c >= total_runs * 0.33 else
                      "#1f77b4" for c in counts]
            ax.bar(range(len(heads_to_show)), counts, color=colors)
            ax.set_xticks(range(len(heads_to_show)))
            ax.set_xticklabels([fmt_head(h) for h in heads_to_show],
                               rotation=90, fontsize=7)
            ax.set_ylabel(f"Appearances in top-{TOP_K} (out of {total_runs} runs)")
            ax.set_title(f"Head Frequency Across All {total_runs} Runs")
            ax.axhline(y=total_runs * 0.67, color="gray", linestyle="--", alpha=0.5,
                       label=f"67% ({total_runs*0.67:.0f})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(ANALYSIS_DIR, "head_frequency_all_runs.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  Saved {path}")

    # --- 7c: Seed variance heatmap (coefficient of variation per head) ---
    for method, info in METHODS.items():
        runs = results.get(method, {})
        if len(runs) < 2:
            continue

        # Stack all scores
        all_scores = torch.stack([d["head_scores"] for d in runs.values()])
        # all_scores: [n_runs, n_layers, n_heads]
        mean_scores = all_scores.mean(dim=0)
        std_scores = all_scores.std(dim=0)

        # Coefficient of variation (avoid div by 0)
        cv = torch.where(mean_scores > 1e-10,
                         std_scores / mean_scores,
                         torch.zeros_like(mean_scores))

        n_layers, n_heads = cv.shape
        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6),
                                        max(6, n_layers * 0.3)))

        try:
            import seaborn as sns
            sns.heatmap(
                cv.numpy(), ax=ax,
                xticklabels=[f"H{h}" for h in range(n_heads)],
                yticklabels=[f"L{l}" for l in range(n_layers)],
                cmap="YlOrRd",
                vmin=0,
            )
        except ImportError:
            im = ax.imshow(cv.numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
            ax.set_xticks(range(n_heads))
            ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
            fig.colorbar(im, ax=ax)

        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{info['label']} — Score CV Across {len(runs)} Runs\n(lower = more stable)")
        fig.tight_layout()
        path = os.path.join(ANALYSIS_DIR, f"seed_variance_{method}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Save aggregated importance for ablation study
# ═══════════════════════════════════════════════════════════════════════

def save_aggregated_importance(results):
    """Average head_scores across all runs per method and save for ablation.

    Saves to: results/.../systematic_base/<method>/aggregated/head_importance.pt
    Format: {head_scores, active_heads (top-20), config}
    """
    print(f"\n\n{'#' * 70}")
    print("  SAVING AGGREGATED IMPORTANCE FILES")
    print(f"{'#' * 70}")

    for method, info in METHODS.items():
        runs = results.get(method, {})
        if not runs:
            print(f"\n  {info['label']}: no runs, skipping")
            continue

        all_scores = torch.stack([d["head_scores"] for d in runs.values()])
        avg_scores = all_scores.mean(dim=0)

        # Top-20 heads from averaged scores
        flat = avg_scores.flatten()
        n_h = avg_scores.shape[1]
        idx = flat.argsort(descending=True)[:TOP_K]
        active_heads = [(ii.item() // n_h, ii.item() % n_h) for ii in idx]

        out_dir = os.path.join(BASE_DIR, method, "aggregated")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "head_importance.pt")

        torch.save({
            "head_scores": avg_scores,
            "active_heads": active_heads,
            "config": {
                "method": method,
                "n_runs_averaged": len(runs),
                "runs_used": sorted(runs.keys()),
                "top_k": TOP_K,
            },
        }, out_path)

        print(f"\n  {info['label']}: averaged {len(runs)} runs → {out_path}")
        print(f"    Top-5: {', '.join(fmt_head(h) for h in active_heads[:5])}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze systematic head identification results")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip token attention profile (GPU-dependent analysis)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for attention profile (default: Qwen/Qwen2.5-Math-1.5B)")
    parser.add_argument("--n-prompts", type=int, default=50,
                        help="Number of GSM8K prompts for attention profile")
    args = parser.parse_args()

    print(SEP)
    print("  SYSTEMATIC HEAD IDENTIFICATION ANALYSIS")
    print(SEP)

    # Load all results
    results, missing = load_all_results()

    total_loaded = sum(len(v) for v in results.values())
    print(f"\n  Loaded {total_loaded}/18 result files")

    if missing:
        print(f"  Missing {len(missing)} results:")
        for method, dataset, seed in missing:
            print(f"    {METHODS[method]['label']} / {dataset} / seed {seed}")

    if total_loaded == 0:
        print("\n  No results found. Exiting.")
        sys.exit(1)

    # Run analyses 1-5
    analysis_per_method_summary(results)
    analysis_seed_stability(results)
    analysis_dataset_stability(results)
    method_robust = analysis_per_method_consensus(results)

    analysis_cross_method_consensus(results, method_robust)

    # Analysis 7: Visualizations (CPU)
    analysis_visualizations(results, method_robust)

    # Save aggregated importance files for ablation study
    save_aggregated_importance(results)

    # Analysis 6: Token attention profile (GPU)
    if not args.skip_attention:
        if not torch.cuda.is_available():
            print("\n  WARNING: No GPU available. Use --skip-attention for CPU-only mode.")
            print("  Skipping token attention profile.")
        else:
            run_token_attention_profile(
                results, method_robust,
                n_prompts=args.n_prompts,
                model_name=args.model,
            )
    else:
        print("\n  Skipping token attention profile (--skip-attention)")

    print(f"\n\n{SEP}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Plots saved to: {ANALYSIS_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
