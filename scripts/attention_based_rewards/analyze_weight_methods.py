#!/usr/bin/env python3
"""Compare token-level weight distributions across credit assignment methods.

Multi-GPU: All GPUs generate solutions in parallel (Phase 1),
then GPU 0 computes weights and generates plots (Phase 2).

Generates 5 outputs:
  1. weight_comparison_single_example.png
  2. weight_correlation_heatmap.png
  3. weight_distributions.png
  4. weight_by_token_category.png
  5. Printed summary statistics table

Usage:
    srun --gpus=4 --time=00:45:00 --account=nn12068k --partition=accel \
        python scripts/attention_based_rewards/analyze_weight_methods.py
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path("/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
OUTPUT_DIR = BASE_DIR / "analysis"
N_SOLUTIONS = 100
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
)


def extract_model_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    nums = re.findall(r'[\d,]+\.?\d*', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


def extract_ground_truth(answer_str):
    m = re.search(r'\\boxed\{([^}]+)\}', answer_str)
    if m:
        return m.group(1).strip()
    return answer_str.strip()


def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return pred.strip() == gold.strip()


def classify_token(token_str):
    t = token_str.strip()
    if not t:
        return "filler"
    if re.fullmatch(r'[\d,]+\.?\d*', t):
        return "number"
    if t in ('+', '-', '*', '/', '=', '>', '<', '>=', '<=', '\\times', '\\div', '\\cdot'):
        return "operator"
    t_lower = t.lower()
    for kw in ["therefore", "since", "because", "so", "let", "thus", "hence",
               "substitut", "simplif", "assume", "consider", "note"]:
        if kw in t_lower:
            return "reasoning"
    if '\n' in token_str or 'Step' in token_str or '\\n' in token_str:
        return "step_boundary"
    return "filler"


# ── Phase 1: Multi-GPU solution generation ──────────────────────────────────

def generate_worker(rank, n_gpus, problems, result_dict):
    """Each GPU generates solutions for its shard of problems, saves correct ones."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    # Each GPU gets a contiguous shard
    shard_size = len(problems) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(problems)
    my_problems = problems[start:end]
    print(f"[GPU {rank}] Generating solutions for problems {start}-{end-1} ({len(my_problems)} total)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    target_per_gpu = (N_SOLUTIONS // n_gpus) + 1
    correct = []
    tried = 0
    for question, gold in my_problems:
        if len(correct) >= target_per_gpu:
            break
        tried += 1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prompt_len = prompt_ids.shape[1]

        try:
            with torch.no_grad():
                output_ids = model.generate(prompt_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            full_ids = output_ids[0]
            response_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
        except Exception as e:
            continue

        pred = extract_model_answer(response_text)
        if answers_match(pred, gold):
            # Save as CPU tensors for serialization
            correct.append({
                "question": question,
                "gold": gold,
                "response": response_text,
                "full_ids": full_ids.cpu(),
                "prompt_len": prompt_len,
            })
            if len(correct) % 5 == 0:
                print(f"[GPU {rank}] Found {len(correct)} correct (tried {tried}/{len(my_problems)})", flush=True)

    print(f"[GPU {rank}] Done: {len(correct)} correct out of {tried} tried", flush=True)
    result_dict[rank] = correct
    del model


# ── Phase 2: Weight computation and plotting (GPU 0 only) ───────────────────

def _normalize_per_seq(weights, mask, eps=1e-8):
    """Normalize weights to mean=1 per sequence."""
    masked = weights * mask
    seq_means = masked.sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
    return (masked / (seq_means + eps)) * mask


def compute_all_weights(attn_model, tokenizer, full_ids, prompt_len, reasoning_heads, head_scores, device):
    """Compute entropy, FAI-all, and FAI-reasoning weights for a single sequence."""

    seq_len = full_ids.shape[0]
    input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones(1, seq_len, device=device)
    response_mask = torch.zeros(1, seq_len, device=device)
    response_mask[0, prompt_len:] = 1.0

    with torch.no_grad():
        outputs = attn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    # Entropy (must use float32 to avoid NaN from float16 underflow: 0 * log(0))
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_w = _normalize_per_seq(entropy.clamp(min=0), response_mask)

    # FAI setup: mask[q, k] = 1 if q > k (future queries for each key position)
    attentions = outputs.attentions
    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)  # number of future queries per key

    # FAI reasoning heads (cast attention to float32 for precision)
    fai_rh = torch.zeros(1, seq_len, device=device, dtype=torch.float32)
    for layer, head in reasoning_heads:
        attn_pattern = attentions[layer][:, head, :, :].float()
        future_attn = attn_pattern * future_mask.unsqueeze(0)
        received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
        fai_rh += received.float() * head_scores[layer, head].item()
    fai_rh_w = _normalize_per_seq(fai_rh, response_mask)

    # FAI all heads
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    weight_per_head = 1.0 / (n_layers * n_heads)
    fai_all = torch.zeros(1, seq_len, device=device, dtype=torch.float32)
    for li in range(n_layers):
        for hi in range(n_heads):
            attn_pattern = attentions[li][:, hi, :, :].float()
            future_attn = attn_pattern * future_mask.unsqueeze(0)
            received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
            fai_all += received.float() * weight_per_head
    fai_all_w = _normalize_per_seq(fai_all, response_mask)

    # Debug: check for NaN on first call
    if not getattr(compute_all_weights, '_debug_done', False):
        compute_all_weights._debug_done = True
        print(f"  DEBUG seq_len={seq_len}, prompt_len={prompt_len}, resp_len={seq_len-prompt_len}", flush=True)
        print(f"  DEBUG entropy raw: min={entropy[0,prompt_len:].min():.4f}, max={entropy[0,prompt_len:].max():.4f}, nan={torch.isnan(entropy[0,prompt_len:]).sum()}", flush=True)
        attn0 = attentions[0][:, 0, :, :].float()
        print(f"  DEBUG attn[0][0]: min={attn0.min():.6f}, max={attn0.max():.6f}, nan={torch.isnan(attn0).sum()}", flush=True)
        print(f"  DEBUG logits: min={logits[0,prompt_len:].min():.4f}, max={logits[0,prompt_len:].max():.4f}, nan={torch.isnan(logits[0,prompt_len:]).sum()}", flush=True)

    del attentions, outputs

    resp_entropy = entropy_w[0, prompt_len:].cpu().numpy()
    resp_fai_all = fai_all_w[0, prompt_len:].cpu().numpy()
    resp_fai_rh = fai_rh_w[0, prompt_len:].cpu().numpy()

    response_ids = full_ids[prompt_len:]
    token_strs = [tokenizer.decode([tid]) for tid in response_ids.cpu().tolist()]

    return resp_entropy, resp_fai_all, resp_fai_rh, token_strs


def plot_single_example(entropy_w, fai_all_w, fai_rh_w, token_strs, output_path):
    n_tokens = len(entropy_w)
    x = np.arange(n_tokens)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.05})

    ax1.plot(x, entropy_w, label='Entropy', color='#1f77b4', alpha=0.8, linewidth=1)
    ax1.plot(x, fai_all_w, label='FAI-AllHeads', color='#ff7f0e', alpha=0.8, linewidth=1)
    ax1.plot(x, fai_rh_w, label='FAI-ReasoningHeads', color='#2ca02c', alpha=0.8, linewidth=1)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Uniform (1.0)')

    for i, t in enumerate(token_strs):
        if '\n' in t:
            ax1.axvline(x=i, color='lightgray', linestyle='--', alpha=0.3)

    ax1.set_ylabel('Normalized Weight', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Token Weight Comparison - Single Correct Solution', fontsize=14)
    ax1.set_xlim(0, n_tokens - 1)

    ax2.set_xlim(0, n_tokens - 1)
    ax2.set_ylim(0, 1)
    step = max(1, n_tokens // 60)
    for i in range(0, n_tokens, step):
        display = token_strs[i].replace('\n', '\\n')
        if len(display) > 8:
            display = display[:7] + '..'
        ax2.text(i, 0.5, display, fontsize=5, ha='center', va='center', rotation=90)
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def plot_correlation_heatmap(all_entropy, all_fai_all, all_fai_rh, per_solution_data, output_path):
    methods = ['Entropy', 'FAI-AllHeads', 'FAI-ReasHeads']
    vectors = [all_entropy, all_fai_all, all_fai_rh]

    n = len(methods)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = stats.spearmanr(vectors[i], vectors[j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    per_sol_corrs = {f'{methods[i]} vs {methods[j]}': [] for i in range(n) for j in range(i + 1, n)}
    for sol in per_solution_data:
        for i in range(n):
            for j in range(i + 1, n):
                if len(sol[i]) > 2:
                    r, _ = stats.spearmanr(sol[i], sol[j])
                    if not np.isnan(r):
                        per_sol_corrs[f'{methods[i]} vs {methods[j]}'].append(r)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_yticklabels(methods, fontsize=11)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i, j]:.3f}', ha='center', va='center', fontsize=12,
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title('Pairwise Spearman Correlation (Global)', fontsize=13)

    text_lines = ["Per-solution correlations (mean +/- std):"]
    for key, vals in per_sol_corrs.items():
        if vals:
            text_lines.append(f"  {key}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    fig.text(0.12, -0.02, '\n'.join(text_lines), fontsize=9, family='monospace',
             verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")
    return corr_matrix, per_sol_corrs


def plot_distributions(all_entropy, all_fai_all, all_fai_rh, output_path):
    data = [all_entropy, all_fai_all, all_fai_rh]
    labels = ['Entropy', 'FAI-AllHeads', 'FAI-ReasHeads']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Normalized Weight', fontsize=12)
    ax.set_title('Weight Distributions Across All Solutions', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Mean=1')
    ax.legend()

    stats_lines = [f"{'Method':<18} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'CV':>6}"]
    stats_lines.append("-" * 52)
    for label, d in zip(labels, data):
        d = np.array(d)
        mean, std = d.mean(), d.std()
        cv = std / mean if mean > 0 else 0
        stats_lines.append(f"{label:<18} {mean:>6.3f} {std:>6.3f} {d.min():>6.3f} {d.max():>6.3f} {cv:>6.3f}")

    fig.text(0.12, -0.02, '\n'.join(stats_lines), fontsize=9, family='monospace', verticalalignment='top')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")
    return stats_lines


def plot_token_categories(per_solution_data, all_token_strs, output_path):
    methods = ['Entropy', 'FAI-AllHeads', 'FAI-ReasHeads']
    categories = ['number', 'operator', 'reasoning', 'step_boundary', 'filler']

    cat_weights = {m: {c: [] for c in categories} for m in methods}
    for sol_idx, sol in enumerate(per_solution_data):
        tokens = all_token_strs[sol_idx]
        for tok_idx, tok in enumerate(tokens):
            cat = classify_token(tok)
            if tok_idx < len(sol[0]):
                cat_weights['Entropy'][cat].append(sol[0][tok_idx])
                cat_weights['FAI-AllHeads'][cat].append(sol[1][tok_idx])
                cat_weights['FAI-ReasHeads'][cat].append(sol[2][tok_idx])

    cat_means = {m: {c: np.mean(cat_weights[m][c]) if cat_weights[m][c] else 0.0 for c in categories} for m in methods}

    x = np.arange(len(categories))
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(methods):
        vals = [cat_means[m][c] for c in categories]
        ax.bar(x + i * width, vals, width, label=m, color=colors[i], alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Mean Normalized Weight', fontsize=12)
    ax.set_title('Mean Weight by Token Category', fontsize=14)
    ax.legend(fontsize=10)

    for c_idx, c in enumerate(categories):
        count = len(cat_weights['Entropy'][c])
        ax.text(c_idx + width, -0.05, f'n={count}', ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")
    return cat_means, cat_weights


def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load reasoning heads
    print("Loading reasoning heads...", flush=True)
    rh_data = torch.load(BASE_DIR / "results" / "reasoning_heads.pt", weights_only=False)
    head_scores = rh_data["head_scores"]
    selected = rh_data["selected_heads"][:10]
    reasoning_heads = [(l, h) for l, h, _ in selected]
    print(f"  {len(reasoning_heads)} reasoning heads loaded", flush=True)

    # Load GSM8K (model gets ~80% — much faster to find correct solutions)
    print("Loading GSM8K...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    problems = []
    for idx in indices:
        question = ds[idx]["question"]
        # Extract numeric answer from #### format
        m = re.search(r"####\s*(.+)", ds[idx]["answer"])
        gold = m.group(1).strip().replace(",", "") if m else ds[idx]["answer"].strip()
        problems.append((question, gold))

    # ── Phase 1: Parallel generation across all GPUs ──
    print(f"\nPhase 1: Generating solutions across {n_gpus} GPUs...", flush=True)
    t0 = time.time()

    result_dict = mp.Manager().dict()
    mp.spawn(generate_worker, args=(n_gpus, problems, result_dict), nprocs=n_gpus, join=True)

    # Collect all correct solutions
    all_correct = []
    for rank in range(n_gpus):
        all_correct.extend(result_dict[rank])
    print(f"Phase 1 done: {len(all_correct)} correct solutions in {(time.time()-t0)/60:.1f} min", flush=True)

    if len(all_correct) < 5:
        print("Too few correct solutions. Exiting.", flush=True)
        sys.exit(1)

    # Cap at N_SOLUTIONS
    if len(all_correct) > N_SOLUTIONS:
        all_correct = all_correct[:N_SOLUTIONS]
    actual_n = len(all_correct)
    print(f"Using {actual_n} solutions for analysis", flush=True)

    # ── Phase 2: Compute weights on GPU 0 ──
    print(f"\nPhase 2: Computing weights on GPU 0...", flush=True)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    attn_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device,
    )
    attn_model.eval()

    all_entropy, all_fai_all, all_fai_rh = [], [], []
    per_solution_data = []
    all_token_strs = []

    for i, sol in enumerate(all_correct):
        if (i + 1) % 10 == 0:
            print(f"  Computing weights {i + 1}/{actual_n}...", flush=True)
        try:
            entropy_w, fai_all_w, fai_rh_w, token_strs = compute_all_weights(
                attn_model, tokenizer, sol["full_ids"], sol["prompt_len"],
                reasoning_heads, head_scores, device,
            )
        except Exception as e:
            print(f"  Error on solution {i}: {e}", flush=True)
            continue

        all_entropy.extend(entropy_w.tolist())
        all_fai_all.extend(fai_all_w.tolist())
        all_fai_rh.extend(fai_rh_w.tolist())
        per_solution_data.append((entropy_w, fai_all_w, fai_rh_w))
        all_token_strs.append(token_strs)

    print(f"Total tokens: {len(all_entropy)}", flush=True)
    n_nan_ent = sum(1 for x in all_entropy if np.isnan(x))
    n_nan_fai = sum(1 for x in all_fai_all if np.isnan(x))
    n_nan_rh = sum(1 for x in all_fai_rh if np.isnan(x))
    print(f"NaN counts: entropy={n_nan_ent}, fai_all={n_nan_fai}, fai_rh={n_nan_rh}", flush=True)

    # ── Generate all outputs ──
    print("\nGenerating plots...", flush=True)

    plot_single_example(
        per_solution_data[0][0], per_solution_data[0][1], per_solution_data[0][2],
        all_token_strs[0], OUTPUT_DIR / "weight_comparison_single_example.png",
    )

    corr_matrix, per_sol_corrs = plot_correlation_heatmap(
        all_entropy, all_fai_all, all_fai_rh, per_solution_data,
        OUTPUT_DIR / "weight_correlation_heatmap.png",
    )

    stats_lines = plot_distributions(
        all_entropy, all_fai_all, all_fai_rh, OUTPUT_DIR / "weight_distributions.png",
    )

    cat_means, cat_weights = plot_token_categories(
        per_solution_data, all_token_strs, OUTPUT_DIR / "weight_by_token_category.png",
    )

    # ── Summary table ──
    methods = ['Entropy', 'FAI-AllHeads', 'FAI-ReasHeads']
    categories = ['number', 'operator', 'reasoning', 'step_boundary', 'filler']

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY STATISTICS", flush=True)
    print("=" * 80, flush=True)
    for line in stats_lines:
        print(line, flush=True)

    print(f"\nGlobal Spearman Correlations:", flush=True)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            print(f"  {methods[i]} vs {methods[j]}: r={corr_matrix[i,j]:.4f}", flush=True)

    print(f"\nPer-solution Correlations (mean +/- std):", flush=True)
    for key, vals in per_sol_corrs.items():
        if vals:
            print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}", flush=True)

    print(f"\nMean Weight by Category:", flush=True)
    header = f"{'Category':<16}" + "".join(f"{m:>16}" for m in methods)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for c in categories:
        row = f"{c:<16}"
        for m in methods:
            row += f"{cat_means[m][c]:>16.3f}"
        row += f"  (n={len(cat_weights['Entropy'][c])})"
        print(row, flush=True)

    print(f"\nStep Boundary vs Mid-step Weights:", flush=True)
    for m in methods:
        sb = cat_means[m]['step_boundary']
        mid_vals = []
        for c in ['number', 'operator', 'reasoning', 'filler']:
            mid_vals.extend(cat_weights[m][c])
        mid = np.mean(mid_vals) if mid_vals else 0
        if mid > 0:
            print(f"  {m}: boundary={sb:.3f}, mid-step={mid:.3f}, ratio={sb/mid:.3f}", flush=True)
        else:
            print(f"  {m}: boundary={sb:.3f}, mid-step={mid:.3f}", flush=True)

    summary = {
        "n_solutions": actual_n,
        "n_tokens_total": len(all_entropy),
        "global_correlations": {
            f"{methods[i]}_vs_{methods[j]}": float(corr_matrix[i, j])
            for i in range(len(methods)) for j in range(i + 1, len(methods))
        },
        "per_solution_correlations": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in per_sol_corrs.items() if v
        },
        "category_means": {m: {c: float(cat_means[m][c]) for c in categories} for m in methods},
    }
    with open(OUTPUT_DIR / "weight_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUTPUT_DIR / 'weight_analysis_summary.json'}", flush=True)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
