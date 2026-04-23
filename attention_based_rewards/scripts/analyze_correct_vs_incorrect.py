#!/usr/bin/env python3
"""Compare FAI weight distributions on correct vs incorrect solutions.

Motivation: The asymmetric condition inverts weights on incorrect solutions.
This analysis checks whether FAI naturally assigns different weights to
reasoning tokens in correct vs incorrect solutions.

Multi-GPU Phase 1 (generation), single-GPU Phase 2 (weight computation).

Outputs:
  1. correct_vs_incorrect_by_category.png — bar chart comparing mean weights
  2. correct_vs_incorrect_distributions.png — violin plots
  3. correct_vs_incorrect_single_example.png — side-by-side token-level plots
  4. Printed summary statistics
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
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path("/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
OUTPUT_DIR = BASE_DIR / "analysis"
N_SOLUTIONS = 30  # per group (correct / incorrect)
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


# ── Phase 1: Multi-GPU solution generation ──

def generate_worker(rank, n_gpus, problems, correct_dict, incorrect_dict):
    """Generate solutions, collecting both correct and incorrect."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    shard_size = len(problems) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(problems)
    my_problems = problems[start:end]
    print(f"[GPU {rank}] Processing problems {start}-{end-1} ({len(my_problems)} total)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    target_per_gpu = (N_SOLUTIONS // n_gpus) + 1
    correct = []
    incorrect = []

    for question, gold in my_problems:
        if len(correct) >= target_per_gpu and len(incorrect) >= target_per_gpu:
            break

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
        except Exception:
            continue

        pred = extract_model_answer(response_text)
        entry = {
            "question": question,
            "gold": gold,
            "response": response_text,
            "full_ids": full_ids.cpu(),
            "prompt_len": prompt_len,
        }

        if answers_match(pred, gold):
            if len(correct) < target_per_gpu:
                correct.append(entry)
        else:
            if len(incorrect) < target_per_gpu:
                incorrect.append(entry)

        total = len(correct) + len(incorrect)
        if total % 10 == 0:
            print(f"[GPU {rank}] {len(correct)} correct, {len(incorrect)} incorrect", flush=True)

    print(f"[GPU {rank}] Done: {len(correct)} correct, {len(incorrect)} incorrect", flush=True)
    correct_dict[rank] = correct
    incorrect_dict[rank] = incorrect
    del model


# ── Phase 2: Weight computation ──

def _normalize_per_seq(weights, mask, eps=1e-8):
    masked = weights * mask
    seq_means = masked.sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
    return (masked / (seq_means + eps)) * mask


def compute_fai_weights(attn_model, tokenizer, full_ids, prompt_len, reasoning_heads, head_scores, device):
    """Compute FAI-ReasHeads weights for a single sequence."""
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

    attentions = outputs.attentions

    # FAI: mask[q, k] = 1 if q > k (future queries attending to key position)
    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)

    fai = torch.zeros(1, seq_len, device=device, dtype=torch.float32)
    for layer, head in reasoning_heads:
        attn_pattern = attentions[layer][:, head, :, :].float()
        future_attn = attn_pattern * future_mask.unsqueeze(0)
        received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
        fai += received * head_scores[layer, head].item()

    fai_w = _normalize_per_seq(fai, response_mask)

    del attentions, outputs

    resp_fai = fai_w[0, prompt_len:].cpu().numpy()
    response_ids = full_ids[prompt_len:]
    token_strs = [tokenizer.decode([tid]) for tid in response_ids.cpu().tolist()]

    return resp_fai, token_strs


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

    # Load GSM8K
    print("Loading GSM8K...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    problems = []
    for idx in indices:
        question = ds[idx]["question"]
        m = re.search(r"####\s*(.+)", ds[idx]["answer"])
        gold = m.group(1).strip().replace(",", "") if m else ds[idx]["answer"].strip()
        problems.append((question, gold))

    # ── Phase 1: Parallel generation ──
    print(f"\nPhase 1: Generating solutions across {n_gpus} GPUs...", flush=True)
    t0 = time.time()

    correct_dict = mp.Manager().dict()
    incorrect_dict = mp.Manager().dict()
    mp.spawn(generate_worker, args=(n_gpus, problems, correct_dict, incorrect_dict),
             nprocs=n_gpus, join=True)

    all_correct = []
    all_incorrect = []
    for rank in range(n_gpus):
        all_correct.extend(correct_dict[rank])
        all_incorrect.extend(incorrect_dict[rank])

    print(f"Phase 1 done: {len(all_correct)} correct, {len(all_incorrect)} incorrect "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)

    if len(all_correct) < 10 or len(all_incorrect) < 10:
        print("Too few solutions. Exiting.", flush=True)
        sys.exit(1)

    all_correct = all_correct[:N_SOLUTIONS]
    all_incorrect = all_incorrect[:N_SOLUTIONS]
    print(f"Using {len(all_correct)} correct, {len(all_incorrect)} incorrect", flush=True)

    # ── Phase 2: Compute FAI weights on GPU 0 ──
    print(f"\nPhase 2: Computing FAI-ReasHeads weights on GPU 0...", flush=True)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    attn_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device,
    )
    attn_model.eval()

    categories = ['number', 'operator', 'reasoning', 'step_boundary', 'filler']

    def process_group(solutions, label):
        all_weights = []
        all_tokens = []
        cat_weights = {c: [] for c in categories}

        for i, sol in enumerate(solutions):
            if (i + 1) % 10 == 0:
                print(f"  [{label}] Computing weights {i + 1}/{len(solutions)}...", flush=True)
            try:
                fai_w, token_strs = compute_fai_weights(
                    attn_model, tokenizer, sol["full_ids"], sol["prompt_len"],
                    reasoning_heads, head_scores, device,
                )
            except Exception as e:
                print(f"  [{label}] Error on {i}: {e}", flush=True)
                continue

            all_weights.extend(fai_w.tolist())
            all_tokens.append(token_strs)

            for tok_idx, tok in enumerate(token_strs):
                if tok_idx < len(fai_w):
                    cat = classify_token(tok)
                    cat_weights[cat].append(fai_w[tok_idx])

        return all_weights, all_tokens, cat_weights

    print("Processing correct solutions...", flush=True)
    corr_weights, corr_tokens, corr_cats = process_group(all_correct, "correct")
    print("Processing incorrect solutions...", flush=True)
    incorr_weights, incorr_tokens, incorr_cats = process_group(all_incorrect, "incorrect")

    n_nan_corr = sum(1 for x in corr_weights if np.isnan(x))
    n_nan_incorr = sum(1 for x in incorr_weights if np.isnan(x))
    print(f"Total tokens: correct={len(corr_weights)}, incorrect={len(incorr_weights)}", flush=True)
    print(f"NaN counts: correct={n_nan_corr}, incorrect={n_nan_incorr}", flush=True)

    # ── Plot 1: Bar chart by category ──
    corr_means = {c: np.mean(corr_cats[c]) if corr_cats[c] else 0 for c in categories}
    incorr_means = {c: np.mean(incorr_cats[c]) if incorr_cats[c] else 0 for c in categories}

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, [corr_means[c] for c in categories], width,
                   label='Correct', color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, [incorr_means[c] for c in categories], width,
                   label='Incorrect', color='#d62728', alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Mean FAI-ReasHeads Weight', fontsize=12)
    ax.set_title('FAI Weight by Token Category: Correct vs Incorrect Solutions', fontsize=14)
    ax.legend(fontsize=11)

    for c_idx, c in enumerate(categories):
        n_c = len(corr_cats[c])
        n_i = len(incorr_cats[c])
        ax.text(c_idx, -0.08, f'n={n_c}/{n_i}', ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correct_vs_incorrect_by_category.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'correct_vs_incorrect_by_category.png'}")

    # ── Plot 2: Violin distributions ──
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [corr_weights, incorr_weights]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    colors = ['#2ca02c', '#d62728']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Correct', 'Incorrect'], fontsize=12)
    ax.set_ylabel('FAI-ReasHeads Weight', fontsize=12)
    ax.set_title('FAI Weight Distribution: Correct vs Incorrect', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    corr_arr = np.array(corr_weights)
    incorr_arr = np.array(incorr_weights)
    stats_text = (f"Correct: mean={corr_arr.mean():.3f}, std={corr_arr.std():.3f}, "
                  f"CV={corr_arr.std()/corr_arr.mean():.3f}\n"
                  f"Incorrect: mean={incorr_arr.mean():.3f}, std={incorr_arr.std():.3f}, "
                  f"CV={incorr_arr.std()/incorr_arr.mean():.3f}")
    fig.text(0.12, -0.02, stats_text, fontsize=9, family='monospace', verticalalignment='top')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correct_vs_incorrect_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'correct_vs_incorrect_distributions.png'}")

    # ── Plot 3: Single example comparison ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Pick first correct and first incorrect with similar length
    corr_example_w = np.array(corr_weights[:len(corr_tokens[0])])
    incorr_example_w = np.array(incorr_weights[:len(incorr_tokens[0])])

    ax1.plot(corr_example_w, color='#2ca02c', alpha=0.8, linewidth=1)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('FAI Weight', fontsize=11)
    ax1.set_title('Correct Solution — Token-level FAI Weights', fontsize=12)

    ax2.plot(incorr_example_w, color='#d62728', alpha=0.8, linewidth=1)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('FAI Weight', fontsize=11)
    ax2.set_xlabel('Token Position', fontsize=11)
    ax2.set_title('Incorrect Solution — Token-level FAI Weights', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correct_vs_incorrect_single_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'correct_vs_incorrect_single_example.png'}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("CORRECT vs INCORRECT — FAI-ReasHeads SUMMARY")
    print("=" * 80)

    print(f"\n{'Category':<16} {'Correct':>10} {'Incorrect':>10} {'Ratio C/I':>10} {'n_corr':>8} {'n_incorr':>8}")
    print("-" * 70)
    for c in categories:
        cm = corr_means[c]
        im = incorr_means[c]
        ratio = cm / im if im > 0 else float('inf')
        print(f"{c:<16} {cm:>10.3f} {im:>10.3f} {ratio:>10.3f} {len(corr_cats[c]):>8} {len(incorr_cats[c]):>8}")

    print(f"\nOverall distribution:")
    print(f"  Correct:   mean={corr_arr.mean():.3f}, std={corr_arr.std():.3f}, CV={corr_arr.std()/corr_arr.mean():.3f}")
    print(f"  Incorrect: mean={incorr_arr.mean():.3f}, std={incorr_arr.std():.3f}, CV={incorr_arr.std()/incorr_arr.mean():.3f}")

    # Save summary
    summary = {
        "n_correct": len(all_correct),
        "n_incorrect": len(all_incorrect),
        "n_tokens_correct": len(corr_weights),
        "n_tokens_incorrect": len(incorr_weights),
        "category_means_correct": {c: float(corr_means[c]) for c in categories},
        "category_means_incorrect": {c: float(incorr_means[c]) for c in categories},
        "overall_correct": {"mean": float(corr_arr.mean()), "std": float(corr_arr.std())},
        "overall_incorrect": {"mean": float(incorr_arr.mean()), "std": float(incorr_arr.std())},
    }
    with open(OUTPUT_DIR / "correct_vs_incorrect_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUTPUT_DIR / 'correct_vs_incorrect_summary.json'}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
