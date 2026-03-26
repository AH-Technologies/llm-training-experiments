#!/usr/bin/env python3
"""Ablation-based head importance ranking.

For each attention head, zero-ablates its output and measures the
accuracy drop on a set of math problems. Heads whose removal causes
the largest accuracy drop are most important for reasoning.

Approach:
1. Generate responses for N problems with no ablation (baseline accuracy)
2. For each head (l, h):
   - Hook the attention layer to zero out head h's contribution
   - Generate responses and measure accuracy
   - importance[l, h] = baseline_acc - ablated_acc

This is O(N_HEADS * N_EXAMPLES) forward passes, so we use greedy
decoding and fewer examples to keep compute manageable.

Usage (1 GPU, ~2-3 hours):
  srun --account=nn12068k --partition=accel --gpus=1 --cpus-per-task=8 \
       --mem=48G --time=04:00:00 \
       python attention_based_rewards/scripts/analyze_head_ablation.py
"""

import os
import random
import re
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/cluster/projects/nn12068k/haaklau/.cache/huggingface")

BASE_DIR = Path("attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATA_PATH = BASE_DIR / "data" / "dapo_math_17k.parquet"
N_EXAMPLES = 80   # problems to evaluate per head
MAX_NEW_TOKENS = 512
N_LAYERS = 28
N_HEADS = 12


def check_answer(response: str, ground_truth: str) -> bool:
    """Simple answer extraction and comparison."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        extracted = numbers[-1] if numbers else ""
    gt = ground_truth.strip().replace(",", "").replace("$", "")
    ext = extracted.strip().replace(",", "").replace("$", "")
    try:
        return abs(float(ext) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return ext == gt


class HeadAblationHook:
    """Context manager that zeros out a specific attention head's output."""

    def __init__(self, model, layer_idx, head_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.handle = None

    def __enter__(self):
        attn_module = self.model.model.layers[self.layer_idx].self_attn

        def hook_fn(module, args, output):
            # output is (attn_output, attn_weights, past_kv) or similar
            # attn_output shape: (bs, seq_len, hidden_dim)
            # Each head contributes hidden_dim // n_heads dimensions
            attn_output = output[0]
            head_dim = attn_output.shape[-1] // N_HEADS
            start = self.head_idx * head_dim
            end = start + head_dim
            modified = attn_output.clone()
            modified[:, :, start:end] = 0.0
            return (modified,) + output[1:]

        self.handle = attn_module.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


def evaluate_accuracy(model, tokenizer, prompts, ground_truths, device, ablation=None):
    """Evaluate accuracy on a set of prompts, optionally with head ablation."""
    correct = 0
    total = 0

    ctx = ablation if ablation else torch.no_grad()

    for prompt_text, gt in zip(prompts, ground_truths):
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)

        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy for deterministic ablation comparison
                pad_token_id=tokenizer.pad_token_id,
            )

        resp = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if check_answer(resp, gt):
            correct += 1
        total += 1

    return correct / max(total, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    df = pd.read_parquet(DATA_PATH)
    random.seed(42)
    indices = random.sample(range(len(df)), min(N_EXAMPLES, len(df)))

    prompts = []
    ground_truths = []
    for data_idx in indices:
        row = df.iloc[data_idx]
        prompt_msgs = row["prompt"]
        gt = row["reward_model"]["ground_truth"]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            for msg in prompt_msgs:
                if msg["role"] == "user":
                    prompt_text = f"Question: {msg['content']}\nLet's solve this step by step.\n"

        prompts.append(prompt_text)
        ground_truths.append(gt)

    print(f"Prepared {len(prompts)} problems")

    # Baseline accuracy (no ablation)
    print("\nComputing baseline accuracy (no ablation)...")
    baseline_acc = evaluate_accuracy(model, tokenizer, prompts, ground_truths, device)
    print(f"Baseline accuracy: {baseline_acc:.3f} ({int(baseline_acc * len(prompts))}/{len(prompts)})")

    # Ablate each head
    importance = torch.zeros(N_LAYERS, N_HEADS)
    total_heads = N_LAYERS * N_HEADS
    print(f"\nAblating {total_heads} heads...")

    for layer_idx in range(N_LAYERS):
        for head_idx in range(N_HEADS):
            head_num = layer_idx * N_HEADS + head_idx + 1

            with HeadAblationHook(model, layer_idx, head_idx):
                ablated_acc = evaluate_accuracy(
                    model, tokenizer, prompts, ground_truths, device
                )

            drop = baseline_acc - ablated_acc
            importance[layer_idx, head_idx] = drop

            status = ""
            if drop > 0.05:
                status = " *** IMPORTANT"
            elif drop < -0.05:
                status = " (improves when removed)"

            print(f"  [{head_num:3d}/{total_heads}] L{layer_idx}H{head_idx}: "
                  f"acc={ablated_acc:.3f} (drop={drop:+.3f}){status}")

    # Rank by importance (largest accuracy drop = most important)
    flat = importance.flatten()
    sorted_idx = flat.argsort(descending=True)

    print(f"\n{'='*60}")
    print("ABLATION-BASED HEAD RANKING (accuracy drop when removed)")
    print(f"{'='*60}")
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    print(f"\n{'Rank':<6} {'Head':<10} {'Acc Drop':>10} {'Ablated Acc':>12}")
    print("-" * 45)

    selected_heads = []
    for rank, idx in enumerate(sorted_idx[:50]):
        l = idx.item() // N_HEADS
        h = idx.item() % N_HEADS
        drop = importance[l, h].item()
        selected_heads.append((l, h, drop))
        if rank < 30:
            print(f"  #{rank+1:<4} L{l}H{h:<6} {drop:>+10.4f} {baseline_acc - drop:>12.3f}")

    # Heads that IMPROVE when removed
    worst_idx = flat.argsort()[:10]
    print(f"\nHeads that IMPROVE accuracy when removed:")
    for idx in worst_idx:
        l = idx.item() // N_HEADS
        h = idx.item() % N_HEADS
        drop = importance[l, h].item()
        if drop < 0:
            print(f"  L{l}H{h}: acc improves by {-drop:.4f}")

    # Compare with EAP-IG
    eapig_path = BASE_DIR / "data" / "base_model_reasoning_heads.pt"
    if eapig_path.exists():
        eapig = torch.load(eapig_path, map_location="cpu", weights_only=False)
        eapig_top20 = {(l, h) for l, h, _ in eapig["selected_heads"][:20]}
        abl_top20 = {(l, h) for l, h, _ in selected_heads[:20]}
        overlap = eapig_top20 & abl_top20
        print(f"\nOverlap with EAP-IG top-20: {len(overlap)}/20")
        if overlap:
            print(f"  Shared: {sorted(overlap)}")

    # Compare with activation-based
    act_path = BASE_DIR / "data" / "base_model_activation_heads.pt"
    if act_path.exists():
        act = torch.load(act_path, map_location="cpu", weights_only=False)
        act_top20 = {(l, h) for l, h, _ in act["selected_heads"][:20]}
        abl_top20 = {(l, h) for l, h, _ in selected_heads[:20]}
        overlap = act_top20 & abl_top20
        print(f"Overlap with activation-based top-20: {len(overlap)}/20")
        if overlap:
            print(f"  Shared: {sorted(overlap)}")

    # Save
    save_data = {
        "head_scores": importance,
        "selected_heads": selected_heads,
        "baseline_accuracy": baseline_acc,
        "method": "zero_ablation",
        "n_examples": N_EXAMPLES,
    }
    save_path = BASE_DIR / "data" / "base_model_ablation_heads.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")

    # Heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(importance.numpy(), cmap="RdBu_r", aspect="auto",
                   vmin=-importance.abs().max().item(), vmax=importance.abs().max().item())
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)])
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)])
    plt.colorbar(im, ax=ax, label="Accuracy drop when ablated")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Ablation-based Head Importance (red = important, blue = harmful)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "ablation_head_importance.png", dpi=150)
    print(f"Saved heatmap")


if __name__ == "__main__":
    main()
