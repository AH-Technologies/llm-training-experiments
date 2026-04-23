#!/usr/bin/env python3
"""Activation-based head importance ranking.

For each attention head, measures how much its behavior differs between
correct and incorrect responses. Heads that behave very differently on
correct vs incorrect responses are likely important for reasoning.

Metrics per head:
  1. FAI divergence: |mean_FAI_correct - mean_FAI_incorrect|
  2. Attention entropy divergence: |mean_entropy_correct - mean_entropy_incorrect|
  3. Activation magnitude divergence: |mean_attn_norm_correct - mean_attn_norm_incorrect|

Combines into a single ranking. Saves results as a .pt file compatible
with the training pipeline.

Usage (1 GPU, ~15 min):
  srun --account=nn12068k --partition=accel --gpus=1 --cpus-per-task=8 \
       --mem=48G --time=01:00:00 \
       python scripts/attention_based_rewards/analyze_head_activation.py
"""

import os
import random
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/cluster/projects/nn12068k/haaklau/.cache/huggingface")

BASE_DIR = Path("attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATA_PATH = BASE_DIR / "data" / "dapo_math_17k.parquet"
N_EXAMPLES = 100  # number of problems to evaluate
N_SAMPLES = 4     # responses per problem
MAX_NEW_TOKENS = 512
N_LAYERS = 28
N_HEADS = 12


def check_answer(response: str, ground_truth: str) -> bool:
    """Simple answer extraction and comparison."""
    import re
    # Try to extract from <answer> tags
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        # Fall back to last number
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        extracted = numbers[-1] if numbers else ""

    # Normalize
    gt = ground_truth.strip().replace(",", "").replace("$", "")
    ext = extracted.strip().replace(",", "").replace("$", "")
    try:
        return abs(float(ext) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return ext == gt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("Loading model (eager attention)...")
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

    # Per-head accumulators
    # FAI: mean future attention received per head
    fai_correct = torch.zeros(N_LAYERS, N_HEADS)
    fai_incorrect = torch.zeros(N_LAYERS, N_HEADS)
    # Attention entropy per head
    ent_correct = torch.zeros(N_LAYERS, N_HEADS)
    ent_incorrect = torch.zeros(N_LAYERS, N_HEADS)
    # Activation magnitude (mean attention weight to response tokens)
    mag_correct = torch.zeros(N_LAYERS, N_HEADS)
    mag_incorrect = torch.zeros(N_LAYERS, N_HEADS)

    n_correct = 0
    n_incorrect = 0

    print(f"\nGenerating and analyzing {N_EXAMPLES} problems x {N_SAMPLES} samples...")

    for idx_i, data_idx in enumerate(indices):
        row = df.iloc[data_idx]
        prompt_msgs = row["prompt"]
        gt = row["reward_model"]["ground_truth"]

        # Build prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            for msg in prompt_msgs:
                if msg["role"] == "user":
                    prompt_text = f"Question: {msg['content']}\nLet's solve this step by step.\n"

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
        prompt_len = prompt_ids["input_ids"].shape[1]

        # Generate N_SAMPLES responses
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids["input_ids"].to(device),
                attention_mask=prompt_ids["attention_mask"].to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=1.0,
                do_sample=True,
                num_return_sequences=N_SAMPLES,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Analyze each response
        for s in range(N_SAMPLES):
            full_ids = outputs[s:s+1]
            resp_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
            correct = check_answer(resp_text, gt)

            seq_len = full_ids.shape[1]
            resp_len = seq_len - prompt_len
            if resp_len < 5:
                continue

            # Forward pass with attention
            with torch.no_grad():
                out = model(
                    input_ids=full_ids.to(device),
                    output_attentions=True,
                    use_cache=False,
                )

            attentions = out.attentions  # tuple of (1, n_heads, seq_len, seq_len)

            # future_mask for FAI
            future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
            future_count = future_mask.sum(dim=0).clamp(min=1)

            for layer_idx in range(N_LAYERS):
                attn = attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)

                for head_idx in range(N_HEADS):
                    a = attn[head_idx]  # (seq_len, seq_len)

                    # FAI: mean future attention received, averaged over response tokens
                    future_attn = a * future_mask
                    received = future_attn.sum(dim=0) / future_count  # (seq_len,)
                    fai_val = received[prompt_len:].mean().item()

                    # Attention entropy over response tokens
                    resp_attn = a[prompt_len:, :]  # (resp_len, seq_len)
                    ent = -(resp_attn * (resp_attn + 1e-10).log()).sum(dim=-1).mean().item()

                    # Magnitude: mean attention weight from response to response
                    resp_to_resp = a[prompt_len:, prompt_len:]
                    mag_val = resp_to_resp.mean().item()

                    if correct:
                        fai_correct[layer_idx, head_idx] += fai_val
                        ent_correct[layer_idx, head_idx] += ent
                        mag_correct[layer_idx, head_idx] += mag_val
                    else:
                        fai_incorrect[layer_idx, head_idx] += fai_val
                        ent_incorrect[layer_idx, head_idx] += ent
                        mag_incorrect[layer_idx, head_idx] += mag_val

            if correct:
                n_correct += 1
            else:
                n_incorrect += 1

            del out, attentions
            torch.cuda.empty_cache()

        if (idx_i + 1) % 10 == 0:
            print(f"  [{idx_i+1}/{N_EXAMPLES}] correct={n_correct}, incorrect={n_incorrect}")

    print(f"\nTotal: {n_correct} correct, {n_incorrect} incorrect")

    # Normalize
    if n_correct > 0:
        fai_correct /= n_correct
        ent_correct /= n_correct
        mag_correct /= n_correct
    if n_incorrect > 0:
        fai_incorrect /= n_incorrect
        ent_incorrect /= n_incorrect
        mag_incorrect /= n_incorrect

    # Compute divergences
    fai_div = (fai_correct - fai_incorrect).abs()
    ent_div = (ent_correct - ent_incorrect).abs()
    mag_div = (mag_correct - mag_incorrect).abs()

    # Normalize each metric to [0, 1] for combining
    def norm01(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    fai_norm = norm01(fai_div)
    ent_norm = norm01(ent_div)
    mag_norm = norm01(mag_div)

    # Combined score (equal weight)
    combined = (fai_norm + ent_norm + mag_norm) / 3.0

    # Rank
    flat = combined.flatten()
    sorted_idx = flat.argsort(descending=True)

    print(f"\n{'='*60}")
    print("ACTIVATION-BASED HEAD RANKING (correct vs incorrect divergence)")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Head':<10} {'Combined':>10} {'FAI div':>10} {'Ent div':>10} {'Mag div':>10}")
    print("-" * 60)

    selected_heads = []
    for rank, idx in enumerate(sorted_idx[:50]):
        l = idx.item() // N_HEADS
        h = idx.item() % N_HEADS
        score = combined[l, h].item()
        selected_heads.append((l, h, score))
        if rank < 30:
            print(f"  #{rank+1:<4} L{l}H{h:<6} {score:>10.4f} "
                  f"{fai_div[l,h]:>10.4f} {ent_div[l,h]:>10.4f} {mag_div[l,h]:>10.4f}")

    # Compare with EAP-IG ranking
    eapig_path = BASE_DIR / "data" / "base_model_reasoning_heads.pt"
    if eapig_path.exists():
        eapig = torch.load(eapig_path, map_location="cpu", weights_only=False)
        eapig_top20 = {(l, h) for l, h, _ in eapig["selected_heads"][:20]}
        act_top20 = {(l, h) for l, h, _ in selected_heads[:20]}
        overlap = eapig_top20 & act_top20
        print(f"\nOverlap with EAP-IG top-20: {len(overlap)}/20")
        if overlap:
            print(f"  Shared: {sorted(overlap)}")

    # Save
    save_data = {
        "head_scores": combined,
        "selected_heads": selected_heads,
        "fai_divergence": fai_div,
        "entropy_divergence": ent_div,
        "magnitude_divergence": mag_div,
        "fai_correct": fai_correct,
        "fai_incorrect": fai_incorrect,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "method": "activation_divergence",
    }
    save_path = BASE_DIR / "data" / "base_model_activation_heads.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
