#!/usr/bin/env python3
"""Diagnostic script for anchor-based circuit credit weights.

Runs compute_anchor_credit_weights on 20 examples from DAPO-Math-17k
for both instruct and base models, printing detailed analysis of
anchor/dependent tokens and weight distributions.

Usage (must run on GPU node via srun):
  srun --account=nn12068k --partition=accel --gpus=1 --cpus-per-task=8 \
       --mem=32G --time=00:30:00 \
       python attention_based_rewards/scripts/diagnose_anchor_credit.py
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention_based_rewards.scripts.token_weighting import compute_anchor_credit_weights


def load_reasoning_heads(path: str, top_k: int = 10):
    data = torch.load(path, map_location="cpu", weights_only=False)
    head_scores = data["head_scores"]
    selected = data["selected_heads"][:top_k]
    reasoning_heads = [(layer, head) for layer, head, _score in selected]
    return reasoning_heads, head_scores


def run_diagnostic(model_name: str, heads_path: str, label: str, device: str = "cuda"):
    print(f"\n{'='*70}")
    print(f"  Diagnostic: {label}")
    print(f"  Model: {model_name}")
    print(f"  Heads: {heads_path}")
    print(f"{'='*70}\n")

    # Load reasoning heads
    reasoning_heads, head_scores = load_reasoning_heads(heads_path)
    print(f"Using {len(reasoning_heads)} reasoning heads:")
    for i, (l, h) in enumerate(reasoning_heads):
        print(f"  #{i+1}: L{l}H{h} (score={head_scores[l,h]:.4f})")

    # Load model
    print(f"\nLoading model (eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data_path = "attention_based_rewards/data/dapo_math_17k.parquet"
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} examples from {data_path}")

    # Take 20 examples
    examples = df.head(20)

    # Generate responses (short, just for diagnostics)
    print("\nGenerating responses for 20 examples...")
    all_weights = []
    all_anchor_masks = []
    all_dep_masks = []

    for idx, row in examples.iterrows():
        # Build prompt
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, list):
            # Chat format
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                # Base model: concatenate messages
                prompt_text = ""
                for msg in prompt_msgs:
                    if msg["role"] == "user":
                        prompt_text += f"Question: {msg['content']}\nLet's solve this step by step.\n"
        else:
            prompt_text = prompt_msgs

        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=1024)

        # Generate a response
        with torch.no_grad():
            output = model.generate(
                prompt_ids.to(device),
                max_new_tokens=512,
                temperature=0.7,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Build full sequence with response mask
        full_ids = output[0:1]  # (1, full_len)
        full_len = full_ids.shape[1]
        prompt_len = prompt_ids.shape[1]
        resp_len = full_len - prompt_len

        attention_mask = torch.ones(1, full_len, device=device, dtype=torch.long)
        response_mask = torch.zeros(1, full_len, device=device, dtype=torch.float32)
        response_mask[0, prompt_len:] = 1.0

        # Compute anchor credit weights
        with torch.no_grad():
            weights, anchor_mask, dep_mask = compute_anchor_credit_weights(
                input_ids=full_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                attn_model=model,
                reasoning_heads=reasoning_heads,
                head_scores=head_scores,
            )

        # Decode tokens for analysis
        resp_ids = full_ids[0, prompt_len:].cpu()
        resp_tokens = [tokenizer.decode([t]) for t in resp_ids]

        anchor_positions = anchor_mask[0, prompt_len:].cpu().nonzero(as_tuple=True)[0]
        dep_positions = dep_mask[0, prompt_len:].cpu().nonzero(as_tuple=True)[0]

        resp_weights = weights[0, prompt_len:].cpu()
        valid_weights = resp_weights[response_mask[0, prompt_len:].bool().cpu()]

        print(f"\n--- Example {idx} ({resp_len} response tokens) ---")
        question = prompt_msgs[-1]["content"] if isinstance(prompt_msgs, list) else prompt_text
        print(f"  Question: {question[:100]}...")

        n_anchors = len(anchor_positions)
        n_deps = len(dep_positions)
        print(f"  Anchors: {n_anchors} ({100*n_anchors/max(resp_len,1):.1f}%)")
        print(f"  Dependents: {n_deps} ({100*n_deps/max(resp_len,1):.1f}%)")
        print(f"  Weights: mean={valid_weights.mean():.3f}, std={valid_weights.std():.3f}, "
              f"min={valid_weights.min():.3f}, max={valid_weights.max():.3f}")

        # Show example anchor tokens
        if n_anchors > 0:
            sample_anchors = anchor_positions[:5].tolist()
            anchor_texts = [repr(resp_tokens[p]) for p in sample_anchors]
            print(f"  Sample anchors: {', '.join(anchor_texts)}")

        # Show example dependent tokens
        if n_deps > 0:
            sample_deps = dep_positions[:5].tolist()
            dep_texts = [repr(resp_tokens[p]) for p in sample_deps]
            print(f"  Sample dependents: {', '.join(dep_texts)}")

        all_weights.append(valid_weights)
        all_anchor_masks.append(n_anchors / max(resp_len, 1))
        all_dep_masks.append(n_deps / max(resp_len, 1))

    # Summary statistics
    all_w = torch.cat(all_weights)
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*70}")
    print(f"  Overall weight stats: mean={all_w.mean():.3f}, std={all_w.std():.3f}, "
          f"min={all_w.min():.3f}, max={all_w.max():.3f}")
    print(f"  Mean anchor fraction: {sum(all_anchor_masks)/len(all_anchor_masks):.3f}")
    print(f"  Mean dependent fraction: {sum(all_dep_masks)/len(all_dep_masks):.3f}")
    frac_above_1 = (all_w > 1.0).float().mean().item()
    frac_eq_1 = (all_w == 1.0).float().mean().item()
    print(f"  Tokens at weight 1.0: {100*frac_eq_1:.1f}%")
    print(f"  Tokens above 1.0: {100*frac_above_1:.1f}%")
    print(f"  No token below 1.0: {(all_w >= 1.0).all().item()}")


if __name__ == "__main__":
    import sys
    import os

    os.environ.setdefault("HF_HOME", "/cluster/projects/nn12068k/haaklau/.cache/huggingface")

    INSTRUCT_MODEL = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
    BASE_MODEL = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    INSTRUCT_HEADS = "attention_based_rewards/results/reasoning_heads.pt"
    BASE_HEADS = "attention_based_rewards/data/base_model_reasoning_heads.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Check which heads files exist
    if Path(INSTRUCT_HEADS).exists():
        run_diagnostic(INSTRUCT_MODEL, INSTRUCT_HEADS, "Instruct Model", device)
    else:
        print(f"Skipping instruct: {INSTRUCT_HEADS} not found")

    if Path(BASE_HEADS).exists():
        run_diagnostic(BASE_MODEL, BASE_HEADS, "Base Model", device)
    else:
        print(f"Skipping base: {BASE_HEADS} not found")
