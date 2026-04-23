#!/usr/bin/env python3
"""Evaluate a checkpoint on MATH500 samples using veRL's math_dapo reward."""

import json
import os
import sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from verl.utils.reward_score.math_dapo import (
    last_boxed_only_string,
    remove_boxed,
    normalize_final_answer,
)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/attention-illuminates-qwen3/rhythm_runA_qwen3_base_v6/global_step_200/actor"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUT = sys.argv[3] if len(sys.argv) > 3 else "attention_sparks_thinking/logs/eval_extraction_v2.jsonl"
BASE_MODEL = "Qwen/Qwen3-4B-Base"


def extract_and_score_dapo(response, ground_truth):
    """Use veRL's math_dapo extraction and normalization."""
    # Extract last \boxed{} from last 300 chars (as veRL does)
    truncated = response[-300:]
    boxed = last_boxed_only_string(truncated)
    if boxed is None:
        # Fallback: try full response
        boxed = last_boxed_only_string(response)

    if boxed is None:
        return None, "", "", False, "no_boxed"

    extracted = remove_boxed(boxed)
    norm_pred = normalize_final_answer(extracted)
    norm_gt = normalize_final_answer(ground_truth)

    match = (norm_pred == norm_gt)
    match_type = "exact_dapo" if match else "no_match_dapo"

    return extracted, norm_pred, norm_gt, match, match_type


def load_fsdp_checkpoint(actor_dir, base_model):
    """Load model from veRL FSDP-sharded checkpoint."""
    print(f"Loading base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, trust_remote_code=True
    )

    shard_files = sorted(
        f for f in os.listdir(actor_dir) if f.startswith("model_world_size_") and f.endswith(".pt")
    )
    print(f"Found {len(shard_files)} FSDP shards, merging...")

    shards = []
    for sf in shard_files:
        path = os.path.join(actor_dir, sf)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shards.append(shard)

    merged = {}
    first_shard = shards[0]
    for key in first_shard:
        tensors = []
        for shard in shards:
            t = shard[key]
            if hasattr(t, '_local_tensor'):
                t = t._local_tensor
            elif hasattr(t, 'full_tensor'):
                t = t.full_tensor()
            tensors.append(t)

        param = dict(model.named_parameters()).get(key)
        if param is None:
            continue

        expected_shape = param.shape
        if all(t.shape == expected_shape for t in tensors):
            merged[key] = tensors[0]
        else:
            for dim in range(len(expected_shape)):
                cat = torch.cat(tensors, dim=dim)
                if cat.shape == expected_shape:
                    merged[key] = cat
                    break
            else:
                merged[key] = tensors[0]

    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys: {missing[:3]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    print("Weights loaded successfully")
    return model


print(f"Checkpoint: {CKPT}")
print(f"Tokenizer: {BASE_MODEL}")
print(f"Samples: {N}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

actor_dir = CKPT
if os.path.isdir(os.path.join(CKPT, "actor")):
    actor_dir = os.path.join(CKPT, "actor")

shard_files = [f for f in os.listdir(actor_dir) if f.startswith("model_world_size_")]
if shard_files:
    model = load_fsdp_checkpoint(actor_dir, BASE_MODEL)
else:
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, trust_remote_code=True
    )

model = model.cuda().eval()

df = pd.read_parquet("data/math500.parquet")
sampled = df.sample(n=min(N, len(df)), random_state=42)

results = []
for idx, row in sampled.iterrows():
    prompt_raw = row["prompt"]
    if isinstance(prompt_raw, str):
        prompt_raw = json.loads(prompt_raw)

    ground_truth = row.get("reward_model", {})
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    gt_answer = ground_truth.get("ground_truth", str(ground_truth))

    prompt_text = "\n".join(m["content"] for m in prompt_raw if m["role"] in ("user", "system"))

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=1.0,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    extracted, norm_pred, norm_gt, match, match_type = extract_and_score_dapo(response, gt_answer)

    entry = {
        "idx": int(idx),
        "ground_truth": str(gt_answer),
        "norm_gt": norm_gt,
        "extracted": extracted,
        "norm_pred": norm_pred,
        "correct": match,
        "match_type": match_type,
        "response": response,
    }
    results.append(entry)

    status = "CORRECT" if match else "WRONG"
    print(f"[{status}] idx={idx}  GT: {gt_answer}  ->  norm_gt: {norm_gt}  |  extracted: {extracted}  ->  norm_pred: {norm_pred}")

with open(OUT, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

correct = sum(1 for r in results if r["correct"])
print(f"\n{'='*60}")
print(f"Results: {correct}/{len(results)} correct ({100*correct/len(results):.1f}%)")
print(f"Saved to {OUT}")
