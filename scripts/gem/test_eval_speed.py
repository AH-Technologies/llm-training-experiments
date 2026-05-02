"""Quick test to measure eval generation speed on MATH500."""

import json
import time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
EVAL_DATASET = "data/math500.parquet"
N_SAMPLES = 10
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8  # batched generation

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()

# Load problems
df = pd.read_parquet(EVAL_DATASET)
import random
random.seed(42)
indices = random.sample(range(len(df)), min(N_SAMPLES, len(df)))

problems = []
for idx in indices:
    row = df.iloc[idx]
    prompt_raw = row["prompt"]
    if isinstance(prompt_raw, str):
        prompt_raw = json.loads(prompt_raw)
    ground_truth = row.get("reward_model", {})
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    gt_answer = ground_truth.get("ground_truth", str(ground_truth))
    prompt_text = "\n".join(m["content"] for m in prompt_raw if m["role"] in ("user", "system"))
    problems.append({"prompt_text": prompt_text, "ground_truth": gt_answer})

print(f"Loaded {len(problems)} problems")

# Test 1: Sequential generation
print(f"\n=== Sequential (1 at a time) ===")
t0 = time.time()
for i, p in enumerate(problems):
    inputs = tokenizer(p["prompt_text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Sample {i}: {len(resp)} chars, {out.shape[1] - inputs['input_ids'].shape[1]} tokens")
seq_time = time.time() - t0
print(f"Sequential: {seq_time:.1f}s total, {seq_time/N_SAMPLES:.1f}s/sample")

# Test 2: Batched generation
print(f"\n=== Batched (batch_size={BATCH_SIZE}) ===")
t0 = time.time()
all_prompts = [p["prompt_text"] for p in problems]
for batch_start in range(0, len(all_prompts), BATCH_SIZE):
    batch_prompts = all_prompts[batch_start:batch_start + BATCH_SIZE]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    for j in range(len(batch_prompts)):
        prompt_len = (inputs["attention_mask"][j] == 1).sum().item()
        resp = tokenizer.decode(out[j][prompt_len:], skip_special_tokens=True)
    print(f"  Batch {batch_start//BATCH_SIZE}: {len(batch_prompts)} samples, {out.shape[1]} max tokens")
batch_time = time.time() - t0
print(f"Batched: {batch_time:.1f}s total, {batch_time/N_SAMPLES:.1f}s/sample")

print(f"\nSpeedup: {seq_time/batch_time:.1f}x")
print(f"\nExtrapolation for 50 samples:")
print(f"  Sequential: {seq_time/N_SAMPLES * 50:.0f}s ({seq_time/N_SAMPLES * 50 / 60:.1f}m)")
print(f"  Batched:    {batch_time/N_SAMPLES * 50:.0f}s ({batch_time/N_SAMPLES * 50 / 60:.1f}m)")
