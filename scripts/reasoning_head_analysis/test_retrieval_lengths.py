#!/usr/bin/env python3
"""Test retrieval at various context lengths to find where the model breaks."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-1.5B"

print("Loading haystack...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
paragraphs = []
total = 0
for item in ds:
    text = item["text"].strip()
    if len(text) > 50:
        paragraphs.append(text)
        total += len(text)
        if total >= 50000:
            break
haystack = "\n\n".join(paragraphs)

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, trust_remote_code=True,
).to("cuda:0")
model.eval()

# Test context lengths from 250 to 4000
for target_tokens in [250, 500, 750, 1000, 1500, 2000, 3000, 4000]:
    suffix = f"\n\nBased on the content above, Question: {QUESTION}\nAnswer:"

    # Estimate chars needed
    target_chars = target_tokens * 4
    hay_budget = max(200, target_chars - len(NEEDLE) - len(suffix) - 50)
    hay_chunk = haystack[:hay_budget]

    # Insert needle at 50% depth
    mid = len(hay_chunk) // 2
    prompt = hay_chunk[:mid] + "\n" + NEEDLE + "\n" + hay_chunk[mid:] + suffix

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=40, do_sample=False, temperature=1.0,
        )
    generated = tokenizer.decode(out[0][n_tokens:], skip_special_tokens=True).strip()

    status = "FAIL" if generated.startswith("!") or len(generated) < 3 else "OK"
    print(f"  {target_tokens:>5} tokens (actual {n_tokens:>5}) | {status} | {generated[:80]}")
