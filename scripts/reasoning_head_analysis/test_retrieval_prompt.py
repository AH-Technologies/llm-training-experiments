#!/usr/bin/env python3
"""Diagnostic: build a few retrieval prompts and see what the base model generates."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-1.5B"

# Load haystack
print("Loading haystack...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
paragraphs = []
total = 0
for item in ds:
    text = item["text"].strip()
    if len(text) > 50:
        paragraphs.append(text)
        total += len(text)
        if total >= 10000:
            break
haystack = "\n\n".join(paragraphs)

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"

# Insert needle at ~50% depth into a short haystack (~500 tokens worth)
hay_chunk = haystack[:2000]
mid = len(hay_chunk) // 2
hay_with_needle = hay_chunk[:mid] + "\n" + NEEDLE + "\n" + hay_chunk[mid:]

# Try several prompt formats
PROMPTS = {
    "original": (
        f"{hay_with_needle}\n\n"
        f"Based on the content above, Question: {QUESTION}\n"
        f"Answer:"
    ),
    "completion_style": (
        f"{hay_with_needle}\n\n"
        f"Q: {QUESTION}\n"
        f"A: The best thing to do in San Francisco is"
    ),
    "extract_style": (
        f"Read the following text and answer the question.\n\n"
        f"Text: {hay_with_needle}\n\n"
        f"Question: {QUESTION}\n"
        f"Answer:"
    ),
    "chat_template": (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{hay_with_needle}\n\n{QUESTION}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    ),
    "short_haystack": (
        f"{haystack[:500]}\n{NEEDLE}\n{haystack[500:1000]}\n\n"
        f"Q: {QUESTION}\nA:"
    ),
}

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, trust_remote_code=True,
).to("cuda:0")
model.eval()

for name, prompt in PROMPTS.items():
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    n_tokens = inputs["input_ids"].shape[1]
    print(f"\n{'='*60}")
    print(f"PROMPT STYLE: {name} ({n_tokens} tokens)")
    print(f"{'='*60}")
    print(f"...last 200 chars of prompt:")
    print(prompt[-200:])
    print(f"--- Generation ---")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=1.0,
        )
    generated = tokenizer.decode(out[0][n_tokens:], skip_special_tokens=True)
    print(generated[:300])
    print()
