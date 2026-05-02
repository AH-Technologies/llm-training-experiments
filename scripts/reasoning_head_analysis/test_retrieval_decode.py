#!/usr/bin/env python3
"""Debug: compare model.generate() vs manual decode loop to find the bug."""
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
        if total >= 10000:
            break
haystack = "\n\n".join(paragraphs)

NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"

hay_chunk = haystack[:2000]
mid = len(hay_chunk) // 2
prompt = hay_chunk[:mid] + "\n" + NEEDLE + "\n" + hay_chunk[mid:]
prompt += f"\n\nBased on the content above, Question: {QUESTION}\nAnswer:"

print(f"Loading model: {MODEL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, trust_remote_code=True,
    attn_implementation="eager",
).to("cuda:0")
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
input_ids = inputs["input_ids"]
prompt_len = input_ids.shape[1]
print(f"Prompt: {prompt_len} tokens")

# --- Method 1: model.generate() ---
print("\n=== model.generate() ===")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
gen_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
print(f"Generated: {gen_text[:200]}")
print(f"Token IDs: {out[0][prompt_len:prompt_len+20].tolist()}")

# --- Method 2: manual loop (matching identify_heads_retrieval.py) ---
print("\n=== Manual decode (as in script) ===")
with torch.no_grad():
    prefill_out = model(
        input_ids=input_ids,
        output_attentions=False,
        use_cache=True,
    )
past_key_values = prefill_out.past_key_values
next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

generated_ids = []
print(f"First token from prefill: id={next_token_id.item()}, "
      f"text='{tokenizer.decode(next_token_id[0])}'")

for step in range(20):
    generated_ids.append(next_token_id.item())
    if next_token_id.item() == tokenizer.eos_token_id:
        break

    with torch.no_grad():
        out = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

gen_text2 = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated: {gen_text2[:200]}")
print(f"Token IDs: {generated_ids}")

# --- Method 3: manual loop WITHOUT output_attentions ---
print("\n=== Manual decode (no output_attentions) ===")
with torch.no_grad():
    prefill_out = model(
        input_ids=input_ids,
        output_attentions=False,
        use_cache=True,
    )
past_key_values = prefill_out.past_key_values
next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

generated_ids3 = []
for step in range(20):
    generated_ids3.append(next_token_id.item())
    if next_token_id.item() == tokenizer.eos_token_id:
        break

    with torch.no_grad():
        out = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

gen_text3 = tokenizer.decode(generated_ids3, skip_special_tokens=True)
print(f"Generated: {gen_text3[:200]}")
print(f"Token IDs: {generated_ids3}")

# --- Method 4: manual loop, output_attentions on BOTH prefill and decode ---
print("\n=== Manual decode (output_attentions=True everywhere) ===")
with torch.no_grad():
    prefill_out = model(
        input_ids=input_ids,
        output_attentions=True,
        use_cache=True,
    )
past_key_values = prefill_out.past_key_values
next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

generated_ids4 = []
for step in range(20):
    generated_ids4.append(next_token_id.item())
    if next_token_id.item() == tokenizer.eos_token_id:
        break

    with torch.no_grad():
        out = model(
            input_ids=next_token_id,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

gen_text4 = tokenizer.decode(generated_ids4, skip_special_tokens=True)
print(f"Generated: {gen_text4[:200]}")
print(f"Token IDs: {generated_ids4}")
