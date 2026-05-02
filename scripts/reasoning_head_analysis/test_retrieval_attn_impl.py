#!/usr/bin/env python3
"""Confirm: eager vs sdpa attention implementation causes the failure."""
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

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

for attn_impl in ["eager", "sdpa", None]:
    label = attn_impl or "default"
    print(f"\n{'='*60}")
    print(f"attn_implementation={label}")
    print(f"{'='*60}")

    kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True)
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs).to("cuda:0")
    model.eval()

    # Check what attention class is being used
    layer0_attn = model.model.layers[0].self_attn
    print(f"  Attention class: {type(layer0_attn).__name__}")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(out[0][n_tokens:], skip_special_tokens=True)
    print(f"  Generated: {gen_text[:100]}")

    # Also test: sdpa model but request output_attentions on decode only
    if attn_impl != "eager":
        print(f"  --- Now test manual decode with output_attentions=True ---")
        inputs2 = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            prefill = model(input_ids=inputs2["input_ids"], output_attentions=False, use_cache=True)
        pkv = prefill.past_key_values
        next_id = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = []
        for step in range(20):
            gen_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                step_out = model(input_ids=next_id, past_key_values=pkv,
                                 output_attentions=True, use_cache=True)
            pkv = step_out.past_key_values
            next_id = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            has_attn = step_out.attentions is not None
        gen_text2 = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  Manual+attentions: {gen_text2[:100]}")
        print(f"  Got attention weights: {has_attn}")

    del model
    torch.cuda.empty_cache()
