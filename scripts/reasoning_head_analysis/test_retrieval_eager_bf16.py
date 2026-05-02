#!/usr/bin/env python3
"""Test if eager attention works with bf16 or fp32."""
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

for dtype_name, dtype in [("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32)]:
    print(f"\n{'='*60}")
    print(f"eager + {dtype_name}")
    print(f"{'='*60}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="eager",
        ).to("cuda:0")
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        n_tokens = inputs["input_ids"].shape[1]

        # Test generate
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        gen_text = tokenizer.decode(out[0][n_tokens:], skip_special_tokens=True)
        print(f"  generate(): {gen_text[:100]}")

        # Test manual decode with output_attentions
        with torch.no_grad():
            prefill = model(input_ids=inputs["input_ids"], output_attentions=False, use_cache=True)
        pkv = prefill.past_key_values
        next_id = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = []
        got_attn = False
        for step in range(20):
            gen_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            with torch.no_grad():
                step_out = model(input_ids=next_id, past_key_values=pkv,
                                 output_attentions=True, use_cache=True)
            pkv = step_out.past_key_values
            next_id = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if step_out.attentions is not None:
                got_attn = True
                if step == 0:
                    a = step_out.attentions[0]
                    print(f"  Attn shape layer 0: {a.shape}")
        gen_text2 = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"  manual+attn: {gen_text2[:100]}")
        print(f"  Got attention weights: {got_attn}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
