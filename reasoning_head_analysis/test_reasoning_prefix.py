#!/usr/bin/env python3
"""Prefix-steering diagnostic + logit KL probe.

For each candidate (clean, corrupt) prefix pair and each question:
  1. Generate a completion from each side (inspect qualitatively).
  2. Forward-pass both prompts and compute KL between their next-token
     distributions at the final prompt position — the quantity EAP-IG cares
     about. Also record top-5 predicted tokens under each condition.

System prompt is the tokenizer's default (unmodified). Only the assistant
prefix varies.

Usage:
  python -m reasoning_head_analysis.test_reasoning_prefix \\
      --model Qwen/Qwen2.5-Math-1.5B \\
      --output reasoning_head_analysis/results/prefix_test_Qwen_Qwen2.5-Math-1.5B
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Length-matched (clean, corrupt) prefix pairs. Tokens counted with
# Qwen2.5-Math tokenizer.
PAIRS = [
    # 6 tokens
    ("To solve this problem, we",        "The answer is \\boxed{"),
    # 7 tokens
    ("Let's think step by step.",        "The final answer is \\boxed{"),
    # 8 tokens
    ("Let me solve this step by step.",  "The answer to this is \\boxed{"),
]

QUESTIONS = [
    "Find the positive integer n such that 1 + 2 + 3 + ... + n = 210.",
    "What is the smallest prime number greater than 50?",
    "A rectangle has area 48 and perimeter 28. What are its dimensions?",
    "If f(x) = 2x + 3, what is f(f(5))?",
]


def build_prompt(tokenizer, question, prefix):
    """Apply chat template (default system) and append the assistant prefix."""
    messages = [{"role": "user", "content": question}]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return chat + prefix


def generate(model, tokenizer, prompts, max_new_tokens, temperature, device):
    """Batched generation. Returns completions (without prompt) for each sample."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # With left-padding, all prompts end at the same column == input_ids.shape[1]-1.
    # New tokens begin at column input_ids.shape[1].
    input_len = inputs["input_ids"].shape[1]
    completions = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return completions


def forward_logits(model, tokenizer, prompt, device):
    """Forward pass on a single prompt; return logits at the final position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1].float()  # shape [vocab]


def topk_tokens(logits, tokenizer, k=5):
    """Return list of (token_str, prob) for the top-k predicted next tokens."""
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    result = []
    for idx, p in zip(top.indices.tolist(), top.values.tolist()):
        result.append({"token": tokenizer.decode([idx]), "prob": round(p, 4)})
    return result


def kl_clean_corrupt(clean_logits, corrupt_logits):
    """KL( softmax(clean) || softmax(corrupt) ) in nats, last-token position."""
    clean_lp = F.log_softmax(clean_logits, dim=-1)
    corrupt_lp = F.log_softmax(corrupt_logits, dim=-1)
    return (clean_lp.exp() * (clean_lp - corrupt_lp)).sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--n_samples", type=int, default=2,
                        help="Generations per (pair, side, question) cell")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {args.model} on {device} ({dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    os.makedirs(args.output, exist_ok=True)

    gen_rows = []
    logit_rows = []

    # ─── Generations (batched per pair × side; n_samples per cell) ──────
    print(f"\n=== Generating completions ({args.n_samples} per cell) ===")
    for pair_i, (clean_pfx, corrupt_pfx) in enumerate(PAIRS):
        for side_name, side_pfx in [("clean", clean_pfx), ("corrupt", corrupt_pfx)]:
            prompts = []
            cell_meta = []
            for q in QUESTIONS:
                for s in range(args.n_samples):
                    prompts.append(build_prompt(tokenizer, q, side_pfx))
                    cell_meta.append({"question": q, "sample": s})
            t = time.time()
            completions = generate(model, tokenizer, prompts,
                                   args.max_new_tokens, args.temperature, device)
            for m, c in zip(cell_meta, completions):
                gen_rows.append({
                    "pair_index": pair_i,
                    "side": side_name,
                    "prefix": side_pfx,
                    "completion": c,
                    "assistant_turn": side_pfx + c,
                    **m,
                })
            print(f"  pair {pair_i} ({side_name}): {len(prompts)} gens in {time.time() - t:.1f}s")

    # ─── Logit diagnostic (one forward pass per prompt) ─────────────────
    print("\n=== Computing end-of-prompt KL and top-k tokens ===")
    for pair_i, (clean_pfx, corrupt_pfx) in enumerate(PAIRS):
        for q in QUESTIONS:
            clean_prompt = build_prompt(tokenizer, q, clean_pfx)
            corrupt_prompt = build_prompt(tokenizer, q, corrupt_pfx)
            clean_logits = forward_logits(model, tokenizer, clean_prompt, device)
            corrupt_logits = forward_logits(model, tokenizer, corrupt_prompt, device)
            kl = kl_clean_corrupt(clean_logits, corrupt_logits)
            logit_rows.append({
                "pair_index": pair_i,
                "clean_prefix": clean_pfx,
                "corrupt_prefix": corrupt_pfx,
                "question": q,
                "kl_nats": round(kl, 4),
                "clean_top5": topk_tokens(clean_logits, tokenizer, 5),
                "corrupt_top5": topk_tokens(corrupt_logits, tokenizer, 5),
            })

    # ─── Save ───────────────────────────────────────────────────────────
    out_path = os.path.join(args.output, "raw.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "pairs": [{"pair_index": i, "clean": c, "corrupt": d}
                      for i, (c, d) in enumerate(PAIRS)],
            "generations": gen_rows,
            "logits": logit_rows,
        }, f, indent=2)
    print(f"\nSaved {len(gen_rows)} generations and {len(logit_rows)} logit rows to {out_path}")

    # ─── Print logit summary ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"LOGIT SUMMARY — {args.model}")
    print("=" * 78)
    for pair_i, (c, d) in enumerate(PAIRS):
        rows = [r for r in logit_rows if r["pair_index"] == pair_i]
        avg_kl = sum(r["kl_nats"] for r in rows) / len(rows)
        print(f"\nPair {pair_i}  avg KL={avg_kl:.3f} nats")
        print(f"  clean   : {c!r}")
        print(f"  corrupt : {d!r}")
        for r in rows:
            ctop = ", ".join(f"{t['token']!r}={t['prob']}" for t in r["clean_top5"][:3])
            dtop = ", ".join(f"{t['token']!r}={t['prob']}" for t in r["corrupt_top5"][:3])
            print(f"  [{r['question'][:40]:<40}]  KL={r['kl_nats']:.3f}")
            print(f"     clean top3   : {ctop}")
            print(f"     corrupt top3 : {dtop}")


if __name__ == "__main__":
    main()
