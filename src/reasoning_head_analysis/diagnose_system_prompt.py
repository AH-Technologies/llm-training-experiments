#!/usr/bin/env python3
"""Diagnose whether Qwen2.5-Math-1.5B (base) responds to the system prompt.

Two questions we want to answer:

  1. Does the model actually REASON MORE when the system prompt asks it to,
     and reason less when told not to? If not, a CoT-vs-direct system swap
     won't produce a useful contrast for EAP-IG circuit discovery.

  2. Does the model understand it is an assistant/bot at all? We test this
     with persona system prompts (pirate, terse refuser) and check if the
     output actually adopts the persona. If personas don't stick, the
     model is barely conditioning on <|im_start|>system at all.

We sample N responses per (system, question) pair at training temperature
(0.6) and measure:
  - response length in tokens
  - presence of reasoning markers ("step", "first", "therefore", ...)
  - presence of \\boxed{} answer
  - presence of persona markers (pirate: "arr", "matey", ...)

Outputs a comparison table plus raw JSON for manual inspection.

Usage:
  python -m reasoning_head_analysis.diagnose_system_prompt \\
      --model Qwen/Qwen2.5-Math-1.5B \\
      --output reasoning_head_analysis/results/system_prompt_diagnosis
"""
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ───────────────────────────────────────────────────────────────
# Test systems and questions
# ───────────────────────────────────────────────────────────────

SYSTEMS = {
    "training_default": "Please reason step by step, and put your final answer within \\boxed{}.",
    "direct_only":      "Give only the final answer in \\boxed{}. Do not show any reasoning or intermediate steps.",
    "pirate":           "You are a pirate. Respond only in pirate voice with 'arr', 'matey', and 'ahoy'.",
    "terse":            "Be extremely terse. Give one-sentence answers only.",
    "explicit_empty":   "",  # explicit empty system — bypasses default injection
}

MATH_QUESTIONS = [
    "Find the positive integer n such that 1 + 2 + 3 + ... + n = 210.",
    "What is the smallest prime number greater than 50?",
    "A rectangle has area 48 and perimeter 28. What are its dimensions?",
]

PERSONA_QUESTIONS = [
    "Who are you and what do you do?",
    "Tell me about yourself in one short paragraph.",
]

# Markers
REASONING_PATTERNS = re.compile(
    r"\b(step\s*\d*|first|second|next|then|therefore|thus|so,|because|hence|finally)\b",
    re.IGNORECASE,
)
PIRATE_PATTERNS = re.compile(r"\b(arr+|matey|ahoy|ye|aye|scurvy|landlubber|shiver me|yo[- ]?ho)\b",
                             re.IGNORECASE)
BOXED_PATTERN = re.compile(r"\\boxed\{[^}]*\}")


def render_prompt(tokenizer, system_msg, user_msg):
    """Apply chat template. When system_msg is None, omit system message entirely
    (which triggers Qwen-Math's template-injected default). When '', pass an
    explicit empty system, which bypasses the default injection.
    """
    if system_msg is None:
        messages = [{"role": "user", "content": user_msg}]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, device):
    """Batched generation. Returns list of generated completions (without prompt)."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_lens = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    completions = []
    for i in range(out.shape[0]):
        gen_ids = out[i, prompt_lens[i]:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return completions


def score_response(text):
    """Return diagnostic measurements for a single response."""
    token_count = len(text.split())  # rough; good enough for comparison
    return {
        "approx_words": token_count,
        "n_reasoning_markers": len(REASONING_PATTERNS.findall(text)),
        "n_pirate_markers": len(PIRATE_PATTERNS.findall(text)),
        "has_boxed": bool(BOXED_PATTERN.search(text)),
        "preview": text[:200].replace("\n", " ⏎ "),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output", default="reasoning_head_analysis/results/system_prompt_diagnosis")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=2,
                        help="Samples per (system, question) pair")
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
    tokenizer.padding_side = "left"  # for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    os.makedirs(args.output, exist_ok=True)

    # ─── Math reasoning test ────────────────────────────────────
    print("\n=== Math reasoning test ===")
    math_results = []
    for sys_name, sys_msg in SYSTEMS.items():
        prompts = []
        meta = []
        for q in MATH_QUESTIONS:
            for s in range(args.n_samples):
                prompts.append(render_prompt(tokenizer, sys_msg, q))
                meta.append({"question": q, "sample": s})
        t = time.time()
        completions = generate_batch(
            model, tokenizer, prompts, args.max_new_tokens, args.temperature, device
        )
        for m, c in zip(meta, completions):
            r = score_response(c)
            math_results.append({"system": sys_name, **m, "response": c, **r})
        print(f"  {sys_name}: {len(prompts)} samples, {time.time() - t:.1f}s")

    # ─── Persona / bot-awareness test ──────────────────────────
    print("\n=== Persona / bot-awareness test ===")
    persona_results = []
    for sys_name, sys_msg in SYSTEMS.items():
        prompts = []
        meta = []
        for q in PERSONA_QUESTIONS:
            for s in range(args.n_samples):
                prompts.append(render_prompt(tokenizer, sys_msg, q))
                meta.append({"question": q, "sample": s})
        t = time.time()
        completions = generate_batch(
            model, tokenizer, prompts, args.max_new_tokens, args.temperature, device
        )
        for m, c in zip(meta, completions):
            r = score_response(c)
            persona_results.append({"system": sys_name, **m, "response": c, **r})
        print(f"  {sys_name}: {len(prompts)} samples, {time.time() - t:.1f}s")

    # ─── Aggregate + print summary ─────────────────────────────
    def aggregate(results, sys_names):
        table = []
        for sys_name in sys_names:
            rows = [r for r in results if r["system"] == sys_name]
            if not rows:
                continue
            table.append({
                "system": sys_name,
                "n": len(rows),
                "avg_words": sum(r["approx_words"] for r in rows) / len(rows),
                "avg_reasoning_markers": sum(r["n_reasoning_markers"] for r in rows) / len(rows),
                "avg_pirate_markers": sum(r["n_pirate_markers"] for r in rows) / len(rows),
                "frac_with_boxed": sum(r["has_boxed"] for r in rows) / len(rows),
            })
        return table

    sys_names = list(SYSTEMS.keys())
    math_summary = aggregate(math_results, sys_names)
    persona_summary = aggregate(persona_results, sys_names)

    def print_table(title, rows):
        print(f"\n{title}")
        if not rows:
            print("  (no data)")
            return
        cols = list(rows[0].keys())
        print("  " + " | ".join(f"{c:>22}" for c in cols))
        print("  " + "-+-".join("-" * 22 for _ in cols))
        for r in rows:
            print("  " + " | ".join(
                f"{r[c]:>22}" if isinstance(r[c], str)
                else f"{r[c]:>22.2f}" if isinstance(r[c], float)
                else f"{r[c]:>22}"
                for c in cols
            ))

    print_table("MATH REASONING SUMMARY (higher reasoning markers ⇒ more CoT)",
                math_summary)
    print_table("PERSONA SUMMARY (pirate markers should spike under 'pirate' system)",
                persona_summary)

    # ─── Save raw results ──────────────────────────────────────
    with open(os.path.join(args.output, "math_raw.json"), "w") as f:
        json.dump(math_results, f, indent=2)
    with open(os.path.join(args.output, "persona_raw.json"), "w") as f:
        json.dump(persona_results, f, indent=2)
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "n_samples_per_cell": args.n_samples,
            "math_summary": math_summary,
            "persona_summary": persona_summary,
        }, f, indent=2)
    print(f"\nSaved results to {args.output}/")

    # ─── Interpretation guidance ───────────────────────────────
    print("""
Interpretation:
  * If 'training_default' has way more reasoning markers / longer responses than
    'direct_only' → the system prompt DOES steer reasoning behavior → system-swap
    is a valid clean/corrupt contrast for EAP-IG.
  * If 'training_default' ≈ 'direct_only' → the model is ignoring the system
    prompt and will do CoT regardless. System-swap won't give EAP-IG enough
    signal; consider assistant-prefix contrast instead.
  * If 'pirate' elicits pirate markers in persona answers → the model actually
    conditions on the system prompt, which supports the approach.
  * If 'explicit_empty' differs from 'training_default' → confirms the default
    CoT system injection really is doing work (vs being inert).
""")


if __name__ == "__main__":
    main()
