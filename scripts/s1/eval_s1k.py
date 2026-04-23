"""Evaluate a model on the s1K verifiable dataset with pass@k.

Usage:
    python scripts/s1/eval_s1k.py [OPTIONS]

Examples:
    python scripts/s1/eval_s1k.py                          # defaults: Qwen3-1.7B, k=1
    python scripts/s1/eval_s1k.py --k 4 --temperature 0.7
    python scripts/s1/eval_s1k.py --model path/to/checkpoint --k 8
"""

import argparse
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams


# ── Answer extraction ──────────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx + len("\\boxed{"):i]
        i += 1
    return None


def extract_answer_from_response(response: str) -> str | None:
    """Extract final answer from model response, trying multiple patterns."""
    response = response.strip()

    # 1. \boxed{...}
    boxed = extract_boxed(response)
    if boxed:
        return boxed.strip()

    # 2. "the answer is X" / "The final answer is X"
    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$|\n)", response)
    if m:
        return m.group(1).strip()

    # 3. "Answer: X"
    m = re.search(r"[Aa]nswer[:\s]+([^\n,]+?)(?:\.|$|\n)", response)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # 4. Last line that looks like a standalone value
    lines = response.strip().split("\n")
    last = lines[-1].strip()
    if len(last) < 30 and not last[0:1].isalpha():
        return last

    return None


# ── Answer comparison ──────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    # Remove $, \text{}, trailing periods
    s = s.replace("$", "").replace("\\text{", "").replace("}", "")
    s = re.sub(r"\.$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove commas in numbers
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    return s


def is_correct(predicted: str | None, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if predicted is None:
        return False

    pred_norm = normalize(predicted)
    gold_norm = normalize(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Numeric comparison
    try:
        p = float(pred_norm.replace(",", ""))
        g = float(gold_norm.replace(",", ""))
        if abs(p - g) < 1e-4 or (g != 0 and abs((p - g) / g) < 1e-4):
            return True
    except ValueError:
        pass

    # Boolean normalization
    bool_map = {"true": "true", "false": "false", "yes": "true", "no": "false"}
    if pred_norm in bool_map and gold_norm in bool_map:
        return bool_map[pred_norm] == bool_map[gold_norm]

    # Multiple choice: sort letters for order-insensitive comparison
    if re.fullmatch(r"[a-e]{1,5}", pred_norm) and re.fullmatch(r"[a-e]{1,5}", gold_norm):
        return "".join(sorted(pred_norm)) == "".join(sorted(gold_norm))

    # List comparison
    list_re = r"\[([^\]]+)\]"
    pm = re.fullmatch(list_re, pred_norm)
    gm = re.fullmatch(list_re, gold_norm)
    if pm and gm:
        try:
            pvals = [float(x.strip()) for x in pm.group(1).split(",")]
            gvals = [float(x.strip()) for x in gm.group(1).split(",")]
            if len(pvals) == len(gvals):
                return all(abs(a - b) < 1e-3 for a, b in zip(pvals, gvals))
        except ValueError:
            pass

    return False


# ── Main ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def build_prompt(question: str) -> str:
    """Build a chat prompt for Qwen3."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on s1K verifiable")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model name or path")
    parser.add_argument("--data", default="data/s1K/s1k_verifiable.parquet", help="Dataset path")
    parser.add_argument("--k", type=int, default=1, help="Number of samples for pass@k")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy when k=1)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of problems (for testing)")
    parser.add_argument("--output", default=None, help="Path to save detailed results JSON")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (adds /think tag)")
    args = parser.parse_args()

    # Auto-adjust temperature for k > 1
    if args.k > 1 and args.temperature == 0.0:
        args.temperature = 0.6
        print(f"Auto-setting temperature={args.temperature} for pass@{args.k}")

    # Load dataset
    table = pq.read_table(args.data)
    questions = table.column("question").to_pylist()
    solutions = table.column("solution").to_pylist()
    if args.max_samples:
        questions = questions[:args.max_samples]
        solutions = solutions[:args.max_samples]
    n = len(questions)
    print(f"Loaded {n} problems from {args.data}")

    # Build prompts
    prompts = []
    for q in questions:
        prompt = build_prompt(q)
        if args.enable_thinking:
            prompt += "/think\n"
        prompts.append(prompt)

    # Repeat each prompt k times for pass@k
    all_prompts = []
    for p in prompts:
        all_prompts.extend([p] * args.k)

    # Load model
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95 if args.temperature > 0 else 1.0,
        max_tokens=args.max_tokens,
    )

    # Generate
    print(f"Generating {len(all_prompts)} responses (pass@{args.k})...")
    t0 = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    # Evaluate pass@k
    correct = 0
    results = []
    for i in range(n):
        gold = solutions[i]
        any_correct = False
        attempts = []
        for j in range(args.k):
            idx = i * args.k + j
            response = outputs[idx].outputs[0].text
            extracted = extract_answer_from_response(response)
            hit = is_correct(extracted, gold)
            if hit:
                any_correct = True
            attempts.append({
                "response": response[:500],
                "extracted": extracted,
                "correct": hit,
            })
        if any_correct:
            correct += 1
        results.append({
            "question": questions[i][:200],
            "gold": gold,
            "pass_at_k": any_correct,
            "attempts": attempts,
        })

    acc = correct / n * 100
    print()
    print("=" * 50)
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.data} ({n} problems)")
    print(f"pass@{args.k}:  {correct}/{n} = {acc:.1f}%")
    print("=" * 50)

    # Category breakdown
    cat_correct = {"numerical": 0, "boolean": 0, "multiple_choice": 0, "number_list": 0}
    cat_total = {"numerical": 0, "boolean": 0, "multiple_choice": 0, "number_list": 0}
    for i, r in enumerate(results):
        gold = solutions[i].strip()
        if gold.lower() in ("true", "false", "yes", "no"):
            cat = "boolean"
        elif re.fullmatch(r"[A-E]{1,5}", gold):
            cat = "multiple_choice"
        elif re.fullmatch(r"\[[\d.,\s-]+\]", gold):
            cat = "number_list"
        else:
            cat = "numerical"
        cat_total[cat] += 1
        if r["pass_at_k"]:
            cat_correct[cat] += 1

    print("\nBreakdown:")
    for cat in cat_correct:
        tot = cat_total[cat]
        if tot > 0:
            c = cat_correct[cat]
            print(f"  {cat:20s}: {c}/{tot} = {c/tot*100:.1f}%")

    # Save results
    output_path = args.output or f"data/s1K/eval_results_k{args.k}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "k": args.k,
            "temperature": args.temperature,
            "n_problems": n,
            "correct": correct,
            "accuracy": acc,
            "category_breakdown": {
                cat: {"correct": cat_correct[cat], "total": cat_total[cat]}
                for cat in cat_correct
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
