"""Evaluate s1-32B on the paper's three benchmarks: MATH500, AIME24, GPQA Diamond.

Follows the s1 paper's evaluation protocol (§4.1):
- Uses vLLM for generation
- Greedy decoding (temperature=0)
- Accuracy metric (equivalent to pass@1)
- lm-evaluation-harness style answer extraction

Usage:
    python -m s1.eval --model checkpoints/s1_sft_qwen32b/checkpoint-epoch5-step315
    python -m s1.eval --model checkpoints/s1_sft_qwen32b/checkpoint-epoch5-step315 --benchmarks math500 aime24
    python -m s1.eval --model Qwen/Qwen2.5-32B-Instruct  # baseline
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

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


def extract_answer(text: str) -> str | None:
    """Extract final answer from model response."""
    text = text.strip()

    boxed = extract_boxed(text)
    if boxed:
        return boxed.strip()

    # "Final Answer: X" or "the answer is X"
    m = re.search(r"[Ff]inal\s+[Aa]nswer[:\s]+(.+?)(?:\.|$|\n)", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$|\n)", text)
    if m:
        return m.group(1).strip()

    return None


def extract_gpqa_answer(text: str) -> str | None:
    """Extract GPQA multiple-choice answer (A/B/C/D)."""
    text = text.strip()

    # \boxed{A}
    boxed = extract_boxed(text)
    if boxed and boxed.strip().upper() in "ABCD":
        return boxed.strip().upper()

    # "answer is (A)" / "Answer: B"
    patterns = [
        r'[Aa]nswer\s*(?:is|:)\s*\(?([A-Da-d])\)?',
        r'[Cc]orrect\s*(?:answer|option)\s*(?:is|:)\s*\(?([A-Da-d])\)?',
        r'\b([A-Da-d])\s*(?:is|is the)\s*(?:correct|right)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).upper()

    # Last standalone letter
    m = re.search(r'\b([A-Da-d])\s*\.?\s*$', text.strip())
    if m:
        return m.group(1).upper()

    return None


def normalize(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip().lower()
    s = s.replace("$", "").replace("\\text{", "").replace("}", "")
    s = re.sub(r"\.$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    return s


def is_correct(predicted: str | None, gold: str, answer_type: str = "boxed") -> bool:
    """Check if predicted answer matches gold."""
    if predicted is None:
        return False

    if answer_type == "mcq":
        return predicted.strip().upper() == gold.strip().upper()

    pred_norm = normalize(predicted)
    gold_norm = normalize(gold)

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

    return False


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_math500():
    """Load MATH-500 benchmark (500 problems)."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for row in ds:
        problems.append({
            "question": row["problem"],
            "answer": row["answer"],
            "answer_type": "boxed",
        })
    return problems


def load_aime24():
    """Load AIME 2024 (30 problems). Answers are integers 000-999."""
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    problems = []
    for row in ds:
        # Filter to AIME 2024 only
        url = row.get("url", "")
        if "2024" in url or "2024" in row.get("problem", ""):
            problems.append({
                "question": row["problem"],
                "answer": str(row["answer"]),
                "answer_type": "boxed",
            })
    # If filtering didn't work (dataset structure varies), take all
    if not problems:
        for row in ds:
            problems.append({
                "question": row["problem"],
                "answer": str(row["answer"]),
                "answer_type": "boxed",
            })
    return problems


def load_gpqa_diamond():
    """Load GPQA Diamond (198 PhD-level science questions)."""
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    problems = []
    for row in ds:
        # Build MCQ prompt with choices
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if key in row and row[key]:
                choices.append(row[key])

        # The correct answer is always the first one; we need to shuffle
        # but keep track of which letter is correct
        import random
        random.seed(42)  # Deterministic shuffle
        correct_answer = choices[0]
        random.shuffle(choices)
        correct_letter = chr(65 + choices.index(correct_answer))  # A, B, C, D

        choice_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        question = f"{row['Question']}\n\n{choice_text}"

        problems.append({
            "question": question,
            "answer": correct_letter,
            "answer_type": "mcq",
        })
    return problems


BENCHMARKS = {
    "math500": ("MATH500", load_math500),
    "aime24": ("AIME24", load_aime24),
    "gpqa": ("GPQA Diamond", load_gpqa_diamond),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """Build chat prompt matching the s1 paper's format."""
    return (
        f"<|im_start|>system\n"
        f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def evaluate_benchmark(
    model: LLM,
    problems: list[dict],
    benchmark_name: str,
    max_tokens: int = 32768,
    max_samples: int | None = None,
) -> dict:
    """Evaluate model on a benchmark."""
    if max_samples:
        problems = problems[:max_samples]

    prompts = [build_prompt(p["question"]) for p in problems]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )

    print(f"  Generating {len(prompts)} responses...")
    start = time.perf_counter()
    outputs = model.generate(prompts, sampling_params)
    gen_time = time.perf_counter() - start
    print(f"  Generation took {gen_time:.1f}s ({gen_time/len(prompts):.1f}s/sample)")

    correct = 0
    results = []
    for i, (problem, output) in enumerate(zip(problems, outputs)):
        response = output.outputs[0].text
        answer_type = problem["answer_type"]

        if answer_type == "mcq":
            predicted = extract_gpqa_answer(response)
        else:
            predicted = extract_answer(response)

        match = is_correct(predicted, problem["answer"], answer_type)
        if match:
            correct += 1

        results.append({
            "question": problem["question"][:200],
            "gold": problem["answer"],
            "predicted": predicted,
            "correct": match,
            "response_length": len(response),
        })

    accuracy = correct / len(problems) * 100
    print(f"  {benchmark_name}: {correct}/{len(problems)} = {accuracy:.1f}%")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "generation_time_s": gen_time,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="S1 evaluation on paper benchmarks")
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "aime24", "gpqa"],
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark (for testing)")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallelism for vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output-dir", default=None, help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_tokens + 2048,  # room for prompt + generation
    )

    all_results = {}
    for bench_key in args.benchmarks:
        bench_name, loader = BENCHMARKS[bench_key]
        print(f"\n{'='*60}")
        print(f"Evaluating: {bench_name}")
        print(f"{'='*60}")
        problems = loader()
        print(f"  Loaded {len(problems)} problems")
        result = evaluate_benchmark(model, problems, bench_name, args.max_tokens, args.max_samples)
        all_results[bench_key] = result

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    for key, result in all_results.items():
        print(f"  {result['benchmark']}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_name = Path(args.model).name
        output_path = os.path.join(args.output_dir, f"eval_{model_name}.json")

        save_data = {
            "model": args.model,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                        for k, v in all_results.items()},
            "detailed_results": {k: v["results"] for k, v in all_results.items()},
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
