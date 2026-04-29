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


_ANSWER_TAG = "<|im_start|>answer"
_END_TAG = "<|im_end|>"


def answer_section(text: str) -> str | None:
    """Content between the last <|im_start|>answer and the next <|im_end|>.

    The s1 training format places the final answer in a dedicated section
    (`<|im_start|>answer\\n...\\n<|im_end|>`). Training data often terminates
    with a bare integer or short expression there rather than \\boxed{...},
    so we look in this section first.
    """
    idx = text.rfind(_ANSWER_TAG)
    if idx < 0:
        return None
    start = idx + len(_ANSWER_TAG)
    end = text.find(_END_TAG, start)
    return text[start:end if end >= 0 else None].strip()


_MATH_GROUP = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$([^\$\n]+?)\$",
    re.DOTALL,
)


def _last_math_expr(text: str) -> str | None:
    """Return content of the last $...$ / \\[...\\] / \\(...\\) in text."""
    matches = list(_MATH_GROUP.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    for g in last.groups():
        if g is not None:
            return g.strip()
    return None


def _rhs_if_equation(expr: str) -> str:
    """If expr has an equals sign that's not part of >=, <=, \\ge, \\le, \\neq, ... ,
    return the RHS of the last real '='. Otherwise return expr unchanged."""
    # Strip LaTeX relational operators that contain '=' so we don't split on them.
    masked = re.sub(r"\\(ge|le|ne|geq|leq|neq|approx|equiv)", "  ", expr)
    masked = masked.replace(">=", "  ").replace("<=", "  ").replace("!=", "  ")
    if "=" not in masked:
        return expr
    # Find the real '=' positions and take after the last one.
    idx = masked.rfind("=")
    return expr[idx + 1:].strip()


def extract_answer(text: str) -> str | None:
    """Extract final answer from model response.

    Priority (based on spot-checks of trained-model MATH500 responses):
      1. \\boxed{...} anywhere — highest-precision signal.
      2. "Final Answer: X" / "the answer is X" / "the result is X" in answer
         section (or full text if no section tag).
      3. Last math-delimited expression ($..$, \\[..\\], \\(..\\)) in the
         answer section, with RHS taken when it's an equation. Handles cases
         like "...so $h = 5$" → "5" or "...sends ... to $-b + ai$" → "-b + ai".
      4. Short (≤100 chars) answer section verbatim — handles bare answers
         like "<|im_start|>answer\\n128<|im_end|>".
    """
    text = text.strip()

    boxed = extract_boxed(text)
    if boxed:
        return boxed.strip()

    section = answer_section(text)
    search = section if section is not None else text

    patterns = [
        r"[Ff]inal\s+[Aa]nswer\s*(?:is|:)\s*(.+?)(?:\.|$|\n)",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*(.+?)(?:\.|$|\n)",
        r"[Tt]he\s+result\s+is\s*(.+?)(?:\.|$|\n)",
    ]
    for pat in patterns:
        m = re.search(pat, search)
        if m:
            cand = m.group(1).strip()
            # If the captured tail starts with $...$ or \(..\), prefer the math content.
            inner = _last_math_expr(cand)
            return inner if inner else cand

    if section is not None:
        expr = _last_math_expr(section)
        if expr:
            return _rhs_if_equation(expr).rstrip(".,;").strip()

    if section is not None and 0 < len(section) <= 100:
        return section.strip()

    return None


def extract_gpqa_answer(text: str) -> str | None:
    """Extract GPQA multiple-choice answer (A/B/C/D)."""
    text = text.strip()
    section = answer_section(text)
    search = section if section is not None else text

    boxed = extract_boxed(search)
    if boxed and boxed.strip().upper() in "ABCD":
        return boxed.strip().upper()

    patterns = [
        r'[Aa]nswer\s*(?:is|:)\s*\(?([A-Da-d])\)?',
        r'[Cc]orrect\s*(?:answer|option)\s*(?:is|:)\s*\(?([A-Da-d])\)?',
        r'\b([A-Da-d])\s*(?:is|is the)\s*(?:correct|right)',
    ]
    for pat in patterns:
        m = re.search(pat, search)
        if m:
            return m.group(1).upper()

    # Section is a single letter like "B"
    if section is not None and section.strip().upper() in "ABCD":
        return section.strip().upper()

    # Last standalone letter in search region
    m = re.search(r'\b([A-Da-d])\s*\.?\s*$', search.strip())
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
    """Load GPQA Diamond (198 PhD-level science questions).

    The dataset stores choices in fixed key order (Correct / Incorrect 1-3),
    so we shuffle them to randomise letter position. Uses a single Random
    instance seeded once — resetting the seed per row would give every row
    the same permutation, pinning every correct answer to the same letter.
    """
    import random as _random
    rng = _random.Random(42)  # one stream across all rows

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    problems = []
    for row in ds:
        choices = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            if key in row and row[key]:
                choices.append(row[key])

        correct_answer = choices[0]
        rng.shuffle(choices)
        correct_letter = chr(65 + choices.index(correct_answer))

        choice_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        question = f"{row['Question']}\n\n{choice_text}"

        problems.append({
            "question": question,
            "answer": correct_letter,
            "answer_type": "mcq",
        })
    return problems


def _load_verl_parquet(path: str, name: str):
    """Load a VERL-format math validation parquet.

    These parquets (data/val/*_verl.parquet) carry chat-style `prompt`
    (system+user dicts) and `reward_model.ground_truth`. We extract the
    user content as the question and the ground truth as the gold answer.
    Used by the screening pipeline for AMC/AIME 2025.
    """
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    problems = []
    for prompt, rm in zip(t.column("prompt").to_pylist(), t.column("reward_model").to_pylist()):
        # prompt is a list of message dicts; pick the user turn
        user_msg = next((m for m in prompt if m.get("role") == "user"), None)
        if user_msg is None:
            continue
        problems.append({
            "question": user_msg["content"],
            "answer": str(rm["ground_truth"]),
            "answer_type": "boxed",
        })
    return problems


def load_amc():
    """AMC validation set (~83 problems, integer/numeric answers)."""
    return _load_verl_parquet("data/val/amc_verl.parquet", "AMC")


def load_aime25():
    """AIME 2025 (~30 problems, integer answers 0–999)."""
    return _load_verl_parquet("data/val/aime_2025_verl.parquet", "AIME25")


BENCHMARKS = {
    "math500": ("MATH500", load_math500),
    "aime24": ("AIME24", load_aime24),
    "gpqa": ("GPQA Diamond", load_gpqa_diamond),
    "amc": ("AMC", load_amc),
    "aime25": ("AIME25", load_aime25),
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """Build chat prompt matching the s1 paper's format (Figure 10, §D).

    Training data (prepare_s1k_sft.py) uses no system prompt — the model is
    fine-tuned on <|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n
    and is expected to auto-generate <|im_start|>think ... <|im_start|>answer
    ... \\boxed{...}<|im_end|>. Prepending Qwen's default system prompt at
    inference breaks the conditioning and suppresses the answer tail.
    """
    return (
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
            # Last 2500 chars — enough to see the <|im_start|>answer section
            # and its tail. Kept small so per-eval JSONs stay manageable.
            "response_tail": response[-2500:] if len(response) > 2500 else response,
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
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "vLLM load dtype. Default bfloat16 — our HF Trainer FSDP checkpoints "
            "save as fp32 even for bf16-trained models; letting vLLM auto-read "
            "config.json triggers an fp32-allocate-then-downcast path that "
            "spikes VRAM during load."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    # Cap max_model_len at the model's max_position_embeddings (Qwen2.5-32B is
    # 32768). Exceeding RoPE's trained range risks NaN / CUDA OOB, so we clamp
    # rather than set VLLM_ALLOW_LONG_MAX_MODEL_LEN.
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cap = getattr(hf_config, "max_position_embeddings", None)
    max_model_len = args.max_tokens + 2048
    if model_cap is not None and max_model_len > model_cap:
        print(f"  Capping max_model_len {max_model_len} → {model_cap} (model limit)")
        max_model_len = model_cap

    model = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=max_model_len,
        # Custom all-reduce uses direct GPU peer access; on this cluster's
        # multi-GPU topology it crashes with "Cuda error ... 'invalid argument'"
        # in the warm-up RPC, killing TP workers silently. NCCL fallback works.
        enable_custom_all_reduce=False,
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
