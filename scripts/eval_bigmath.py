"""Evaluate Qwen3-8B on bigmath_filtered with pass@k, saving per-problem solve rates.

Outputs a JSON with per-problem solve_rate (correct_count / k) for downstream
filtering. Supports sharding and checkpointing (processes in batches, resumes
from last checkpoint on restart).

Uses the same grading function as the training pipeline (deepscaler_reward.py)
which handles LaTeX normalization and SymPy symbolic comparison.

Usage:
    python scripts/eval_bigmath.py --model Qwen/Qwen3-8B --k 8
    python scripts/eval_bigmath.py --model Qwen/Qwen3-8B --k 8 --shard 0 --num-shards 8
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams

# Add project root to path so we can import the reward function
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rlvr_grokking.rewards.deepscaler_reward import compute_score as grade_solution


# ── Main ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the problem and provide your final answer. "
    "Put your final answer within \\boxed{}."
)


def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on bigmath_filtered")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name or path")
    parser.add_argument("--data", default="data/bigmath/bigmath_filtered.parquet")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Number of problems per checkpoint batch")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.k > 1 and args.temperature == 0.0:
        args.temperature = 0.6
        print(f"Auto-setting temperature={args.temperature} for pass@{args.k}")

    # Load dataset
    table = pq.read_table(args.data)
    questions = table.column("question").to_pylist()
    solutions = table.column("solution").to_pylist()
    total = len(questions)
    print(f"Loaded {total} problems from {args.data}")

    # Shard if requested
    if args.shard is not None and args.num_shards is not None:
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        questions = questions[start:end]
        solutions = solutions[start:end]
        print(f"Shard {args.shard}/{args.num_shards}: problems [{start}, {end}) = {len(questions)}")

    n = len(questions)

    # Output path
    shard_suffix = f"_shard{args.shard}" if args.shard is not None else ""
    output_path = Path(args.output or f"data/bigmath/eval_results_k{args.k}{shard_suffix}.json")
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if it exists
    results = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        print(f"Resumed from checkpoint: {len(results)} problems already done")

    start_idx = len(results)
    if start_idx >= n:
        print("All problems already evaluated, skipping to summary.")
    else:
        # Load model
        print(f"Loading model: {args.model}")
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=4096,
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95 if args.temperature > 0 else 1.0,
            max_tokens=args.max_tokens,
        )

        # Process in batches with checkpointing
        t0 = time.time()
        total_generated = 0

        for batch_start in range(start_idx, n, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n)
            batch_questions = questions[batch_start:batch_end]
            batch_solutions = solutions[batch_start:batch_end]
            batch_n = len(batch_questions)

            # Build prompts for this batch
            batch_prompts = []
            for q in batch_questions:
                prompt = build_prompt(q)
                if args.enable_thinking:
                    prompt += "/think\n"
                batch_prompts.extend([prompt] * args.k)

            # Generate
            bt0 = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            bt1 = time.time()
            total_generated += len(batch_prompts)

            # Evaluate batch using the same grader as training
            batch_results = []
            for i in range(batch_n):
                gold = batch_solutions[i]
                correct_count = 0
                for j in range(args.k):
                    idx = i * args.k + j
                    response = outputs[idx].outputs[0].text
                    score = grade_solution(
                        data_source="bigmath",
                        solution_str=response,
                        ground_truth=gold,
                    )
                    if score >= 0.5:
                        correct_count += 1
                batch_results.append({
                    "question": batch_questions[i][:300],
                    "gold": gold,
                    "correct_count": correct_count,
                    "solve_rate": correct_count / args.k,
                })

            # Append to checkpoint file
            with open(checkpoint_path, "a") as f:
                for r in batch_results:
                    f.write(json.dumps(r) + "\n")

            results.extend(batch_results)
            elapsed = time.time() - t0
            speed = total_generated / elapsed
            remaining = (n - batch_end) * args.k / speed if speed > 0 else 0
            print(
                f"  [{batch_end}/{n}] "
                f"batch {bt1-bt0:.0f}s, "
                f"overall {speed:.1f} prompts/s, "
                f"ETA {remaining/3600:.1f}h"
            )

        elapsed = time.time() - t0
        print(f"\nGeneration done in {elapsed:.1f}s ({total_generated/elapsed:.1f} prompts/s)")

    # Summary
    total_correct = sum(1 for r in results if r["correct_count"] > 0)
    acc = total_correct / n * 100
    print(f"\npass@{args.k}: {total_correct}/{n} = {acc:.1f}%")

    print("\nSolve rate distribution:")
    brackets = [(0, 0), (0.001, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    labels = ["0 (never)", "(0, 0.125)", "[0.125, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1.0]"]
    for (lo, hi), label in zip(brackets, labels):
        if lo == 0 and hi == 0:
            count = sum(1 for r in results if r["solve_rate"] == 0)
        else:
            count = sum(1 for r in results if lo <= r["solve_rate"] < hi)
        print(f"  {label:20s}: {count:>6d} ({count/n*100:.1f}%)")

    # Save final results
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "k": args.k,
            "temperature": args.temperature,
            "n_problems": n,
            "pass_at_k": total_correct,
            "accuracy": acc,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint removed: {checkpoint_path}")


if __name__ == "__main__":
    main()
