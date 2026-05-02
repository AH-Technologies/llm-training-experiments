#!/usr/bin/env python3
"""Evaluate saved checkpoints on MATH500, AIME 2024, and AMC 2023.

Usage:
  python scripts/attention_sparks_thinking/evaluate_checkpoints.py \
    --checkpoint_dir /path/to/checkpoints \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --benchmarks math500 aime amc
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"

BENCHMARK_FILES = {
    "math500": os.path.join(PROJECT_DIR, "data/math500.parquet"),
    "aime": os.path.join(PROJECT_DIR, "attention_based_rewards/data/aime_2025.parquet"),
    "amc": os.path.join(PROJECT_DIR, "attention_based_rewards/data/amc_2023.parquet"),
}

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def load_benchmark(name: str) -> list[dict]:
    """Load benchmark data from parquet."""
    path = BENCHMARK_FILES[name]
    df = pd.read_parquet(path)
    items = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        ground_truth = row["reward_model"]
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        items.append({
            "prompt": prompt,
            "ground_truth": ground_truth.get("ground_truth", ""),
        })
    return items


def format_prompt(tokenizer, messages: list[dict]) -> str:
    """Format chat messages into a single prompt string."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_checkpoint(
    checkpoint_path: str,
    model_name: str,
    benchmarks: list[str],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    n_samples: int = 1,
) -> dict:
    """Evaluate a single checkpoint on specified benchmarks.

    Returns dict of {benchmark_name: accuracy}.
    """
    # Load reward function
    sys.path.insert(0, PROJECT_DIR)
    from src.rlvr_grokking.rewards.verl_reward import compute_score

    # Load model with vLLM
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=5120,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        n=n_samples,
    )

    results = {}

    for bench_name in benchmarks:
        items = load_benchmark(bench_name)
        logger.info(f"Evaluating {bench_name}: {len(items)} problems")

        # Format prompts
        prompts = []
        for item in items:
            prompt_str = format_prompt(tokenizer, item["prompt"])
            prompts.append(prompt_str)

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Score
        correct = 0
        total = 0
        for item, output in zip(items, outputs):
            for completion in output.outputs:
                response_text = completion.text
                score = compute_score("math_dapo", response_text, item["ground_truth"])
                if score > 0:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        results[bench_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        logger.info(f"  {bench_name}: {accuracy:.1%} ({correct}/{total})")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return results


def find_checkpoints(checkpoint_dir: str) -> list[str]:
    """Find all checkpoint directories sorted by step number."""
    checkpoints = []
    if not os.path.isdir(checkpoint_dir):
        logger.warning(f"Checkpoint dir not found: {checkpoint_dir}")
        return checkpoints

    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and ("step" in name or "checkpoint" in name or "global_step" in name):
            checkpoints.append(path)

    # Sort by step number
    def extract_step(p):
        base = os.path.basename(p)
        for part in base.split("_"):
            if part.isdigit():
                return int(part)
        return 0

    checkpoints.sort(key=extract_step)
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Base model name (for tokenizer)")
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "aime", "amc"],
                        choices=["math500", "aime", "amc"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        logger.error(f"No checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    logger.info(f"Found {len(checkpoints)} checkpoints")

    all_results = {}
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {ckpt_name}")
        logger.info(f"{'='*60}")

        results = evaluate_checkpoint(
            ckpt_path,
            args.model_name,
            args.benchmarks,
            temperature=args.temperature,
        )
        all_results[ckpt_name] = results

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for ckpt_name, results in all_results.items():
        print(f"\n{ckpt_name}:")
        for bench, metrics in results.items():
            print(f"  {bench}: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")

    # Save results
    output_path = args.output or os.path.join(args.checkpoint_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
