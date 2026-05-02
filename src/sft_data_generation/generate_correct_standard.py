"""Generate a dataset of correct standard-sampled answers.

Batched vLLM generation at temp=1 — resample until num_correct are collected.
"""

import argparse
import os
import sys
import time

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlvr_grokking.rewards.deepscaler_reward import compute_score, extract_answer

from sft_data_generation.utils import format_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correct answers via standard sampling")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_correct", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128, help="Samples per batch")
    parser.add_argument("--max_batches", type=int, default=100, help="Max resampling rounds")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="sft_data_generation/results/generate_standard")
    parser.add_argument("--problem_idx", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Temperature: {args.temperature}")
    print(f"Target: {args.num_correct} correct, batch size: {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    print(f"Loading vLLM model: {args.model}")
    vllm_model = LLM(
        model=args.model, dtype=args.dtype, trust_remote_code=True,
        gpu_memory_utilization=0.9, enable_prefix_caching=True,
    )

    # Load dataset
    if args.dataset.endswith(".json"):
        df = pd.read_json(args.dataset)
    else:
        df = pd.read_parquet(args.dataset)

    row = df.iloc[args.problem_idx]
    if "reward_model.ground_truth" in df.columns:
        ground_truth = str(row["reward_model.ground_truth"])
    else:
        ground_truth = str(row["reward_model"]["ground_truth"])

    raw_prompt = row["prompt"]
    if isinstance(raw_prompt, str):
        prompt_text = raw_prompt
        prompt_str = raw_prompt
    else:
        msgs = list(raw_prompt)
        prompt_text = msgs[-1]["content"] if msgs else ""
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    print(f"\nProblem {args.problem_idx}: ground_truth={ground_truth}")
    print(f"Prompt: {prompt_text[:200]}...")

    params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_new_tokens)

    results = []
    total_attempts = 0

    for batch_i in range(args.max_batches):
        remaining = args.num_correct - len(results)
        if remaining <= 0:
            break

        n_samples = min(args.batch_size, remaining * 5)  # oversample a bit
        prompts = [prompt_str] * n_samples

        t0 = time.time()
        outputs = vllm_model.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - t0
        total_attempts += n_samples

        batch_correct = 0
        for out in outputs:
            completion = out.outputs[0].text
            score = compute_score("math", completion, ground_truth)
            if score > 0:
                batch_correct += 1
                results.append({
                    "problem_idx": args.problem_idx,
                    "prompt": prompt_text,
                    "ground_truth": ground_truth,
                    "completion": completion,
                    "extracted_answer": extract_answer(completion),
                    "attempt_num": total_attempts,
                })
                if len(results) >= args.num_correct:
                    break

        print(f"  Batch {batch_i}: {batch_correct}/{n_samples} correct, "
              f"{elapsed:.1f}s, total correct: {len(results)}/{args.num_correct}")

    print(f"\nCollected {len(results)}/{args.num_correct} correct in {total_attempts} total attempts")

    if results:
        out_df = pd.DataFrame(results[:args.num_correct])
        out_path = os.path.join(args.output_dir, f"problem_{args.problem_idx:04d}.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
