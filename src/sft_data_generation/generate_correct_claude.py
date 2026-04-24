"""Generate a dataset of correct answers using Claude API.

Async API calls with concurrency control — resample until num_correct are collected.
"""

import argparse
import asyncio
import os
import sys
import time

import pandas as pd
from anthropic import AsyncAnthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlvr_grokking.rewards.deepscaler_reward import compute_score, extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correct answers via Claude API")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_correct", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=10, help="Samples per batch")
    parser.add_argument("--max_batches", type=int, default=100, help="Max resampling rounds")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="sft_data_generation/results/generate_claude")
    parser.add_argument("--problem_idx", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent API requests")
    return parser.parse_args()


async def call_claude(client, model, prompt_text, temperature, max_tokens, semaphore, max_retries=5):
    """Make a single Claude API call with concurrency control and retry on rate limits."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                return response.content[0].text
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise
        raise Exception(f"Max retries ({max_retries}) exceeded due to rate limiting")


async def generate_batch(client, model, prompt_text, temperature, max_tokens, n_samples, semaphore):
    """Generate n_samples completions concurrently."""
    tasks = [
        call_claude(client, model, prompt_text, temperature, max_tokens, semaphore)
        for _ in range(n_samples)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def async_main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Target: {args.num_correct} correct, batch size: {args.batch_size}")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.concurrency)

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
    else:
        msgs = list(raw_prompt)
        prompt_text = msgs[-1]["content"] if msgs else ""

    # Instruct model to use \boxed{} format and give numeric answers
    prompt_text = prompt_text.rstrip() + "\n\nPlease solve this step by step. Give your final answer as a simplified decimal or fraction (not a symbolic expression) in \\boxed{}."

    print(f"\nProblem {args.problem_idx}: ground_truth={ground_truth}")
    print(f"Prompt: {prompt_text[:200]}...")

    results = []
    total_attempts = 0

    for batch_i in range(args.max_batches):
        remaining = args.num_correct - len(results)
        if remaining <= 0:
            break

        n_samples = min(args.batch_size, remaining * 5)

        t0 = time.time()
        completions = await generate_batch(
            client, args.model, prompt_text, args.temperature,
            args.max_new_tokens, n_samples, semaphore,
        )
        elapsed = time.time() - t0
        total_attempts += n_samples

        batch_correct = 0
        for i, completion in enumerate(completions):
            if isinstance(completion, Exception):
                print(f"  API error: {completion}")
                continue
            extracted = extract_answer(completion)
            score = compute_score("math", completion, ground_truth)
            if i < 3 and batch_i == 0:  # debug first 3 of first batch
                print(f"  [DEBUG] extracted={extracted}, score={score}, tail={completion[-200:]}")
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


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
