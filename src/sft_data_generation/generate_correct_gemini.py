"""Generate a dataset of correct answers using Google Gemini API.

Sequential requests with rate-limit-aware pacing for the free tier (15 RPM).
"""

import argparse
import asyncio
import os
import sys
import time

import pandas as pd
from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlvr_grokking.rewards.deepscaler_reward import compute_score, extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correct answers via Gemini API")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_correct", type=int, default=128)
    parser.add_argument("--max_attempts", type=int, default=500, help="Max total API calls")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="data/sft_1shot_datasets/gemini")
    parser.add_argument("--problem_idx", type=int, default=0)
    parser.add_argument("--rpm", type=int, default=14, help="Requests per minute (stay under 15 free tier limit)")
    return parser.parse_args()


async def call_gemini(client, model, prompt_text, temperature, max_tokens, max_retries=5):
    """Make a single Gemini API call with retry on rate limits."""
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt_text,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RATE" in err_str.upper() or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                wait = 2 ** attempt * 10
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries}): {err_str[:150]}")
                await asyncio.sleep(wait)
            else:
                print(f"  Non-rate-limit error: {err_str[:300]}")
                return e
    return Exception(f"Max retries ({max_retries}) exceeded due to rate limiting")


async def async_main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    delay = 60.0 / args.rpm  # seconds between requests

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Rate: {args.rpm} RPM (delay: {delay:.1f}s between requests)")
    print(f"Target: {args.num_correct} correct, max attempts: {args.max_attempts}")

    client = genai.Client()

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

    # Instruct model to use \boxed{} format for answer extraction
    prompt_text = prompt_text.rstrip() + "\n\nPlease solve this step by step and put your final answer in \\boxed{}."

    print(f"\nProblem {args.problem_idx}: ground_truth={ground_truth}")
    print(f"Prompt: {prompt_text[:200]}...")

    results = []
    total_attempts = 0
    errors = 0
    t_start = time.time()

    for i in range(args.max_attempts):
        if len(results) >= args.num_correct:
            break

        completion = await call_gemini(
            client, args.model, prompt_text, args.temperature, args.max_new_tokens,
        )
        total_attempts += 1

        if isinstance(completion, Exception):
            errors += 1
            print(f"  [{total_attempts}] API error: {completion}")
            await asyncio.sleep(delay)
            continue

        if completion is None:
            errors += 1
            print(f"  [{total_attempts}] Empty response (safety filter?)")
            await asyncio.sleep(delay)
            continue

        extracted = extract_answer(completion)
        score = compute_score("math", completion, ground_truth)

        if total_attempts <= 3:  # debug first few
            print(f"  [DEBUG] extracted={extracted}, score={score}, tail={completion[-200:]}")

        if score > 0:
            results.append({
                "problem_idx": args.problem_idx,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "completion": completion,
                "extracted_answer": extracted,
                "attempt_num": total_attempts,
            })

        if total_attempts % 10 == 0 or score > 0:
            elapsed = time.time() - t_start
            print(f"  [{total_attempts}] correct: {len(results)}/{args.num_correct}, "
                  f"errors: {errors}, elapsed: {elapsed:.0f}s")

        # Pace requests to stay under RPM limit
        await asyncio.sleep(delay)

    elapsed = time.time() - t_start
    print(f"\nCollected {len(results)}/{args.num_correct} correct in {total_attempts} attempts "
          f"({errors} errors, {elapsed:.0f}s)")

    if results:
        out_df = pd.DataFrame(results[:args.num_correct])
        out_path = os.path.join(args.output_dir, f"problem_{args.problem_idx:04d}.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
