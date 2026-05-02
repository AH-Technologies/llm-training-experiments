"""Generate correct answers via best-of-N sampling (log-likelihood reranking).

Approximates power sampling by generating many samples at temp=1,
scoring by total log-likelihood, and keeping the top correct ones.

Selecting top-k by log p(x) from N samples approximates sampling from p^alpha
as N grows.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlvr_grokking.rewards.deepscaler_reward import compute_score, extract_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate correct answers via best-of-N")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_correct", type=int, default=1024, help="Correct answers to collect before ranking")
    parser.add_argument("--num_keep", type=int, default=128, help="Top-k by log-likelihood to keep")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="sft_data_generation/results/generate_best_of_n")
    parser.add_argument("--problem_idx", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=4096, help="Samples per vLLM call")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Best-of-N sampling: collect {args.num_correct} correct, keep top {args.num_keep} by log-likelihood")
    print(f"Temperature: {args.temperature}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    print(f"Loading vLLM model: {args.model}")
    vllm_model = LLM(
        model=args.model, dtype=args.dtype, trust_remote_code=True,
        gpu_memory_utilization=0.9, enable_prefix_caching=True,
        tensor_parallel_size=args.tp,
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
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=1,  # get log-prob for each generated token
    )

    # Generate in batches until we have enough correct
    results = []
    total_samples = 0
    batch_idx = 0

    while len(results) < args.num_correct:
        prompts = [prompt_str] * args.batch_size

        print(f"\n  Batch {batch_idx}: generating {args.batch_size} samples...", end=" ", flush=True)
        t0 = time.time()
        outputs = vllm_model.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - t0
        total_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"{elapsed:.1f}s ({total_tok/elapsed:.0f} tok/s)")

        total_samples += len(outputs)
        batch_correct = 0

        for out in outputs:
            completion = out.outputs[0].text
            token_ids = out.outputs[0].token_ids
            logprobs = out.outputs[0].logprobs

            score = compute_score("math", completion, ground_truth)
            if score <= 0:
                continue

            batch_correct += 1
            log_ll = sum(
                lp_dict[token_ids[i]].logprob
                for i, lp_dict in enumerate(logprobs)
            )

            results.append({
                "problem_idx": args.problem_idx,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "completion": completion,
                "extracted_answer": extract_answer(completion),
                "log_likelihood": log_ll,
                "log_likelihood_per_token": log_ll / len(token_ids),
                "num_tokens": len(token_ids),
            })

        print(f"  {batch_correct} correct this batch, {len(results)}/{args.num_correct} total")
        batch_idx += 1

    print(f"\nCollected {len(results)} correct in {total_samples} total samples "
          f"({100*len(results)/total_samples:.1f}% pass rate)")

    # Sort by log-likelihood (highest first)
    results.sort(key=lambda r: r["log_likelihood"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i

    # Save all correct (sorted by log-likelihood)
    if results:
        out_df = pd.DataFrame(results)
        all_path = os.path.join(args.output_dir, f"problem_{args.problem_idx:04d}_all.parquet")
        out_df.to_parquet(all_path, index=False)
        print(f"Saved {len(results)} correct (all, ranked) to {all_path}")

        # Save top-k
        top_results = results[:args.num_keep]
        top_df = pd.DataFrame(top_results)
        top_path = os.path.join(args.output_dir, f"problem_{args.problem_idx:04d}.parquet")
        top_df.to_parquet(top_path, index=False)
        print(f"Saved top {len(top_results)} to {top_path}")

        # Stats
        ll = np.array([r["log_likelihood"] for r in results])
        ll_per_tok = np.array([r["log_likelihood_per_token"] for r in results])
        print(f"\nLog-likelihood stats ({len(results)} correct):")
        print(f"  Total:     mean={ll.mean():.1f}, std={ll.std():.1f}, "
              f"min={ll.min():.1f}, max={ll.max():.1f}")
        print(f"  Per-token: mean={ll_per_tok.mean():.3f}, std={ll_per_tok.std():.3f}, "
              f"min={ll_per_tok.min():.3f}, max={ll_per_tok.max():.3f}")

        top_ll = np.array([r["log_likelihood"] for r in top_results])
        rest_ll = np.array([r["log_likelihood"] for r in results[args.num_keep:]])
        print(f"\n  Top-{args.num_keep} mean ll: {top_ll.mean():.1f}")
        if len(rest_ll) > 0:
            print(f"  Rest mean ll:     {rest_ll.mean():.1f}")
    else:
        print("No correct answers found!")


if __name__ == "__main__":
    main()
