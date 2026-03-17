"""Generate rollouts for one-shot RLVR metric evaluation.

Runs vLLM inference on a set of examples, producing k completions per example.
Saves raw rollouts to jsonl files for downstream metric computation.

The --examples flag takes pi-numbers (Wang et al. ranking), e.g. 1, 13, 1207.
These are mapped to parquet indices via acc_step_500.json ranking (sorted by std dev).
If --examples is omitted, all 1209 examples are processed.

Usage:
    python compute_rollouts.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --data_path One-Shot-RLVR/data/train/one_shot_rlvr/dsr_sub.parquet \
        --ranking_path One-Shot-RLVR/data/acc_step_500.json \
        --k 128 \
        --examples 1 13 1207 1208 \
        --output_dir results/run_01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


def build_pi_to_index_mapping(ranking_path: str) -> dict[int, int]:
    """Build mapping from pi-number (1-indexed rank) to parquet index.

    Wang et al. rank examples by std dev of accuracy curve (descending).
    pi_1 = highest std, pi_2 = second highest, etc.
    """
    with open(ranking_path) as f:
        data = json.load(f)

    scores = {k: np.std(v) for k, v in data.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # pi_N is rank N (1-indexed), maps to parquet index
    return {rank: int(parquet_idx) for rank, (parquet_idx, _) in enumerate(ranked, start=1)}


def load_examples(
    data_path: str,
    ranking_path: str,
    pi_numbers: list[int] | None,
) -> list[dict]:
    """Load examples from parquet, optionally filtering by pi-number.

    Returns deduplicated examples with fields: pi_number, parquet_index, prompt, ground_truth, metadata.
    """
    pi_to_idx = build_pi_to_index_mapping(ranking_path)
    df = pd.read_parquet(data_path)

    # Build index -> row lookup
    index_to_row = {}
    for _, row in df.iterrows():
        idx = row.get("extra_info", {}).get("index", None)
        if idx is not None:
            index_to_row[idx] = row

    # Determine which pi-numbers to process
    if pi_numbers is not None:
        target_pis = pi_numbers
    else:
        target_pis = sorted(pi_to_idx.keys())

    examples = []
    for pi_num in target_pis:
        if pi_num not in pi_to_idx:
            print(f"Warning: pi_{pi_num} not in ranking (max={len(pi_to_idx)})", file=sys.stderr)
            continue

        parquet_idx = pi_to_idx[pi_num]
        if parquet_idx not in index_to_row:
            print(f"Warning: pi_{pi_num} (parquet index {parquet_idx}) not found in data", file=sys.stderr)
            continue

        row = index_to_row[parquet_idx]
        prompt_raw = row["prompt"]
        prompt_content = prompt_raw[0]["content"] if isinstance(prompt_raw, (list, np.ndarray)) else str(prompt_raw)
        ground_truth = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else str(row["reward_model"])

        examples.append({
            "pi_number": pi_num,
            "parquet_index": parquet_idx,
            "prompt": prompt_content,
            "ground_truth": ground_truth,
            "data_source": row.get("data_source", "math"),
            "metadata": dict(row.get("extra_info", {})),
        })

    examples.sort(key=lambda e: e["pi_number"])
    print(f"Loaded {len(examples)} examples from {data_path}")
    for e in examples:
        print(f"  pi_{e['pi_number']} (idx={e['parquet_index']}): gt={e['ground_truth']}")
    return examples


def generate_rollouts(
    model_name: str,
    examples: list[dict],
    k: int,
    temperature: float = 0.6,
    max_tokens: int = 3072,
    tensor_parallel_size: int = 1,
) -> dict[int, list[str]]:
    """Run vLLM inference, k completions per example. Keyed by pi_number."""
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    prompts = [e["prompt"] for e in examples]
    print(f"Generating {k} rollouts for {len(prompts)} examples...")

    outputs = llm.generate(prompts, sampling_params)

    results = {}
    for example, output in zip(examples, outputs):
        completions = [o.text for o in output.outputs]
        results[example["pi_number"]] = completions

    return results


def save_rollouts(
    examples: list[dict],
    rollouts: dict[int, list[str]],
    output_dir: Path,
    model_name: str,
    k: int,
):
    """Save rollouts as one jsonl file per example."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for example in examples:
        pi_num = example["pi_number"]
        completions = rollouts[pi_num]
        out_path = output_dir / f"pi_{pi_num}.jsonl"

        with open(out_path, "w") as f:
            # Write metadata as first line
            meta = {
                "type": "metadata",
                "pi_number": pi_num,
                "parquet_index": example["parquet_index"],
                "example_name": f"pi_{pi_num}",
                "prompt": example["prompt"],
                "ground_truth": example["ground_truth"],
                "model": model_name,
                "k": k,
                "data_source": example["data_source"],
            }
            f.write(json.dumps(meta) + "\n")

            for i, completion in enumerate(completions):
                record = {
                    "type": "rollout",
                    "rollout_index": i,
                    "completion": completion,
                }
                f.write(json.dumps(record) + "\n")

        print(f"  Saved {len(completions)} rollouts to {out_path}")

    # Save run config
    config = {
        "model": model_name,
        "k": k,
        "num_examples": len(examples),
        "pi_numbers": [e["pi_number"] for e in examples],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate rollouts for one-shot RLVR metrics")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HuggingFace model name or path")
    parser.add_argument("--data_path", type=str,
                        default="One-Shot-RLVR/data/train/one_shot_rlvr/dsr_sub.parquet",
                        help="Path to parquet file with all examples")
    parser.add_argument("--ranking_path", type=str,
                        default="One-Shot-RLVR/data/acc_step_500.json",
                        help="Path to acc_step_500.json for pi-number ranking")
    parser.add_argument("--k", type=int, default=128,
                        help="Number of rollouts per example")
    parser.add_argument("--examples", type=int, nargs="*", default=None,
                        help="Pi-numbers to process (e.g. 1 13 1207). If omitted, process all.")
    parser.add_argument("--output_dir", type=str, default="results/rollouts",
                        help="Directory to save rollout outputs")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    examples = load_examples(args.data_path, args.ranking_path, args.examples)
    if not examples:
        print("No examples to process. Exiting.")
        return

    rollouts = generate_rollouts(
        model_name=args.model,
        examples=examples,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    output_dir = Path(args.output_dir)
    save_rollouts(examples, rollouts, output_dir, args.model, args.k)
    print(f"\nDone. Rollouts saved to {output_dir}")


if __name__ == "__main__":
    main()
