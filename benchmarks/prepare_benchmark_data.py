#!/usr/bin/env python3
"""
Prepare benchmark datasets for SFT and GRPO throughput testing.

Downloads OpenMathInstruct-2 for SFT and analyzes token lengths
to enable accurate throughput calculations.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


def download_openmath_sft(
    output_dir: Path,
    num_samples: int = 10000,
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B",
    max_length: int = 4096,
):
    """
    Download and prepare OpenMathInstruct-2 for SFT benchmarking.

    Args:
        output_dir: Where to save the parquet files
        num_samples: Number of samples to download
        tokenizer_name: Tokenizer to use for length analysis
        max_length: Maximum sequence length (prompt + response)
    """
    print(f"Loading OpenMathInstruct-2 dataset...")

    # Load dataset (streaming to avoid downloading all 14M samples)
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train",
        streaming=True,
    )

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Collect samples
    samples = []
    prompt_lengths = []
    response_lengths = []

    print(f"Collecting {num_samples} samples...")
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        problem = item.get("problem", "")
        solution = item.get("generated_solution", "")

        if not problem or not solution:
            continue

        # Tokenize to get lengths
        prompt_tokens = tokenizer.encode(problem, add_special_tokens=False)
        response_tokens = tokenizer.encode(solution, add_special_tokens=False)

        total_len = len(prompt_tokens) + len(response_tokens)

        # Skip if too long
        if total_len > max_length:
            continue

        # Store in verl-compatible format for SFT
        samples.append({
            "prompt": problem,
            "response": solution,
            "prompt_length": len(prompt_tokens),
            "response_length": len(response_tokens),
            "total_length": total_len,
        })

        prompt_lengths.append(len(prompt_tokens))
        response_lengths.append(len(response_tokens))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} samples, kept {len(samples)}")

    print(f"\nCollected {len(samples)} samples")

    # Calculate statistics
    stats = {
        "num_samples": len(samples),
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "prompt_length": {
            "mean": sum(prompt_lengths) / len(prompt_lengths),
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "median": sorted(prompt_lengths)[len(prompt_lengths) // 2],
        },
        "response_length": {
            "mean": sum(response_lengths) / len(response_lengths),
            "min": min(response_lengths),
            "max": max(response_lengths),
            "median": sorted(response_lengths)[len(response_lengths) // 2],
        },
        "total_length": {
            "mean": sum(prompt_lengths) / len(prompt_lengths) + sum(response_lengths) / len(response_lengths),
        },
        "response_ratio": sum(response_lengths) / (sum(prompt_lengths) + sum(response_lengths)),
    }

    print(f"\n=== Token Length Statistics ===")
    print(f"Prompt length:   mean={stats['prompt_length']['mean']:.0f}, "
          f"median={stats['prompt_length']['median']}, "
          f"range=[{stats['prompt_length']['min']}, {stats['prompt_length']['max']}]")
    print(f"Response length: mean={stats['response_length']['mean']:.0f}, "
          f"median={stats['response_length']['median']}, "
          f"range=[{stats['response_length']['min']}, {stats['response_length']['max']}]")
    print(f"Response ratio:  {stats['response_ratio']:.1%} of tokens are response (trained)")

    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(samples)

    # Save train/val split
    train_size = int(len(df) * 0.9)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    train_path = output_dir / "openmath_sft_train.parquet"
    val_path = output_dir / "openmath_sft_val.parquet"
    stats_path = output_dir / "openmath_sft_stats.json"

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Files Saved ===")
    print(f"Train: {train_path} ({len(train_df)} samples)")
    print(f"Val:   {val_path} ({len(val_df)} samples)")
    print(f"Stats: {stats_path}")

    return stats


def analyze_existing_data(data_path: Path, tokenizer_name: str = "Qwen/Qwen2.5-1.5B"):
    """Analyze token lengths in existing GRPO dataset."""

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)

    print(f"Columns: {list(df.columns)}")
    print(f"Samples: {len(df)}")

    # Check format
    if "prompt" in df.columns:
        first_prompt = df.iloc[0]["prompt"]
        if isinstance(first_prompt, list):
            # Multi-turn format
            print("Format: Multi-turn (list of messages)")
            prompts = [p[0]["content"] if isinstance(p, list) else str(p) for p in df["prompt"]]
        else:
            print("Format: Single-turn")
            prompts = df["prompt"].tolist()
    else:
        print("No 'prompt' column found")
        return

    # Tokenize and analyze
    lengths = []
    for prompt in prompts[:100]:  # Sample first 100
        tokens = tokenizer.encode(str(prompt), add_special_tokens=False)
        lengths.append(len(tokens))

    print(f"\nPrompt token lengths (first 100 samples):")
    print(f"  Mean: {sum(lengths) / len(lengths):.0f}")
    print(f"  Min:  {min(lengths)}")
    print(f"  Max:  {max(lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download OpenMathInstruct-2")
    download_parser.add_argument("--output-dir", type=Path, default=Path("data/benchmark"))
    download_parser.add_argument("--num-samples", type=int, default=10000)
    download_parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B")
    download_parser.add_argument("--max-length", type=int, default=4096)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing dataset")
    analyze_parser.add_argument("data_path", type=Path)
    analyze_parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B")

    args = parser.parse_args()

    if args.command == "download":
        download_openmath_sft(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
        )
    elif args.command == "analyze":
        analyze_existing_data(args.data_path, args.tokenizer)
    else:
        parser.print_help()
