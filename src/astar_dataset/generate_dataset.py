"""Main dataset generation script for A* grokking experiments.

Usage:
    # Token length test (run first!)
    python -m astar_dataset.generate_dataset --test-tokens --num-samples 50

    # Full generation
    python -m astar_dataset.generate_dataset \
        --grid-sizes 7,7 9,9 \
        --obstacle-ratios 0.20 0.25 0.30 \
        --num-instances 1000 \
        --train-fraction 0.40 \
        --output-dir data/astar_grokking_dataset \
        --seed 42
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .config import DatasetConfig
from .grid import generate_grid, solve_astar
from .trace_formatter import format_messages, format_trace


def generate_single_instance(
    rows: int,
    cols: int,
    obstacle_ratio: float,
    seed: int,
    config: DatasetConfig,
) -> dict | None:
    """Generate a single dataset instance.

    Returns dict with all columns, or None on failure.
    """
    grid = generate_grid(rows, cols, obstacle_ratio, seed)
    if grid is None:
        return None

    result = solve_astar(grid)
    if result is None:
        return None

    messages = format_messages(grid, result, config)
    path = result["path"]
    path_str = " ".join(f"({r},{c})" for r, c in path)

    instance_id = f"grid_{rows}x{cols}_obs{obstacle_ratio:.2f}_seed{seed:05d}"

    return {
        "id": instance_id,
        "messages": messages,
        "grid_rows": rows,
        "grid_cols": cols,
        "obstacle_ratio": grid.obstacle_ratio,
        "start": f"({grid.start[0]},{grid.start[1]})",
        "goal": f"({grid.goal[0]},{grid.goal[1]})",
        "optimal_path": path_str,
        "optimal_path_length": len(path) - 1,
        "num_search_steps": result["num_steps"],
        "grid_string": grid.to_string(),
        "astar_trace_json": json.dumps(result["trace"], default=str),
        "seed": seed,
    }


def generate_instances_for_config(
    rows: int,
    cols: int,
    obstacle_ratio: float,
    num_instances: int,
    base_seed: int,
    config: DatasetConfig,
) -> list[dict]:
    """Generate all instances for one (grid_size, obstacle_ratio) configuration."""
    instances = []
    seed = base_seed
    retries = 0
    max_retries = num_instances * config.max_retries_per_instance

    while len(instances) < num_instances and retries < max_retries:
        inst = generate_single_instance(rows, cols, obstacle_ratio, seed, config)
        if inst is not None:
            instances.append(inst)
        else:
            retries += 1
        seed += 1

    return instances


def split_instances(
    instances: list[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split instances into train/val/test sets."""
    rng = random.Random(seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def estimate_token_length(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English + math."""
    return len(text) // 4


def run_token_length_test(config: DatasetConfig, num_samples: int = 50) -> None:
    """Generate sample instances and report token length statistics."""
    print("=" * 60)
    print("TOKEN LENGTH TEST")
    print("=" * 60)
    print(f"Generating {num_samples} samples per configuration...\n")

    for rows, cols in config.grid_sizes:
        for obs_ratio in config.obstacle_ratios:
            lengths = []
            assistant_lengths = []
            seed = config.seed

            generated = 0
            while generated < num_samples:
                grid = generate_grid(rows, cols, obs_ratio, seed)
                seed += 1
                if grid is None:
                    continue

                result = solve_astar(grid)
                if result is None:
                    continue

                messages = format_messages(grid, result, config)
                # Total length = all messages concatenated
                total_text = "".join(m["content"] for m in messages)
                assistant_text = messages[2]["content"]

                lengths.append(estimate_token_length(total_text))
                assistant_lengths.append(estimate_token_length(assistant_text))
                generated += 1

            lengths.sort()
            assistant_lengths.sort()

            p50 = lengths[len(lengths) // 2]
            p95 = lengths[int(len(lengths) * 0.95)]

            print(f"  {rows}x{cols}, obstacles={obs_ratio:.0%}:")
            print(f"    Total tokens  — min={lengths[0]}, median={p50}, "
                  f"mean={sum(lengths)//len(lengths)}, "
                  f"p95={p95}, max={lengths[-1]}")

            a_p50 = assistant_lengths[len(assistant_lengths) // 2]
            a_p95 = assistant_lengths[int(len(assistant_lengths) * 0.95)]
            print(f"    Asst. tokens  — min={assistant_lengths[0]}, "
                  f"median={a_p50}, "
                  f"mean={sum(assistant_lengths)//len(assistant_lengths)}, "
                  f"p95={a_p95}, max={assistant_lengths[-1]}")

            over_budget = sum(1 for l in lengths if l > config.max_token_length)
            print(f"    Over {config.max_token_length} tokens: "
                  f"{over_budget}/{num_samples} "
                  f"({100*over_budget/num_samples:.1f}%)")

            # Show a sample search step count
            print(f"    Search steps  — "
                  f"min to max across samples (from A* trace)")
            print()

    print("=" * 60)
    print("If many instances exceed the token budget, consider:")
    print("  - Reducing grid size")
    print("  - Reducing show_top_k_candidates")
    print("  - Increasing max_token_length")
    print("=" * 60)


def tokenizer_length(messages: list[dict], tokenizer) -> int:
    """Compute exact token length using the real tokenizer with chat template."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text))


def generate_full_dataset(config: DatasetConfig, model_name: str = "Qwen/Qwen2.5-1.5B") -> None:
    """Generate the complete dataset and save to disk."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for accurate length filtering
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        print(f"Loaded tokenizer: {model_name} (filtering at {config.max_token_length} tokens)")
    except Exception as e:
        print(f"WARNING: Could not load tokenizer ({e}), using heuristic filtering")

    all_instances = []
    config_seed_offset = 0

    print(f"Generating dataset with {config.num_instances_per_config} "
          f"instances per configuration...")
    print(f"Grid sizes: {config.grid_sizes}")
    print(f"Obstacle ratios: {config.obstacle_ratios}")
    print()

    for rows, cols in config.grid_sizes:
        for obs_ratio in config.obstacle_ratios:
            base_seed = config.seed + config_seed_offset
            config_seed_offset += config.num_instances_per_config * 100

            t0 = time.time()
            instances = generate_instances_for_config(
                rows, cols, obs_ratio,
                config.num_instances_per_config,
                base_seed, config,
            )
            elapsed = time.time() - t0

            all_instances.extend(instances)
            print(f"  {rows}x{cols} obs={obs_ratio:.0%}: "
                  f"{len(instances)} instances in {elapsed:.1f}s")

    print(f"\nTotal before filtering: {len(all_instances)}")

    # Filter by actual token length
    if tokenizer is not None:
        filtered = []
        token_lengths = []
        for inst in all_instances:
            tlen = tokenizer_length(inst["messages"], tokenizer)
            if tlen <= config.max_token_length:
                inst["token_length"] = tlen
                filtered.append(inst)
                token_lengths.append(tlen)

        removed = len(all_instances) - len(filtered)
        print(f"Filtered out {removed} instances exceeding {config.max_token_length} tokens")
        all_instances = filtered

        if token_lengths:
            token_lengths.sort()
            print(f"Token lengths: min={token_lengths[0]}, "
                  f"median={token_lengths[len(token_lengths)//2]}, "
                  f"max={token_lengths[-1]}")
    else:
        over_budget = 0
        for inst in all_instances:
            total_text = "".join(m["content"] for m in inst["messages"])
            if estimate_token_length(total_text) > config.max_token_length:
                over_budget += 1
        if over_budget > 0:
            print(f"WARNING: ~{over_budget}/{len(all_instances)} instances "
                  f"may exceed {config.max_token_length} token budget (heuristic)")

    print(f"Total after filtering: {len(all_instances)}")

    # Split
    train, val, test = split_instances(
        all_instances,
        config.train_fraction,
        config.val_fraction,
        config.seed,
    )
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Path length stats
    path_lengths = [inst["optimal_path_length"] for inst in all_instances]
    path_lengths.sort()
    print(f"\nPath length stats: min={path_lengths[0]}, "
          f"median={path_lengths[len(path_lengths)//2]}, "
          f"max={path_lengths[-1]}")

    search_steps = [inst["num_search_steps"] for inst in all_instances]
    search_steps.sort()
    print(f"Search step stats: min={search_steps[0]}, "
          f"median={search_steps[len(search_steps)//2]}, "
          f"max={search_steps[-1]}")

    # Save as parquet
    try:
        import pandas as pd

        for split_name, split_data in [
            ("train", train), ("val", val), ("test", test)
        ]:
            df = pd.DataFrame(split_data)
            path = output_dir / f"astar_{split_name}.parquet"
            df.to_parquet(path)
            print(f"Saved {path} ({len(split_data)} rows)")

    except ImportError:
        print("pandas not available, saving as JSON instead")
        for split_name, split_data in [
            ("train", train), ("val", val), ("test", test)
        ]:
            path = output_dir / f"astar_{split_name}.json"
            with open(path, "w") as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {path} ({len(split_data)} rows)")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved config to {config_path}")


def parse_grid_size(s: str) -> tuple[int, int]:
    """Parse 'R,C' into (R, C)."""
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Generate A* grokking dataset"
    )
    parser.add_argument(
        "--test-tokens", action="store_true",
        help="Run token length test only (no full generation)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Samples per config for token test (default: 50)",
    )
    parser.add_argument(
        "--grid-sizes", nargs="+", type=parse_grid_size,
        default=None,
        help="Grid sizes as R,C pairs (e.g., 7,7 9,9)",
    )
    parser.add_argument(
        "--obstacle-ratios", nargs="+", type=float,
        default=None,
        help="Obstacle ratios (e.g., 0.20 0.25 0.30)",
    )
    parser.add_argument(
        "--num-instances", type=int, default=None,
        help="Instances per (size, ratio) config",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=None,
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k candidates to show per step")
    parser.add_argument("--max-token-length", type=int, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Model name for tokenizer-based length filtering")

    args = parser.parse_args()

    config = DatasetConfig()

    # Override config from CLI args
    if args.grid_sizes is not None:
        config.grid_sizes = args.grid_sizes
    if args.obstacle_ratios is not None:
        config.obstacle_ratios = args.obstacle_ratios
    if args.num_instances is not None:
        config.num_instances_per_config = args.num_instances
    if args.train_fraction is not None:
        config.train_fraction = args.train_fraction
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.top_k is not None:
        config.show_top_k_candidates = args.top_k
    if args.max_token_length is not None:
        config.max_token_length = args.max_token_length

    if args.test_tokens:
        run_token_length_test(config, num_samples=args.num_samples)
    else:
        generate_full_dataset(config, model_name=args.model)


if __name__ == "__main__":
    main()
