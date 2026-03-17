#!/usr/bin/env python3
"""Extract examples from dsr_sub.parquet for entropy profiling."""

import json

import numpy as np
import pandas as pd
from pathlib import Path

# Mapping: example name -> index in dsr_sub.parquet (0-based)
# Wang et al. name pi_N = rank N by historical variance (descending).
# The actual parquet indices were found by sorting acc_step_500 by std descending.
WANG_EXAMPLES = {
    "pi_1": 124,
    "pi_2": 267,
    "pi_4": 931,
    "pi_7": 568,
    "pi_11": 875,
    "pi_13": 682,
    "pi_16": 50,
    "pi_17": 870,
    "pi_605": 60,
    "pi_606": 906,
    "pi_1201": 847,
    "pi_1207": 174,
    "pi_1208": 770,
    "pi_1209": 789,
}

# MATH500 scores from wang_examples_benchmark.csv
MATH500_SCORES = {
    "pi_1": 74.0,
    "pi_2": 70.6,
    "pi_4": 65.6,
    "pi_7": 64.0,
    "pi_11": 64.0,
    "pi_13": 74.4,
    "pi_16": 67.0,
    "pi_17": 67.2,
    "pi_605": 71.8,
    "pi_606": 64.4,
    "pi_1201": 71.4,
    "pi_1207": 54.0,
    "pi_1208": 45.0,
    "pi_1209": 72.2,
}

PARQUET_PATH = Path(__file__).resolve().parent.parent / "One-Shot-RLVR" / "data" / "train" / "one_shot_rlvr" / "dsr_sub.parquet"
ACC_STEP_500_PATH = Path(__file__).resolve().parent.parent / "One-Shot-RLVR" / "data" / "acc_step_500.json"
LI_PRIME_PARQUET_PATH = Path(__file__).resolve().parent.parent / "polymath-learning" / "data" / "polymath_synthetic_prime" / "train.parquet"

# Li et al. examples: CSV name -> directory name under polymath-learning/data/
LI_EXAMPLES = {
    "natural_geometry": "polymath_natural_geometry",
    "natural_prealgebra": "polymath_natural_prealgebra",
    "natural_algebra": "polymath_natural_algebra",
    "natural_intermediate_algebra": "polymath_natural_inter_algebra",
    "natural_number_theory": "polymath_natural_number_theory",
    "natural_precalculus": "polymath_natural_precalculus",
    "natural_probability": "polymath_natural_counting",
    "synthetic_prime": "polymath_synthetic_prime",
    "pi_1": "pi_1",
}
LI_BENCHMARK_CSV = Path(__file__).resolve().parent.parent / "li_examples_benchmark.csv"
LI_DATA_DIR = Path(__file__).resolve().parent.parent / "polymath-learning" / "data"


def extract_examples(parquet_path: Path = PARQUET_PATH) -> list[dict]:
    """Extract 14 Wang examples with prompt text, ground truth, and MATH500 scores.

    Returns list of dicts with keys:
        name, index, prompt_text, ground_truth, math500_score
    """
    df = pd.read_parquet(parquet_path)

    examples = []
    for name, idx in WANG_EXAMPLES.items():
        row = df[df.apply(lambda r, i=idx: r["extra_info"]["index"] == i, axis=1)].iloc[0]

        # Extract prompt text from the chat-format prompt
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt_text = prompt[0]["content"]

        ground_truth = row["reward_model"]["ground_truth"]

        examples.append({
            "name": name,
            "index": idx,
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "math500_score": MATH500_SCORES[name],
        })

    # Sort by MATH500 score descending
    examples.sort(key=lambda x: x["math500_score"], reverse=True)
    return examples


def extract_examples_large(
    num_examples: int = 400,
    seed: int = 42,
    parquet_path: Path = PARQUET_PATH,
    acc_step_path: Path = ACC_STEP_500_PATH,
) -> list[dict]:
    """Extract a large stratified sample of examples with historical variance.

    Always includes the 14 Wang examples. Remaining slots are stratified-sampled
    from DSR-sub based on historical variance (low/mid/high bins).

    Returns list of dicts with keys:
        name, index, prompt_text, ground_truth, historical_variance, math500_score
    """
    # Load accuracy trajectories and compute historical variance (std)
    with open(acc_step_path) as f:
        acc_data = json.load(f)

    hist_variance = {int(k): float(np.std(v)) for k, v in acc_data.items()}

    # Load parquet
    df = pd.read_parquet(parquet_path)

    # Build index -> row mapping
    all_indices = []
    for _, row in df.iterrows():
        idx = row["extra_info"]["index"]
        all_indices.append(idx)

    wang_indices = set(WANG_EXAMPLES.values())
    wang_name_by_idx = {v: k for k, v in WANG_EXAMPLES.items()}

    # Helper to extract example dict from a row
    def make_example(row, idx):
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt_text = prompt[0]["content"]
        ground_truth = row["reward_model"]["ground_truth"]

        is_wang = idx in wang_indices
        name = wang_name_by_idx[idx] if is_wang else f"ex_{idx}"

        return {
            "name": name,
            "index": idx,
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "historical_variance": hist_variance.get(idx, 0.0),
            "math500_score": MATH500_SCORES.get(wang_name_by_idx.get(idx, ""), None),
        }

    # Extract Wang examples first
    examples = []
    for _, row in df.iterrows():
        idx = row["extra_info"]["index"]
        if idx in wang_indices:
            examples.append(make_example(row, idx))

    # Get non-Wang indices that have historical variance data
    non_wang_indices = [idx for idx in hist_variance.keys() if idx not in wang_indices]
    non_wang_variances = np.array([hist_variance[idx] for idx in non_wang_indices])

    # Stratified sampling: split into 3 bins by variance
    num_remaining = num_examples - len(examples)
    rng = np.random.default_rng(seed)

    sorted_order = np.argsort(non_wang_variances)
    third = len(sorted_order) // 3
    bins = [
        sorted_order[:third],           # low variance
        sorted_order[third:2*third],    # mid variance
        sorted_order[2*third:],         # high variance
    ]

    per_bin = num_remaining // 3
    extra = num_remaining - per_bin * 3  # distribute remainder

    sampled_positions = []
    for i, bin_indices in enumerate(bins):
        n_sample = per_bin + (1 if i < extra else 0)
        n_sample = min(n_sample, len(bin_indices))
        chosen = rng.choice(bin_indices, size=n_sample, replace=False)
        sampled_positions.extend(chosen)

    sampled_indices = set(non_wang_indices[pos] for pos in sampled_positions)

    # Extract sampled examples from dataframe
    for _, row in df.iterrows():
        idx = row["extra_info"]["index"]
        if idx in sampled_indices:
            examples.append(make_example(row, idx))

    # Append Li et al. synthetic Prime example
    prime_example = extract_li_prime()
    if prime_example is not None:
        examples.append(prime_example)

    print(f"Extracted {len(examples)} examples: "
          f"{sum(1 for e in examples if e['math500_score'] is not None)} with MATH500 + "
          f"{sum(1 for e in examples if e['math500_score'] is None)} sampled")

    return examples


def extract_li_prime(parquet_path: Path = LI_PRIME_PARQUET_PATH) -> dict | None:
    """Extract the Li et al. synthetic Prime example (DNA + H-bonds + photons).

    Assumed MATH500 score of 75.0 based on Li et al. results.
    """
    if not parquet_path.exists():
        print(f"WARNING: Li Prime parquet not found at {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)
    row = df.iloc[0]

    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    # Li parquet has system + user messages
    prompt_text = next(m["content"] for m in prompt if m["role"] == "user")

    ground_truth = row["reward_model"]["ground_truth"]

    return {
        "name": "li_prime",
        "index": -1,  # Not from DSR-sub
        "prompt_text": prompt_text,
        "ground_truth": ground_truth,
        "historical_variance": None,
        "math500_score": 75.0,
    }


def extract_li_examples() -> list[dict]:
    """Extract all 9 available Li et al. examples with avg_all scores.

    Returns list of dicts with keys:
        name, prompt_text, ground_truth, avg_all
    """
    # Load benchmark scores
    bench_df = pd.read_csv(LI_BENCHMARK_CSV)
    bench_by_name = {row["example"]: row for _, row in bench_df.iterrows()}

    examples = []
    for csv_name, dir_name in LI_EXAMPLES.items():
        parquet_path = LI_DATA_DIR / dir_name / "train.parquet"
        if not parquet_path.exists():
            print(f"WARNING: Li parquet not found at {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        row = df.iloc[0]

        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        # Li parquets have system + user messages
        prompt_text = next(m["content"] for m in prompt if m["role"] == "user")

        ground_truth = row["reward_model"]["ground_truth"]

        bench_row = bench_by_name.get(csv_name)
        avg_all = float(bench_row["avg_all"]) if bench_row is not None else None

        examples.append({
            "name": csv_name,
            "prompt_text": prompt_text,
            "ground_truth": ground_truth,
            "avg_all": avg_all,
        })

    # Sort by avg_all descending
    examples.sort(key=lambda x: x["avg_all"] or 0, reverse=True)
    print(f"Extracted {len(examples)} Li examples")
    return examples


if __name__ == "__main__":
    import sys

    if "--large" in sys.argv:
        examples = extract_examples_large()
        for ex in examples[:20]:
            wang_tag = f"MATH500={ex['math500_score']:.1f}" if ex['math500_score'] else "sampled"
            print(f"{ex['name']:>10} (idx={ex['index']:>4d}) | hvar={ex['historical_variance']:.4f} | {wang_tag}")
        if len(examples) > 20:
            print(f"  ... and {len(examples) - 20} more")
    elif "--li" in sys.argv:
        examples = extract_li_examples()
        for ex in examples:
            avg_str = f"avg_all={ex['avg_all']:.1f}" if ex['avg_all'] else "N/A"
            print(f"{ex['name']:>30} | {avg_str} | gt={ex['ground_truth'][:30]}")
            print(f"{'':>30}   prompt: {ex['prompt_text'][:80]}...")
            print()
    else:
        examples = extract_examples()
        for ex in examples:
            print(f"{ex['name']:>10} (idx={ex['index']:>4d}) | MATH500={ex['math500_score']:.1f} | gt={ex['ground_truth'][:30]}")
            print(f"           prompt: {ex['prompt_text'][:80]}...")
            print()
