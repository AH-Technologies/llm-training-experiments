"""Compute metrics on saved rollouts.

Loads rollout jsonl files, extracts answers, runs all registered metrics,
and outputs a CSV with one row per example and metric columns.

Usage:
    python compute_metrics.py \
        --rollout_dir results/rollouts \
        --output results/metrics.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path for reward imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import ExampleRollouts, compute_all_metrics
from src.rlvr_grokking.rewards.deepscaler_reward import extract_answer, grade_answer_mathd, grade_answer_sympy


def load_rollouts(rollout_path: Path) -> tuple[dict, list[str]]:
    """Load a single example's rollouts from jsonl.

    Returns (metadata_dict, list_of_completions).
    """
    metadata = None
    completions = []

    with open(rollout_path) as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "metadata":
                metadata = record
            elif record["type"] == "rollout":
                completions.append(record["completion"])

    return metadata, completions


def extract_and_grade(completions: list[str], ground_truth: str) -> tuple[list[str | None], list[bool]]:
    """Extract boxed answers and grade correctness for each completion."""
    extracted = []
    correct = []

    for completion in completions:
        answer = extract_answer(completion)
        extracted.append(answer)

        if answer is None:
            correct.append(False)
        else:
            is_correct = (
                grade_answer_mathd(answer, ground_truth)
                or grade_answer_sympy(answer, ground_truth)
            )
            correct.append(is_correct)

    return extracted, correct


def process_example(rollout_path: Path) -> dict[str, float] | None:
    """Load rollouts for one example, compute all metrics."""
    metadata, completions = load_rollouts(rollout_path)
    if metadata is None:
        print(f"Warning: no metadata in {rollout_path}, skipping", file=sys.stderr)
        return None

    extracted, correct = extract_and_grade(completions, metadata["ground_truth"])

    rollouts = ExampleRollouts(
        example_id=metadata["example_name"],
        prompt=metadata["prompt"],
        ground_truth=metadata["ground_truth"],
        completions=completions,
        extracted_answers=extracted,
        is_correct=correct,
    )

    metrics = compute_all_metrics(rollouts)
    metrics["example"] = metadata["example_name"]
    metrics["ground_truth"] = metadata["ground_truth"]
    metrics["k"] = len(completions)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute metrics on saved rollouts")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory containing rollout jsonl files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <rollout_dir>/metrics.csv)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    rollout_files = sorted(rollout_dir.glob("pi_*.jsonl"))

    if not rollout_files:
        print(f"No rollout files found in {rollout_dir}")
        return

    print(f"Found {len(rollout_files)} rollout files in {rollout_dir}")

    all_results = []
    for path in rollout_files:
        print(f"  Processing {path.name}...")
        result = process_example(path)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No results to write.")
        return

    # Write CSV
    output_path = Path(args.output) if args.output else rollout_dir / "metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all metric keys (preserving order)
    info_keys = ["example", "ground_truth", "k"]
    metric_keys = [k for k in all_results[0] if k not in info_keys]
    fieldnames = info_keys + sorted(metric_keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nMetrics written to {output_path}")

    # Print summary table
    print(f"\n{'Example':<15} {'pass_rate':>10} {'entropy':>10} {'wrong_ent':>10} {'uniq_ans':>10} {'uniq_wrong':>10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['example']:<15} {r.get('pass_rate', 0):>10.3f} {r.get('answer_entropy', 0):>10.3f} "
              f"{r.get('wrong_answer_entropy', 0):>10.3f} {r.get('unique_answer_count', 0):>10.0f} "
              f"{r.get('unique_wrong_answer_count', 0):>10.0f}")


if __name__ == "__main__":
    main()
