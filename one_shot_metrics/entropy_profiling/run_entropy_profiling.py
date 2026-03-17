#!/usr/bin/env python3
"""Main orchestrator for per-token entropy profiling of Wang et al. examples."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from extract_examples import extract_examples, extract_examples_large, extract_li_examples
from compute_entropy import run_entropy_profiling
from entropy_features import extract_all_features
from visualize_entropy import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Per-Token Entropy Profiling for One-Shot RLVR Examples")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="HuggingFace model name")
    parser.add_argument("--num-rollouts", type=int, default=32, help="Number of rollouts per example")
    parser.add_argument("--max-new-tokens", type=int, default=3072, help="Max new tokens per rollout")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--large", action="store_true", help="Use large stratified sample from DSR-sub instead of 14 Wang examples")
    parser.add_argument("--li", action="store_true", help="Profile 9 Li et al. examples on 7B model")
    parser.add_argument("--num-examples", type=int, default=400, help="Number of examples for --large mode")
    parser.add_argument("--visualize-only", action="store_true", help="Skip inference, only generate plots from existing results")
    args = parser.parse_args()

    if args.large and args.li:
        parser.error("--large and --li are mutually exclusive")

    # Auto-set model for Li mode
    if args.li and args.model == "Qwen/Qwen2.5-Math-1.5B":
        args.model = "Qwen/Qwen2.5-7B"

    output_dir = Path(args.output_dir)
    if args.large:
        output_dir = output_dir / "large_run"
    elif args.li:
        output_dir = output_dir / "li_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "entropy_profiles"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract examples
    print("=" * 60)
    if args.li:
        print("Step 1: Extracting Li et al. examples")
    elif args.large:
        print(f"Step 1: Extracting {args.num_examples} stratified examples from DSR-sub")
    else:
        print("Step 1: Extracting Wang et al. examples")
    print("=" * 60)
    if args.li:
        examples = extract_li_examples()
    elif args.large:
        examples = extract_examples_large(num_examples=args.num_examples, seed=args.seed)
    else:
        examples = extract_examples()
    print(f"Extracted {len(examples)} examples")
    for ex in examples[:20]:
        if args.li:
            score_str = f"avg_all={ex['avg_all']:.1f}" if ex.get('avg_all') else "N/A"
        else:
            score_str = f"MATH500={ex['math500_score']:.1f}" if ex.get('math500_score') else "sampled"
        print(f"  {ex['name']:>30}: {score_str}, gt={ex['ground_truth'][:20]}")
    if len(examples) > 20:
        print(f"  ... and {len(examples) - 20} more")
    print()

    if not args.visualize_only:
        # Step 2: Run entropy profiling
        print("=" * 60)
        print("Step 2: Running entropy profiling")
        print("=" * 60)
        all_results = run_entropy_profiling(
            examples=examples,
            model_name=args.model,
            num_rollouts=args.num_rollouts,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            tensor_parallel_size=args.tensor_parallel_size,
            output_dir=results_dir,
        )

        # Save combined results
        with open(output_dir / "entropy_profiles.pkl", "wb") as f:
            pickle.dump(all_results, f)
        print(f"\nSaved combined results to {output_dir / 'entropy_profiles.pkl'}")
    else:
        # Load from individual pickles
        print("Loading existing results...")
        all_results = {}
        for ex in examples:
            pkl_path = results_dir / f"entropy_{ex['name']}.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    all_results[ex["name"]] = pickle.load(f)
                print(f"  Loaded {ex['name']}")
            else:
                print(f"  WARNING: Missing {pkl_path}")

        if not all_results:
            print("ERROR: No results found. Run without --visualize-only first.")
            sys.exit(1)

    # Step 3: Extract features
    print()
    print("=" * 60)
    print("Step 3: Extracting entropy features")
    print("=" * 60)
    features = extract_all_features(all_results)

    # Save features as JSON
    with open(output_dir / "entropy_features.json", "w") as f:
        json.dump(features, f, indent=2)
    print(f"Saved features to {output_dir / 'entropy_features.json'}")

    # Step 4: Create summary CSV
    print()
    print("=" * 60)
    print("Step 4: Creating summary CSV")
    print("=" * 60)
    rows = []
    for name, feat in features.items():
        row = {"example": name}
        ex_info = all_results[name]["example"]
        if args.li:
            row["avg_all"] = ex_info.get("avg_all")
        else:
            row["math500_score"] = ex_info.get("math500_score")
            row["historical_variance"] = ex_info.get("historical_variance")
        row["ground_truth"] = ex_info["ground_truth"]
        row.update(feat)
        rows.append(row)

    if args.li:
        sort_col = "avg_all"
    elif args.large:
        sort_col = "historical_variance"
    else:
        sort_col = "math500_score"
    summary_df = pd.DataFrame(rows).sort_values(sort_col, ascending=False, na_position="last")
    summary_df.to_csv(output_dir / "entropy_summary.csv", index=False)
    print(f"Saved summary to {output_dir / 'entropy_summary.csv'}")
    print()
    summary_cols = ["example", "pass_rate", "mean_entropy_mean", "num_spikes_mean", "entropy_trend_mean", "answer_entropy", "num_entropy_clusters"]
    if args.li:
        summary_cols.insert(1, "avg_all")
    elif args.large:
        summary_cols.insert(1, "historical_variance")
    else:
        summary_cols.insert(1, "math500_score")
    print(summary_df[summary_cols].head(20).to_string(index=False))

    # Step 5: Generate plots
    print()
    print("=" * 60)
    print("Step 5: Generating visualizations")
    print("=" * 60)
    if args.li:
        generate_all_plots(all_results, features, output_dir,
                           score_key="avg_all", score_label="Avg All Domains")
    else:
        generate_all_plots(all_results, features, output_dir)

    print()
    print("=" * 60)
    print("Done! Check output directory:")
    print(f"  Results:  {output_dir / 'entropy_profiles.pkl'}")
    print(f"  Features: {output_dir / 'entropy_features.json'}")
    print(f"  Summary:  {output_dir / 'entropy_summary.csv'}")
    print(f"  Figures:  {output_dir / 'figures/'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
