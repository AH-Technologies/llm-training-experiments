#!/usr/bin/env python3
"""Plot eval results from all JSON files in the results directory.

Reads all *_step*.json files and base_model_results.json, then plots
accuracy over training steps for each benchmark.

Usage:
  python scripts/gem/plot_eval_results.py --results_dir results/sft_1shot_eval
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/sft_1shot_eval")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Collect results: {model_name: {step: {bench: {accuracy, correct, total}}}}
    all_results: dict[str, dict[int, dict]] = defaultdict(dict)
    benchmarks = set()

    for f in sorted((results_dir / "json").glob("*_step*.json")):
        m = re.match(r"(.+)_step(\d+)\.json", f.name)
        if not m:
            continue
        model_name = m.group(1)
        step = int(m.group(2))
        with open(f) as fh:
            data = json.load(fh)
        all_results[model_name][step] = data
        benchmarks.update(data.keys())

    # Load base model results if available
    base_path = results_dir / "json" / "base_model_results.json"
    base_results = None
    if base_path.exists():
        with open(base_path) as fh:
            base_results = json.load(fh)

    benchmarks = sorted(benchmarks)
    print(f"Found {len(all_results)} models, benchmarks: {benchmarks}")
    for name, steps in sorted(all_results.items()):
        print(f"  {name}: steps {sorted(steps.keys())}")

    if not all_results:
        print("No results found!")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Colors and markers
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

    # Group models by type
    sft_models = {k: v for k, v in all_results.items() if k.startswith("sft_")}
    grpo_models = {k: v for k, v in all_results.items() if k.startswith("grpo_")}

    groups = [
        ("all_models", all_results, "All Models"),
        ("sft_only", sft_models, "SFT Models"),
        ("grpo_only", grpo_models, "GRPO Models"),
    ]

    for group_name, group_results, group_title in groups:
        if not group_results:
            continue

        # One plot per benchmark
        for bench_name in benchmarks:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, (model_name, step_results) in enumerate(sorted(group_results.items())):
                steps = sorted(step_results.keys())
                accs = []
                for s in steps:
                    if bench_name in step_results[s]:
                        accs.append(step_results[s][bench_name]["accuracy"] * 100)
                    else:
                        accs.append(None)
                valid = [(s, a) for s, a in zip(steps, accs) if a is not None]
                if valid:
                    vs, va = zip(*valid)
                    ax.plot(
                        vs, va,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        label=model_name,
                        linewidth=2,
                        markersize=5,
                    )

            if base_results and bench_name in base_results:
                base_acc = base_results[bench_name]["accuracy"] * 100
                ax.axhline(y=base_acc, color="black", linestyle="--", linewidth=1.5,
                           label=f"base model ({base_acc:.1f}%)")

            ax.set_xlabel("Training Step (epochs)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{group_title} — {bench_name}")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            out_path = plots_dir / f"{group_name}_{bench_name}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_path}")

        # Combined subplot for this group
        n_bench = len(benchmarks)
        fig, axes = plt.subplots(1, n_bench, figsize=(7 * n_bench, 6), squeeze=False)
        for j, bench_name in enumerate(benchmarks):
            ax = axes[0][j]
            for i, (model_name, step_results) in enumerate(sorted(group_results.items())):
                steps = sorted(step_results.keys())
                accs = []
                for s in steps:
                    if bench_name in step_results[s]:
                        accs.append(step_results[s][bench_name]["accuracy"] * 100)
                    else:
                        accs.append(None)
                valid = [(s, a) for s, a in zip(steps, accs) if a is not None]
                if valid:
                    vs, va = zip(*valid)
                    ax.plot(
                        vs, va,
                        color=colors[i % len(colors)],
                        marker=markers[i % len(markers)],
                        label=model_name,
                        linewidth=2,
                        markersize=4,
                    )

            if base_results and bench_name in base_results:
                base_acc = base_results[bench_name]["accuracy"] * 100
                ax.axhline(y=base_acc, color="black", linestyle="--", linewidth=1.5,
                           label=f"base ({base_acc:.1f}%)")

            ax.set_xlabel("Training Step")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(bench_name)
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        fig.suptitle(group_title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        out_path = plots_dir / f"{group_name}_combined.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for model_name in sorted(all_results.keys()):
        step_results = all_results[model_name]
        print(f"\n{model_name}:")
        header = f"  {'Step':>6}"
        for bench in benchmarks:
            header += f"  {bench:>12}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*12}" * len(benchmarks))
        for step in sorted(step_results.keys()):
            row = f"  {step:>6}"
            for bench in benchmarks:
                if bench in step_results[step]:
                    acc = step_results[step][bench]["accuracy"] * 100
                    row += f"  {acc:>11.1f}%"
                else:
                    row += f"  {'N/A':>12}"
            print(row)

    if base_results:
        print(f"\nBase model:")
        for bench in benchmarks:
            if bench in base_results:
                acc = base_results[bench]["accuracy"] * 100
                print(f"  {bench}: {acc:.1f}%")


if __name__ == "__main__":
    main()
