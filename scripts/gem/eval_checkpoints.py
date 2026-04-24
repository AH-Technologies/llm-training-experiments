#!/usr/bin/env python3
"""Evaluate SFT checkpoints on MATH-500, AIME 2025, and AMC 2023.

Uses vLLM for fast batched generation. Evaluates all checkpoint-* dirs
in each model path, then plots accuracy over training steps.

Usage:
  python scripts/gem/eval_checkpoints.py \
      --model_dirs models/sft_1shot_ce models/sft_1shot_gem \
      --benchmarks math500 aime_2025 amc_2023

  # Evaluate only specific checkpoints:
  python scripts/gem/eval_checkpoints.py \
      --model_dirs models/sft_1shot_ce \
      --checkpoints 100 300 500 700
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rlvr_grokking.rewards.verl_reward import compute_score

# ── Benchmark loaders ──────────────────────────────────────────────────

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."


def load_math500() -> list[dict]:
    """Load MATH-500 from local parquet."""
    path = Path("data/math500.parquet")
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        prompt_raw = row["prompt"]
        if isinstance(prompt_raw, str):
            prompt_raw = json.loads(prompt_raw)
        gt = row["reward_model"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt_answer = gt.get("ground_truth", str(gt))
        data.append({
            "messages": prompt_raw,
            "answer": str(gt_answer),
            "data_source": "math500",
        })
    return data


def load_aime_2025() -> list[dict]:
    """Load AIME 2025 (30 problems) from HuggingFace."""
    ds = load_dataset("MathArena/aime_2025", split="train")
    data = []
    for ex in ds:
        data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "answer": str(int(ex["answer"])),
            "data_source": "aime_2025",
        })
    return data


def load_amc_2023() -> list[dict]:
    """Load AMC 2022-2023 (83 problems) from HuggingFace."""
    ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
    data = []
    for ex in ds:
        data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["problem"]},
            ],
            "answer": str(int(ex["answer"])),
            "data_source": "amc_2023",
        })
    return data


BENCHMARK_LOADERS = {
    "math500": load_math500,
    "aime_2025": load_aime_2025,
    "amc_2023": load_amc_2023,
}


# ── Evaluation ─────────────────────────────────────────────────────────


def find_checkpoints(model_dir: Path, filter_steps: list[int] | None = None) -> list[tuple[int, Path]]:
    """Find checkpoint-N or global_step_N directories, return sorted (step, path) pairs."""
    checkpoints = []
    for d in model_dir.iterdir():
        m = re.match(r"(?:checkpoint-|global_step_)(\d+)", d.name)
        if m and d.is_dir():
            step = int(m.group(1))
            if filter_steps is None or step in filter_steps:
                checkpoints.append((step, d))
    return sorted(checkpoints, key=lambda x: x[0])


def evaluate_checkpoint(
    model_path: Path,
    benchmarks: dict[str, list[dict]],
    max_new_tokens: int = 2048,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, dict]:
    """Evaluate a single checkpoint on all benchmarks using vLLM."""
    print(f"\n{'='*60}")
    print(f"Loading: {model_path}")
    print(f"{'='*60}")

    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_new_tokens + 1024,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    results = {}
    for bench_name, bench_data in benchmarks.items():
        print(f"\n  Evaluating {bench_name} ({len(bench_data)} problems)...")

        # Build prompts using chat template
        prompts = []
        for item in bench_data:
            prompt = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Batched generation
        outputs = llm.generate(prompts, sampling_params)

        # Score
        correct = 0
        total = len(bench_data)
        per_sample = []
        for item, output in zip(bench_data, outputs):
            response = output.outputs[0].text
            reward = compute_score(
                data_source=item["data_source"],
                solution_str=response,
                ground_truth=item["answer"],
            )
            is_correct = reward > 0.5
            correct += int(is_correct)
            per_sample.append({
                "correct": is_correct,
                "gold": item["answer"],
                "response_preview": response[:300],
            })

        acc = correct / total if total > 0 else 0
        print(f"  {bench_name}: {correct}/{total} = {acc*100:.1f}%")
        results[bench_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "per_sample": per_sample,
        }

    # Free GPU memory
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ── Plotting ───────────────────────────────────────────────────────────


def plot_results(
    all_results: dict[str, dict[int, dict[str, dict]]],
    output_dir: Path,
    benchmarks: list[str],
):
    """Plot accuracy over checkpoints for each benchmark.

    all_results: {model_name: {step: {bench_name: {accuracy, ...}}}}
    """
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]

    # One plot per benchmark
    for bench_name in benchmarks:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (model_name, step_results) in enumerate(all_results.items()):
            steps = sorted(step_results.keys())
            accs = [step_results[s][bench_name]["accuracy"] * 100 for s in steps]
            ax.plot(
                steps, accs,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=model_name,
                linewidth=2,
                markersize=6,
            )
        ax.set_xlabel("Training Step (epochs)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{bench_name} Accuracy over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        out_path = output_dir / f"{bench_name}_accuracy.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    # Combined plot (all benchmarks, one subplot per benchmark)
    n_bench = len(benchmarks)
    fig, axes = plt.subplots(1, n_bench, figsize=(6 * n_bench, 5), squeeze=False)
    for j, bench_name in enumerate(benchmarks):
        ax = axes[0][j]
        for i, (model_name, step_results) in enumerate(all_results.items()):
            steps = sorted(step_results.keys())
            accs = [step_results[s][bench_name]["accuracy"] * 100 for s in steps]
            ax.plot(
                steps, accs,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=model_name,
                linewidth=2,
                markersize=5,
            )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(bench_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    fig.suptitle("SFT Checkpoint Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "combined_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoints on math benchmarks")
    parser.add_argument(
        "--model_dirs", nargs="+", required=True,
        help="Paths to model directories containing checkpoint-N subdirs",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["math500", "aime_2025", "amc_2023"],
        choices=list(BENCHMARK_LOADERS.keys()),
    )
    parser.add_argument("--checkpoints", nargs="+", type=int, default=None,
                        help="Only evaluate these checkpoint steps (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="results/sft_1shot_eval")
    parser.add_argument("--also_eval_base", action="store_true",
                        help="Also evaluate the base model (Qwen/Qwen2.5-Math-1.5B-Instruct)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmarks once
    print("Loading benchmarks...")
    benchmarks = {}
    for name in args.benchmarks:
        print(f"  Loading {name}...")
        benchmarks[name] = BENCHMARK_LOADERS[name]()
        print(f"  {name}: {len(benchmarks[name])} problems")

    # all_results: {model_name: {step: {bench: {accuracy, correct, total}}}}
    all_results: dict[str, dict[int, dict]] = {}

    # Optionally evaluate base model
    if args.also_eval_base:
        print(f"\nEvaluating base model: {args.base_model}")
        base_results = evaluate_checkpoint(
            Path(args.base_model), benchmarks,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        # Save base results
        json_dir = output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        base_out = {bench: {k: v for k, v in res.items() if k != "per_sample"}
                    for bench, res in base_results.items()}
        with open(json_dir / "base_model_results.json", "w") as f:
            json.dump(base_out, f, indent=2)
        print(f"\nBase model results:")
        for bench, res in base_results.items():
            print(f"  {bench}: {res['accuracy']*100:.1f}%")

    # Evaluate each model directory
    for model_dir_str in args.model_dirs:
        model_dir = Path(model_dir_str)
        model_name = model_dir.name
        print(f"\n{'#'*60}")
        print(f"Model: {model_name} ({model_dir})")
        print(f"{'#'*60}")

        checkpoints = find_checkpoints(model_dir, args.checkpoints)
        if not checkpoints:
            print(f"  No checkpoints found in {model_dir}!")
            continue

        print(f"  Found checkpoints: {[s for s, _ in checkpoints]}")
        all_results[model_name] = {}

        for step, ckpt_path in checkpoints:
            results = evaluate_checkpoint(
                ckpt_path, benchmarks,
                max_new_tokens=args.max_new_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            all_results[model_name][step] = results

            # Save per-checkpoint results (without per_sample for brevity)
            json_dir = output_dir / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            ckpt_out = {bench: {k: v for k, v in res.items() if k != "per_sample"}
                        for bench, res in results.items()}
            out_file = json_dir / f"{model_name}_step{step}.json"
            with open(out_file, "w") as f:
                json.dump(ckpt_out, f, indent=2)

    # Save combined results
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model_name, step_results in all_results.items():
        summary[model_name] = {}
        for step, bench_results in step_results.items():
            summary[model_name][step] = {
                bench: {"accuracy": r["accuracy"], "correct": r["correct"], "total": r["total"]}
                for bench, r in bench_results.items()
            }
    with open(json_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, step_results in all_results.items():
        print(f"\n{model_name}:")
        header = f"  {'Step':>6}"
        for bench in args.benchmarks:
            header += f"  {bench:>12}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*12}" * len(args.benchmarks))
        for step in sorted(step_results.keys()):
            row = f"  {step:>6}"
            for bench in args.benchmarks:
                acc = step_results[step][bench]["accuracy"] * 100
                row += f"  {acc:>11.1f}%"
            print(row)

    # Plot
    if all_results:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_results(all_results, plots_dir, args.benchmarks)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
