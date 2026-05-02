#!/usr/bin/env python3
"""Post-training evaluation for circuit-guided GRPO experiments.

Evaluates all saved checkpoints across conditions on GSM8K test and MATH-500,
then produces comparison plots.

Usage:
  python scripts/attention_based_rewards/evaluate.py \
    --checkpoint_dir outputs/ \
    --conditions uniform attention entropy combined \
    --output_dir results/attention_based_rewards/evaluation
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Reuse reward function for answer extraction
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rlvr_grokking.rewards.verl_reward import compute_score


SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
)


def find_checkpoints(checkpoint_dir: Path, condition: str) -> list[tuple[int, Path]]:
    """Find all checkpoints for a condition, sorted by step number."""
    pattern = f"grpo_{condition}"
    checkpoints = []

    for d in checkpoint_dir.rglob("*"):
        if not d.is_dir():
            continue
        # verl saves checkpoints as global_step_XXX directories
        match = re.search(r"global_step_(\d+)", d.name)
        if match and pattern in str(d):
            step = int(match.group(1))
            # Check for model files
            if (d / "actor").exists() or (d / "huggingface_model").exists():
                checkpoints.append((step, d))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(
    model_path: Path,
    tokenizer: AutoTokenizer,
    eval_data: list[dict],
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    batch_size: int = 8,
) -> dict:
    """Evaluate a single checkpoint on a dataset.

    Returns dict with accuracy and per-example results.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    correct = 0
    total = 0
    results = []

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size]

        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["question"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            reward = compute_score(
                data_source=item.get("data_source", "gsm8k"),
                solution_str=response,
                ground_truth=item["answer"],
            )

            is_correct = reward > 0.5
            correct += int(is_correct)
            total += 1

            results.append({
                "question": item["question"][:200],
                "answer": item["answer"],
                "response": response[:500],
                "correct": is_correct,
            })

        print(f"  Evaluated {min(i + batch_size, len(eval_data))}/{len(eval_data)}, "
              f"accuracy so far: {correct}/{total} ({correct/total*100:.1f}%)")

    del model
    torch.cuda.empty_cache()

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "results": results,
    }


def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test set."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    data = []
    for ex in ds:
        match = re.search(r"####\s*(.+)", ex["answer"])
        answer = match.group(1).strip().replace(",", "") if match else ex["answer"]
        data.append({
            "question": ex["question"],
            "answer": answer,
            "data_source": "gsm8k",
        })
    return data


def load_math500() -> list[dict]:
    """Load MATH-500 from existing parquet."""
    import pandas as pd
    df = pd.read_parquet("data/math500.parquet")
    data = []
    for _, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"] if isinstance(row["reward_model"], dict) else str(row["reward_model"])
        question = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else str(row["prompt"])
        data.append({
            "question": question,
            "answer": gt,
            "data_source": "math500",
        })
    return data


def plot_learning_curves(
    all_results: dict[str, list[tuple[int, float]]],
    dataset_name: str,
    output_path: Path,
):
    """Plot learning curves for all conditions on same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "uniform": "#1f77b4",
        "attention": "#ff7f0e",
        "entropy": "#2ca02c",
        "combined": "#d62728",
    }
    markers = {"uniform": "o", "attention": "s", "entropy": "^", "combined": "D"}

    for condition, results in all_results.items():
        if not results:
            continue
        steps, accs = zip(*results)
        ax.plot(
            steps, [a * 100 for a in accs],
            label=f"GRPO-{condition.capitalize()}",
            color=colors.get(condition, "gray"),
            marker=markers.get(condition, "o"),
            markersize=6,
            linewidth=2,
        )

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(f"{dataset_name} Accuracy (%)", fontsize=12)
    ax.set_title(f"Circuit-Guided GRPO: {dataset_name} Performance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate circuit-guided GRPO checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/", help="Base directory for checkpoints")
    parser.add_argument("--conditions", nargs="+", default=["uniform", "attention", "entropy", "combined"])
    parser.add_argument("--output_dir", type=str, default="results/attention_based_rewards/evaluation")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"], choices=["gsm8k", "math500"])
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="Limit eval samples (-1 = all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Load eval datasets
    eval_datasets = {}
    if "gsm8k" in args.datasets:
        eval_datasets["GSM8K"] = load_gsm8k_test()
    if "math500" in args.datasets:
        eval_datasets["MATH-500"] = load_math500()

    if args.max_eval_samples > 0:
        for name in eval_datasets:
            eval_datasets[name] = eval_datasets[name][: args.max_eval_samples]

    # Load tokenizer (same for all checkpoints)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    # Evaluate each condition
    all_results = {}  # {dataset_name: {condition: [(step, accuracy)]}}
    for ds_name in eval_datasets:
        all_results[ds_name] = {}

    for condition in args.conditions:
        checkpoints = find_checkpoints(checkpoint_dir, condition)
        if not checkpoints:
            print(f"No checkpoints found for condition: {condition}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating condition: {condition} ({len(checkpoints)} checkpoints)")
        print(f"{'='*60}")

        for ds_name, eval_data in eval_datasets.items():
            condition_results = []

            for step, ckpt_path in checkpoints:
                print(f"\n  Step {step} on {ds_name}...")
                # Find model path within checkpoint
                model_path = ckpt_path / "actor" if (ckpt_path / "actor").exists() else ckpt_path

                result = evaluate_checkpoint(model_path, tokenizer, eval_data)
                condition_results.append((step, result["accuracy"]))

                # Save per-checkpoint results
                result_file = output_dir / f"{condition}_step{step}_{ds_name.lower()}.json"
                with open(result_file, "w") as f:
                    json.dump({
                        "condition": condition,
                        "step": step,
                        "dataset": ds_name,
                        **result,
                    }, f, indent=2)

                print(f"    {ds_name}: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")

            all_results[ds_name][condition] = condition_results

    # Plot comparison curves
    for ds_name, condition_results in all_results.items():
        if any(condition_results.values()):
            plot_learning_curves(
                condition_results,
                ds_name,
                output_dir / f"learning_curves_{ds_name.lower()}.png",
            )

    # Save summary
    summary = {}
    for ds_name, condition_results in all_results.items():
        summary[ds_name] = {}
        for condition, results in condition_results.items():
            if results:
                best_step, best_acc = max(results, key=lambda x: x[1])
                final_step, final_acc = results[-1]
                summary[ds_name][condition] = {
                    "best_accuracy": best_acc,
                    "best_step": best_step,
                    "final_accuracy": final_acc,
                    "final_step": final_step,
                    "n_checkpoints": len(results),
                }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for ds_name, conditions in summary.items():
        print(f"\n{ds_name}:")
        print(f"  {'Condition':<12} {'Best Acc':>10} {'@ Step':>8} {'Final Acc':>10} {'@ Step':>8}")
        print(f"  {'-'*50}")
        for cond, stats in conditions.items():
            print(f"  {cond:<12} {stats['best_accuracy']*100:>9.1f}% {stats['best_step']:>8} "
                  f"{stats['final_accuracy']*100:>9.1f}% {stats['final_step']:>8}")


if __name__ == "__main__":
    main()
