#!/usr/bin/env python3
"""Evaluate a single GRPO condition checkpoint on hard math benchmarks.

Designed to be launched 4x in parallel (one per GPU) by the SLURM script.
Each instance handles one condition on one GPU.

Usage:
  # Single condition on a specific GPU:
  CUDA_VISIBLE_DEVICES=0 python eval_checkpoints.py --condition uniform --step 100

  # The SLURM script launches all 4 in parallel.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rlvr_grokking.rewards.verl_reward import compute_score

CHECKPOINT_BASE = Path("checkpoints/attention-grpo")
MERGED_BASE = Path("attention_based_rewards/merged_models")
RESULTS_DIR = Path("attention_based_rewards/results/eval_hard")

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
)


def find_steps(condition: str) -> list[int]:
    """Find available checkpoint steps for a condition."""
    cond_dir = CHECKPOINT_BASE / f"grpo_{condition}"
    if not cond_dir.exists():
        return []
    steps = []
    for d in cond_dir.iterdir():
        m = re.match(r"global_step_(\d+)", d.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def find_common_steps(conditions: list[str]) -> list[int]:
    """Find checkpoint steps that exist for ALL given conditions."""
    step_sets = [set(find_steps(c)) for c in conditions]
    if not step_sets:
        return []
    common = step_sets[0]
    for s in step_sets[1:]:
        common = common & s
    return sorted(common)


def merge_fsdp_checkpoint(condition: str, step: int) -> Path:
    """Merge FSDP shards into HF model using verl's model_merger."""
    src = CHECKPOINT_BASE / f"grpo_{condition}" / f"global_step_{step}" / "actor"
    target = MERGED_BASE / f"grpo_{condition}" / f"step_{step}"

    if (target / "config.json").exists():
        print(f"[{condition}] Already merged: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"[{condition}] Merging FSDP: {src} -> {target}")

    result = subprocess.run(
        [
            sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", str(src),
            "--target_dir", str(target),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[{condition}] Merge stderr: {result.stderr[-1000:]}")
        raise RuntimeError(f"Failed to merge {src}")

    print(f"[{condition}] Merged successfully")
    return target


def load_eval_dataset(name: str) -> list[dict]:
    """Load an evaluation dataset from parquet."""
    path = Path(f"attention_based_rewards/data/{name}.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python attention_based_rewards/scripts/prepare_eval_data.py"
        )
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        prompt_msgs = row["prompt"]
        question = prompt_msgs[-1]["content"]
        data.append({
            "question": question,
            "answer": str(gt),
            "data_source": row["data_source"],
            "extra_info": row.get("extra_info", {}),
        })
    return data


@torch.no_grad()
def evaluate_model(
    model_path: Path,
    eval_data: list[dict],
    dataset_name: str,
    condition: str,
    max_new_tokens: int = 3072,
) -> dict:
    """Evaluate a single model on a dataset with greedy decoding."""
    print(f"[{condition}] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(eval_data):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        reward = compute_score(
            data_source=item["data_source"],
            solution_str=response,
            ground_truth=item["answer"],
        )
        is_correct = reward > 0.5
        correct += int(is_correct)
        total += 1

        results.append({
            "index": i,
            "question": item["question"][:200],
            "gold": item["answer"],
            "response": response[:500],
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(eval_data):
            print(f"[{condition}] {dataset_name} [{i+1}/{len(eval_data)}] "
                  f"acc={correct}/{total} ({correct/total*100:.1f}%)")

    del model
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "dataset": dataset_name,
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True,
                        choices=["uniform", "attention", "entropy", "combined"])
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: latest available)")
    parser.add_argument("--datasets", nargs="+", default=["aime_2025", "amc_2023"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cond = args.condition

    # Find step
    available = find_steps(cond)
    print(f"[{cond}] Available steps: {available}")

    if args.step is not None:
        eval_step = args.step
    else:
        eval_step = max(available) if available else None

    if eval_step is None or eval_step not in available:
        print(f"[{cond}] Step {eval_step} not available. Exiting.")
        sys.exit(1)

    print(f"[{cond}] Evaluating step {eval_step} on GPU {torch.cuda.current_device()}")

    # Merge checkpoint
    model_path = merge_fsdp_checkpoint(cond, eval_step)

    # Evaluate on each dataset
    for ds_name in args.datasets:
        print(f"\n[{cond}] Loading {ds_name}...")
        eval_data = load_eval_dataset(ds_name)
        print(f"[{cond}] {len(eval_data)} problems")

        result = evaluate_model(model_path, eval_data, ds_name, cond)

        out_file = RESULTS_DIR / f"{cond}_step{eval_step}_{ds_name}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{cond}] {ds_name}: {result['accuracy']*100:.1f}% "
              f"({result['correct']}/{result['total']})")

    print(f"\n[{cond}] Done!")


if __name__ == "__main__":
    main()
