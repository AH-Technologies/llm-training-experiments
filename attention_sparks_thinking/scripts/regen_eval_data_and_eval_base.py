#!/usr/bin/env python3
"""Regenerate eval parquets with the Attention Illuminates system prompt,
then evaluate base Qwen3-4B-Base on MATH500, AIME 2025, AMC 2023.

System prompt (from paper):
  "Please reason step by step, and put your final answer within \\boxed{}."

Usage:
  python attention_sparks_thinking/scripts/regen_eval_data_and_eval_base.py
"""

import json
import os
import sys

import pandas as pd
import torch

PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"
sys.path.insert(0, PROJECT_DIR)

SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."

# ── Step 1: Regenerate eval parquets ──────────────────────────────────────────

def fix_parquet(path: str):
    """Rewrite a parquet file replacing the system prompt."""
    df = pd.read_parquet(path)
    print(f"\nFixing {path} ({len(df)} rows)")

    fixed = 0
    for i in range(len(df)):
        prompt = df.at[i, "prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        # Replace or add system message
        new_prompt = []
        has_system = False
        for msg in prompt:
            msg = dict(msg)  # copy
            if msg["role"] == "system":
                msg["content"] = SYSTEM_PROMPT
                has_system = True
            new_prompt.append(msg)

        if not has_system:
            new_prompt.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        df.at[i, "prompt"] = new_prompt
        fixed += 1

    df.to_parquet(path)
    print(f"  Updated {fixed} rows. System prompt: {SYSTEM_PROMPT[:60]}...")

    # Verify
    df2 = pd.read_parquet(path)
    p = df2.iloc[0]["prompt"]
    if hasattr(p, "tolist"):
        p = p.tolist()
    sys_msg = [m for m in p if m["role"] == "system"]
    print(f"  Verify: system msg = {sys_msg[0]['content'][:60]}...")


def regen_all_eval_data():
    """Fix system prompts in all eval parquets."""
    paths = [
        os.path.join(PROJECT_DIR, "data/math500.parquet"),
        os.path.join(PROJECT_DIR, "attention_based_rewards/data/aime_2025.parquet"),
        os.path.join(PROJECT_DIR, "attention_based_rewards/data/amc_2023.parquet"),
    ]
    for p in paths:
        if os.path.exists(p):
            fix_parquet(p)
        else:
            print(f"  SKIP (not found): {p}")


# ── Step 2: Also fix training data ───────────────────────────────────────────

def fix_training_data():
    """Fix system prompt in DAPO training parquet."""
    path = os.path.join(PROJECT_DIR, "attention_based_rewards/data/dapo_math_17k.parquet")
    if os.path.exists(path):
        fix_parquet(path)


# ── Step 3: Evaluate base model ──────────────────────────────────────────────

def evaluate_base_model(model_name: str = "Qwen/Qwen3-4B-Base"):
    """Evaluate base model on MATH500, AIME 2025, AMC 2023 using vLLM."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from src.rlvr_grokking.rewards.verl_reward import compute_score

    print(f"\n{'='*60}")
    print(f"Evaluating base model: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8000,
        n=1,
    )

    benchmarks = {
        "math500": os.path.join(PROJECT_DIR, "data/math500.parquet"),
        "aime_2025": os.path.join(PROJECT_DIR, "attention_based_rewards/data/aime_2025.parquet"),
        "amc_2023": os.path.join(PROJECT_DIR, "attention_based_rewards/data/amc_2023.parquet"),
    }

    results = {}

    for bench_name, path in benchmarks.items():
        df = pd.read_parquet(path)
        print(f"\n--- {bench_name}: {len(df)} problems ---")

        # Format prompts
        prompts = []
        ground_truths = []
        for _, row in df.iterrows():
            prompt = row["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            if isinstance(prompt, str):
                prompt = json.loads(prompt)

            prompt_str = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_str)

            gt = row["reward_model"]
            if isinstance(gt, str):
                gt = json.loads(gt)
            if hasattr(gt, "tolist"):
                gt = gt.tolist()
            ground_truths.append(gt.get("ground_truth", gt) if isinstance(gt, dict) else str(gt))

        # Print first prompt for verification
        print(f"  First prompt:\n{prompts[0][:300]}...")

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Score and save full outputs
        correct = 0
        total = 0
        all_outputs = []
        has_boxed = 0
        total_resp_len = 0
        max_len_hit = 0
        for gt, output in zip(ground_truths, outputs):
            response_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            score = compute_score("math_dapo", response_text, gt)
            if score > 0:
                correct += 1
            total += 1
            total_resp_len += len(response_text)
            if r"\boxed" in response_text:
                has_boxed += 1
            if finish_reason == "length":
                max_len_hit += 1

            all_outputs.append({
                "ground_truth": gt,
                "response": response_text,
                "score": score,
                "finish_reason": finish_reason,
                "response_len": len(response_text),
            })

            # Print first few examples (full response)
            if total <= 3:
                print(f"\n  Example {total} (finish={finish_reason}, len={len(response_text)}):")
                print(f"    Response: {response_text[:500]}")
                print(f"    [...end:] {response_text[-200:]}")
                print(f"    Ground truth: {gt}")
                print(f"    Score: {score}")

        # Save full outputs
        out_dir = os.path.join(PROJECT_DIR, "attention_sparks_thinking/logs/base_eval_outputs")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{bench_name}.json"), "w") as f:
            json.dump(all_outputs, f, indent=2)

        avg_len = total_resp_len / total if total > 0 else 0
        print(f"\n  Stats: {has_boxed}/{total} have \\boxed{{}}, {max_len_hit}/{total} hit max_tokens, avg response len={avg_len:.0f} chars")

        accuracy = correct / total if total > 0 else 0.0
        results[bench_name] = {"accuracy": accuracy, "correct": correct, "total": total}
        print(f"\n  {bench_name}: {accuracy:.1%} ({correct}/{total})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Base model (no training)")
    print(f"{'='*60}")
    for bench, r in results.items():
        print(f"  {bench}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    # Save results
    out_path = os.path.join(PROJECT_DIR, "attention_sparks_thinking/logs/base_model_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Step 1: Regenerating eval data with new system prompt...")
    regen_all_eval_data()

    print("\nStep 2: Fixing training data system prompt...")
    fix_training_data()

    print("\nStep 3: Evaluating base Qwen3-4B-Base model...")
    evaluate_base_model("Qwen/Qwen3-4B-Base")
