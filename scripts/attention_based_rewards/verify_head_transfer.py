#!/usr/bin/env python3
"""Verify reasoning heads (identified on GSM8K) transfer to DAPO-Math-17k.

Multi-GPU: GPU 0-1 run baseline, GPU 2-3 run ablated (10 problems each).

Usage:
    srun --gpus=4 --time=00:20:00 --account=nn12068k --partition=accel \
        python scripts/attention_based_rewards/verify_head_transfer.py
"""

import json
import random
import re
import time

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformer_lens import HookedTransformer

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
N_PROBLEMS = 20
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
    "You first think about the reasoning process as an internal monologue and then provide "
    "the user with the answer. Respond in the following format:\n"
    "<think>\\n...\\n</think>\\n<answer>\\n...\\n</answer>"
)


def make_chat_prompt(question):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_model_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    nums = re.findall(r'[\d,]+\.?\d*', text)
    if nums:
        return nums[-1].replace(',', '')
    return None


def extract_ground_truth(answer_str):
    m = re.search(r'\\boxed\{([^}]+)\}', answer_str)
    if m:
        return m.group(1).strip()
    return answer_str.strip()


def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return pred.strip() == gold.strip()


def generate_with_ablation(model, prompt, ablate_heads=None):
    hooks = []
    if ablate_heads:
        for layer, head in ablate_heads:
            hook_name = f"blocks.{layer}.attn.hook_result"
            def make_hook(h):
                def hook_fn(activation, hook):
                    activation[:, :, h, :] = 0.0
                    return activation
                return hook_fn
            hooks.append((hook_name, make_hook(head)))

    tokens = model.to_tokens(prompt)
    for _ in range(MAX_NEW_TOKENS):
        with model.hooks(fwd_hooks=hooks):
            logits = model(tokens)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        tokens = torch.cat([tokens, next_token], dim=1)
        decoded = model.to_string(next_token[0])
        if "<|im_end|>" in decoded or "<|endoftext|>" in decoded:
            break

    full_text = model.to_string(tokens[0])
    if "<|im_start|>assistant\n" in full_text:
        response = full_text.split("<|im_start|>assistant\n")[-1]
    else:
        response = full_text[len(prompt):]
    return response.replace("<|im_end|>", "").strip()


def worker(rank, n_gpus, problems, top_heads, result_dict):
    """GPU 0..n/2-1 run baseline, GPU n/2..n-1 run ablated."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    half = n_gpus // 2
    is_ablated = rank >= half
    ablate_heads = top_heads if is_ablated else None
    label = "ablated" if is_ablated else "baseline"

    # Split problems across GPUs in each group
    group_rank = rank - half if is_ablated else rank
    group_size = half if half > 0 else 1
    chunk = len(problems) // group_size
    start = group_rank * chunk
    end = start + chunk if group_rank < group_size - 1 else len(problems)
    my_problems = problems[start:end]

    print(f"[GPU {rank}] {label}, problems {start}-{end-1} ({len(my_problems)} total)", flush=True)

    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", device=device, dtype=torch.float16,
    )
    model.cfg.use_attn_result = True
    model.setup()

    correct = 0
    for i, (question, gold) in enumerate(my_problems):
        prompt = make_chat_prompt(question)
        response = generate_with_ablation(model, prompt, ablate_heads=ablate_heads)
        pred = extract_model_answer(response)
        match = answers_match(pred, gold)
        if match:
            correct += 1
        print(f"  [GPU {rank}] [{i+1}/{len(my_problems)}] pred={pred} gold={gold} {'Y' if match else 'N'}", flush=True)

    result_dict[rank] = {"label": label, "correct": correct, "total": len(my_problems)}


def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Need at least 2 GPUs, got {n_gpus}"
    print(f"Using {n_gpus} GPUs ({n_gpus//2} baseline, {n_gpus//2} ablated)", flush=True)

    # Load head scores
    data = torch.load(f"{BASE_DIR}/results/reasoning_heads.pt", weights_only=False)
    head_scores = data["head_scores"]
    n_layers, n_heads = head_scores.shape
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    top_heads = [(idx.item() // n_heads, idx.item() % n_heads) for idx in sorted_idx[:10]]

    print("Top 10 reasoning heads:", flush=True)
    for i, (l, h) in enumerate(top_heads):
        print(f"  #{i+1}: L{l}H{h} = {head_scores[l,h]:.2f}", flush=True)

    # Load DAPO problems
    print(f"\nLoading {N_PROBLEMS} DAPO-Math-17k problems...", flush=True)
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    problems = []
    for i in indices[:N_PROBLEMS]:
        q = ds[i]["prompt"][0]["content"] if isinstance(ds[i]["prompt"], list) else ds[i]["prompt"]
        gold = extract_ground_truth(str(ds[i]["reward_model"]["ground_truth"]))
        problems.append((q, gold))

    print(f"Sample: {problems[0][0][:80]}... -> {problems[0][1]}", flush=True)

    t0 = time.time()
    result_dict = mp.Manager().dict()
    mp.spawn(worker, args=(n_gpus, problems, top_heads, result_dict), nprocs=n_gpus, join=True)
    elapsed = time.time() - t0

    # Aggregate
    baseline_correct = sum(r["correct"] for r in result_dict.values() if r["label"] == "baseline")
    ablated_correct = sum(r["correct"] for r in result_dict.values() if r["label"] == "ablated")
    baseline_acc = baseline_correct / N_PROBLEMS
    ablated_acc = ablated_correct / N_PROBLEMS
    drop = baseline_acc - ablated_acc

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS ({elapsed/60:.1f} min)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline: {baseline_acc:.1%} ({baseline_correct}/{N_PROBLEMS})", flush=True)
    print(f"Ablated:  {ablated_acc:.1%} ({ablated_correct}/{N_PROBLEMS})", flush=True)
    print(f"Drop:     {drop:.1%}", flush=True)

    transfer = drop > 0
    if drop > 0.05:
        print(f"\nHEADS TRANSFER: {drop:.1%} drop. Proceed.", flush=True)
    elif drop > 0:
        print(f"\nMARGINAL TRANSFER: {drop:.1%} drop.", flush=True)
    else:
        print(f"\nNO TRANSFER detected.", flush=True)

    results = {
        "dataset": "DAPO-Math-17k (en)", "n_problems": N_PROBLEMS,
        "top_heads": top_heads,
        "baseline_accuracy": baseline_acc, "ablated_accuracy": ablated_acc,
        "accuracy_drop": drop, "heads_transfer": transfer,
        "time_minutes": elapsed / 60,
    }
    output_path = f"{BASE_DIR}/results/head_transfer_dapo.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
