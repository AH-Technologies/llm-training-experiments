"""Step 3: Ablation validation of discovered reasoning heads.

Multi-GPU: each GPU generates responses for a shard of problems.
Zero out top-10 reasoning heads and measure accuracy drop on GSM8K.
Compare against random head ablation as control.
"""
import json
import os
import random
import re
import sys
import time
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformer_lens import HookedTransformer

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
N_PROBLEMS = 50
N_RANDOM_CONTROLS = 3
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

# ── Answer extraction ────────────────────────────────────────────────
def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,.\-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None

def extract_model_answer(text):
    # Try <answer> tags
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    # Try \boxed{}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        nums = re.findall(r'[\d,]+\.?\d*', m.group(1))
        if nums:
            return nums[-1].replace(',', '')
    # Fallback: last number
    nums = re.findall(r'[\d,]+\.?\d*', text)
    if nums:
        return nums[-1].replace(',', '')
    return None

def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return pred.strip() == gold.strip()

# ── Generation with optional head ablation ───────────────────────────
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


# ── Per-GPU worker ───────────────────────────────────────────────────
def worker(rank, n_gpus, problems, gold_answers, conditions, result_dict):
    """Run generation for a shard of problems across all conditions on one GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Shard problems
    shard_size = len(problems) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(problems)
    my_problems = problems[start:end]
    my_gold = gold_answers[start:end]
    my_indices = list(range(start, end))

    print(f"[GPU {rank}] Loading model, processing problems {start}-{end} ({len(my_problems)} problems)", flush=True)

    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        dtype=torch.float16,
    )
    model.cfg.use_attn_result = True
    model.setup()

    gpu_results = {}

    for cond_name, ablate_heads in conditions:
        print(f"[GPU {rank}] Condition: {cond_name}", flush=True)
        cond_results = []
        t0 = time.time()

        for i, ((q, a), gold) in enumerate(zip(my_problems, my_gold)):
            prompt = make_chat_prompt(q)
            response = generate_with_ablation(model, prompt, ablate_heads=ablate_heads)
            pred = extract_model_answer(response)
            correct = answers_match(pred, gold)
            cond_results.append({
                "idx": my_indices[i],
                "question": q[:80],
                "gold": gold,
                "pred": pred,
                "correct": correct,
            })
            if (i + 1) % 5 == 0:
                n_correct = sum(1 for r in cond_results if r["correct"])
                elapsed = time.time() - t0
                print(f"[GPU {rank}] {cond_name}: {i+1}/{len(my_problems)} ({n_correct}/{i+1} correct, {elapsed:.0f}s)", flush=True)

        n_correct = sum(1 for r in cond_results if r["correct"])
        elapsed = time.time() - t0
        print(f"[GPU {rank}] {cond_name} DONE: {n_correct}/{len(my_problems)} correct in {elapsed:.0f}s", flush=True)
        gpu_results[cond_name] = cond_results

    result_dict[rank] = gpu_results


# ── Main ─────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}", flush=True)

    # Load head scores from Step 2
    print("Loading reasoning head scores...", flush=True)
    data = torch.load(f"{BASE_DIR}/results/reasoning_heads.pt", weights_only=False)
    head_scores = data["head_scores"]
    n_layers = head_scores.shape[0]
    n_heads = head_scores.shape[1]

    # Top 10 reasoning heads
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    top10_heads = []
    for idx in sorted_idx[:10]:
        l = idx.item() // n_heads
        h = idx.item() % n_heads
        top10_heads.append((l, h))
    print(f"Top 10 reasoning heads: {top10_heads}", flush=True)
    print(f"  Scores: {[f'{head_scores[l,h]:.2f}' for l,h in top10_heads]}", flush=True)

    # Load GSM8K held-out split
    print("\nLoading GSM8K (held-out split)...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    held_out_indices = indices[500:500 + N_PROBLEMS]
    problems = [(ds[i]["question"], ds[i]["answer"]) for i in held_out_indices]
    gold_answers = [extract_gsm8k_answer(a) for _, a in problems]
    print(f"Using {len(problems)} held-out problems", flush=True)

    # Build random control head sets
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    top10_set = set(top10_heads)
    candidate_heads = [h for h in all_heads if h not in top10_set]

    random_head_sets = []
    for run in range(N_RANDOM_CONTROLS):
        random.seed(run + 100)
        random_head_sets.append(random.sample(candidate_heads, 10))

    # Build conditions list: (name, ablate_heads)
    conditions = [
        ("baseline", None),
        ("top10_ablation", top10_heads),
    ]
    for run in range(N_RANDOM_CONTROLS):
        conditions.append((f"random_{run}", random_head_sets[run]))

    print(f"\nConditions: {[c[0] for c in conditions]}", flush=True)
    print(f"Total generations: {N_PROBLEMS} problems x {len(conditions)} conditions = {N_PROBLEMS * len(conditions)}", flush=True)
    print(f"Parallelized across {n_gpus} GPUs\n", flush=True)

    # Run on all GPUs
    t0 = time.time()
    if n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(
            worker,
            args=(n_gpus, problems, gold_answers, conditions, result_dict),
            nprocs=n_gpus,
            join=True,
        )
    else:
        result_dict = {}
        worker(0, 1, problems, gold_answers, conditions, result_dict)

    total_time = time.time() - t0
    print(f"\nAll GPUs done in {total_time/60:.1f} min", flush=True)

    # Aggregate results from all GPUs
    print("\nAggregating results...", flush=True)
    aggregated = {cond_name: [] for cond_name, _ in conditions}
    for rank in range(n_gpus):
        for cond_name in aggregated:
            aggregated[cond_name].extend(result_dict[rank][cond_name])

    # Sort by original index
    for cond_name in aggregated:
        aggregated[cond_name].sort(key=lambda x: x["idx"])

    # Compute accuracies
    def compute_acc(results):
        n = sum(1 for r in results if r["correct"])
        return n, len(results), n / len(results)

    baseline_n, baseline_total, baseline_acc = compute_acc(aggregated["baseline"])
    top10_n, top10_total, top10_acc = compute_acc(aggregated["top10_ablation"])

    random_accs = []
    for run in range(N_RANDOM_CONTROLS):
        _, _, racc = compute_acc(aggregated[f"random_{run}"])
        random_accs.append(racc)
    avg_random_acc = sum(random_accs) / len(random_accs)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("ABLATION VALIDATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Baseline accuracy:           {baseline_acc:.1%} ({baseline_n}/{baseline_total})", flush=True)
    print(f"Top-10 head ablation:        {top10_acc:.1%} ({top10_n}/{top10_total})", flush=True)
    for run in range(N_RANDOM_CONTROLS):
        rn, rt, ra = compute_acc(aggregated[f"random_{run}"])
        print(f"Random control {run+1}:           {ra:.1%} ({rn}/{rt})  heads={random_head_sets[run][:3]}...", flush=True)
    print(f"Random ablation (avg):       {avg_random_acc:.1%}", flush=True)
    print(f"", flush=True)
    print(f"Drop from top-10 ablation:   {baseline_acc - top10_acc:.1%}", flush=True)
    print(f"Drop from random ablation:   {baseline_acc - avg_random_acc:.1%}", flush=True)
    denom = max(baseline_acc - avg_random_acc, 0.001)
    causal_ratio = (baseline_acc - top10_acc) / denom
    print(f"Causal effect ratio:         {causal_ratio:.1f}x", flush=True)
    print(f"", flush=True)
    print(f"Top 10 heads ablated: {top10_heads}", flush=True)
    ts_heads = {(0,8),(5,1),(7,1),(18,11),(11,8)}
    overlap = set(top10_heads) & ts_heads
    print(f"Thinking Sparks GRPO+OpenR1: {ts_heads}", flush=True)
    print(f"Overlap with TS heads: {overlap} ({len(overlap)}/5)", flush=True)

    if baseline_acc - top10_acc > baseline_acc - avg_random_acc + 0.05:
        conclusion = "Top reasoning heads are causally important for reasoning!"
    elif baseline_acc - top10_acc > baseline_acc - avg_random_acc:
        conclusion = "Moderate causal effect detected."
    else:
        conclusion = "No clear causal effect. May need more problems or different ablation."
    print(f"\nCONCLUSION: {conclusion}", flush=True)

    # Save
    results = {
        "config": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "n_problems": N_PROBLEMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_random_controls": N_RANDOM_CONTROLS,
            "n_gpus": n_gpus,
            "total_time_minutes": total_time / 60,
        },
        "top10_heads": top10_heads,
        "top10_scores": [float(head_scores[l, h]) for l, h in top10_heads],
        "baseline": {
            "accuracy": baseline_acc,
            "n_correct": baseline_n,
            "results": aggregated["baseline"],
        },
        "top10_ablation": {
            "accuracy": top10_acc,
            "n_correct": top10_n,
            "accuracy_drop": baseline_acc - top10_acc,
            "results": aggregated["top10_ablation"],
        },
        "random_ablation": {
            "accuracies": random_accs,
            "mean_accuracy": avg_random_acc,
            "mean_drop": baseline_acc - avg_random_acc,
            "heads_used": random_head_sets,
        },
        "causal_effect_ratio": causal_ratio,
        "conclusion": conclusion,
    }

    with open(f"{BASE_DIR}/results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation_results.json", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 3 COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
