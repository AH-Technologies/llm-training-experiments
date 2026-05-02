"""Step 3: Ablation validation for base model reasoning heads.

Uses few-shot prompting since this is a base (completion) model.
Three conditions:
  1. Baseline: no ablation
  2. Top-10 heads ablated (from step 2)
  3. Random 10 heads ablated (5 repeats)

Multi-GPU: splits problems across GPUs for each condition.
"""

import json
import random
import re
import time

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
TL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # TL architecture name (Math variant loaded via hf_model)
N_PROBLEMS = 50
MAX_NEW_TOKENS = 256
N_RANDOM_TRIALS = 5

# 3 easy GSM8K few-shot examples (short answers, simple arithmetic)
FEW_SHOT_EXAMPLES = [
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?", "18"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "3"),
    ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "70000"),
]


def make_fewshot_prompt(question):
    """Build a few-shot prompt for the base model."""
    prompt = ""
    for q, a in FEW_SHOT_EXAMPLES:
        prompt += f"Question: {q}\nAnswer: {a}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def extract_answer(text):
    """Extract the first number from the model's completion."""
    # Look for numbers (possibly with commas/decimals)
    nums = re.findall(r'[\d,]+\.?\d*', text)
    if nums:
        return nums[0].replace(',', '')
    return None


def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,.\-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


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
        if "\n" in decoded:  # Stop at newline (answer should be on one line)
            break

    full_text = model.to_string(tokens[0])
    # Extract text after "Answer:" for the test question
    if "Answer:" in full_text:
        parts = full_text.split("Answer:")
        response = parts[-1].strip()
    else:
        response = full_text[len(prompt):]
    return response.strip()


def evaluate_problems(model, problems, gold_answers, ablate_heads, label, rank):
    """Evaluate accuracy on problems."""
    correct = 0
    for i, (q, gold) in enumerate(zip(problems, gold_answers)):
        prompt = make_fewshot_prompt(q)
        response = generate_with_ablation(model, prompt, ablate_heads=ablate_heads)
        pred = extract_answer(response)
        match = answers_match(pred, gold)
        if match:
            correct += 1
        if i < 5 or (i + 1) % 10 == 0:
            print(f"  [GPU {rank}] [{i+1}/{len(problems)}] pred={pred} gold={gold} {'Y' if match else 'N'} | {response[:50]}", flush=True)
    return correct


def worker(rank, n_gpus, problems, gold_answers, top_heads, all_random_sets, result_dict):
    """Each GPU handles a subset of problems for all conditions."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Split problems across GPUs
    chunk = len(problems) // n_gpus
    start = rank * chunk
    end = start + chunk if rank < n_gpus - 1 else len(problems)
    my_problems = problems[start:end]
    my_golds = gold_answers[start:end]

    print(f"[GPU {rank}] Problems {start}-{end-1} ({len(my_problems)} total)", flush=True)

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME, hf_model=hf_model, device=device, dtype=torch.float16,
    )
    del hf_model
    model.cfg.use_attn_result = True
    model.setup()

    results = {}

    # 1. Baseline
    print(f"\n[GPU {rank}] === BASELINE ===", flush=True)
    t0 = time.time()
    baseline_correct = evaluate_problems(model, my_problems, my_golds, None, "baseline", rank)
    results["baseline"] = {"correct": baseline_correct, "total": len(my_problems), "time": time.time() - t0}

    # 2. Top-10 ablated
    print(f"\n[GPU {rank}] === TOP-10 ABLATED ===", flush=True)
    t0 = time.time()
    top10_correct = evaluate_problems(model, my_problems, my_golds, top_heads[:10], "top10", rank)
    results["top10"] = {"correct": top10_correct, "total": len(my_problems), "time": time.time() - t0}

    # 3. Random-10 ablated (N_RANDOM_TRIALS repeats)
    random_results = []
    for trial, random_heads in enumerate(all_random_sets):
        print(f"\n[GPU {rank}] === RANDOM-10 trial {trial+1}/{N_RANDOM_TRIALS} ===", flush=True)
        t0 = time.time()
        rand_correct = evaluate_problems(model, my_problems, my_golds, random_heads, f"random_{trial}", rank)
        random_results.append({"correct": rand_correct, "total": len(my_problems), "time": time.time() - t0})

    results["random"] = random_results
    result_dict[rank] = results


def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Load base model reasoning heads from step 2
    heads_path = f"{BASE_DIR}/data/base_model_reasoning_heads.pt"
    print(f"Loading reasoning heads from {heads_path}...")
    data = torch.load(heads_path, weights_only=False)
    head_scores = data["head_scores"]
    n_layers, n_heads = head_scores.shape

    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    top_heads = [(idx.item() // n_heads, idx.item() % n_heads) for idx in sorted_idx[:20]]

    print("Top 10 reasoning heads (base model):")
    for i, (l, h) in enumerate(top_heads[:10]):
        print(f"  #{i+1}: L{l}H{h} = {head_scores[l,h]:.4f}")

    # Generate random head sets (excluding top-10 to ensure fair comparison)
    top10_set = set(top_heads[:10])
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in top10_set]
    all_random_sets = []
    for _ in range(N_RANDOM_TRIALS):
        all_random_sets.append(random.sample(all_heads, 10))

    # Load GSM8K problems
    print(f"\nLoading {N_PROBLEMS} GSM8K problems...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    # Use indices 700+ to avoid overlap with previous experiments
    held_out = indices[700:700 + N_PROBLEMS]
    problems = [ds[i]["question"] for i in held_out]
    gold_answers = [extract_gsm8k_answer(ds[i]["answer"]) for i in held_out]

    # Quick check: does few-shot prompting work?
    print(f"\nSample prompt (truncated):")
    sample = make_fewshot_prompt(problems[0])
    print(f"  {sample[:200]}...")
    print(f"  ...{sample[-100:]}")

    # Run evaluation
    t0 = time.time()
    if n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(worker, args=(n_gpus, problems, gold_answers, top_heads, all_random_sets, result_dict),
                 nprocs=n_gpus, join=True)
    else:
        result_dict = {}
        worker(0, 1, problems, gold_answers, top_heads, all_random_sets, result_dict)

    total_elapsed = time.time() - t0

    # Aggregate results across GPUs
    baseline_correct = sum(result_dict[r]["baseline"]["correct"] for r in range(n_gpus))
    top10_correct = sum(result_dict[r]["top10"]["correct"] for r in range(n_gpus))

    random_corrects = []
    for trial in range(N_RANDOM_TRIALS):
        trial_correct = sum(result_dict[r]["random"][trial]["correct"] for r in range(n_gpus))
        random_corrects.append(trial_correct)

    baseline_acc = baseline_correct / N_PROBLEMS
    top10_acc = top10_correct / N_PROBLEMS
    random_accs = [c / N_PROBLEMS for c in random_corrects]
    random_mean = sum(random_accs) / len(random_accs)

    # Results
    print(f"\n{'='*60}")
    print(f"ABLATION RESULTS ({total_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Problems: {N_PROBLEMS} GSM8K (few-shot)")
    print(f"")
    print(f"Baseline accuracy:     {baseline_acc:.1%} ({baseline_correct}/{N_PROBLEMS})")
    print(f"Top-10 ablated:        {top10_acc:.1%} ({top10_correct}/{N_PROBLEMS})")
    print(f"Random-10 ablated:     {random_mean:.1%} (range: {min(random_accs):.1%}-{max(random_accs):.1%})")

    if baseline_acc > 0 and random_mean < baseline_acc:
        causal_ratio = (baseline_acc - top10_acc) / (baseline_acc - random_mean)
        print(f"Causal effect ratio:   {causal_ratio:.2f}x")
    else:
        causal_ratio = None
        print(f"Causal effect ratio:   N/A (baseline={baseline_acc:.1%}, random_mean={random_mean:.1%})")

    drop = baseline_acc - top10_acc
    if drop > 0.05:
        print(f"\nREASONING HEADS VALIDATED: {drop:.1%} accuracy drop from top-10 ablation.")
    elif baseline_acc < 0.2:
        print(f"\nWARNING: Baseline accuracy very low ({baseline_acc:.1%}).")
        print("  Few-shot format may not be working. Consider more examples or different format.")
    else:
        print(f"\nWEAK/NO EFFECT: Only {drop:.1%} drop from top-10 ablation.")

    # Save
    results = {
        "model": MODEL_NAME,
        "n_problems": N_PROBLEMS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_fewshot": len(FEW_SHOT_EXAMPLES),
        "top_heads": top_heads[:10],
        "baseline_accuracy": baseline_acc,
        "top10_ablated_accuracy": top10_acc,
        "random_ablated_accuracies": random_accs,
        "random_ablated_mean": random_mean,
        "causal_effect_ratio": causal_ratio,
        "accuracy_drop": drop,
        "time_minutes": total_elapsed / 60,
    }

    output_path = f"{BASE_DIR}/data/base_model_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    print(f"\n{'='*60}")
    print("STEP 3 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
