"""Step 3b: Incremental ablation curve.

Remove top reasoning heads one at a time and measure accuracy.
Produces data for plotting acc vs n_heads_removed.

Multi-GPU: each GPU handles a different ablation level.
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
N_PROBLEMS = 50
MAX_NEW_TOKENS = 512
N_TOP_HEADS = 12  # test removing up to 12 heads

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

def extract_gsm8k_answer(text):
    match = re.search(r'####\s*([\d,.\-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None

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


def evaluate_condition(model, problems, gold_answers, ablate_heads, label, rank):
    """Evaluate accuracy on all problems with given ablation."""
    correct = 0
    for i, ((q, _), gold) in enumerate(zip(problems, gold_answers)):
        prompt = make_chat_prompt(q)
        response = generate_with_ablation(model, prompt, ablate_heads=ablate_heads)
        pred = extract_model_answer(response)
        if answers_match(pred, gold):
            correct += 1
    acc = correct / len(problems)
    print(f"[GPU {rank}] {label}: {correct}/{len(problems)} = {acc:.1%}", flush=True)
    return correct, acc


def worker(rank, n_gpus, problems, gold_answers, all_levels, top_heads_ordered, result_dict):
    """Each GPU processes a contiguous block of ablation levels, with early stopping."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Give each GPU a contiguous block so early stopping works
    chunk_size = len(all_levels) // n_gpus
    start = rank * chunk_size
    end = start + chunk_size if rank < n_gpus - 1 else len(all_levels)
    my_levels = all_levels[start:end]
    print(f"[GPU {rank}] Handling ablation levels: {my_levels}", flush=True)

    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        dtype=torch.float16,
    )
    model.cfg.use_attn_result = True
    model.setup()

    gpu_results = {}
    for n_remove in my_levels:
        if n_remove == 0:
            ablate_heads = None
            label = "baseline (0 removed)"
        else:
            ablate_heads = top_heads_ordered[:n_remove]
            label = f"top-{n_remove} removed"

        t0 = time.time()
        n_correct, acc = evaluate_condition(model, problems, gold_answers, ablate_heads, label, rank)
        elapsed = time.time() - t0
        gpu_results[n_remove] = {
            "n_removed": n_remove,
            "heads_removed": top_heads_ordered[:n_remove] if n_remove > 0 else [],
            "n_correct": n_correct,
            "accuracy": acc,
            "time_seconds": elapsed,
        }

        # Early stop: if accuracy is 0, all higher levels will also be 0 or near 0
        if n_remove > 0 and acc == 0.0:
            print(f"[GPU {rank}] Accuracy reached 0 at level {n_remove}, filling remaining levels with 0", flush=True)
            for remaining in my_levels[my_levels.index(n_remove) + 1:]:
                gpu_results[remaining] = {
                    "n_removed": remaining,
                    "heads_removed": top_heads_ordered[:remaining],
                    "n_correct": 0,
                    "accuracy": 0.0,
                    "time_seconds": 0,
                    "early_stopped": True,
                }
            break

    result_dict[rank] = gpu_results


def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}", flush=True)

    # Load head scores
    print("Loading reasoning head scores...", flush=True)
    data = torch.load(f"{BASE_DIR}/results/reasoning_heads.pt", weights_only=False)
    head_scores = data["head_scores"]
    n_layers, n_heads = head_scores.shape

    # Get top heads in order
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    top_heads_ordered = []
    for idx in sorted_idx[:N_TOP_HEADS]:
        l = idx.item() // n_heads
        h = idx.item() % n_heads
        top_heads_ordered.append((l, h))

    print(f"Top {N_TOP_HEADS} heads (in order of importance):", flush=True)
    for i, (l, h) in enumerate(top_heads_ordered):
        print(f"  #{i+1}: L{l}H{h} = {head_scores[l,h]:.2f}", flush=True)

    # Load held-out GSM8K problems (different from Step 3 to avoid overlap)
    print(f"\nLoading {N_PROBLEMS} GSM8K problems...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    # Use indices 600+ to avoid overlap with step2 (0-500) and step3 (500-550)
    held_out = indices[600:600 + N_PROBLEMS]
    problems = [(ds[i]["question"], ds[i]["answer"]) for i in held_out]
    gold_answers = [extract_gsm8k_answer(a) for _, a in problems]
    print(f"Using {len(problems)} problems", flush=True)

    # Ablation levels: 0, 1, 2, 3, ..., N_TOP_HEADS
    all_levels = list(range(N_TOP_HEADS + 1))  # 0 through 20
    print(f"\nAblation levels: {all_levels}", flush=True)
    print(f"Total evaluations: {len(all_levels)} levels x {N_PROBLEMS} problems = {len(all_levels) * N_PROBLEMS}", flush=True)

    t0 = time.time()
    if n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(
            worker,
            args=(n_gpus, problems, gold_answers, all_levels, top_heads_ordered, result_dict),
            nprocs=n_gpus,
            join=True,
        )
    else:
        result_dict = {}
        worker(0, 1, problems, gold_answers, all_levels, top_heads_ordered, result_dict)

    total_time = time.time() - t0
    print(f"\nAll done in {total_time/60:.1f} min", flush=True)

    # Aggregate
    all_results = {}
    for rank in range(n_gpus):
        all_results.update(result_dict[rank])

    # Sort by n_removed
    curve = []
    print("\n" + "=" * 60, flush=True)
    print("INCREMENTAL ABLATION CURVE", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Heads removed':>15} {'Accuracy':>10} {'Correct':>10} {'Head removed':>20}", flush=True)
    print("-" * 60, flush=True)
    for n in sorted(all_results.keys()):
        r = all_results[n]
        head_label = f"L{top_heads_ordered[n-1][0]}H{top_heads_ordered[n-1][1]}" if n > 0 else "-"
        print(f"{n:>15} {r['accuracy']:>10.1%} {r['n_correct']:>10}/{N_PROBLEMS}   {head_label:>20}", flush=True)
        curve.append(r)

    # Save
    output = {
        "config": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "n_problems": N_PROBLEMS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_top_heads": N_TOP_HEADS,
            "n_gpus": n_gpus,
            "total_time_minutes": total_time / 60,
        },
        "top_heads_ordered": top_heads_ordered,
        "top_heads_scores": [float(head_scores[l, h]) for l, h in top_heads_ordered],
        "curve": curve,
    }

    with open(f"{BASE_DIR}/results/incremental_ablation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved incremental_ablation.json", flush=True)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["n_removed"] for r in curve]
    ys = [r["accuracy"] for r in curve]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, 'o-', color='#d62728', linewidth=2, markersize=6)
    ax.set_xlabel("Number of top reasoning heads removed", fontsize=12)
    ax.set_ylabel("GSM8K Accuracy", fontsize=12)
    ax.set_title("Incremental Ablation of Reasoning Heads", fontsize=14)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=ys[0], color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({ys[0]:.0%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate key heads
    for i in range(min(5, len(xs) - 1)):
        l, h = top_heads_ordered[i]
        ax.annotate(f'L{l}H{h}', (xs[i+1], ys[i+1]),
                    textcoords="offset points", xytext=(5, 10), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/incremental_ablation_curve.png", dpi=150)
    print(f"Saved incremental_ablation_curve.png", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("STEP 3b COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
