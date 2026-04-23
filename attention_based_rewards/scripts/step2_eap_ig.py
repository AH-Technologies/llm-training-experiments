"""Step 2: Run EAP-IG circuit discovery following Thinking Sparks methodology.

Multi-GPU: each GPU processes a shard of pairs independently, then scores
are aggregated. Uses torch.multiprocessing.spawn for parallelism.

EAP-IG config (from Thinking Sparks):
- method: EAP-IG-inputs, ig_steps=100, top_n=5000, threshold tau=0.1
- metric: KL divergence (task-agnostic)
"""
import os
import random
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
N_PAIRS = 300
IG_STEPS = 100
TOP_N = 5000

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

# ── Prefix matching ───────────────────────────────────────────────────
reasoning_candidates = [
    "<think>\nOkay, so I have this problem",
    "<think>\nAlright, let me think about this",
    "<think>\nOkay, so I need to find",
    "<think>\nLet me work through this step by step",
    "<think>\nHmm, let me think about this carefully",
    "<think>\nOkay, let me break this down",
]
direct_candidates = [
    "To solve this problem, we need to",
    "The answer can be found by",
    "We can solve this by computing",
    "Let me calculate this directly. First",
    "Using the given information, we",
    "To determine the answer, first",
]

def find_matched_prefixes(model):
    """Find or create prefix pairs that tokenize to the same length."""
    matched = []
    for r in reasoning_candidates:
        rl = model.to_tokens(r).shape[1]
        for d in direct_candidates:
            dl = model.to_tokens(d).shape[1]
            if rl == dl:
                matched.append((r, d))
            elif abs(rl - dl) <= 5:
                short, long_len = (r, dl) if rl < dl else (d, rl)
                padded = short
                for _ in range(20):
                    padded += "."
                    if model.to_tokens(padded).shape[1] == long_len:
                        if rl < dl:
                            matched.append((padded, d))
                        else:
                            matched.append((r, padded))
                        break
    return matched

# ── Dataset ───────────────────────────────────────────────────────────
class EAPDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

# ── KL metric ─────────────────────────────────────────────────────────
def kl_divergence_metric(logits, clean_logits, input_lengths, labels):
    """KL(clean || model) at the last non-padded position."""
    batch_size = logits.shape[0]
    kl_total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    for i in range(batch_size):
        last_pos = input_lengths[i] - 1
        clean_lp = F.log_softmax(clean_logits[i, last_pos].float(), dim=-1)
        model_lp = F.log_softmax(logits[i, last_pos].float(), dim=-1)
        kl = (clean_lp.exp() * (clean_lp - model_lp)).sum()
        kl_total = kl_total + kl
    return kl_total / batch_size

# ── Per-GPU worker ────────────────────────────────────────────────────
def worker(rank, n_gpus, all_pairs, result_dict):
    """Run EAP-IG on a shard of the data on GPU `rank`."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Shard the data
    shard_size = len(all_pairs) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(all_pairs)
    shard = all_pairs[start:end]
    print(f"[GPU {rank}] Processing pairs {start}-{end} ({len(shard)} pairs)")

    # Load model on this GPU
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device=device,
        dtype=torch.float16,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.setup()

    graph = Graph.from_model(model)
    dataset = EAPDataset(shard)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    t0 = time.time()
    attribute(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric=kl_divergence_metric,
        method="EAP-IG-inputs",
        ig_steps=IG_STEPS,
        quiet=(rank != 0),  # only show progress for GPU 0
    )
    elapsed = time.time() - t0
    print(f"[GPU {rank}] Done in {elapsed/60:.1f} min")

    # Store scores (weighted by shard size for proper averaging)
    result_dict[rank] = graph.scores.cpu() * len(shard)

# ── Main ──────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Build dataset on CPU (avoid CUDA init in parent before mp.spawn)
    print("\nLoading model on CPU for dataset construction...")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device="cpu",
        dtype=torch.float32,
    )

    matched = find_matched_prefixes(model)
    assert matched, "No matched-length prefix pairs found!"
    print(f"Found {len(matched)} matched prefix pairs")

    print("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    questions = [ds[i]["question"] for i in indices[:500]]

    all_pairs = []
    for q in questions:
        if len(all_pairs) >= N_PAIRS:
            break
        r, d = random.choice(matched)
        chat = make_chat_prompt(q)
        clean, corrupt = chat + r, chat + d
        if model.to_tokens(clean).shape[1] != model.to_tokens(corrupt).shape[1]:
            continue
        all_pairs.append((clean, corrupt, 0))

    print(f"Dataset: {len(all_pairs)} pairs")
    del model

    # Run on all GPUs
    print(f"\nLaunching EAP-IG on {n_gpus} GPU(s)...")
    print(f"  ig_steps={IG_STEPS}, pairs={len(all_pairs)}")
    print(f"  ~{len(all_pairs) // n_gpus} pairs per GPU")
    print(f"  Estimated: ~{len(all_pairs) * IG_STEPS * 1.5 / n_gpus / 60:.0f} min")

    t0 = time.time()
    if n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(worker, args=(n_gpus, all_pairs, result_dict), nprocs=n_gpus, join=True)
    else:
        result_dict = {}
        worker(0, 1, all_pairs, result_dict)

    total_elapsed = time.time() - t0
    print(f"\nAll GPUs done in {total_elapsed/60:.1f} min")

    # Aggregate scores: weighted average across shards
    print("Aggregating scores...")
    agg_scores = sum(result_dict[r] for r in range(n_gpus)) / len(all_pairs)

    # Apply to a fresh graph
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", device="cpu", dtype=torch.float16,
    )
    graph = Graph.from_model(model)
    graph.scores[:] = agg_scores

    # Top-n selection and pruning
    print(f"\nApplying top-{TOP_N} edge selection...")
    graph.apply_topn(n=TOP_N)
    print(f"Edges in circuit: {graph.count_included_edges()}")
    print(f"Nodes in circuit: {graph.count_included_nodes()}")

    # Extract per-head importance
    print("\nExtracting per-head importance scores...")
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    head_scores = torch.zeros(n_layers, n_heads)

    for layer in range(n_layers):
        for head in range(n_heads):
            node = graph.nodes[f"a{layer}.h{head}"]
            fwd_idx = graph.forward_index(node, attn_slice=False)
            outgoing = graph.scores[fwd_idx, :].abs().sum().item()
            incoming = 0
            for letter in "qkv":
                bwd_idx = graph.backward_index(node, qkv=letter, attn_slice=False)
                incoming += graph.scores[:, bwd_idx].abs().sum().item()
            head_scores[layer, head] = outgoing + incoming

    # Rank and print top heads
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    print(f"\nHead scores range: [{head_scores.min():.4f}, {head_scores.max():.4f}]")
    print("\nTop 20 attention heads by importance:")
    selected_heads = []
    for rank, idx in enumerate(sorted_idx[:20]):
        l, h = idx.item() // n_heads, idx.item() % n_heads
        score = flat[idx].item()
        in_circ = graph.nodes[f"a{l}.h{h}"].in_graph
        selected_heads.append((l, h, score))
        print(f"  {'*' if in_circ else ' '} #{rank+1}: L{l}H{h} = {score:.4f}")

    # Compare with Thinking Sparks Table 1
    ts_openr1 = {(0,8),(5,1),(7,1),(18,11),(11,8)}
    ts_gsm8k = {(0,8),(5,1),(7,2),(3,3),(21,2)}
    our = {(l,h) for l,h,_ in selected_heads}
    print(f"\nOverlap with TS GRPO+OpenR1: {our & ts_openr1} ({len(our & ts_openr1)}/5)")
    print(f"Overlap with TS GRPO+GSM8K:  {our & ts_gsm8k} ({len(our & ts_gsm8k)}/5)")

    # Heatmap
    # Save results FIRST before plotting (so we don't lose data if plotting fails)
    torch.save({
        "head_scores": head_scores,
        "selected_heads": selected_heads,
        "scores_matrix": graph.scores.cpu(),
        "in_graph": graph.in_graph.cpu(),
        "nodes_in_graph": graph.nodes_in_graph.cpu(),
        "config": {
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "method": "EAP-IG-inputs",
            "ig_steps": IG_STEPS,
            "top_n": TOP_N,
            "n_pairs": len(all_pairs),
            "n_gpus": n_gpus,
            "metric": "KL_divergence",
            "elapsed_minutes": total_elapsed / 60,
        },
    }, f"{BASE_DIR}/results/reasoning_heads.pt")
    graph.to_json(f"{BASE_DIR}/results/circuit.json")
    print(f"Saved reasoning_heads.pt and circuit.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
        print("seaborn not available, using matplotlib imshow")

    plt.figure(figsize=(14, 8))
    if has_seaborn:
        sns.heatmap(
            head_scores.numpy(),
            xticklabels=[f"H{h}" for h in range(n_heads)],
            yticklabels=[f"L{l}" for l in range(n_layers)],
            cmap="Reds",
        )
    else:
        plt.imshow(head_scores.numpy(), cmap="Reds", aspect="auto")
        plt.xticks(range(n_heads), [f"H{h}" for h in range(n_heads)])
        plt.yticks(range(n_layers), [f"L{l}" for l in range(n_layers)])
        plt.colorbar()
    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title("EAP-IG: Attention Head Importance (reasoning vs direct)")
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/plots/head_importance_heatmap.png", dpi=150)
    print(f"Saved heatmap")

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
