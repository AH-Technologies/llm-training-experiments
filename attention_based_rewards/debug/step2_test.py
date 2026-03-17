"""Quick test of EAP-IG pipeline with 5 pairs and 5 IG steps."""
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute

random.seed(42)
torch.manual_seed(42)

# ── 1. Load model ─────────────────────────────────────────────────────
print("Loading model with EAP-IG config flags...")
try:
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        dtype=torch.float16,
        attn_result=True,
        split_qkv_input=True,
        hook_mlp_in=True,
        ungroup_grouped_query_attention=True,
    )
except TypeError as e:
    print(f"kwargs not supported, trying cfg overrides: {e}")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        dtype=torch.float16,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.setup()

print(f"use_attn_result={model.cfg.use_attn_result}")
print(f"use_split_qkv_input={model.cfg.use_split_qkv_input}")
print(f"use_hook_mlp_in={model.cfg.use_hook_mlp_in}")
print(f"n_key_value_heads={model.cfg.n_key_value_heads}")
if model.cfg.n_key_value_heads is not None:
    print(f"ungroup_grouped_query_attention={model.cfg.ungroup_grouped_query_attention}")

# ── 2. Find matched-length prefixes ──────────────────────────────────
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

reasoning_candidates = [
    "<think>\nOkay, so I have this problem",
    "<think>\nAlright, let me think about this",
    "<think>\nOkay, so I need to find",
    "<think>\nLet me work through this step by step",
]
direct_candidates = [
    "To solve this problem, we need to",
    "The answer can be found by",
    "We can solve this by computing",
    "Let me calculate this directly. First",
]

print("\nPrefix token lengths:")
for p in reasoning_candidates:
    print(f"  R({model.to_tokens(p).shape[1]}): {p}")
for p in direct_candidates:
    print(f"  D({model.to_tokens(p).shape[1]}): {p}")

# Find matches or pad
matched = []
for r in reasoning_candidates:
    rl = model.to_tokens(r).shape[1]
    for d in direct_candidates:
        dl = model.to_tokens(d).shape[1]
        if rl == dl:
            matched.append((r, d))
            print(f"\n  EXACT MATCH ({rl} toks): R='{r[:30]}' D='{d[:30]}'")

if not matched:
    print("\n  No exact matches. Padding shorter prefixes...")
    for r in reasoning_candidates:
        rl = model.to_tokens(r).shape[1]
        for d in direct_candidates:
            dl = model.to_tokens(d).shape[1]
            diff = abs(rl - dl)
            if diff <= 5:
                if rl < dl:
                    padded = r
                    for _ in range(20):
                        padded += "."
                        if model.to_tokens(padded).shape[1] == dl:
                            matched.append((padded, d))
                            print(f"  PADDED ({dl} toks): R='{padded[:40]}' D='{d[:40]}'")
                            break
                else:
                    padded = d
                    for _ in range(20):
                        padded += "."
                        if model.to_tokens(padded).shape[1] == rl:
                            matched.append((r, padded))
                            print(f"  PADDED ({rl} toks): R='{r[:40]}' D='{padded[:40]}'")
                            break

if not matched:
    print("\nFAILED: Could not find any matched-length prefix pairs!")
    import sys; sys.exit(1)

print(f"\n{len(matched)} matched pairs found")

# ── 3. Build small test dataset ───────────────────────────────────────
ds = load_dataset("openai/gsm8k", "main", split="train")
questions = [ds[i]["question"] for i in range(20)]

class TestDataset(Dataset):
    def __init__(self, questions, matched_prefixes, n=5):
        self.data = []
        for q in questions:
            if len(self.data) >= n:
                break
            r, d = random.choice(matched_prefixes)
            chat = make_chat_prompt(q)
            clean = chat + r
            corrupt = chat + d
            cl = model.to_tokens(clean).shape[1]
            cr = model.to_tokens(corrupt).shape[1]
            if cl != cr:
                print(f"  LENGTH MISMATCH: clean={cl}, corrupt={cr}")
                continue
            self.data.append((clean, corrupt, 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

dataset = TestDataset(questions, matched, n=5)
print(f"\nTest dataset: {len(dataset)} pairs")
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

# Verify lengths
for clean, corrupt, _ in dataloader:
    cl = model.to_tokens(clean).shape[1]
    cr = model.to_tokens(corrupt).shape[1]
    print(f"  clean={cl}, corrupt={cr}, match={cl==cr}")

# ── 4. KL metric ──────────────────────────────────────────────────────
def kl_metric(logits, clean_logits, input_lengths, labels):
    batch_size = logits.shape[0]
    kl_total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    for i in range(batch_size):
        last_pos = input_lengths[i] - 1
        cp = F.log_softmax(clean_logits[i, last_pos].float(), dim=-1)
        mp = F.log_softmax(logits[i, last_pos].float(), dim=-1)
        kl = (cp.exp() * (cp - mp)).sum()
        kl_total = kl_total + kl
    return kl_total / batch_size

# ── 5. Build graph and test attribution ───────────────────────────────
graph = Graph.from_model(model)
print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

print(f"\nRunning test attribution (5 pairs, 5 IG steps)...")
t0 = time.time()
attribute(
    model=model,
    graph=graph,
    dataloader=dataloader,
    metric=kl_metric,
    method="EAP-IG-inputs",
    ig_steps=5,
    quiet=False,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Estimated full run (300 pairs, 100 steps): {elapsed * (300/5) * (100/5) / 3600:.1f} hours")

# Check scores
print(f"\nScore matrix: min={graph.scores.min():.6f}, max={graph.scores.max():.6f}")
print(f"Non-zero scores: {(graph.scores != 0).sum().item()}")

graph.apply_topn(n=5000)
print(f"After top-5000: {graph.count_included_edges()} edges, {graph.count_included_nodes()} nodes")

print("\nTEST PASSED - EAP-IG pipeline works!")
