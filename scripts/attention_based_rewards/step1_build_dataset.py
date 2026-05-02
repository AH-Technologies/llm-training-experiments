"""Step 1: Build diagnostic dataset following Thinking Sparks (Park et al., 2025) A.2.

Clean = chat prompt + reasoning prefix (<think>Okay, so I have this problem...)
Corrupted = chat prompt + direct prefix (To solve this problem, we...)

Uses GSM8K from HuggingFace as the source of math questions.
"""
import random
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards/results"

# ── 1. Load model ──────────────────────────────────────────────────────
print("Loading model...")
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cuda",
    dtype=torch.float16,
)
print(f"Model loaded: {model.cfg.n_layers}L, {model.cfg.n_heads}H")

# ── 2. Load GSM8K ─────────────────────────────────────────────────────
print("\nLoading GSM8K...")
ds = load_dataset("openai/gsm8k", "main", split="train")
print(f"GSM8K train: {len(ds)} problems")

# Shuffle and take first 400 (we'll filter some out)
indices = list(range(len(ds)))
random.shuffle(indices)
questions = [ds[i]["question"] for i in indices[:400]]

# ── 3. Build chat-template prompts ────────────────────────────────────
# System prompt from paper A.1
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

# From paper A.2: reasoning model responses start with <think>
REASONING_PREFIXES = [
    "<think>\nOkay, so I have this problem",
    "<think>\nAlright, let me think about this",
    "<think>\nOkay, so I need to find",
    "<think>\nLet me work through this step by step",
    "<think>\nHmm, let me think about this carefully",
]

# Baseline model responses are direct (no <think>)
DIRECT_PREFIXES = [
    "To solve this problem, we",
    "The answer is",
    "We can solve this by",
    "Let me calculate this directly.",
    "Using the formula,",
]

# ── 4. Build pairs ────────────────────────────────────────────────────
print("\nBuilding clean/corrupted pairs...")
pairs = []
skipped = 0

for question in questions:
    if len(pairs) >= 300:
        break

    chat_prompt = make_chat_prompt(question)
    r_prefix = random.choice(REASONING_PREFIXES)
    d_prefix = random.choice(DIRECT_PREFIXES)

    clean_text = chat_prompt + r_prefix
    corrupt_text = chat_prompt + d_prefix

    clean_toks = model.to_tokens(clean_text)
    corrupt_toks = model.to_tokens(corrupt_text)

    max_len = max(clean_toks.shape[1], corrupt_toks.shape[1])
    if max_len > 512:
        skipped += 1
        continue

    # Pad shorter sequence to match (EAP-IG needs same length)
    if clean_toks.shape[1] != corrupt_toks.shape[1]:
        target_len = max_len
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        device = clean_toks.device
        if clean_toks.shape[1] < target_len:
            pad = torch.full((1, target_len - clean_toks.shape[1]), pad_id, dtype=clean_toks.dtype, device=device)
            clean_toks = torch.cat([pad, clean_toks], dim=1)
        if corrupt_toks.shape[1] < target_len:
            pad = torch.full((1, target_len - corrupt_toks.shape[1]), pad_id, dtype=corrupt_toks.dtype, device=device)
            corrupt_toks = torch.cat([pad, corrupt_toks], dim=1)

    pairs.append({
        "question": question,
        "clean_text": clean_text,
        "corrupt_text": corrupt_text,
        "clean_tokens": clean_toks.cpu(),
        "corrupt_tokens": corrupt_toks.cpu(),
        "reasoning_prefix": r_prefix,
        "direct_prefix": d_prefix,
    })

print(f"Generated {len(pairs)} pairs (skipped {skipped} too-long)")

# ── 5. Samples ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE PAIRS")
print("=" * 60)
for p in pairs[:3]:
    print(f"\nQ: {p['question'][:80]}...")
    print(f"Clean ends with:   ...{p['reasoning_prefix']}")
    print(f"Corrupt ends with: ...{p['direct_prefix']}")
    print(f"Token length: {p['clean_tokens'].shape[1]}")

# ── 6. Sanity: KL divergence between clean and corrupt outputs ────────
print("\n" + "=" * 60)
print("SANITY: KL divergence at last token position")
print("=" * 60)
diffs = []
for p in pairs[:20]:
    cl = model(p["clean_tokens"].to("cuda"))
    cr = model(p["corrupt_tokens"].to("cuda"))
    cp = torch.softmax(cl[0, -1].float(), dim=-1)
    crp = torch.softmax(cr[0, -1].float(), dim=-1)
    kl = (cp * (cp / crp).log()).sum().item()
    diffs.append(kl)
print(f"Mean KL: {sum(diffs)/len(diffs):.4f} (should be > 0)")

# ── 7. Stats ──────────────────────────────────────────────────────────
tok_lens = [p["clean_tokens"].shape[1] for p in pairs]
print(f"\nToken lengths: min={min(tok_lens)}, max={max(tok_lens)}, "
      f"mean={sum(tok_lens)/len(tok_lens):.0f}")

# ── 8. Save ───────────────────────────────────────────────────────────
save_path = f"{RESULTS_DIR}/diagnostic_pairs.pt"
torch.save({
    "pairs": pairs,
    "n_pairs": len(pairs),
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "methodology": "Thinking Sparks A.2: reasoning (<think>) vs direct response",
    "system_prompt": SYSTEM_PROMPT,
}, save_path)
print(f"\nSaved {len(pairs)} pairs to {save_path}")
