# Implementation Plan v2: Circuit-Guided Credit Assignment for GRPO

## Context & Motivation

Standard GRPO applies uniform reward to all tokens in a response. We want to use
mechanistic interpretability to identify which attention heads perform reasoning, then
use those heads' attention patterns to weight per-token rewards during GRPO training.

A key reference paper — "Thinking Sparks!: Emergent Attention Heads in Reasoning
Models During Post Training" (Park et al., 2025) — has already done circuit analysis
on Qwen2.5-Math-1.5B during GRPO training using EAP-IG. They identified emergent
reasoning heads and called for exactly the kind of reward-shaping work we're doing,
but nobody has implemented it yet. We follow their methodology for circuit discovery
and extend it by feeding the results into GRPO credit assignment.

Their key findings relevant to us:
- GRPO produces ~19 emergent reasoning heads (small, sparse, targeted)
- These heads track reward fluctuations during training
- Heads can be validated by ablation (zeroing out and measuring performance drop)
- The EAP-IG method with ig_steps=100, top_n=5000, threshold=0.1 works well
- They used Qwen2.5-Math-1.5B-Instruct as the base model

---

## Step 0 — Environment Setup

### Dependencies

```bash
# Core libraries
pip install transformer_lens torch einops jaxtyping

# EAP-IG library (the circuit discovery tool used by Thinking Sparks paper)
pip install eap-ig
# OR install from source for latest version:
# git clone https://github.com/hannamw/EAP-IG.git
# cd EAP-IG && pip install .

# For visualization of circuits (optional but nice)
pip install eap-ig[viz]
# This may require graphviz system package:
# apt-get install graphviz  (Ubuntu)
# brew install graphviz     (Mac)

# For generating diagnostic tasks
pip install reasoning_gym

# Standard scientific computing
pip install matplotlib seaborn pandas
```

### Verify EAP-IG installation

```python
# Quick smoke test
from eap.graph import Graph
from transformer_lens import HookedTransformer

print("EAP-IG imported successfully")
print("TransformerLens imported successfully")
```

### Model Choice

Use **Qwen2.5-Math-1.5B-Instruct** — same model as the Thinking Sparks paper.
This lets us directly compare our discovered heads against their published Table 1.

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    device="cuda"
)

print(f"Layers: {model.cfg.n_layers}")
print(f"Heads per layer: {model.cfg.n_heads}")
print(f"Model dim: {model.cfg.d_model}")
print(f"Head dim: {model.cfg.d_head}")
```

### CRITICAL: Verify TransformerLens + EAP-IG compatibility with Qwen

Before doing anything else, run this compatibility check. If this fails, we need
to either fix the integration or fall back to a supported model (e.g., Llama-3.2-1B).

```python
# Test 1: Can TransformerLens load the model and run inference?
tokens = model.to_tokens("3 + 5 =")
logits = model(tokens)
print(f"Output shape: {logits.shape}")
print(f"Top prediction: {model.to_string(logits[0, -1].argmax())}")

# Test 2: Can we cache activations?
logits, cache = model.run_with_cache(tokens)
print(f"Cache keys: {len(cache)}")
# Check that attention head outputs are accessible
hook_name = "blocks.0.attn.hook_result"
if hook_name in cache:
    print(f"Head output shape: {cache[hook_name].shape}")
else:
    print("WARNING: Expected hook not found. Check TransformerLens hook names for this model.")
    print(f"Available hooks containing 'attn': {[k for k in cache.keys() if 'attn' in k][:10]}")

# Test 3: Can EAP-IG build a graph from this model?
from eap.graph import Graph
graph = Graph.from_model(model)
print(f"Graph nodes: {len(graph.nodes)}")
print(f"Graph edges: {len(graph.edges)}")
```

If Test 3 fails, the EAP-IG library may not support the Qwen architecture out of
the box. In that case, check the EAP-IG GitHub issues and TransformerLens model
support page. Fallback options:
- Llama-3.2-1B (well supported by both libraries)
- Qwen2.5-1.5B base (non-Math, non-Instruct — might have different support)
- Pythia-2.8B (guaranteed to work but less relevant to our GRPO setup)

---

## Step 1 — Build Diagnostic Dataset

### Requirements

We need clean/corrupted prompt pairs where:
- Clean prompt has a definite correct answer as the next token(s)
- Corrupted prompt has the same structure but a DIFFERENT correct answer
- Both tokenize to the same length

### Task types

Start with arithmetic since it's the most likely to work on a 1.5B model
in single-pass (no chain-of-thought needed for circuit discovery).

```python
import random
import torch

def generate_arithmetic_pairs(model, n=300, max_val=20):
    """
    Generate clean/corrupted arithmetic pairs.
    Validates that both prompts tokenize to same length
    and that the answer is a single token.
    """
    pairs = []
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}

    while len(pairs) < n:
        op_sym = random.choice(list(ops.keys()))
        op_fn = ops[op_sym]

        a1, b1 = random.randint(1, max_val), random.randint(1, max_val)
        a2, b2 = random.randint(1, max_val), random.randint(1, max_val)

        ans1 = op_fn(a1, b1)
        ans2 = op_fn(a2, b2)

        if ans1 == ans2:
            continue

        clean_prompt = f"{a1} {op_sym} {b1} ="
        corrupt_prompt = f"{a2} {op_sym} {b2} ="

        # Tokenize and validate
        clean_toks = model.to_tokens(clean_prompt)
        corrupt_toks = model.to_tokens(corrupt_prompt)

        # Must be same length
        if clean_toks.shape[1] != corrupt_toks.shape[1]:
            continue

        # Answer should ideally be a single token
        ans1_toks = model.to_tokens(f" {ans1}")
        ans2_toks = model.to_tokens(f" {ans2}")

        # We want the first real answer token (skip BOS if present)
        # Get the last token of the answer encoding
        ans1_id = ans1_toks[0, -1].item()
        ans2_id = ans2_toks[0, -1].item()

        pairs.append({
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "clean_answer": str(ans1),
            "corrupt_answer": str(ans2),
            "clean_answer_id": ans1_id,
            "corrupt_answer_id": ans2_id,
            "clean_tokens": clean_toks,
            "corrupt_tokens": corrupt_toks,
        })

    return pairs

pairs = generate_arithmetic_pairs(model, n=300)
print(f"Generated {len(pairs)} valid pairs")
print(f"Example: {pairs[0]['clean_prompt']} -> {pairs[0]['clean_answer']}")
print(f"Corrupt: {pairs[0]['corrupt_prompt']} -> {pairs[0]['corrupt_answer']}")
```

### Sanity check: can the model solve these?

```python
correct = 0
for pair in pairs[:100]:
    logits = model(pair["clean_tokens"])
    pred_id = logits[0, -1].argmax().item()
    if pred_id == pair["clean_answer_id"]:
        correct += 1

accuracy = correct / 100
print(f"Model accuracy on clean prompts: {accuracy:.1%}")

# We want 30-70% accuracy. If too low, simplify tasks (smaller numbers).
# If too high, the tasks are too easy and all heads contribute trivially.
```

### Optional: add more task types

If arithmetic works, consider adding comparison and simple logic tasks:

```python
def generate_comparison_pairs(model, n=100):
    pairs = []
    while len(pairs) < n:
        a1, b1 = random.randint(1, 100), random.randint(1, 100)
        a2, b2 = random.randint(1, 100), random.randint(1, 100)
        if a1 == b1 or a2 == b2:
            continue
        ans1 = str(max(a1, b1))
        ans2 = str(max(a2, b2))
        if ans1 == ans2:
            continue

        clean = f"Which is larger, {a1} or {b1}? Answer: "
        corrupt = f"Which is larger, {a2} or {b2}? Answer: "

        clean_toks = model.to_tokens(clean)
        corrupt_toks = model.to_tokens(corrupt)
        if clean_toks.shape[1] != corrupt_toks.shape[1]:
            continue

        pairs.append({
            "clean_prompt": clean,
            "corrupt_prompt": corrupt,
            "clean_answer": ans1,
            "corrupt_answer": ans2,
            "clean_answer_id": model.to_tokens(f" {ans1}")[0, -1].item(),
            "corrupt_answer_id": model.to_tokens(f" {ans2}")[0, -1].item(),
            "clean_tokens": clean_toks,
            "corrupt_tokens": corrupt_toks,
        })
    return pairs
```

---

## Step 2 — Run EAP-IG Circuit Discovery

This is the core step. We use the EAP-IG library following the same methodology
and hyperparameters as the Thinking Sparks paper (Park et al., 2025).

### Approach

The EAP-IG library handles the heavy lifting:
1. Build a computational graph from the model
2. Run attribution with integrated gradients on clean/corrupted pairs
3. Select top edges by score
4. Prune isolated nodes
5. Extract the attention heads that appear in the circuit

### Implementation

```python
from eap.graph import Graph
from eap import attribute, evaluate_graph
import torch

# Build the computational graph
graph = Graph.from_model(model)

# Define the metric function
# This should return a scalar we want to maximize on clean and minimize on corrupt
def logit_diff_metric(logits, clean_answer_id, corrupt_answer_id):
    """
    Logit difference: how much does the model prefer the correct answer
    over the incorrect answer at the last token position.
    """
    last_logits = logits[:, -1, :]
    return (last_logits[:, clean_answer_id] - last_logits[:, corrupt_answer_id]).mean()

# Run EAP-IG attribution across all pairs
# Following Thinking Sparks: ig_steps=100
#
# The exact API depends on the EAP-IG version. The general pattern is:
#
# Option A: If EAP-IG has a high-level attribute function
all_edge_scores = []

for i, pair in enumerate(pairs):
    if i % 50 == 0:
        print(f"Processing pair {i}/{len(pairs)}")

    # Reset graph scores
    graph.reset()

    # Run attribution
    # The attribute function takes clean tokens, corrupted tokens,
    # the model, and the metric, then scores all edges
    attribute(
        model=model,
        graph=graph,
        clean_input=pair["clean_tokens"],
        corrupt_input=pair["corrupt_tokens"],
        metric=lambda logits: logit_diff_metric(
            logits, pair["clean_answer_id"], pair["corrupt_answer_id"]
        ),
        method="EAP-IG",   # or "eap-ig"
        ig_steps=100,       # integrated gradient steps, same as Thinking Sparks
    )

    # Collect edge scores
    # The graph object now has scores on each edge
    all_edge_scores.append(graph.get_scores())

# Aggregate scores across all pairs
# Average the absolute scores (both positive and negative effects matter)
aggregated_scores = aggregate_edge_scores(all_edge_scores)  # implementation depends on API

# Select top edges following Thinking Sparks: top_n=5000
graph.apply_topn(n=5000)

# Prune isolated nodes (threshold tau=0.1)
graph.prune(threshold=0.1)
```

### IMPORTANT: API adaptation note

The exact function signatures above are approximate. The EAP-IG library's API
may differ. Key things to check:

1. Look at the demo notebooks: `greater_than.ipynb` and `ioi.ipynb` in the repo
2. Check whether `attribute()` is a standalone function or a method on Graph
3. Check how the metric/loss function should be defined (some versions expect
   a loss that should be minimized, others expect a metric to maximize)
4. Check whether you pass tokens or strings as input

The notebooks are the ground truth for the API. Adapt accordingly.

### Extract attention head importance scores

After running EAP-IG and getting the circuit, extract which attention heads
are in the circuit and their aggregate importance:

```python
def extract_head_scores(graph, n_layers, n_heads):
    """
    From the EAP-IG graph, extract a per-head importance score.
    A head's importance = sum of absolute edge scores of all edges
    connected to that head.
    """
    head_scores = torch.zeros(n_layers, n_heads)

    for node in graph.nodes:
        # Check if this node is an attention head
        # Node naming convention in EAP-IG: "a{layer}.{head}" or similar
        # Check the actual naming by inspecting graph.nodes
        if node.node_type == "attention":
            layer = node.layer
            head = node.head
            # Sum all incoming and outgoing edge scores for this head
            score = sum(
                abs(edge.score) for edge in graph.edges
                if edge.source == node or edge.target == node
            )
            head_scores[layer, head] = score

    return head_scores

head_scores = extract_head_scores(graph, model.cfg.n_layers, model.cfg.n_heads)
```

### Visualize

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.heatmap(
    head_scores.numpy(),
    xticklabels=[f"H{h}" for h in range(model.cfg.n_heads)],
    yticklabels=[f"L{l}" for l in range(model.cfg.n_layers)],
    cmap="Reds",
)
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("EAP-IG: Attention Head Importance for Reasoning")
plt.tight_layout()
plt.savefig("head_importance_heatmap.png", dpi=150)
```

### Compare against Thinking Sparks results

Their GRPO-emergent heads for Qwen2.5-Math-1.5B (from Table 1):
- GRPO with OpenR1-Math-220k: L0H8, L5H1, L7H1, L18H11, L11H8 (19 total)
- GRPO with GSM8K: L0H8, L5H1, L7H2, L3H3, L21H2 (20 total)

Note: Their heads are EMERGENT heads (new after training vs baseline).
Our heads are REASONING heads in the base model. These may overlap but
are not the same thing. The comparison is still informative:
- Overlap suggests these heads are already important before training
  and GRPO strengthens them
- Non-overlap suggests GRPO discovers new circuits we didn't find,
  or our tasks activate different circuits than AIME

```python
# Thinking Sparks emergent heads (GRPO with OpenR1-Math-220k)
thinking_sparks_heads = {
    (0, 8), (5, 1), (7, 1), (18, 11), (11, 8)
    # ... add rest from their Table 1 if available
}

our_top_heads = set()
for layer, head, score in selected_heads[:20]:
    our_top_heads.add((layer, head))

overlap = our_top_heads & thinking_sparks_heads
print(f"Our top 20 heads: {our_top_heads}")
print(f"Thinking Sparks heads: {thinking_sparks_heads}")
print(f"Overlap: {overlap} ({len(overlap)} heads)")
```

---

## Step 3 — Select and Validate Reasoning Heads

### Selection

```python
# Rank all heads and select top 15%
flat_scores = head_scores.flatten()
n_total = flat_scores.numel()
n_select = int(n_total * 0.15)

threshold = flat_scores.sort(descending=True).values[n_select]
reasoning_mask = head_scores >= threshold

selected_heads = []
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        if reasoning_mask[layer, head]:
            selected_heads.append((layer, head, head_scores[layer, head].item()))

selected_heads.sort(key=lambda x: -x[2])
print(f"Selected {len(selected_heads)} reasoning heads (top 15%):")
for layer, head, score in selected_heads:
    print(f"  Layer {layer}, Head {head}: score = {score:.4f}")
```

### Validation via ablation (same approach as Thinking Sparks)

Following Park et al., we validate by zeroing out the identified heads
and measuring performance degradation.

```python
def ablation_test(model, heads_to_ablate, test_pairs):
    """
    Zero out specific attention heads and measure accuracy drop.
    This is the same validation approach used in Thinking Sparks Table 2.
    """
    def ablation_hook(activation, hook, head_idx):
        activation[:, :, head_idx, :] = 0
        return activation

    # Build hook list
    fwd_hooks = []
    for layer, head in heads_to_ablate:
        hook_name = f"blocks.{layer}.attn.hook_result"
        fwd_hooks.append((hook_name, lambda act, hook, h=head: ablation_hook(act, hook, h)))

    # Test with ablation
    correct_ablated = 0
    correct_normal = 0

    for pair in test_pairs:
        # Normal
        logits = model(pair["clean_tokens"])
        if logits[0, -1].argmax().item() == pair["clean_answer_id"]:
            correct_normal += 1

        # Ablated
        logits_ablated = model.run_with_hooks(
            pair["clean_tokens"],
            fwd_hooks=fwd_hooks
        )
        if logits_ablated[0, -1].argmax().item() == pair["clean_answer_id"]:
            correct_ablated += 1

    n = len(test_pairs)
    print(f"Normal accuracy: {correct_normal/n:.1%}")
    print(f"Ablated accuracy: {correct_ablated/n:.1%}")
    print(f"Drop: {(correct_normal - correct_ablated)/n:.1%}")

    return correct_normal / n, correct_ablated / n

# Test on held-out pairs
test_pairs = pairs[250:]  # use last 50 as test
heads_to_ablate = [(l, h) for l, h, s in selected_heads[:10]]

normal_acc, ablated_acc = ablation_test(model, heads_to_ablate, test_pairs)

# If ablated accuracy drops significantly, our heads are causally important.
# If not, our selection threshold may need adjustment.
```

### Save results

```python
torch.save({
    "reasoning_mask": reasoning_mask,
    "head_scores": head_scores,
    "selected_heads": selected_heads,
    "validation": {
        "normal_accuracy": normal_acc,
        "ablated_accuracy": ablated_acc,
    },
    "config": {
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "method": "EAP-IG",
        "ig_steps": 100,
        "top_n": 5000,
        "threshold": 0.1,
        "n_pairs": len(pairs),
        "selection_percentile": 0.15,
    }
}, "reasoning_heads.pt")

print("Saved reasoning_heads.pt")
```

---

## Step 4 — Validate Circuit Transfer to Chain-of-Thought

Check that the heads found on simple direct-answer tasks are also active
during longer chain-of-thought generation (which GRPO will produce).

```python
def analyze_cot_attention(model, prompt, reasoning_mask):
    """
    Generate a CoT response and check whether reasoning heads
    attend more to computation-relevant tokens than filler tokens.
    """
    tokens = model.to_tokens(prompt)
    generated = model.generate(tokens, max_new_tokens=200, temperature=0.7)

    # Decode to see what was generated
    text = model.to_string(generated[0])
    print(f"Generated: {text[:500]}")

    # Forward pass with caching
    logits, cache = model.run_with_cache(generated)

    # For each reasoning head, look at its attention pattern
    reasoning_head_attention = {}

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            if not reasoning_mask[layer, head]:
                continue

            # Get attention pattern: [seq_len, seq_len]
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]

            # For the last few generated tokens, which positions do they attend to?
            # Average attention from generated tokens to all positions
            prompt_len = tokens.shape[1]
            if generated.shape[1] > prompt_len:
                gen_attention = attn[prompt_len:, :].mean(dim=0)  # [seq_len]
                reasoning_head_attention[(layer, head)] = gen_attention

    # Decode attended-to tokens and classify as reasoning vs filler
    # This is qualitative — look at the top-attended positions
    for (layer, head), attn_weights in reasoning_head_attention.items():
        top_positions = attn_weights.topk(10).indices
        top_tokens = [model.to_string(generated[0, pos:pos+1]) for pos in top_positions]
        print(f"L{layer}H{head} attends to: {top_tokens}")

# Test with a few reasoning prompts
test_prompts = [
    "Solve step by step: What is 15 * 7 + 23?",
    "Think carefully: If x + 5 = 12, what is x?",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    analyze_cot_attention(model, prompt, reasoning_mask)
```

If reasoning heads attend primarily to numbers, operators, and intermediate
results rather than filler words, the circuit transfer assumption holds.

---

## Step 5 — Output Artifacts

At the end of this pipeline, you should have:

### Files
1. **`reasoning_heads.pt`** — Contains:
   - `reasoning_mask`: boolean tensor [n_layers, n_heads]
   - `head_scores`: float tensor [n_layers, n_heads] with importance scores
   - `selected_heads`: sorted list of (layer, head, score) tuples
   - `validation`: ablation test results
   - `config`: hyperparameters used

2. **`head_importance_heatmap.png`** — Visualization

### Key numbers to report
- How many heads selected and at what threshold
- Ablation accuracy drop (should be significant if heads are real)
- Overlap with Thinking Sparks Table 1 heads
- Qualitative CoT attention analysis results

### These artifacts feed directly into Phase 2:
Modifying GRPO to use `reasoning_mask` and `head_scores` for per-token
credit assignment during training. The mask tells us WHICH heads to
extract attention from during rollouts, and the scores tell us how to
weight each head's contribution.

---

## Compute Estimates

- Model loading: ~3GB GPU memory for Qwen2.5-Math-1.5B
- EAP-IG with ig_steps=100 on 300 pairs: ~4-8 hours on single A100
  (100 gradient steps per pair × 300 pairs is the bottleneck)
- Ablation validation: ~30 minutes
- CoT attention analysis: ~30 minutes
- Total: comfortably doable in a single day on one IDUN GPU node

---

## Key References

- Park, Y., Jeong, M., & Kang, J. (2025). "Thinking Sparks!: Emergent
  Attention Heads in Reasoning Models During Post Training." arXiv:2509.25758
  — Primary reference for methodology and Qwen circuit analysis

- Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). "Have Faith in Faithfulness:
  Going Beyond Circuit Overlap When Finding Model Mechanisms." COLM 2024
  — EAP-IG method paper, also the library we're using

- Nanda, N. (2023). "Attribution Patching: Activation Patching At Industrial Scale."
  — Foundational reference for the patching approach

- Tigges, C. et al. (2024). "LLM Circuit Analyses Are Consistent Across
  Training and Scale." NeurIPS 2024
  — Evidence that circuits remain stable during training

---

## Troubleshooting

### EAP-IG doesn't support Qwen
Try loading via TransformerLens with `from_pretrained_no_processing` or check
if a custom model wrapper is needed. The EAP-IG library builds on TransformerLens
so model support depends on TL's model list.

### Model accuracy is too low on diagnostic tasks
Try simpler arithmetic (single digit only), or switch to Qwen2.5-Math-7B
(more capable but more expensive to run EAP-IG on).

### Token length mismatch between clean/corrupt pairs
Pad shorter sequences or only generate pairs where both prompts use the
same number of digits. For arithmetic: restrict to same-digit-count numbers.

### EAP-IG runs out of GPU memory
Reduce ig_steps from 100 to 50 (less precise but still useful).
Or process pairs in smaller batches with gradient accumulation.
Or use standard EAP (no integrated gradients, single gradient step) as
a first pass, then do EAP-IG on the top candidate heads only.
