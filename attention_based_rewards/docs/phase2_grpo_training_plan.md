# Phase 2: Circuit-Guided Credit Assignment in GRPO Training

## Overview

We run four GRPO training conditions that are identical in every way except how
the per-token advantage is computed. We then compare training efficiency and
final accuracy across conditions.

### Conditions

1. **GRPO-Uniform** — Standard GRPO. Every token gets the same advantage.
   This is the control.

2. **GRPO-Attention** — Our method. Per-token advantage is weighted by how
   much the identified reasoning heads attend to each token.

3. **GRPO-Entropy** — GTPO-style baseline. Per-token advantage is weighted
   by the policy entropy at each token (high entropy = high weight).

4. **GRPO-Combined** — Attention + entropy combined. Per-token advantage
   is weighted by a combination of both signals.

---

## Training Configuration

All four conditions use identical hyperparameters. The ONLY difference is
the per-token advantage weighting function.

### Model
- Qwen2.5-Math-1.5B-Instruct (same model used for circuit discovery)
- Full parameter training (no LoRA)
- bfloat16 precision

### Hardware
- 4x GH200 per run
- DeepSpeed ZeRO-3 or FSDP for distributed training

### Training Framework
- Verifiers library (same as previous paper) OR verl framework
- Check which one supports custom advantage modification more easily
- The key requirement: we need to be able to modify the per-token
  advantage computation without rewriting the whole training loop

### Dataset
- **Training:** GSM8K train split (7,473 examples)
- Same dataset used for circuit discovery, and same dataset the
  Thinking Sparks paper used for one of their GRPO experiments

### System Prompt
Following the Thinking Sparks paper / OpenR1 standard:
```
You are a helpful AI Assistant that provides well-reasoned and detailed
responses. You first think about the reasoning process as an internal
monologue and then provide the user with the answer. Respond in the
following format:
<think>\n...\n</think>\n<answer>\n...\n</answer>
```

### Hyperparameters
These should match across all 4 conditions. Starting point based on
the Thinking Sparks paper's GRPO config and our previous paper:

```
algorithm: GRPO
learning_rate: 5e-6          # Thinking Sparks used 5e-6 for GSM8K
epochs: 1                    # Thinking Sparks used 1 epoch for GSM8K
warmup_ratio: 0.1
num_generations: 16          # responses per prompt (group size G)
max_new_tokens: 2048
temperature: 0.7
kl_penalty_beta: 0.001
max_grad_norm: 0.2
optimizer: AdamW
precision: bfloat16
checkpoint_interval: 100     # save every 100 steps for analysis
```

Note: Adjust if needed based on what works with the framework and
hardware. The critical thing is that ALL FOUR conditions use the
exact same hyperparameters.

### Reward Function
Binary correctness reward — same for all conditions:
- R = 1 if the model's answer matches the ground truth
- R = 0 otherwise
- Answer extracted from <answer> tags or \boxed{} format

---

## The Four Advantage Computation Methods

### Standard GRPO Advantage (all conditions start here)

For each prompt q, sample G=16 responses. Each response i gets reward R_i.
Group-normalized advantage:

```
A_i = (R_i - mean(R_1...R_G)) / std(R_1...R_G)
```

This A_i is the SAME scalar for every token in response i.
The four conditions differ in what happens next.

### Condition 1: GRPO-Uniform

No modification. Every token t in response i gets:

```
advantage(i, t) = A_i
```

This is standard GRPO. Equation 6 from the Thinking Sparks paper.

### Condition 2: GRPO-Attention

For each generated response, extract attention patterns from the top-k
reasoning heads (identified in Phase 1). For each token t, compute how
much the reasoning heads attend to it:

```python
# During rollout generation, cache attention patterns for reasoning heads
# reasoning_heads = [(11,8), (15,7), (10,5), (18,11), (18,10), ...]
# loaded from reasoning_heads.pt

def compute_attention_weights(attention_cache, reasoning_heads, head_scores):
    """
    For each token position t in the response, compute a reasoning
    importance weight based on how much the identified reasoning heads
    attend to that token.

    Args:
        attention_cache: dict of attention patterns from the forward pass
            Each entry shape: [batch, n_heads, seq_len, seq_len]
        reasoning_heads: list of (layer, head) tuples
        head_scores: importance score for each head (from EAP-IG)

    Returns:
        weights: tensor of shape [seq_len] with per-token importance
    """
    seq_len = attention_cache[list(attention_cache.keys())[0]].shape[-1]
    token_importance = torch.zeros(seq_len)

    for (layer, head) in reasoning_heads:
        # Get this head's attention pattern: [seq_len, seq_len]
        attn = attention_cache[f"layer_{layer}"][0, head]  # exact key depends on framework

        # How much does each token get attended TO by later tokens?
        # Sum attention received from all subsequent positions
        # attn[i, j] = how much position i attends to position j
        # We want: for each j, sum of attn[i, j] for all i > j
        received_attention = attn.sum(dim=0)  # [seq_len]

        # Weight by this head's importance score
        importance = head_scores[layer, head]
        token_importance += received_attention * importance

    # Normalize so mean weight = 1 (preserves overall advantage scale)
    token_importance = token_importance / token_importance.mean()

    return token_importance

# Then multiply into advantage:
# advantage(i, t) = A_i * w_t
```

Key implementation detail: the attention patterns are already computed
during the forward pass that generates the response. We just need to
cache them for the reasoning heads rather than discarding them. This
adds minimal memory overhead since we only cache ~10-20 heads out of
the full set.

### Condition 3: GRPO-Entropy

Following the GTPO paper (Tan et al., 2025). For each token t, compute
the policy entropy — how uncertain the model was at that position:

```python
def compute_entropy_weights(logits):
    """
    For each token position, compute the policy entropy.
    High entropy = model was uncertain = important decision point.

    Args:
        logits: tensor of shape [seq_len, vocab_size]

    Returns:
        weights: tensor of shape [seq_len]
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]

    # Normalize so mean weight = 1
    entropy_weights = entropy / entropy.mean()

    return entropy_weights

# advantage(i, t) = A_i * entropy_weight_t
```

### Condition 4: GRPO-Combined

Combine both signals with a mixing coefficient alpha:

```python
def compute_combined_weights(attention_cache, reasoning_heads, head_scores, logits, alpha=0.5):
    """
    Combine attention-based and entropy-based weights.

    Args:
        alpha: mixing coefficient. 0.5 = equal weight to both signals.
    """
    attn_weights = compute_attention_weights(attention_cache, reasoning_heads, head_scores)
    entropy_weights = compute_entropy_weights(logits)

    # Both are already normalized to mean=1
    combined = alpha * attn_weights + (1 - alpha) * entropy_weights

    # Re-normalize to mean=1
    combined = combined / combined.mean()

    return combined

# advantage(i, t) = A_i * combined_weight_t
```

For alpha, start with 0.5. If time permits, try 0.3 and 0.7 as well
to see if one signal dominates.

---

## Evaluation

### Benchmarks
1. **GSM8K test split** (1,319 examples) — in-distribution
2. **MATH-500** (500 examples) — out-of-distribution, harder

### Evaluation Protocol
- Evaluate every 100 training steps (matching checkpoint_interval)
- Temperature 0.1 for near-deterministic evaluation (same as our previous paper)
- Extract answer from <answer> tags or \boxed{}
- Exact match accuracy
- Generate 1 response per problem for efficiency (pass@1)

### Metrics to Report
1. **Learning curves** — accuracy vs training step for all 4 conditions
   on both benchmarks. This is the PRIMARY result. Plot all 4 on the
   same graph.

2. **Final accuracy** — accuracy at the end of training for all conditions
   on both benchmarks.

3. **Steps to threshold** — how many training steps each condition takes
   to reach a given accuracy (e.g., 50% on GSM8K). This directly
   measures whether better credit assignment speeds up learning.

4. **Reward curve** — average reward per training step for all conditions.
   Should correlate with accuracy but might show different dynamics.

---

## Where to Modify the Code

The key modification happens at the advantage computation step in the
GRPO training loop. In pseudocode, standard GRPO does:

```python
for batch in training_data:
    prompts = batch["prompts"]

    # 1. Generate G responses per prompt
    responses = model.generate(prompts, num_return_sequences=G)

    # 2. Score each response
    rewards = verifier.score(responses)

    # 3. Compute group-normalized advantage (SAME FOR ALL TOKENS)
    advantages = (rewards - rewards.mean()) / rewards.std()

    # 4. Compute per-token loss
    for response_i, advantage_i in zip(responses, advantages):
        for token_t in response_i:
            ratio = pi_theta(token_t) / pi_old(token_t)
            loss += -clip(ratio) * advantage_i  # <-- uniform advantage
```

Our modification is at step 3-4. We add:

```python
    # 3b. Compute per-token weights (THIS IS THE NEW PART)
    if method == "attention":
        token_weights = compute_attention_weights(...)
    elif method == "entropy":
        token_weights = compute_entropy_weights(...)
    elif method == "combined":
        token_weights = compute_combined_weights(...)
    else:
        token_weights = ones(seq_len)  # uniform

    # 4. Compute per-token loss WITH WEIGHTS
    for response_i, advantage_i in zip(responses, advantages):
        for t, token_t in enumerate(response_i):
            ratio = pi_theta(token_t) / pi_old(token_t)
            loss += -clip(ratio) * advantage_i * token_weights[t]  # <-- weighted
```

### Framework-Specific Implementation

For **Verifiers library**:
- Find where the GRPO loss is computed
- Look for where the advantage tensor is applied to the per-token loss
- Inject the token_weights multiplication there
- The attention cache needs to be retained during generation
  (check if Verifiers already caches attention or if we need to
  modify the generation step)

For **verl framework**:
- Similar approach but verl may have different abstractions
- Check verl's GRPO implementation for the advantage application point

IMPORTANT: Before writing training code, READ the framework source
to understand exactly where the advantage is applied. Don't guess.

---

## Attention Extraction During Training

This is the trickiest implementation detail. During GRPO rollouts,
the model generates responses token by token. We need to extract
attention patterns from the reasoning heads during this generation.

### Option A: Post-generation forward pass
After generating a response, run one additional forward pass on the
full response with attention caching enabled for the reasoning heads.
This is simpler to implement but adds ~50% compute overhead per rollout.

### Option B: Cache during generation
Modify the generation loop to cache attention patterns for the
reasoning heads as tokens are generated. More efficient but requires
deeper integration with the framework's generation code.

### Recommendation
Start with Option A. It's simpler and the overhead is acceptable
for a research prototype. If training is too slow, optimize to
Option B later.

```python
# Option A pseudocode
responses = model.generate(prompts, num_return_sequences=G)

# Run forward pass on complete responses to get attention patterns
with torch.no_grad():
    outputs = model(
        responses,
        output_attentions=True,
        # Only need attention from reasoning head layers
    )
    attention_patterns = extract_reasoning_head_attention(outputs.attentions)
    token_weights = compute_attention_weights(attention_patterns, reasoning_heads, head_scores)
```

---

## Pre-Training Artifacts Required

Before starting training, we need:

1. **reasoning_heads.pt** — from Phase 1 (already have this)
   Contains: reasoning_mask, head_scores, selected_heads

2. **Ablation validation results** — from Step 3 (in progress)
   Confirms the heads are causally important

3. **Training script** — modified GRPO with the 4 advantage methods

4. **Evaluation script** — runs all checkpoints on GSM8K test and MATH-500

---

## Execution Plan

### Run Order
Run all 4 conditions sequentially (or in parallel if nodes available):
1. GRPO-Uniform (control) — ~2-4 hours on 4x GH200
2. GRPO-Attention (our method) — ~3-5 hours (extra forward pass overhead)
3. GRPO-Entropy (GTPO baseline) — ~2-4 hours (negligible overhead)
4. GRPO-Combined — ~3-5 hours

### Total Compute
Roughly 12-20 hours of 4x GH200 time for all 4 conditions.
Plus evaluation time: ~30 min per checkpoint per benchmark.

### Checkpointing
Save model checkpoints every 100 steps.
Save training metrics (loss, reward, advantage stats) every step.
Save per-condition token weight statistics (mean, std, distribution)
for analysis.

---

## Analysis Beyond Accuracy

In addition to the main accuracy comparison, report:

1. **Token weight distribution analysis** — For the attention-weighted
   conditions, show what kinds of tokens get high vs low weights.
   Are numbers weighted more? Operators? Reasoning keywords?
   This is the qualitative interpretability story.

2. **Head importance stability** — If time permits, re-run the circuit
   discovery on the trained models (like Thinking Sparks Figure 2)
   to see if the reasoning heads changed during training.

3. **Response length analysis** — Does attention weighting change the
   average response length? If it makes responses shorter without
   losing accuracy, that's evidence it reduces overthinking.

4. **Correlation between attention weights and correctness** — In
   correct vs incorrect responses, do the attention weight patterns
   differ? This validates whether the signal is meaningful.

---

## Key References for This Phase

- Shao et al. (2024) — GRPO algorithm, DeepSeekMath paper
- Tan et al. (2025) — GTPO/GRPO-S, entropy-based token credit
- Park et al. (2025) — Thinking Sparks, GRPO training config for GSM8K
- Lauvrak & Auren (2025) — Our previous paper, Verifiers setup

---

## Risk Mitigation

**Risk: Attention extraction is too slow**
Mitigation: Start with top 5 heads instead of top 20. Fewer heads
means less attention to cache. Or batch the forward pass efficiently.

**Risk: Token weights are too noisy to help**
Mitigation: Apply a temperature/smoothing parameter to the weights
before multiplying into the advantage. E.g., w_t = softmax(raw_w / tau)
with tau controlling how peaked vs uniform the weights are.

**Risk: The method works on GSM8K but not MATH-500**
Mitigation: This is actually an interesting finding about the
task-specificity of circuit-based credit assignment. Report it.

**Risk: All methods converge to similar final accuracy**
Mitigation: Focus on training SPEED (steps to reach X% accuracy)
rather than final accuracy. Also report reward curve dynamics.
