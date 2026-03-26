# Implementation Spec: Attention-Rhythm-Guided GRPO

## Project Overview

We are implementing a modified GRPO training pipeline that uses attention-based signals to perform fine-grained credit assignment during RL training of language models. The project involves three training runs that share identical hyperparameters but differ only in how per-token advantages are scaled.

**Base model:** Qwen2.5-Math-1.5B (or Qwen2.5-3B)
**Training data:** DAPO-Math-17K or OpenR1-Math-220k
**Framework:** Build on top of an existing GRPO implementation (e.g., TRL GRPOTrainer, veRL, or OpenRLHF)

## The Three Runs

- **Run A (Baseline):** Standard GRPO with uniform advantage across all tokens. No attention analysis during training.
- **Run B (Static Rhythm):** Compute head classification and attention metrics ONCE on the base model before training. Use those fixed classifications for credit assignment throughout training. Never update them.
- **Run C (Adaptive Rhythm):** Same as Run B, but re-compute head classification and attention metrics every K=20 training steps using the current model weights.

Everything else is identical: same seeds, same data order, same learning rate, same batch size, same number of steps.

---

## PART 1: Attention Metrics Module

Create a standalone module `attention_rhythm.py` that takes attention maps and returns per-token scaling coefficients. This is the mathematical core of the project.

### Step 1: Head Classification

Given a model, run a forward pass on a batch of prompts+responses with `output_attentions=True`. This gives attention tensors `A[l,h]` of shape `(seq_len, seq_len)` for each layer `l` and head `h`. The attention is causal (lower-triangular) and row-stochastic (each row sums to 1).

**Compute the average backward distance for each head:**

```
For each head (l, h):
    d[l,h] = (1 / |R|) * sum over t in R of:
        sum over s=1..t of:  A[l,h][t,s] * (t - s)
```

Where `R` is the set of response token positions (excluding prompt tokens). This `d[l,h]` measures how far back each head looks on average. Heads with small `d` are "local" (attend nearby), heads with large `d` are "global" (attend far back).

**Classify heads:**
- Sort all `(l, h)` pairs by their `d[l,h]` value
- Bottom 30% by `d` → `H_loc` (local-focused heads)
- Top 30% by `d` → `H_glob` (global-focused heads)

**Important:** The paper samples from 5 evenly spaced layers in the middle third of the network (layers `floor(L/3)` to `floor(2L/3)` where `L` is total layers). For Qwen2.5-Math-1.5B with 28 layers, that's layers 9-18, sample 5 evenly spaced = layers 9, 11, 13, 16, 18 approximately. However, for head classification, you compute `d[l,h]` across ALL layers and heads, then select the 30% quantiles globally.

### Step 2: Aggregate Attention Maps

Once you have `H_loc` and `H_glob`, compute two aggregated attention matrices:

```
A_bar_loc[t, s] = (1 / |H_loc|) * sum over (l,h) in H_loc of: A[l,h][t, s]

A_bar_glob[t, s] = (1 / |H_glob|) * sum over (l,h) in H_glob of: A[l,h][t, s]
```

These are both `(seq_len, seq_len)` matrices. `A_bar_loc` captures local phrasal patterns. `A_bar_glob` captures global influence patterns.

### Step 3: Compute WAAD (Windowed Average Attention Distance)

For each response token position `t`, compute:

```
WAAD[t] = sum over s=1..t of:  A_bar_loc[t, s] * min(t - s, W)
```

Where `W = 10` is the clipping window.

**What this does:** For each token, it measures how far back the local heads are looking, but clipped at window W. High WAAD = the model is reaching far back (a "preplan" moment at a chunk boundary). Low WAAD = the model is doing local continuation within a chunk.

**Implementation note:** This is a simple weighted sum. For a token at position t, you multiply each element of row t of `A_bar_loc` by the clipped distance to that position, then sum. In PyTorch:

```python
def compute_waad(A_bar_loc, response_start, W=10):
    """
    A_bar_loc: (seq_len, seq_len) aggregated local attention matrix
    response_start: int, index where response tokens begin
    W: int, clipping window (default 10)
    Returns: (num_response_tokens,) tensor of WAAD values
    """
    seq_len = A_bar_loc.shape[0]
    # Create distance matrix: dist[t, s] = t - s
    positions = torch.arange(seq_len, device=A_bar_loc.device)
    dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
    dist_clipped = dist.clamp(min=0, max=W).float()  # clamp negatives to 0 (causal), cap at W

    # WAAD[t] = sum_s A_bar_loc[t,s] * min(t-s, W)
    waad = (A_bar_loc * dist_clipped).sum(dim=-1)  # (seq_len,)

    # Return only response token positions
    return waad[response_start:]
```

### Step 4: Compute FAI (Future Attention Influence)

For each token position `s`, compute:

```
FAI[s] = (1 / |T(s)|) * sum over t in T(s) of:  A_bar_glob[t, s]
```

Where `T(s) = {t : t is a response token, s + H_lo <= t <= min(N, s + H_hi)}`. The paper uses `H_lo = 10` and `H_hi = 50` (influence horizon).

**What this does:** For each token, it measures how much attention it receives FROM future tokens. High FAI = this token is an "anchor" that future tokens keep referring back to. Low FAI = this token is locally consumed and forgotten.

**Implementation:**

```python
def compute_fai(A_bar_glob, response_start, H_lo=10, H_hi=50):
    """
    A_bar_glob: (seq_len, seq_len) aggregated global attention matrix
    response_start: int, index where response tokens begin
    H_lo, H_hi: int, horizon bounds for influence calculation
    Returns: (num_response_tokens,) tensor of FAI values
    """
    seq_len = A_bar_glob.shape[0]
    num_response = seq_len - response_start
    fai = torch.zeros(num_response, device=A_bar_glob.device)

    for idx, s in enumerate(range(response_start, seq_len)):
        # T(s) = {t in response : s + H_lo <= t <= min(N-1, s + H_hi)}
        t_lo = max(s + H_lo, response_start)
        t_hi = min(s + H_hi, seq_len - 1)
        if t_lo > t_hi:
            fai[idx] = 0.0
            continue
        # Average attention received by position s from positions t_lo..t_hi
        fai[idx] = A_bar_glob[t_lo:t_hi+1, s].mean()

    return fai
```

**Note:** There is a more efficient vectorized implementation using masking, but the above makes the logic clear. Optimize later.

### Step 5: Compute Per-Token Scaling Coefficients (γ_t)

The paper defines three credit assignment strategies. We implement the **coupled rhythm credit** (strategy 3) which is the best-performing:

**Step 5a: Identify preplan tokens (from WAAD)**

```
delta[t] = |WAAD[t] - WAAD[t+1]|    for each response token t

T_loc = top 40% of tokens by delta value (indices of the top quantile)
```

Large `delta[t]` means a sharp transition in WAAD between consecutive tokens — a chunk boundary where the model shifts from long-range consultation to local continuation.

**Step 5b: Identify anchor tokens (from FAI)**

```
T_glob = top 40% of tokens by FAI value
```

**Step 5c: Identify locally-dominated anchors and back-allocate credit**

An anchor token `t` is "locally dominated" if:
```
t is in T_glob  AND
WAAD[t] <= tau_waad  AND
max(delta[u] for u in {t-k, ..., t-1}) >= tau_delta
```

Where `tau_waad` and `tau_delta` are thresholds (use median of WAAD and delta respectively as defaults), and `k` is the neighborhood lookback (use `k=3`).

When this condition holds, it means: the anchor token has high future influence BUT low local attention distance (it's locally licensed — its immediate predecessor already set things up). Meanwhile, a recent WAAD peak (the preplan token) did the heavy lifting. So we back-allocate some of the anchor's credit bonus to the preplan token.

Let `D` = set of locally-dominated anchors.
Let `I(D)` = the associated introductory (preplan) tokens. For each `t in D`, `intro(t) = argmax_{u in {t-k,...,t-1}} delta[u]`.

**Step 5d: Compute final γ_t**

```
gamma_amp = 1.5   (amplification factor)
alpha = 0.5       (back-allocation fraction)

For each response token t:
    if t in T_glob and t NOT in D:
        # Regular anchor: full amplification
        gamma[t] = 1 + (gamma_amp - 1) = 1.5

    elif t in D:
        # Locally-dominated anchor: reduced amplification
        gamma[t] = 1 + (1 - alpha) * (gamma_amp - 1) = 1.25

    elif t in I(D):
        # Preplan token associated with a dominated anchor: receives back-allocated credit
        gamma[t] = 1 + alpha * (gamma_amp - 1) = 1.25

    else:
        # Regular token: no amplification
        gamma[t] = 1.0
```

**Critical:** The paper notes that "all shaping signals are detached from gradients and applied only to nonnegative advantages." This means:
1. The γ_t computation is done with `torch.no_grad()` — no gradients flow through the attention metrics
2. γ_t is only applied when the advantage `A_t >= 0`. For negative advantages (incorrect responses), keep `γ_t = 1.0`

### Step 6: Apply to GRPO Objective

The standard GRPO advantage for each token in response `i` is:

```
A_hat[i] = (R_i - mean(R)) / std(R)
```

where `R_i` is the binary reward for response `i`, and the mean/std are over the group of G responses.

The modified advantage becomes:

```
A_tilde[i, t] = gamma[t] * A_hat[i]    if A_hat[i] >= 0
A_tilde[i, t] = A_hat[i]               if A_hat[i] < 0
```

This `A_tilde[i, t]` replaces `A_hat[i]` in the standard GRPO loss:

```
L = (1/G) * sum_i (1/|o_i|) * sum_t min(
    r[i,t](theta) * A_tilde[i,t],
    clip(r[i,t](theta), 1-eps, 1+eps) * A_tilde[i,t]
)
```

where `r[i,t](theta) = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})`.

---

## PART 2: Integration into the Training Loop

### Attention Extraction

Modern training frameworks use flash attention which discards attention maps. We need a separate forward pass with eager attention to get the maps.

**Option A (recommended for small models):** After generating responses with the main model (vLLM or standard generation), concatenate prompt + response into a single sequence and run ONE additional forward pass through a copy of the model loaded with `attn_implementation="eager"` and `output_attentions=True`. Extract attention maps from 5 evenly spaced layers in the middle third.

```python
# After response generation
with torch.no_grad():
    outputs = model_eager(
        input_ids=full_sequence,  # prompt + response concatenated
        attention_mask=attention_mask,
        output_attentions=True,
    )
    # outputs.attentions is a tuple of (batch, num_heads, seq_len, seq_len)
    # one per layer

    # Sample 5 layers from middle third
    L = len(outputs.attentions)
    mid_start = L // 3
    mid_end = 2 * L // 3
    layer_indices = torch.linspace(mid_start, mid_end, 5).long().tolist()
    sampled_attentions = [outputs.attentions[i] for i in layer_indices]
```

**Option B (memory-constrained):** Only compute attention metrics every K=20 steps. Cache the γ_t values and reuse them for K steps. For the base model's initial classification (Run B), compute once before training starts.

### Differences Between Runs

**Run A:** Skip attention extraction entirely. All `gamma[t] = 1.0`.

**Run B:**
1. Before training: run head classification on base model using a batch of ~50-100 prompts. Store `H_loc` and `H_glob` sets.
2. During training: at every step, extract attention maps using the CURRENT model weights BUT use the FROZEN `H_loc` and `H_glob` from step 0 to aggregate into `A_bar_loc` and `A_bar_glob`.
3. Compute WAAD, FAI, and γ_t from these aggregated maps.

**Run C:**
1. Same as Run B, but every K=20 steps, also re-run head classification using the current model weights. Update `H_loc` and `H_glob`.
2. Between reclassification steps, use the most recent `H_loc` and `H_glob`.

### Checkpointing

Save model checkpoints every 50 training steps for all three runs. These will be used for post-hoc circuit analysis (separate from this implementation).

Also log at every step:
- Training reward (mean accuracy)
- The γ_t distribution statistics (mean, std, fraction of tokens with γ > 1)
- For Runs B and C: the head sets `H_loc` and `H_glob` (just save the list of (layer, head) tuples)
- Policy entropy (mean token-level entropy across the batch)

---

## PART 3: Hyperparameters

All three runs share these settings:

```yaml
# Model
base_model: "Qwen/Qwen2.5-Math-1.5B"  # or Qwen2.5-3B

# Training
learning_rate: 1.0e-6
batch_size: 128          # number of prompts per batch
group_size: 8            # number of responses per prompt (G)
max_response_length: 4096
num_training_steps: 500
warmup_ratio: 0.1
gradient_accumulation_steps: as needed for your GPU count
bf16: true

# GRPO specific
clip_eps_low: 0.2
clip_eps_high: 0.28      # slightly asymmetric like DAPO's clip-higher
kl_coeff: 0.0            # no KL penalty (following DAPO recommendation)
temperature: 1.0
top_p: 1.0

# Attention rhythm specific (Runs B and C only)
waad_window_W: 10
fai_horizon_lo: 10
fai_horizon_hi: 50
quantile_q: 0.4          # top 40% for T_loc and T_glob
gamma_amp: 1.5
alpha_backalloc: 0.5     # back-allocation fraction
neighborhood_k: 3        # lookback for locally-dominated check
reclassify_every_K: 20   # only for Run C
head_quantile: 0.3       # bottom/top 30% for H_loc/H_glob
num_classification_prompts: 100  # prompts used for head classification

# Checkpointing
save_every: 50
```

---

## PART 4: Evaluation

Evaluate all three runs on these benchmarks at every saved checkpoint:

- MATH500 (Pass@1, temperature 0.0)
- AIME 2024 (Pass@1 and Pass@8, temperature 0.7)
- AMC 2023 (Pass@1, temperature 0.0)

Use the standard math verifier (exact match on boxed answer).

---

## PART 5: File Structure

```
project/
├── src/
│   ├── attention_rhythm.py      # WAAD, FAI, gamma computation (Part 1)
│   ├── head_classifier.py       # Head classification logic
│   ├── rhythm_grpo_trainer.py   # Modified GRPO trainer (Part 2)
│   ├── training_configs/
│   │   ├── run_a_baseline.yaml
│   │   ├── run_b_static.yaml
│   │   └── run_c_adaptive.yaml
│   └── eval/
│       └── evaluate_checkpoint.py
├── scripts/
│   ├── run_training.sh
│   └── run_eval.sh
└── analysis/                    # Post-hoc analysis (separate phase)
    └── circuit_analysis.py
```

---

## Implementation Priority

1. **First:** Implement and unit test `attention_rhythm.py` standalone. Test it by loading the base model, running a few prompts through it, and verifying that WAAD shows the expected sawtooth pattern and FAI highlights meaningful tokens. Visualize these to confirm correctness.

2. **Second:** Implement `head_classifier.py` and verify that the local/global head split looks reasonable (local heads should be in early-mid layers, global heads in mid-late layers typically).

3. **Third:** Integrate into the GRPO training loop. Start with Run A (baseline) to make sure training works. Then add the attention extraction for Run B. Finally add the reclassification for Run C.

4. **Fourth:** Run all three experiments and collect results.

---

## Key Mathematical Summary

All equations in one place for reference:

**Head distance (Eq 7):**
`d[l,h] = (1/|R|) * Σ_{t∈R} Σ_{s=1}^{t} A[l,h][t,s] * (t-s)`

**Aggregated attention (Eq 8):**
`A_bar_loc[t,s] = (1/|H_loc|) * Σ_{(l,h)∈H_loc} A[l,h][t,s]`
`A_bar_glob[t,s] = (1/|H_glob|) * Σ_{(l,h)∈H_glob} A[l,h][t,s]`

**WAAD (Eq 9):**
`WAAD[t] = Σ_{s=1}^{t} A_bar_loc[t,s] * min(t-s, W)`   where W=10

**FAI (Eq 10):**
`FAI[s] = (1/|T(s)|) * Σ_{t∈T(s)} A_bar_glob[t,s]`   where T(s) = {t : s+H_lo ≤ t ≤ min(N, s+H_hi)}

**Preplan detection (Eq 13):**
`delta[t] = |WAAD[t] - WAAD[t+1]|`
`T_loc = TopQuantile(delta, q=0.4)`

**Anchor detection (Eq 15):**
`T_glob = TopQuantile(FAI, q=0.4)`

**Locally-dominated anchor test (Eq 17):**
`t ∈ D iff: t ∈ T_glob AND WAAD[t] ≤ τ_waad AND max_{u∈{t-k,...,t-1}} delta[u] ≥ τ_delta`

**Coupled credit assignment (Eq 18):**
```
γ[t] = 1 + (γ_amp-1)·𝟙{t ∈ T_glob \ D}
     + (1-α)(γ_amp-1)·𝟙{t ∈ D}
     + α(γ_amp-1)·𝟙{t ∈ I(D)}
```
where γ_amp=1.5, α=0.5

**Modified GRPO objective (Eq 12):**
```
J(θ) = E[ (1/|o|) Σ_t min(
    π_θ(o_t|q,o_{<t}) / π_old(o_t|q,o_{<t}) · A_t · γ_t,
    clip(ratio, 1-ε, 1+ε) · A_t · γ_t
)]
```
γ_t is detached from gradients. Only applied when A_t ≥ 0.