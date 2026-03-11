# Metrics for Predicting Good 1-Shot RLVR Examples from Base Model Outputs

All of these metrics are measured on the untrained base model before any RL training begins. The goal is to cheaply identify which examples will produce the best training dynamics during 1-shot RLVR.

---

## Output-Based Metrics (from sampling rollouts)

These require sampling the base model multiple times (typically 64–128) on each candidate example and analyzing the outputs.

### 1. Pass Rate @k
Sample the example k times, compute the fraction that get the correct answer. This is the most basic filter. Examples at 0% give no positive GRPO signal, examples at 100% give no negative signal. The sweet spot is likely somewhere around 5–30%. Cheap and essential, but weak as a standalone predictor since many examples with the same pass rate perform very differently in RLVR.

### 2. Answer Distribution Entropy
From your k samples, extract all final \boxed{} answers. Compute Shannon entropy over the distribution of distinct answers (both correct and incorrect). High entropy means the model produces many different answers, indicating a large landscape of different reasoning outcomes. An example where the model gives 15 different wrong answers is fundamentally richer than one where it always gives the same wrong answer. This is just a histogram over strings — very cheap to compute.

### 3. Unique Wrong Answer Count × Pass Rate
Count the number of distinct incorrect answers from your samples, then multiply by the pass rate. The unique wrong count measures how many different error modes exist (the "error surface" for training to explore). The pass rate multiplier filters out impossible examples where the model never gets a positive reward. This is essentially a simplified, more interpretable version of answer entropy that also accounts for difficulty. The strongest candidate for a single practical metric.

### 4. Suffix-Conditioned Pass Rate Variance
Pick 3–5 different prompt suffixes (e.g., "think step by step," "repeat the problem then solve," "solve and verify"). Compute pass@k under each suffix independently. Take the variance of the pass rates across suffixes. High variance means the example's difficulty is sensitive to how you frame the reasoning instruction, implying the problem has multiple reasoning paths that get activated or deactivated by the prompt. This reuses the same sampling pipeline you already have, just run multiple times with different suffixes.

### 5. Salient Skill Count (from the Alibaba paper)
Use a strong LLM (like Qwen2.5-72B-Instruct) to analyze the problem text and identify which mathematical skills are needed to solve it (e.g., unit conversion, polynomial manipulation, trigonometry). Count the total number of distinct skills and how many math categories they span. Problems with more skills across more categories tend to produce better cross-domain generalization. This measures the problem's intrinsic complexity rather than the model's interaction with it — it's complementary to the other metrics.

---

## Activation-Based Metrics (from model internals)

These require access to logits, hidden states, or attention patterns during the forward pass. More informative than output-only metrics but require slightly more infrastructure.

### 6. Per-Token Entropy Profile
During generation (even a single greedy decode), record the entropy of the output probability distribution at each token position. This gives you an "uncertainty curve" over the reasoning chain. Good examples should show low entropy for most of the chain (the model confidently handles setup and intermediate steps) with sharp high-entropy spikes at specific "crux points" (where the model genuinely doesn't know what to do). From this single curve you can extract several sub-features:

- **Number of entropy spikes**: How many genuine decision points the problem has. More spikes = more places for the model to explore alternative strategies during training.
- **Spike magnitude**: How uncertain the model is at the crux points. Higher magnitude = more room for diverse exploration at that step.
- **Spike position** (early vs. late): Late spikes mean the model understands the problem but struggles with execution, which matches the profile of the best examples (like π₁). Early spikes suggest the model doesn't even understand the setup.
- **High-entropy token ratio**: Fraction of tokens where entropy exceeds some threshold. This measures the density of decision points in the reasoning chain.

This is the single most informative internal metric. It requires saving logits during generation, which most inference frameworks support natively. One forward pass per rollout.

### 7. Hidden State Variance Across Rollouts
Sample ~16 rollouts. At each layer of the model, extract the hidden state at the final token (or at key computation points). Compute the variance of these hidden states across the 16 rollouts. High variance at a particular layer means the model is taking genuinely different internal computational paths across different attempts — different "circuits" are activating.

Measuring this at different layers tells you different things:
- High variance in early layers: the model can't even consistently parse the problem (probably too hard or ambiguous).
- High variance in late layers: the model understands the problem but is uncertain about the computation (this is the profile you want).

Cost: 16 forward passes per candidate + simple tensor variance computation.

### 8. Attention Pattern Diversity
Across multiple rollouts (~16), extract attention matrices at a few key layers. For each rollout, the model "looks at" different parts of the problem when making decisions. Compute the average pairwise distance (e.g., cosine distance) between attention distributions across rollouts.

High attention diversity means the model is literally interpreting and approaching the problem in fundamentally different ways across attempts. This is the closest internal proxy for "reasoning strategy diversity" — which is hard to measure from text output but straightforward from attention matrices.

The risk: attention patterns are noisy and it's unclear which layers/heads carry the reasoning-relevant signal. This could be very powerful or just noise. Most speculative of the activation-based metrics.

### 9. Gradient Norm of the Correct Answer
Format the correct answer as a target sequence, compute cross-entropy loss against the model's output distribution, and backpropagate once. The gradient norm tells you how far the model's current weights are from producing the correct answer.

Very high gradient norm = the model is far from the answer (too hard). Very low = already there (too easy). You want something in between. However, this is essentially a continuous version of pass rate and may not add much beyond it. It also requires a backward pass, making it more expensive than the other metrics. Least promising of the activation-based metrics, but included for completeness.

---

## Composite Metrics (combining multiple signals)

### 10. Error Surface Score
**unique_wrong_answers × pass_rate × num_entropy_spikes**

Combines the three most promising individual metrics. Unique wrong answers captures how many different failure modes exist (from sampling). Pass rate ensures the example is solvable (from sampling). Number of entropy spikes captures how many internal decision points the problem has (from one forward pass). Together they measure: "the model fails in diverse ways, at a solvable difficulty, because it has multiple genuine decision points."

### 11. Problem Richness Score
**salient_skill_count × answer_entropy × f(pass_rate)**

Combines the Alibaba paper's approach (problem-intrinsic complexity) with model-behavior metrics. Salient skill count measures the problem's structural complexity. Answer entropy measures the diversity of the model's outputs. f(pass_rate) filters difficulty. This captures both "the problem is inherently rich" and "the model can productively engage with that richness."

---

## Summary Table

| # | Metric | What It Measures | Cost | Confidence |
|---|--------|-----------------|------|------------|
| 1 | Pass rate @k | Basic difficulty | 64–128 samples | Filter only |
| 2 | Answer distribution entropy | Output diversity | Same samples | High |
| 3 | Unique wrong × pass rate | Error surface + difficulty | Same samples | High |
| 4 | Suffix-conditioned variance | Sensitivity to framing | 3–5× sampling cost | Medium-High |
| 5 | Salient skill count | Problem complexity | 1 LLM call | Medium |
| 6 | Per-token entropy profile | Internal uncertainty map | 1 forward pass | High |
| 7 | Hidden state variance | Computational path diversity | 16 forward passes | Medium-High |
| 8 | Attention pattern diversity | Reasoning strategy diversity | 16 forward passes | Speculative |
| 9 | Gradient norm | Distance to correct answer | 1 backward pass | Low |
| 10 | Error Surface Score (composite) | Combined prediction | Sampling + 1 forward | High |
| 11 | Problem Richness Score (composite) | Structure + behavior | Sampling + 1 LLM call | Medium-High |
