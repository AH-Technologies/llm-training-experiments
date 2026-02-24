# Bidirectional Experience Replay for Signal Maintenance in Data-Efficient RLVR

## Thesis Proposal Summary — February 2026

---

## 1. Background: 1-Shot RLVR and Post-Saturation Generalization

Wang et al. (NeurIPS 2025; arXiv:2504.20571) demonstrated a remarkable finding: RLVR training on a **single math example** can match the performance of training on 1,200+ examples. Using Qwen2.5-Math-1.5B with GRPO, a single example (π₁) elevates MATH500 from 36.0% to 73.6%, matching the full DeepScaleR subset (73.6%).

The training dynamics reveal three distinct phases on a single example:

**Phase 1 (Steps 0–~20): Pre-competence.** The model cannot yet solve the training example. Most or all of the 1024 rollouts per step are incorrect. The model is learning basic approach strategies.

**Phase 2 (Steps ~20–~100): Rapid learning.** A mix of correct and incorrect rollouts provides strong gradient signal. Training accuracy climbs rapidly. GRPO works as designed — correct rollouts get positive advantages, incorrect ones get negative advantages.

**Phase 3 (Steps ~100–2000+): Post-saturation.** Training accuracy hits ~99.x% and stays there. Nearly all rollouts are correct. Yet test accuracy **continues improving** — gaining 3.4% (π₁) to 9.9% (π₁₃) over the next 1000-1900 steps. The authors call this **post-saturation generalization**.

The puzzle is: where does the learning signal come from in Phase 3? GRPO computes advantages by normalizing rewards within a group. When all 8 rollouts in a group are correct (reward = 1), the mean is 1, variance is 0, and every advantage is zero. No gradient. No learning.

The 1-shot paper identifies entropy loss as critical: it maintains output diversity, which means the model occasionally (at rate ~0.x%) produces an incorrect rollout even post-saturation. That rare incorrect rollout among 7 correct ones creates non-zero variance and produces a learning signal. Without entropy loss, performance plateaus immediately at saturation.

But this is a thin, stochastic signal. It depends entirely on the entropy coefficient being tuned just right and on random chance producing occasional failures. We propose to make this signal robust and reliable.

---

## 2. The Idea: Bidirectional Adaptive Replay

### 2.1 The Mechanism

We maintain two caches — one for correct responses, one for incorrect responses — and inject the appropriate one depending on the current training phase:

**When all rollouts are incorrect (Phase 1):** We replace one rollout with a cached correct response. This creates reward variance (7 wrong + 1 right), giving the correct response a strong positive advantage. The model has something to learn *toward*. This is conceptually similar to existing positive replay methods (AR3PO, RLEP), but we use a specific high-quality source (see Section 3).

**When rollouts are mixed (Phase 2):** We train normally — GRPO works as designed. However, we **cache the most recent incorrect rollout** from this phase. This is critical: these are on-policy errors that the model was genuinely capable of producing, representing realistic failure modes at the model's current skill level.

**When all rollouts are correct (Phase 3):** We replace one rollout with the cached incorrect response. This creates reward variance (7 correct + 1 wrong), giving the incorrect response a negative advantage and the correct ones positive advantages. The model now has an explicit signal about *what to avoid*, maintaining the NSR (Negative Sample Reinforcement) signal that Zhu et al. (NeurIPS 2025) showed is crucial for exploration and diversity.

### 2.2 Concrete Example: Training on π₁

Consider training Qwen2.5-Math-1.5B on π₁ (a Level 3 algebra problem about finding integer values satisfying an inequality).

**Steps 0–15 (Phase 1):** The base model solves π₁ roughly 30% of the time at temperature 0.6. In early training, many groups of 8 rollouts come back all-wrong. Without intervention, these steps produce zero gradient. With our mechanism, we inject a power-sampled correct solution (generated once before training), creating a 1/8 correct group. The model gets a clear signal: "this correct approach is better than your 7 failed attempts."

**Steps 15–80 (Phase 2):** The model now solves π₁ roughly 50-90% of the time. Groups naturally have a mix of correct and incorrect rollouts. GRPO works normally. Meanwhile, we continuously update our error cache with the freshest incorrect rollout — capturing the model's *current* failure modes, not stale ones from early training.

**Steps 80–2000 (Phase 3):** Training accuracy hits ~99.5%. In a batch of 128 groups (1024 rollouts), perhaps 127 groups are all-correct and produce zero gradient. Only 1 group has a natural error. With our mechanism, we inject the cached error into every all-correct group. Now all 128 groups produce non-zero gradient. The signal isn't just 128× stronger in expectation — it's also **reliable** rather than stochastic. Every step produces learning, not just the lucky ones where entropy happened to cause a failure.

### 2.3 Why the Cached Error is Informative

A critical question: isn't the cached error stale? It was generated steps or hundreds of steps ago. Why would it still be useful?

The answer connects to what Zhu et al. found about NSR: negative reinforcement works by **suppressing incorrect patterns and redistributing probability mass** toward alternatives guided by the model's prior. The cached error represents a reasoning path the model used to take but has since moved away from. By continuing to present it as "wrong," the model continues to push probability mass away from that path and toward other (potentially novel) correct paths. As the model's distribution evolves, the *meaning* of "push away from this error" changes — it redirects mass to different alternatives at different points in training.

Furthermore, Tang et al. (A3PO, arXiv:2512.21625) showed that negative samples specifically encourage **exploration of new reasoning paths**, while positive samples sharpen existing ones. Post-saturation is precisely when you want exploration — the model has already learned one way to solve the problem, and generalization comes from exploring diverse solution strategies. The cached error keeps the exploration pressure on.

That said, staleness is a real concern. We track cache age and measure its correlation with learning rate. If the error becomes too stale (the model has fully internalized avoiding it), we can explore refresh strategies — though the 1-shot paper suggests that even the current thin signal from entropy maintains learning for 2000 steps, so staleness may decay slowly.

### 2.4 The Algorithm

```
# Pre-training: generate high-quality correct response
correct_cache = power_sample_correct_response(π_base, prompt)

# Initialize error cache as empty
error_cache = None

for step in training:
    rollouts = generate_rollouts(current_model, prompt, n=8)
    rewards = evaluate(rollouts)  # binary: 1 correct, 0 incorrect
    n_correct = sum(rewards)

    if n_correct == 0:                              # Phase 1: all wrong
        rollouts[-1] = correct_cache
        rewards[-1] = 1
        # Now group has 7 wrong + 1 right → positive signal

    elif n_correct == len(rollouts):                # Phase 3: all correct
        if error_cache is not None:
            rollouts[-1] = error_cache
            rewards[-1] = 0
            # Now group has 7 right + 1 wrong → negative signal

    else:                                           # Phase 2: mixed
        # Normal GRPO training, but cache the newest error
        wrong_indices = where(rewards == 0)
        error_cache = rollouts[wrong_indices[-1]]
        # Keep the most recent error (closest to current policy)

    # Standard GRPO update with modified rollouts/rewards
    train_step(rollouts, rewards)
```

### 2.5 Advantage Clamping: Stabilising Phase 3 Training

BER's injection creates a predictable but extreme advantage distribution within each group. In Phase 3, a group of 8 rollouts has 7 correct (reward=1) and 1 injected incorrect (reward=0):

```
mean = 7/8 = 0.875,  std ≈ 0.331

advantage_correct  = (1.0 - 0.875) / 0.331 = +0.378   (×7 rollouts)
advantage_injected = (0.0 - 0.875) / 0.331 = -2.645   (×1 rollout)
```

The injected negative gets a **-2.645 advantage** — 7× stronger per-rollout than each positive signal. While this is the intended learning signal (push away from incorrect reasoning), the magnitude can cause periodic policy instability: the model collapses to ~20% accuracy for a few steps, then recovers via Phase 1 positive injection, then re-saturates and collapses again.

**Advantage clamping** addresses this by bounding the advantage tensor before the PPO policy loss is computed. This is implemented as an override of `_update_actor()` in `BERRayPPOTrainer`, which clamps `batch.batch["advantages"]` to `[adv_clamp_min, adv_clamp_max]` (default `[-2.0, 2.0]`) before calling the parent actor update.

With default bounds of [-2.0, 2.0]:
- The injected negative advantage is clamped from -2.645 → -2.0 (a 24% reduction)
- All positive advantages (+0.378) are unaffected
- Normal Phase 2 groups (mixed correct/incorrect) are also mostly unaffected, since their advantages are typically within [-2, 2]

The clamping is **global** (applied to all advantages, not just BER-injected indices), which is both simpler to implement and more principled — any extreme advantage can cause instability, regardless of source.

**Configuration:**
```bash
+ber.adv_clamp_enabled=True    # Enable/disable (default: False)
+ber.adv_clamp_min=-2.0        # Lower bound (default: -2.0)
+ber.adv_clamp_max=2.0         # Upper bound (default: 2.0)
```

**Logged metrics:** `ber/adv_clamped_low`, `ber/adv_clamped_high`, `ber/adv_min_pre_clamp`, `ber/adv_max_pre_clamp`.

---

## 3. Power-Sampled Seeding for Difficult Problems

### 3.1 The Problem with Hard Examples

The 1-shot paper shows that most examples in the DSR-sub pool work for 1-shot RLVR — but not all. Hard examples like π₁₂₀₈, where the base model almost never produces a correct answer, fail because the model is stuck in Phase 1 indefinitely. All rollouts are wrong, advantages are zero, nothing is learned.

Existing positive replay (AR3PO, RLEP) solves this by injecting a correct response. But *where does that correct response come from*? The options matter:

**Ground truth solution from the dataset:** The correct answer, but written in a style and vocabulary that may be completely unlike the model's own reasoning. When injected into a GRPO batch, the model sees a correct response that looks alien — high reward, but the path to producing it is opaque. The gradient points toward a region of response space the model has no idea how to reach.

**Base model rejection sampling:** Sample at standard temperature until a correct response appears. This is on-distribution by definition — it's something the model actually produced. But for hard examples, rejection sampling may require thousands of attempts, and the resulting correct response may be a low-probability fluke using an unusual reasoning path.

**Power sampling (our proposal):** Karan & Du (2025; arXiv:2510.14901) showed that sampling from p^α (the power distribution with α > 1) produces responses from high-likelihood regions while maintaining diversity. A power-sampled correct response is one the base model assigns high probability to — it represents the model's *best accessible reasoning*, not a random walk that happened to reach the right answer.

### 3.2 Why Power Sampling is Ideal for Seeding

Power sampling with α=4.0 amplifies the model's own probability distribution, favoring sequences where the model's "future planning" leads to good outcomes. The key insight from the power sampling paper is that these high-likelihood correct solutions exist even when standard sampling rarely finds them — the model "knows" a good solution but can't reliably produce it because the probability mass is spread too thin.

For the Phase 1 cache, this means:

1. The correct response is **on-distribution** — it's a sequence the model itself considers likely
2. It's **high quality** — it represents the model's best reasoning, not a fluke
3. For hard examples where rejection sampling fails entirely, power sampling may be the **only way** to get a correct on-distribution response
4. The computational cost is ~9× standard inference, but it's a **one-time cost** before training begins — for 1-shot RLVR, that's literally one power-sampled generation

### 3.3 Rescuing Hard Examples

This creates a concrete path to extending 1-shot RLVR to harder problems:

**Without our framework:** π₁₂₀₈ fails in 1-shot RLVR because the base model rarely (or never) produces correct rollouts. The model is stuck in Phase 1 forever. Zero gradient. No learning.

**With power-sampled positive replay only:** The model now has a correct exemplar to learn from in Phase 1. It can bootstrap to Phase 2, where natural mixing of correct/incorrect rollouts drives normal GRPO learning. But once it saturates in Phase 3, we're back to the same problem as π₁ — all-correct groups, vanishing signal.

**With full bidirectional replay:** Phase 1 is bootstrapped by the power-sampled correct response. Phase 2 errors are cached. Phase 3 uses those cached errors to maintain signal. The model can now successfully train on examples that were previously intractable, potentially expanding the set of viable 1-shot RLVR training examples significantly.

If we can show that the fraction of DSR-sub examples that work for 1-shot RLVR increases from, say, 30% to 60% with our framework, that's a significant practical result.

---

## 4. Existing Work Landscape

### 4.1 Approaches that Rescue All-Incorrect Groups (Positive-Only Replay)

All of these solve Phase 1 only. None address Phase 3.

**AR3PO** (arXiv:2509.25808, Sept 2025) — Adaptive rollout allocation + injects previously generated correct responses when groups are all-wrong. Reduces rollout cost 4.2× vs DAPO. Positive replay only.

**RLEP** (arXiv:2507.07451, July 2025) — Stores correct trajectories from a preliminary run, merges them with on-policy rollouts. Qwen2.5-7B: +1.7pp AIME24, +2.5pp AIME25, +5.2pp AMC23. Positive replay only.

**ARPO** (arXiv:2505.16282, May 2025) — GUI agent experience replay. Caches successful trajectories per task; injects into all-failed GRPO groups. Positive replay only.

**NGRPO** (arXiv:2509.18851, Sept 2025) — Virtual maximum-reward sample analytically adjusts advantage for all-incorrect groups. No actual injection. All-incorrect only.

**LENS** (arXiv:2510.08696, Oct 2025) — Confidence-dependent non-zero rewards for incorrect generations. MLE gradient as modified policy gradient. All-incorrect only.

**SGPO** (arXiv:2505.11595, May 2025) — Step-wise judge creates reward diversity among incorrect responses. Requires judge model. All-incorrect only.

**EDGE-GRPO** (arXiv:2507.21848, July 2025) — Guided Error Correction + Entropy-Driven Advantage. General advantage collapse, not specifically targeting all-correct.

### 4.2 Approaches that Skip/Filter All-Correct Groups

All of these treat saturated prompts as waste. None try to extract signal from them.

**DAPO** (Yu et al., 2025) — Dynamic sampling discards zero-variance groups and resamples. Requires large prompt pool (1536). Cannot work in 1-shot — there are no alternative prompts.

**GRESO** (arXiv:2506.02177, June 2025) — Predicts and skips uninformative prompts before rollout. Assumes multi-prompt setting.

**SRPO / History Resampling** (Kwai AI, April 2025) — Explicitly filters out all-correct samples between epochs. Directly discards the regime our method targets.

**F-GRPO** (arXiv:2602.06717, Feb 2026) — Focal weighting downweights high-success prompts. Reduces contribution but doesn't create new signal.

### 4.3 General Replay (Not Phase-Aware)

**RePO** (arXiv:2506.09340, June 2025) — General replay buffer with diverse retrieval strategies (recency, reward-oriented, variance-driven). +18.4pp Qwen2.5-Math-1.5B vs GRPO. Closest architecturally to our approach, but: (a) general-purpose, not phase-aware; (b) designed for multi-prompt efficiency; (c) does not specifically target negative injection into all-correct groups; (d) does not distinguish between positive and negative cached responses.

**PACED-RL** (arXiv:2602.12642, Feb 2026) — GFlowNet-based with accuracy-error-prioritized replay. Prompt selection focus, not directional replay.

### 4.4 Theoretical Motivation (Why Negative Signal Matters)

**"The Surprising Effectiveness of Negative Reinforcement"** (Zhu et al., NeurIPS 2025; arXiv:2506.01347) — Decomposes RLVR into Positive Sample Reinforcement (PSR) and Negative Sample Reinforcement (NSR). Key finding: training with *only negative samples* — penalizing errors without reinforcing correct answers — matches or surpasses PPO and GRPO across the entire Pass@k spectrum (k up to 256). NSR works by suppressing incorrect generations and redistributing probability mass toward plausible alternatives guided by the model's prior. PSR improves Pass@1 but *reduces diversity* and hurts Pass@k. Their W-REINFORCE (λ=0.1 downweighting PSR) yields best overall results. **This directly motivates our approach: the negative replay mechanism guarantees NSR signal is always available, even when the model no longer naturally produces errors.**

**"Rethinking Sample Polarity" / A3PO** (arXiv:2512.21625, Dec 2025) — Systematic analysis of polarity effects: positive samples sharpen existing correct reasoning patterns; negative samples encourage exploration of new reasoning paths. Their A3PO adaptively weights tokens across polarities. **Supports our Phase 3 mechanism: negative replay maintains the exploration-promoting signal that drives post-saturation generalization.**

**"Two-Stage Dynamic View"** (arXiv:2510.04028, Oct 2025) — Formalizes exploitation → exploration dynamic. Initial training sharpens existing high-reward tokens (exploitation). Prolonged training shifts mass to previously low-probability but high-reward tokens (exploration). **Aligns with our framework: Phase 3 is when exploration matters most, and negative replay accelerates it.**

### 4.5 The 1-Shot RLVR Testbed

**Wang et al.** (NeurIPS 2025; arXiv:2504.20571) — 1-shot RLVR with 1024 rollouts per step, GRPO + entropy loss. Post-saturation generalization from step ~100 to ~2000. Without entropy loss, performance plateaus immediately at saturation. The training signal comes from 99.x% ≠ 100%. Entropy loss alone (no reward) improves MATH500 by 27.4%. **Our testbed: the cleanest setting to study signal maintenance, where all three phases occur sequentially on the same prompt.**

### 4.6 Power Sampling

**Karan & Du** (2025; arXiv:2510.14901) — Training-free inference from p^α. 74.8% MATH500 vs 78.5% GRPO vs 49.6% base (Qwen2.5-Math-7B). MCMC with Metropolis-Hastings, α=4.0, ~8.84× token cost. Pass@k superior to GRPO (avoids diversity collapse). **Our use: one-time generation of high-quality on-distribution correct response for Phase 1 seeding.**

---

## 5. Novelty Analysis

### What has been done

| Problem | Existing Solutions | Count |
| --- | --- | --- |
| All-incorrect groups | AR3PO, RLEP, ARPO, NGRPO, LENS, SGPO, EDGE-GRPO | 7+ |
| All-correct groups | DAPO, GRESO, SRPO, F-GRPO (all skip/filter) | 4+ |
| General replay | RePO, PACED-RL | 2 |
| Polarity analysis | Zhu et al., A3PO, Two-Stage Dynamic | 3 |

### What has NOT been done

1. **Injecting cached incorrect responses into all-correct groups** — zero prior work
2. **Bidirectional (positive + negative) replay in a unified framework** — all replay is positive-only
3. **Phase-aware adaptive replay** that detects the training regime and switches direction
4. **Replay designed for 1-shot/few-shot RLVR** — all existing replay assumes multi-prompt
5. **Power sampling to seed a replay buffer** — no connection made previously
6. **Testing whether replay can substitute for entropy loss** in maintaining post-saturation signal

### Why this gap exists

In multi-prompt RLVR with thousands of training prompts, the all-correct problem is a **resource allocation** problem: just skip easy prompts and focus compute on harder ones. There is no need to maintain signal on a saturated prompt when you can switch to an unsaturated one. The 1-shot setting transforms this into a **signal maintenance** problem — you have one prompt, and once it saturates, you cannot switch. This is why nobody has needed negative replay before, and why the 1-shot setting is the ideal testbed for this idea.

---

## 6. Experimental Design

### 6.1 Main Ablation Matrix

| Method | Entropy Loss | Positive Replay | Negative Replay |
| --- | --- | --- | --- |
| Baseline (reproducing 1-shot paper) | ✓ | ✗ | ✗ |
| No entropy baseline | ✗ | ✗ | ✗ |
| Positive only | ✓ | ✓ | ✗ |
| Negative only | ✓ | ✗ | ✓ |
| Full bidirectional | ✓ | ✓ | ✓ |
| Full bidirectional, no entropy | ✗ | ✓ | ✓ |

### 6.2 Phase 1 Source Ablation

| Source | Method | On-distribution? |
| --- | --- | --- |
| Ground truth | Dataset answer | No |
| Rejection sampling | Temp 0.6, filter correct | Yes |
| **Power sampling** | **p^α, α=4.0, MCMC** | **Yes, high-likelihood** |
| Teacher model | Qwen2.5-Math-7B | No |

### 6.3 Key Hypotheses

1. **Substitution:** Full bidirectional + no entropy ≈ baseline → replay can substitute for entropy loss as a signal maintenance mechanism
2. **Complementarity:** Full bidirectional + entropy > baseline → the mechanisms are complementary, each providing signal the other doesn't
3. **Hard example rescue:** Full bidirectional with power-sampled seeding enables training on π₁₂₀₈ and other examples that fail under standard 1-shot RLVR
4. **Power sampling advantage:** Power-sampled Phase 1 cache outperforms other sources, especially on hard examples

### 6.4 Measurements

- Post-saturation generalization curves (extending Figure 2 from 1-shot paper)
- Advantage signal variance over training steps (directly measuring signal maintenance)
- Cache age vs learning effectiveness (staleness analysis)
- Phase transition timing: when does the model enter each phase?
- Pass@k spectrum following Zhu et al. — does negative replay preserve diversity like NSR?
- Trainable fraction: what percentage of DSR-sub examples succeed with vs without our framework?

### 6.5 Setup

- **Models:** Qwen2.5-Math-1.5B (primary), Qwen2.5-Math-7B (scaling validation)
- **Training examples:** π₁ (easy, base model ~30% correct), π₁₃ (medium), π₁₂₀₈ (hard, base model rarely correct)
- **Framework:** VERL (same as 1-shot paper for direct comparison)
- **Benchmarks:** MATH500, AIME24, AIME25, AMC23, GSM8K, OlympiadBench

---

## 7. Thesis Narrative

**Part 1 — The Problem:** We formalize the post-saturation generalization phenomenon in 1-shot RLVR as a signal maintenance problem. When training accuracy saturates, GRPO's advantage normalization produces zero gradients. The 1-shot paper shows entropy loss is critical but doesn't explain the mechanism or propose alternatives. We analyze the three training phases and identify Phase 3 (all-correct) as the underserved regime.

**Part 2 — The Mechanism:** We propose bidirectional adaptive replay. Positive replay (seeded via power sampling from the base model) bootstraps Phase 1, enabling training on hard examples. Negative replay (injecting cached on-policy errors) maintains Phase 3 signal. We connect this to Zhu et al.'s finding that NSR drives exploration, and A3PO's finding that negative samples encourage new reasoning paths. We argue that our mechanism *guarantees* these beneficial signals remain available throughout training, rather than depending on entropy-induced stochastic failures.

**Part 3 — The Impact:** We demonstrate that bidirectional replay (a) extends post-saturation generalization beyond what entropy alone achieves, (b) can partially or fully substitute for entropy loss as a signal maintenance mechanism, (c) rescues hard examples that are intractable under standard 1-shot RLVR via power-sampled seeding, and (d) provides a clean experimental framework for disentangling the contributions of positive signal, negative signal, and entropy loss to RLVR training dynamics — a question that existing multi-prompt work cannot cleanly address.

---

## 8. Complete References

**Testbed + Tools:**
Wang et al. (2025). 1-shot RLVR. NeurIPS 2025. arXiv:2504.20571 •
Karan & Du (2025). Power Sampling. arXiv:2510.14901

**Positive Replay (Phase 1 prior work):**
AR3PO: arXiv:2509.25808 • RLEP: arXiv:2507.07451 • ARPO: arXiv:2505.16282 • VL-Rethinker SSR: arXiv:2504.08837

**All-Incorrect Solutions (Analytical):**
NGRPO: arXiv:2509.18851 • LENS: arXiv:2510.08696 • SGPO: arXiv:2505.11595 • EDGE-GRPO: arXiv:2507.21848

**Skip/Filter (All-Correct Handling):**
DAPO: Yu et al. 2025 • GRESO: arXiv:2506.02177 • SRPO: Kwai 2025 • F-GRPO: arXiv:2602.06717

**General Replay:**
RePO: arXiv:2506.09340 • PACED-RL: arXiv:2602.12642

**Theoretical Motivation:**
Zhu et al. NSR: NeurIPS 2025, arXiv:2506.01347 • A3PO: arXiv:2512.21625 • Two-Stage Dynamic: arXiv:2510.04028

**Data Selection:**
DEPO: arXiv:2509.01321 • LIMR: arXiv:2502.11886

**GRPO Analysis:**
2-GRPO: arXiv:2510.00977 • Exploration vs Exploitation: arXiv:2512.16912 • GSPO: arXiv:2507.18071