# Bidirectional Experience Replay (BER) — Implementation Guide

## Overview

BER is implemented as a lightweight extension to verl's `RayPPOTrainer`. Rather than copying or modifying verl's 380-line `fit()` method, we override a single method (`_compute_or_extract_reward`) to inject BER logic at exactly the right point in the training loop — after rewards are computed but before `old_log_probs` are recomputed on the (now-modified) batch.

## Architecture

```
src/rlvr_grokking/ber/
├── __init__.py          # Empty
├── ber_module.py        # Core BER algorithm (classify, inject, cache)
├── ber_trainer.py       # BERRayPPOTrainer + BERTaskRunner (verl subclasses)
└── main_ber.py          # Hydra entry point

scripts/
├── generate_correct_cache.py   # Pre-training: rejection-sample a correct response
└── run_grpo_ber.sh             # Training launch script with BER config
```

## How It Works

### The Hook Point

verl's training loop in `RayPPOTrainer.fit()` follows this sequence:

```
1. Generate rollouts (1024 = 128 prompts × 8 rollouts)
2. Compute rewards via _compute_or_extract_reward()    ← BER hooks here
3. Recompute old_log_probs on batch                    ← sees modified batch
4. Compute ref_log_probs, values, advantages
5. Update actor and critic
```

`BERRayPPOTrainer` overrides `_compute_or_extract_reward()`. After calling the parent to get the reward tensor, it:
1. Classifies each group of 8 rollouts into Phase 1/2/3
2. Replaces response tensors in-place for Phase 1 and Phase 3 groups
3. Updates the reward tensor for replaced slots
4. Returns the modified reward tensor

Because `batch` is a mutable `DataProto` (tensors modified in-place), the subsequent `old_log_probs` recomputation at step 3 automatically operates on the modified responses. This means injected responses get proper log-probabilities — no off-policy correction needed.

### Phase Classification

For each group of `n_rollouts` (default 8):

```python
scores = reward_tensor.sum(dim=-1)      # per-rollout scalar scores
scores_grouped = scores.view(B, n)      # [num_groups, n_rollouts]

n_correct = (scores_grouped[i] > 0.5).sum()

if n_correct == 0:           → Phase 1 (all-incorrect)
elif n_correct == n_rollouts: → Phase 3 (all-correct)
else:                         → Phase 2 (mixed)
```

### Response Replacement

When replacing a rollout slot (always the last in each group, index `n-1`):

1. **`responses[idx]`** — cached tokens, right-padded to `response_len`
2. **`input_ids[idx]`** — reconstructed as `prompts[idx] + padded_response`
3. **`attention_mask[idx]`** — rebuilt: 1 for real prompt/response tokens, 0 for padding
4. **`position_ids[idx]`** — cumulative sum of attention_mask, clamped to 0
5. **`response_mask`** — recomputed globally from attention_mask after all replacements

The reward tensor is zeroed for the replaced slot and the new reward (1.0 or 0.0) is placed at the last valid response token position (matching verl's convention from `NaiveRewardManager`).

### Cache Management

**Correct cache** — generated once before training via rejection sampling:
```bash
python scripts/generate_correct_cache.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --train_file data/pi13_r128.parquet \
    --output data/ber_correct_cache_pi13.pt
```
Loads the training prompt, samples at temperature 0.6 until a correct response is found (graded by the same `compute_score` reward function used in training), saves the tokenized response tensor.

**Error cache** — populated during training from Phase 2 groups. Each time a mixed group is encountered, the last incorrect rollout's response tokens are cloned and stored. The cache is a single dict:
```python
{"response_tokens": tensor, "reward": 0.0, "step_cached": int}
```
If the error cache becomes too stale (age > `max_error_cache_age` steps), it's discarded.

## Configuration

BER is configured via Hydra overrides (added with `+` prefix since they're not in verl's base config):

```bash
+ber.enabled=True
+ber.correct_cache_path=data/ber_correct_cache_pi13.pt
+ber.max_error_cache_age=500
```

These are parsed in `BERTaskRunner.run()` and passed as a `BERConfig` dataclass to the trainer.

## Metrics (wandb)

BER logs these metrics at each training step:

| Metric | Description |
|--------|-------------|
| `ber/phase1_groups` | Count of all-incorrect groups |
| `ber/phase2_groups` | Count of mixed groups |
| `ber/phase3_groups` | Count of all-correct groups |
| `ber/injected_positive` | Phase 1 injections (correct cache → all-incorrect group) |
| `ber/injected_negative` | Phase 3 injections (error cache → all-correct group) |
| `ber/error_cache_age` | Steps since error cache was last updated |
| `ber/error_cache_available` | 1.0 if error cache exists, 0.0 if not |

Metrics are logged to console (prefixed `[BER step N]`) and directly to wandb.

## Expected Training Dynamics

With 128 groups of 8 rollouts per step on pi13:

| Training Phase | Steps (approx) | BER Behavior | `critic/score/mean` |
|---------------|----------------|--------------|---------------------|
| Pre-competence | 0–20 | Phase 1 dominant. Correct cache injected into all-wrong groups. | 0.1–0.3 |
| Rapid learning | 20–100 | Phase 2 dominant. Normal GRPO. Error cache continuously refreshed. | 0.3–0.9 |
| Post-saturation | 100–2000 | Phase 3 dominant. Error cache injected into all-correct groups. | ~0.875 (7/8) |

The key difference from baseline: `critic/score/mean` stabilizes at ~0.875 instead of ~1.0, because each group always has one injected incorrect response. This maintains non-zero advantage variance and keeps the learning signal alive.

## Usage

### Full training run via SLURM

```bash
# Submits the job (auto-generates correct cache if needed)
sbatch scripts/submit_training.slurm ber

# With multi-prompt validation (3 prompting strategies × 500 questions)
sbatch scripts/submit_training.slurm ber MULTI_PROMPT_VAL=1
```

### Manual steps

```bash
# Step 1: Generate correct cache (one-time, ~1 min on GPU)
python scripts/generate_correct_cache.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --train_file data/pi13_r128.parquet \
    --output data/ber_correct_cache_pi13.pt

# Step 2: Run training
bash scripts/run_grpo_ber.sh
```

### Monitoring

```bash
# Watch BER metrics in real-time
tail -f logs/slurm-<jobid>.out | grep BER

# Example output:
# [BER step 1] phase1_groups=128, phase2_groups=0, phase3_groups=0, ...
# [BER step 50] phase1_groups=12, phase2_groups=98, phase3_groups=18, ...
# [BER step 200] phase1_groups=0, phase2_groups=1, phase3_groups=127, ...
```

## Design Decisions

1. **Override `_compute_or_extract_reward` instead of copying `fit()`** — The parent's `fit()` is ~380 lines and changes between verl versions. By overriding a single method and relying on in-place batch mutation, we avoid fragile copy-paste and stay compatible with verl updates. The only constraint: `launch_reward_fn_async` must be False (checked at init).

2. **Replace last rollout in each group (index n-1), not random** — Deterministic and simple. No need for randomness since all rollouts in a homogeneous group are equivalent.

3. **Single error cache** — Simplest viable design. Could extend to multi-error cache (ring buffer of recent errors) if staleness becomes an issue.

4. **Error cache on CPU** — The cached response tokens are stored on CPU and moved to device during injection. This avoids holding GPU memory for cached tensors.

5. **Rejection sampling for correct cache (not power sampling)** — The thesis proposes power sampling for hard problems, but for pi13 (base model ~30% correct at temp 0.6), simple rejection sampling finds a correct response in 2-3 attempts. Power sampling can be added later for hard examples like pi1208.

## File Details

### `ber_module.py` — Core Algorithm

- `BERCache` dataclass — holds correct/error caches, loads from disk
- `classify_and_inject()` — main loop over groups, delegates to helpers
- `_replace_response()` — replaces all 5 batch tensors for one slot
- `_extract_response()` — clones response tokens to CPU for caching
- `_set_reward()` — zeros reward row, places new score at correct position
- `_compute_response_mask()` — recomputes from attention_mask

### `ber_trainer.py` — verl Integration

- `BERConfig` dataclass — enabled, correct_cache_path, max_error_cache_age
- `BERRayPPOTrainer(RayPPOTrainer)` — overrides `_compute_or_extract_reward`
- `BERTaskRunner(TaskRunner)` — overrides `run()` to instantiate BER trainer

### `main_ber.py` — Entry Point

- Resolves Hydra config path to verl's config directory
- Passes `BERTaskRunner` to `run_ppo()`
