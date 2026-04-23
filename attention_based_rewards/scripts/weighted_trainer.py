"""Monkey-patches for circuit-guided GRPO token weighting.

Strategy:
  1. Patch `compute_advantage` in ray_trainer module to multiply per-token weights
     into the advantages tensor after standard GRPO advantage computation.
  2. For entropy-based conditions, patch the fit loop to preserve the entropy tensor
     (which verl normally pops after logging) by saving it under a different key.
  3. For attention-based conditions, load a separate HF model with eager attention
     and run a forward pass to extract attention weights from reasoning heads.

Usage:
    from weighted_trainer import apply_patches
    apply_patches(condition="attention", reasoning_heads=..., head_scores=...)
    # Then run verl training as normal
"""

import logging

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Will be set by apply_patches()
_CONDITION = "uniform"
_ATTN_MODEL = None
_REASONING_HEADS = None
_HEAD_SCORES = None
_ALPHA = 0.5
_MODEL_NAME = None
_MAX_TOKEN_WEIGHT = 5.0  # clip per-token weights to this value after normalization
_LIKELIHOOD_LAMBDA = 0.2
_LIKELIHOOD_BETA = 5.0


import os as _os
_CONFIG_PATH = f"/tmp/weighted_trainer_config_{_os.environ.get('SLURM_JOB_ID', 'default')}.pt"


def apply_patches(
    condition: str,
    model_name: str = None,
    reasoning_heads: list[tuple[int, int]] = None,
    head_scores: torch.Tensor = None,
    alpha: float = 0.5,
    anchor_percentile: float = 90.0,
    likelihood_lambda: float = 0.2,
    likelihood_beta: float = 5.0,
):
    """Configure token-weighted advantages for verl GRPO training.

    Saves config to disk so Ray workers (separate processes) can load it.
    The actual weight computation is triggered by a hook in verl's
    compute_advantage() which calls apply_token_weights().

    Args:
        condition: One of "uniform", "attention", "entropy", "combined",
                   "fai", "fai_allheads", "fai_asymmetric"
        model_name: HF model name (needed for attention/fai conditions)
        reasoning_heads: list of (layer, head) tuples from Phase 1
        head_scores: (n_layers, n_heads) importance scores
        alpha: mixing coefficient for combined condition
    """
    global _CONDITION, _ATTN_MODEL, _REASONING_HEADS, _HEAD_SCORES, _ALPHA, _MODEL_NAME

    _CONDITION = condition
    _REASONING_HEADS = reasoning_heads
    _HEAD_SCORES = head_scores
    _ALPHA = alpha
    _MODEL_NAME = model_name

    # Conditions that need attention model
    _ATTN_CONDITIONS = ("attention", "combined", "fai", "asymmetric", "attention_top5", "fai_allheads", "fai_asymmetric", "anchor_credit", "circuit_reward", "activation_entropy", "mlp_circuit_reward", "layerwise_slope")
    # Conditions that need entropy preservation
    _ENTROPY_CONDITIONS = ("entropy", "combined")

    if condition in _ATTN_CONDITIONS:
        if condition not in ("fai_allheads", "activation_entropy", "layerwise_slope"):
            if reasoning_heads is None or head_scores is None:
                raise ValueError(f"Condition '{condition}' requires reasoning_heads and head_scores")
        if model_name is None:
            raise ValueError(f"Condition '{condition}' requires model_name for attention extraction")

    # Save config to disk so Ray workers can load it
    config = {
        "condition": condition,
        "model_name": model_name,
        "reasoning_heads": reasoning_heads,
        "head_scores": head_scores,
        "alpha": alpha,
        "anchor_percentile": anchor_percentile,
        "likelihood_lambda": likelihood_lambda,
        "likelihood_beta": likelihood_beta,
    }
    torch.save(config, _CONFIG_PATH)
    logger.info(f"Saved weighted_trainer config to {_CONFIG_PATH}")
    logger.info(f"Condition: {condition}, model: {model_name}")


def _ensure_config_loaded():
    """Load config from disk in the Ray worker process (lazy, once)."""
    global _CONDITION, _REASONING_HEADS, _HEAD_SCORES, _ALPHA, _MODEL_NAME
    global _LIKELIHOOD_LAMBDA, _LIKELIHOOD_BETA

    if _CONDITION != "uniform":
        return  # Already loaded

    import os
    if not os.path.exists(_CONFIG_PATH):
        return  # No config file, stay uniform

    config = torch.load(_CONFIG_PATH, weights_only=False)
    _CONDITION = config["condition"]
    _MODEL_NAME = config["model_name"]
    _REASONING_HEADS = config["reasoning_heads"]
    _HEAD_SCORES = config["head_scores"]
    _ALPHA = config["alpha"]
    _LIKELIHOOD_LAMBDA = config.get("likelihood_lambda", 0.2)
    _LIKELIHOOD_BETA = config.get("likelihood_beta", 5.0)
    _weight_log(f"Loaded config: condition={_CONDITION}, model={_MODEL_NAME}")


def _ensure_attention_model():
    """Lazily load the attention model on first use.

    With RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0, the TaskRunner sees GPUs
    even with num_gpus=0. Model loads on CUDA (~3GB bfloat16).
    FAI uses hooks to process attention one layer at a time.
    """
    global _ATTN_MODEL
    if _ATTN_MODEL is not None:
        return

    if _MODEL_NAME is None:
        raise RuntimeError("No model_name configured for attention extraction")

    if torch.cuda.is_available():
        device_map = "cuda"
        dtype = torch.bfloat16
        _weight_log(f"Loading attention model on CUDA (bfloat16): {_MODEL_NAME}")
    else:
        device_map = "cpu"
        dtype = torch.bfloat16
        _weight_log(f"Loading attention model on CPU (bfloat16): {_MODEL_NAME}")

    _ATTN_MODEL = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=device_map,
    )
    _ATTN_MODEL.eval()
    _weight_log(f"Attention model loaded on {next(_ATTN_MODEL.parameters()).device}")


def _patch_entropy_preservation():
    """Patch to save entropy before verl pops it from old_log_prob.

    verl computes entropy at line ~1456 and pops it at line ~1470.
    We wrap the pop to save a copy first.
    """
    import verl.trainer.ppo.ray_trainer as rt_module

    _original_fit = rt_module.RayPPOTrainer.fit

    def _patched_fit(self, *args, **kwargs):
        """Wrap fit to prevent entropy from being popped."""
        # Patch the internal _compute_old_log_prob to preserve entropy
        _orig_compute_olp = self._compute_old_log_prob

        def _compute_olp_keep_entropy(batch):
            old_log_prob, mfu = _orig_compute_olp(batch)
            # Save entropy under a different key so it survives the pop
            if "entropys" in old_log_prob.batch.keys():
                old_log_prob.batch["_saved_entropys"] = old_log_prob.batch["entropys"].clone()
            return old_log_prob, mfu

        self._compute_old_log_prob = _compute_olp_keep_entropy
        return _original_fit(self, *args, **kwargs)

    rt_module.RayPPOTrainer.fit = _patched_fit


_LOG_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards/logs"


def _weight_log(msg):
    """Write to a log file reliably (stdout/logger may be swallowed by Ray)."""
    import os
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    log_path = f"{_LOG_DIR}/weight_check_{_CONDITION}_{job_id}.log"
    with open(log_path, "a") as _f:
        _f.write(msg + "\n")


def apply_token_weights(data):
    """Called from verl's compute_advantage (ray_trainer.py) to apply token weights.

    This function is called directly from the verl source code (not via monkey-patching)
    so it runs in the correct Ray worker process.
    """
    import traceback

    _ensure_config_loaded()

    try:
        weights = _compute_token_weights(data)

        if weights is not None:
            response_mask = data.batch["response_mask"]

            # ── Clip extreme weights and re-normalize to mean=1 per seq ──
            # anchor_credit already clips internally and must NOT be normalized
            # (normalizing to mean=1 would push non-anchor tokens below 1.0)
            if _CONDITION not in ("uniform", "anchor_credit", "circuit_reward", "activation_entropy", "mlp_circuit_reward", "layerwise_slope") and _MAX_TOKEN_WEIGHT is not None:
                weights = weights.clamp(max=_MAX_TOKEN_WEIGHT)
                # Re-normalize so mean over response tokens = 1 per sequence
                masked_w = weights * response_mask
                seq_means = masked_w.sum(dim=-1, keepdim=True) / (
                    response_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                )
                weights = (masked_w / seq_means.clamp(min=1e-8)) * response_mask

            masked_weights = weights * response_mask
            valid_weights = masked_weights[response_mask.bool()]
            if valid_weights.numel() > 0:
                w_mean = valid_weights.mean().item()
                w_std = valid_weights.std().item()
                w_min = valid_weights.min().item()
                w_max = valid_weights.max().item()
                n_nan = torch.isnan(valid_weights).sum().item()

                _weight_log(
                    f"mean={w_mean:.4f}, std={w_std:.4f}, "
                    f"min={w_min:.4f}, max={w_max:.4f}, "
                    f"nan={n_nan}"
                )

                if n_nan > 0:
                    _weight_log(f"ERROR: {n_nan} NaN values — falling back to uniform")
                    weights = torch.ones_like(weights)
                if w_std < 1e-6 and _CONDITION not in ("uniform",):
                    _weight_log(f"ERROR: weights are constant (std={w_std:.8f})")

            # Store weights in batch for metric_utils to pick up
            data.batch["_token_weights"] = weights
            data.batch["advantages"] = data.batch["advantages"] * weights
        else:
            _weight_log("weights is None")

    except Exception as e:
        _weight_log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        raise

    return data


def _compute_token_weights(data) -> torch.Tensor | None:
    """Compute per-token weights based on the active condition.

    Args:
        data: DataProto with batch containing input_ids, attention_mask, response_mask, etc.

    Returns:
        (bs, seq_len) tensor of weights, or None for uniform
    """
    from attention_based_rewards.scripts.token_weighting import (
        compute_asymmetric_weights,
        compute_attention_weights,
        compute_combined_weights,
        compute_entropy_weights,
        compute_fai_weights,
        compute_uniform_weights,
    )

    response_mask = data.batch["response_mask"]

    if _CONDITION in ("uniform", "circuit_reward", "activation_entropy", "mlp_circuit_reward", "layerwise_slope", "likelihood_bonus"):
        # These conditions modify rewards (sequence-level blending), not per-token weights
        return compute_uniform_weights(response_mask)

    if _CONDITION == "entropy":
        entropy = _get_entropy(data)
        return compute_entropy_weights(entropy, response_mask)

    if _CONDITION in ("attention", "attention_top5"):
        return _get_attention_weights(data)

    if _CONDITION == "combined":
        attn_w = _get_attention_weights(data)
        entropy = _get_entropy(data)
        entropy_w = compute_entropy_weights(entropy, response_mask)
        return compute_combined_weights(attn_w, entropy_w, alpha=_ALPHA)

    if _CONDITION == "fai":
        return _get_fai_weights(data)

    if _CONDITION == "asymmetric":
        attn_w = _get_attention_weights(data)
        advantages = data.batch["advantages"]
        return compute_asymmetric_weights(attn_w, advantages, response_mask)

    if _CONDITION == "fai_allheads":
        return _get_fai_allheads_weights(data)

    if _CONDITION == "fai_asymmetric":
        fai_w = _get_fai_weights(data)
        advantages = data.batch["advantages"]
        return compute_asymmetric_weights(fai_w, advantages, response_mask)

    if _CONDITION == "anchor_credit":
        return _get_anchor_credit_weights(data)

    raise ValueError(f"Unknown condition: {_CONDITION}")


def _get_entropy(data) -> torch.Tensor:
    """Get per-token entropy from the batch.

    Tries _saved_entropys (from our patch), falls back to recomputing from old_log_probs.
    """
    if "_saved_entropys" in data.batch.keys():
        return data.batch["_saved_entropys"]

    # Fallback: estimate entropy from old_log_probs
    # H(p) ≈ -log_prob (for a rough per-token estimate)
    # This is the negative log-probability, which correlates with entropy
    if "old_log_probs" in data.batch.keys():
        logger.warning("Using -old_log_probs as entropy proxy (saved entropy not available)")
        return -data.batch["old_log_probs"]

    raise RuntimeError(
        "Cannot compute entropy weights: neither _saved_entropys nor old_log_probs in batch. "
        "Make sure _patch_entropy_preservation() was called."
    )


def _get_fai_weights(data) -> torch.Tensor:
    """Get FAI-based weights, preferring pre-computed from GPU workers."""
    # Pre-computed by actor workers via compute_fai_weights (no separate model needed)
    if "_precomputed_fai_weights" in data.batch.keys():
        _weight_log("Using pre-computed FAI weights from GPU workers")
        return data.batch["_precomputed_fai_weights"].to(data.batch["input_ids"].device)

    # Fallback: compute locally (requires GPU access in driver — may fail on HPC)
    _weight_log("WARNING: _precomputed_fai_weights not found, falling back to local computation")
    from attention_based_rewards.scripts.token_weighting import compute_fai_weights

    _ensure_attention_model()

    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    model_device = next(_ATTN_MODEL.parameters()).device

    chunk_size = 4
    bs = input_ids.shape[0]
    all_weights = []

    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)
        chunk_weights = compute_fai_weights(
            input_ids=input_ids[start:end].to(model_device),
            attention_mask=attention_mask[start:end].to(model_device),
            response_mask=response_mask[start:end].to(model_device),
            attn_model=_ATTN_MODEL,
            reasoning_heads=_REASONING_HEADS,
            head_scores=_HEAD_SCORES,
        )
        all_weights.append(chunk_weights.to(input_ids.device))

    return torch.cat(all_weights, dim=0)


def _get_fai_allheads_weights(data) -> torch.Tensor:
    """Get FAI weights across ALL heads, preferring pre-computed from GPU workers."""
    # Pre-computed by actor workers (same path as FAI — worker checks condition)
    if "_precomputed_fai_weights" in data.batch.keys():
        _weight_log("Using pre-computed FAI-allheads weights from GPU workers")
        return data.batch["_precomputed_fai_weights"].to(data.batch["input_ids"].device)

    # Fallback: compute locally
    _weight_log("WARNING: _precomputed_fai_weights not found, falling back to local computation")
    from attention_based_rewards.scripts.token_weighting import compute_fai_weights_allheads

    _ensure_attention_model()

    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    model_device = next(_ATTN_MODEL.parameters()).device

    chunk_size = 2
    bs = input_ids.shape[0]
    all_weights = []

    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)
        chunk_weights = compute_fai_weights_allheads(
            input_ids=input_ids[start:end].to(model_device),
            attention_mask=attention_mask[start:end].to(model_device),
            response_mask=response_mask[start:end].to(model_device),
            attn_model=_ATTN_MODEL,
        )
        all_weights.append(chunk_weights.to(input_ids.device))

    return torch.cat(all_weights, dim=0)


def _get_attention_weights(data) -> torch.Tensor:
    """Get attention-based weights, preferring pre-computed from GPU workers."""
    # Pre-computed by actor workers (FAI weights serve as attention proxy)
    if "_precomputed_fai_weights" in data.batch.keys():
        _weight_log("Using pre-computed attention weights from GPU workers")
        return data.batch["_precomputed_fai_weights"].to(data.batch["input_ids"].device)

    # Fallback: compute locally
    _weight_log("WARNING: _precomputed_fai_weights not found, falling back to local computation")
    from attention_based_rewards.scripts.token_weighting import compute_attention_weights

    _ensure_attention_model()

    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    model_device = next(_ATTN_MODEL.parameters()).device

    chunk_size = 16
    bs = input_ids.shape[0]
    all_weights = []

    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)
        chunk_weights = compute_attention_weights(
            input_ids=input_ids[start:end].to(model_device),
            attention_mask=attention_mask[start:end].to(model_device),
            response_mask=response_mask[start:end].to(model_device),
            attn_model=_ATTN_MODEL,
            reasoning_heads=_REASONING_HEADS,
            head_scores=_HEAD_SCORES,
        )
        all_weights.append(chunk_weights.to(input_ids.device))

    return torch.cat(all_weights, dim=0)


def _get_anchor_credit_weights(data) -> torch.Tensor:
    """Get anchor-credit weights: pre-computed on GPU workers, then gate by correctness.

    Pre-computed weights come from compute_anchor_credit_weights on actor GPUs.
    Here we apply the correctness gate: only boost correct responses (advantage > 0),
    incorrect responses get uniform weight (1.0).
    """
    response_mask = data.batch["response_mask"]
    advantages = data.batch["advantages"]

    # Get pre-computed anchor credit weights from GPU workers
    if "_precomputed_fai_weights" in data.batch.keys():
        _weight_log("Using pre-computed anchor credit weights from GPU workers")
        weights = data.batch["_precomputed_fai_weights"].to(data.batch["input_ids"].device)
    else:
        _weight_log("WARNING: _precomputed_fai_weights not found for anchor_credit, falling back to local")
        weights = _compute_anchor_credit_local(data)

    # Get diagnostic masks if available
    anchor_mask = data.batch.get("_anchor_mask", None)
    dependent_mask = data.batch.get("_dependent_mask", None)

    # ── Correctness gate: only apply non-uniform weights to correct responses ──
    # In GRPO, advantage > 0 means correct (above-group-mean reward)
    is_correct = (advantages[:, 0] > 0).unsqueeze(1)  # (bs, 1)

    # Where incorrect, reset to uniform (1.0)
    weights = torch.where(is_correct, weights, response_mask.float())

    # ── Log diagnostics ──
    resp_bool = response_mask.bool()
    n_correct = is_correct.sum().item()
    n_total = is_correct.shape[0]

    if anchor_mask is not None:
        anchor_frac = (anchor_mask & resp_bool).float().sum() / resp_bool.float().sum().clamp(min=1)
        dep_frac = (dependent_mask & resp_bool).float().sum() / resp_bool.float().sum().clamp(min=1) if dependent_mask is not None else 0
        _weight_log(f"anchor_credit: correct={n_correct}/{n_total}, "
                    f"anchor_frac={anchor_frac:.3f}, dependent_frac={dep_frac:.3f}")

    correct_weights = weights[is_correct.expand_as(weights) & resp_bool]
    if correct_weights.numel() > 0:
        _weight_log(f"  correct weights: mean={correct_weights.mean():.3f}, "
                    f"min={correct_weights.min():.3f}, max={correct_weights.max():.3f}")

    return weights


def _compute_anchor_credit_local(data) -> torch.Tensor:
    """Fallback: compute anchor credit weights locally (requires GPU)."""
    from attention_based_rewards.scripts.token_weighting import compute_anchor_credit_weights

    _ensure_attention_model()

    input_ids = data.batch["input_ids"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    model_device = next(_ATTN_MODEL.parameters()).device

    chunk_size = 4
    bs = input_ids.shape[0]
    all_weights = []

    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)
        w, _, _ = compute_anchor_credit_weights(
            input_ids=input_ids[start:end].to(model_device),
            attention_mask=attention_mask[start:end].to(model_device),
            response_mask=response_mask[start:end].to(model_device),
            attn_model=_ATTN_MODEL,
            reasoning_heads=_REASONING_HEADS,
            head_scores=_HEAD_SCORES,
        )
        all_weights.append(w.to(input_ids.device))

    return torch.cat(all_weights, dim=0)


def apply_likelihood_bonus(batch, uid_key="uid"):
    """Add a likelihood bonus to token_level_scores using frozen base model log-probs.

    Uses ref_log_prob (frozen base model) to compute length-normalized log-likelihood
    per rollout. Centers within each prompt group, passes through sigmoid, and adds
    as a bonus scaled by lambda.

    Final reward = correctness + lambda * sigmoid(beta * (norm_lp - group_mean))

    This ensures correct rollouts always beat incorrect ones (lambda < 0.5),
    while differentiating within each group.

    Must be called after ref_log_prob is in the batch and token_level_scores are set.
    """
    from collections import defaultdict

    _ensure_config_loaded()

    if _CONDITION != "likelihood_bonus":
        return batch

    lam = _LIKELIHOOD_LAMBDA
    beta = _LIKELIHOOD_BETA

    if "ref_log_prob" not in batch.batch:
        _weight_log("WARNING: ref_log_prob not in batch, skipping likelihood bonus")
        return batch

    ref_lp = batch.batch["ref_log_prob"]        # (bs, resp_len)
    resp_mask = batch.batch["response_mask"]     # (bs, resp_len)
    tls = batch.batch["token_level_scores"]      # (bs, resp_len)
    uid = batch.non_tensor_batch[uid_key]        # (bs,)

    bs = ref_lp.shape[0]

    # Length-normalized log-likelihood per sequence
    total_lp = (ref_lp * resp_mask).sum(dim=-1)          # (bs,)
    seq_len = resp_mask.sum(dim=-1).clamp(min=1)          # (bs,)
    norm_lp = total_lp / seq_len                          # (bs,)

    # Center around group mean (per prompt)
    id2scores = defaultdict(list)
    for i in range(bs):
        id2scores[uid[i]].append((i, norm_lp[i].item()))

    centered = torch.zeros_like(norm_lp)
    for prompt_id, entries in id2scores.items():
        group_mean = sum(v for _, v in entries) / len(entries)
        for idx, val in entries:
            centered[idx] = val - group_mean

    # Sigmoid maps centered values to (0, 1)
    bonus = torch.sigmoid(beta * centered)  # (bs,)

    # Original correctness (binary 0/1)
    correctness = tls.sum(dim=-1)  # (bs,)

    # Save original for logging
    batch.batch["_original_correctness"] = correctness.clone()
    batch.batch["_likelihood_bonus"] = bonus.clone()

    # Final reward = correctness + lambda * bonus
    final_reward = correctness + lam * bonus

    # Put blended reward at last response token (verl convention)
    new_tls = torch.zeros_like(tls)
    last_idx = resp_mask.sum(dim=-1).long() - 1
    for i in range(bs):
        new_tls[i, last_idx[i]] = final_reward[i]
    batch.batch["token_level_scores"] = new_tls

    # Logging
    correct_mask = correctness > 0.5
    incorrect_mask = ~correct_mask
    n_correct = correct_mask.sum().item()
    n_incorrect = incorrect_mask.sum().item()

    log_parts = [
        f"likelihood_bonus: lambda={lam}, beta={beta}, bs={bs}",
        f"  correct={n_correct}, incorrect={n_incorrect}",
        f"  bonus: mean={bonus.mean():.4f}, std={bonus.std():.4f}",
        f"  norm_lp: mean={norm_lp.mean():.4f}, std={norm_lp.std():.4f}",
        f"  final_reward: mean={final_reward.mean():.4f}",
    ]
    if n_correct > 0:
        log_parts.append(f"  correct_bonus: mean={bonus[correct_mask].mean():.4f}")
        log_parts.append(f"  correct_reward: mean={final_reward[correct_mask].mean():.4f}")
    if n_incorrect > 0:
        log_parts.append(f"  incorrect_bonus: mean={bonus[incorrect_mask].mean():.4f}")
        log_parts.append(f"  incorrect_reward: mean={final_reward[incorrect_mask].mean():.4f}")

    _weight_log("\n".join(log_parts))

    return batch
