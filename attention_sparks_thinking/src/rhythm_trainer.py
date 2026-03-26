"""veRL integration for attention-rhythm-guided GRPO.

Architecture overview:
  - Driver process (CPU): calls compute_advantage → apply_rhythm_weights
    reads pre-computed gammas from batch and applies them to advantages.
  - Worker processes (GPU): compute_rhythm_gammas runs the eager model,
    syncs weights from FSDP training model, computes WAAD/FAI/gamma.

The split is necessary because the driver has no GPU access; all model
inference must happen in Ray workers.

Config flow:
  1. Driver: apply_patches() saves config to /tmp/
  2. Workers: lazy-load config on first compute_rhythm_gammas call
  3. Workers: lazy-load eager model, sync weights from FSDP training model
  4. Workers: compute gammas, return to driver as _rhythm_gammas tensor
  5. Driver: apply_rhythm_weights reads _rhythm_gammas, multiplies into advantages
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# ── Config path shared between driver and workers ──
_CONFIG_PATH = f"/tmp/rhythm_trainer_config_{os.environ.get('SLURM_JOB_ID', 'default')}.pt"


# ═══════════════════════════════════════════════════════════════════════
# DRIVER-SIDE: called from compute_advantage() in ray_trainer.py
# ═══════════════════════════════════════════════════════════════════════

_DRIVER_CONFIG: dict | None = None
_DRIVER_STEP_COUNTER: int = 0


def _ensure_driver_config():
    global _DRIVER_CONFIG
    if _DRIVER_CONFIG is not None:
        return
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Rhythm config not found at {_CONFIG_PATH}")
    _DRIVER_CONFIG = torch.load(_CONFIG_PATH, map_location="cpu", weights_only=False)


def _reload_driver_config():
    """Re-read config from disk to pick up worker-written reclassification stats."""
    global _DRIVER_CONFIG
    if os.path.exists(_CONFIG_PATH):
        _DRIVER_CONFIG = torch.load(_CONFIG_PATH, map_location="cpu", weights_only=False)


_RHYTHM_METRICS: dict = {}


def get_rhythm_metrics() -> dict:
    """Return rhythm metrics for the current step (consumed by ray_trainer for wandb)."""
    return _RHYTHM_METRICS.copy()


def apply_rhythm_weights(data) -> object:
    """Driver-side hook: apply pre-computed gammas to advantages.

    Called from ray_trainer.py's compute_advantage(), AFTER standard
    advantage computation. Expects _rhythm_gammas to already be in
    data.batch (put there by the worker-side compute_rhythm_gammas).

    For Run A, this is a no-op.
    """
    global _DRIVER_STEP_COUNTER, _RHYTHM_METRICS

    _ensure_driver_config()
    cfg = _DRIVER_CONFIG
    _RHYTHM_METRICS = {}

    if cfg["run_type"] == "A":
        return data

    # Read pre-computed gammas from worker dispatch
    if "_rhythm_gammas" not in data.batch:
        logger.warning("No _rhythm_gammas in batch — worker dispatch may have failed. Skipping gamma application.")
        return data

    gammas = data.batch["_rhythm_gammas"]
    advantages = data.batch["advantages"]
    response_mask = data.batch.get("response_mask", None)

    # Gammas should already be (bs, resp_len) but trim if needed.
    resp_len = advantages.shape[1]
    if gammas.shape[1] != resp_len:
        gammas = gammas[:, -resp_len:]

    gamma_mode = cfg.get("gamma_mode", "raw")  # "raw" or "normalized"

    if gamma_mode == "normalized":
        # Normalize gamma per-response so mean(gamma * A) = mean(A)
        # This redistributes credit without changing total gradient magnitude.
        rmask = response_mask.float() if response_mask is not None else torch.ones_like(gammas)
        positive_mask = (advantages >= 0).float()
        apply_mask = positive_mask * rmask  # only positive-advantage, non-padding tokens

        for i in range(advantages.shape[0]):
            mask_i = apply_mask[i]
            n_tokens = mask_i.sum()
            if n_tokens < 2:
                continue
            g_i = gammas[i]
            a_i = advantages[i]
            # mean(g * a * mask) should equal mean(a * mask)
            weighted_sum = (g_i * a_i * mask_i).sum()
            original_sum = (a_i * mask_i).sum()
            if weighted_sum.abs() > 1e-8:
                scale = original_sum / weighted_sum
                # Rescale gamma: new_g = 1 + scale * (g - 1)
                gammas[i] = 1.0 + scale * (g_i - 1.0)

        gamma_effective = gammas * positive_mask + (1.0 - positive_mask)
        data.batch["advantages"] = advantages * gamma_effective
    else:
        # Raw mode: direct multiplication (original behavior)
        positive_mask = (advantages >= 0).float()
        gamma_effective = gammas * positive_mask + (1.0 - positive_mask)
        data.batch["advantages"] = advantages * gamma_effective

    # Compute stats
    _DRIVER_STEP_COUNTER += 1
    response_mask = data.batch.get("response_mask", None)
    gamma_mean = gammas.mean().item()
    gamma_std = gammas.std().item()
    gamma_max = gammas.max().item()
    gamma_min = gammas.min().item()
    n_gt1 = (gammas > 1.0).sum().item()
    n_total = gammas.numel()
    frac_gt1 = n_gt1 / n_total
    adv_before = advantages.mean().item()
    adv_after = data.batch['advantages'].mean().item()

    # DEBUG: detailed breakdown to diagnose low gamma_mean
    # (1) gamma over non-padding response tokens only
    if response_mask is not None:
        rmask = response_mask.float()
        n_valid = rmask.sum().item()
        gamma_valid_mean = (gammas * rmask).sum().item() / max(n_valid, 1)
        gamma_valid_gt1 = ((gammas > 1.0).float() * rmask).sum().item()
    else:
        n_valid = n_total
        gamma_valid_mean = gamma_mean
        gamma_valid_gt1 = n_gt1

    # (2) gamma over only tokens with positive advantage
    pos_mask = (advantages > 0).float()
    n_pos = pos_mask.sum().item()
    if n_pos > 0:
        gamma_pos_mean = (gammas * pos_mask).sum().item() / n_pos
        gamma_pos_gt1 = ((gammas > 1.0).float() * pos_mask).sum().item()
    else:
        gamma_pos_mean = 1.0
        gamma_pos_gt1 = 0

    # (3) distribution of gamma values
    gamma_eq1 = (gammas == 1.0).sum().item()
    gamma_125 = ((gammas > 1.24) & (gammas < 1.26)).sum().item()  # ~1.25
    gamma_15 = ((gammas > 1.49) & (gammas < 1.51)).sum().item()   # ~1.5

    debug_msg = (
        f"DEBUG step={_DRIVER_STEP_COUNTER} "
        f"gamma_shape={list(gammas.shape)} "
        f"n_total={n_total} n_gt1={n_gt1} n_eq1={gamma_eq1} "
        f"n_~1.25={gamma_125} n_~1.5={gamma_15} | "
        f"valid_tokens={n_valid:.0f} gamma_valid_mean={gamma_valid_mean:.4f} valid_gt1={gamma_valid_gt1:.0f} | "
        f"pos_adv_tokens={n_pos:.0f} gamma_pos_mean={gamma_pos_mean:.4f} pos_gt1={gamma_pos_gt1:.0f} | "
        f"gamma_min={gamma_min:.4f} gamma_max={gamma_max:.4f}"
    )
    _driver_log(debug_msg)
    logger.info(f"[RHYTHM-DEBUG] {debug_msg}")

    # Fraction amplified over valid tokens only
    frac_amplified_valid = gamma_valid_gt1 / max(n_valid, 1)

    # Store metrics for wandb (picked up by ray_trainer via get_rhythm_metrics)
    _RHYTHM_METRICS = {
        "rhythm/gamma_valid_mean": gamma_valid_mean,
        "rhythm/gamma_pos_adv_mean": gamma_pos_mean,
        "rhythm/gamma_max": gamma_max,
        "rhythm/gamma_min": gamma_min,
        "rhythm/frac_amplified_valid": frac_amplified_valid,
        "rhythm/n_amplified_tokens": n_gt1,
        "rhythm/n_valid_tokens": n_valid,
        "rhythm/n_pos_adv_tokens": n_pos,
        "rhythm/adv_mean_before": adv_before,
        "rhythm/adv_mean_after": adv_after,
        "rhythm/run_type": {"A": 0, "B": 1, "C": 2}.get(cfg["run_type"], -1),
        "rhythm/gamma_mode": {"raw": 0, "normalized": 1}.get(gamma_mode, -1),
    }

    # Check for reclassification stats from workers (saved to config file)
    _reload_driver_config()
    reclass = _DRIVER_CONFIG.get("_last_reclassification")
    if reclass and reclass.get("step") == _DRIVER_STEP_COUNTER:
        _RHYTHM_METRICS.update({
            "rhythm/reclass_loc_turnover": reclass["loc_turnover"],
            "rhythm/reclass_glob_turnover": reclass["glob_turnover"],
            "rhythm/reclass_loc_jaccard": reclass["loc_jaccard"],
            "rhythm/reclass_glob_jaccard": reclass["glob_jaccard"],
            "rhythm/reclass_n_loc": reclass["n_loc"],
            "rhythm/reclass_n_glob": reclass["n_glob"],
        })

    log_msg = (
        f"run={cfg['run_type']} step={_DRIVER_STEP_COUNTER} "
        f"gamma_valid_mean={gamma_valid_mean:.4f} gamma_pos_adv_mean={gamma_pos_mean:.4f} "
        f"gamma_max={gamma_max:.4f} frac_amplified_valid={frac_amplified_valid:.3f} "
        f"adv_mean_before={adv_before:.4f} adv_mean_after={adv_after:.4f}"
    )
    _driver_log(log_msg)
    logger.info(f"[RHYTHM] {log_msg}")

    return data


_DRIVER_LOG_FILE = None

def _driver_log(msg):
    global _DRIVER_LOG_FILE
    if _DRIVER_LOG_FILE is None:
        job_id = os.environ.get("SLURM_JOB_ID", "default")
        log_dir = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_sparks_thinking/logs"
        os.makedirs(log_dir, exist_ok=True)
        _DRIVER_LOG_FILE = open(f"{log_dir}/rhythm_driver_{job_id}.log", "a")
    _DRIVER_LOG_FILE.write(f"{msg}\n")
    _DRIVER_LOG_FILE.flush()


# ═══════════════════════════════════════════════════════════════════════
# DRIVER-SIDE: apply_patches (called from train.py before verl starts)
# ═══════════════════════════════════════════════════════════════════════

def apply_patches(
    run_type: str,
    model_name: str,
    H_loc: list[tuple[int, int]] | None = None,
    H_glob: list[tuple[int, int]] | None = None,
    classification_prompts: list[str] | None = None,
    waad_W: int = 10,
    fai_H_lo: int = 10,
    fai_H_hi: int = 50,
    quantile_q: float = 0.4,
    gamma_amp: float = 1.5,
    alpha: float = 0.5,
    neighborhood_k: int = 3,
    reclassify_every_K: int = 20,
    head_quantile: float = 0.3,
    gamma_mode: str = "raw",
    dry_run: bool = False,
):
    """Save rhythm config to disk for Ray workers to load."""
    config = {
        "run_type": run_type,
        "model_name": model_name,
        "H_loc": H_loc,
        "H_glob": H_glob,
        "classification_prompts": classification_prompts,
        "waad_W": waad_W,
        "fai_H_lo": fai_H_lo,
        "fai_H_hi": fai_H_hi,
        "quantile_q": quantile_q,
        "gamma_amp": gamma_amp,
        "alpha": alpha,
        "neighborhood_k": neighborhood_k,
        "reclassify_every_K": reclassify_every_K,
        "head_quantile": head_quantile,
        "gamma_mode": gamma_mode,
        "dry_run": dry_run,
        "step_counter": 0,
    }
    torch.save(config, _CONFIG_PATH)
    logger.info(f"Saved rhythm_trainer config to {_CONFIG_PATH}")
    logger.info(f"Run type: {run_type}, model: {model_name}")
    if H_loc is not None:
        logger.info(f"H_loc: {len(H_loc)} heads, H_glob: {len(H_glob)} heads")


def needs_rhythm_dispatch() -> bool:
    """Check if rhythm gamma dispatch is needed (called from driver fit loop)."""
    try:
        _ensure_driver_config()
        return _DRIVER_CONFIG["run_type"] in ("B", "C")
    except FileNotFoundError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# WORKER-SIDE: called from compute_rhythm_gammas in fsdp_workers.py
# ═══════════════════════════════════════════════════════════════════════

def compute_gammas_on_worker(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    eager_model,
    log_fn=None,
) -> torch.Tensor:
    """Compute rhythm gammas for a batch of sequences on a GPU worker.

    This is called from fsdp_workers.py's compute_rhythm_gammas method,
    which handles eager model loading and weight sync.

    Sequences are batched into mini-batches for efficient forward passes
    (8 sequences per pass instead of 1), sorted by length to minimize
    padding overhead. Output is identical to the unbatched version.

    Args:
        input_ids: (bs, seq_len) input token ids
        attention_mask: (bs, seq_len) attention mask
        response_mask: (bs, seq_len) binary response mask
        eager_model: eager-attention model (already weight-synced)
        log_fn: optional logging function

    Returns:
        gammas: (bs, seq_len) tensor with gamma values
    """
    from attention_sparks_thinking.src.attention_rhythm import (
        compute_fai_vectorized as _compute_fai,
        compute_gamma as _compute_gamma,
        compute_waad as _compute_waad,
    )
    from attention_sparks_thinking.src.head_classifier import (
        aggregate_attention_hooks as _aggregate,
        classify_heads as _classify,
    )

    # Load config
    cfg = torch.load(_CONFIG_PATH, map_location="cpu", weights_only=False)
    H_loc = cfg["H_loc"]
    H_glob = cfg["H_glob"]

    bs, full_seq_len = input_ids.shape
    resp_len = response_mask.shape[1]
    # Gammas match response_mask shape (bs, resp_len), NOT full seq_len.
    # response_mask is attention_mask[:, -resp_len:] so indices 0..resp_len-1
    # correspond to the LAST resp_len tokens of input_ids.
    gammas = torch.ones(bs, resp_len, device=input_ids.device)
    device = next(eager_model.parameters()).device

    # Response starts at (full_seq_len - resp_len) in the full sequence
    prompt_len_default = full_seq_len - resp_len

    # ── Phase 1: Pre-compute valid samples and metadata ──
    samples = []
    for i in range(bs):
        resp_mask_i = response_mask[i]
        n_resp_tokens = int(resp_mask_i.sum().item())
        if n_resp_tokens == 0:
            continue

        # veRL layout: [LEFT_PAD | prompt_tokens | response_tokens | RIGHT_PAD]
        nonzero_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(nonzero_positions) == 0:
            continue
        first_real = nonzero_positions[0].item()
        last_real = nonzero_positions[-1].item()

        response_start = prompt_len_default - first_real
        actual_seq_len = last_real - first_real + 1

        if response_start < 0 or response_start >= actual_seq_len:
            if log_fn:
                log_fn(f"Sample {i}: invalid response_start={response_start} "
                       f"(first_real={first_real}, actual_seq_len={actual_seq_len})")
            continue

        # Diagnostic for first sample
        if i == 0 and log_fn:
            log_fn(f"DIAG sample 0: full_seq_len={full_seq_len} resp_len={resp_len} "
                   f"first_real={first_real} last_real={last_real} "
                   f"actual_seq_len={actual_seq_len} response_start={response_start} "
                   f"n_resp_tokens={n_resp_tokens}")

        samples.append({
            "batch_idx": i,
            "ids": input_ids[i, first_real:last_real + 1],
            "seq_len": actual_seq_len,
            "response_start": response_start,
            "n_resp_tokens": n_resp_tokens,
            "resp_mask": resp_mask_i,
        })

    if not samples:
        return gammas

    # ── Phase 2: Sort by length and process in mini-batches ──
    samples.sort(key=lambda s: s["seq_len"])
    MINI_BATCH = 8  # sequences per forward pass
    all_stats = []

    with torch.no_grad():
        for mb_start in range(0, len(samples), MINI_BATCH):
            mb_samples = samples[mb_start:mb_start + MINI_BATCH]
            mb_size = len(mb_samples)
            max_len = max(s["seq_len"] for s in mb_samples)

            # Right-pad to max length in this mini-batch
            mb_ids = torch.zeros(mb_size, max_len, dtype=input_ids.dtype, device=device)
            mb_mask = torch.zeros(mb_size, max_len, dtype=attention_mask.dtype, device=device)
            for j, s in enumerate(mb_samples):
                L = s["seq_len"]
                mb_ids[j, :L] = s["ids"].to(device)
                mb_mask[j, :L] = 1

            # Batched forward pass with hooks — returns (mb_size, max_len, max_len)
            try:
                A_bar_loc_batch, A_bar_glob_batch = _aggregate(
                    eager_model, mb_ids, mb_mask, H_loc, H_glob
                )
            except Exception as e:
                if log_fn:
                    log_fn(f"Batched aggregate failed for mb_start={mb_start}: {e}")
                # Fallback: process individually
                for s in mb_samples:
                    try:
                        ids_i = s["ids"].unsqueeze(0).to(device)
                        mask_i = torch.ones_like(ids_i, device=device)
                        A_loc, A_glob = _aggregate(eager_model, ids_i, mask_i, H_loc, H_glob)
                        waad = _compute_waad(A_loc, s["response_start"], W=cfg["waad_W"])
                        fai = _compute_fai(A_glob, s["response_start"], H_lo=cfg["fai_H_lo"], H_hi=cfg["fai_H_hi"])
                        gamma, stats = _compute_gamma(waad, fai, q=cfg["quantile_q"],
                                                      gamma_amp=cfg["gamma_amp"], alpha=cfg["alpha"], k=cfg["neighborhood_k"])
                        n_use = min(s["n_resp_tokens"], len(gamma))
                        if n_use > 0:
                            resp_positions = s["resp_mask"].nonzero(as_tuple=True)[0]
                            gammas[s["batch_idx"], resp_positions[:n_use]] = gamma[:n_use].to(gammas.device)
                        all_stats.append(stats)
                    except Exception as e2:
                        if log_fn:
                            log_fn(f"Gamma failed for sample {s['batch_idx']}: {e2}")
                continue

            # Process each sample's attention maps
            for j, s in enumerate(mb_samples):
                try:
                    L = s["seq_len"]
                    # Trim padded attention back to real sequence length
                    A_loc = A_bar_loc_batch[j, :L, :L]
                    A_glob = A_bar_glob_batch[j, :L, :L]

                    waad = _compute_waad(A_loc, s["response_start"], W=cfg["waad_W"])
                    fai = _compute_fai(A_glob, s["response_start"], H_lo=cfg["fai_H_lo"], H_hi=cfg["fai_H_hi"])
                    gamma, stats = _compute_gamma(
                        waad, fai,
                        q=cfg["quantile_q"],
                        gamma_amp=cfg["gamma_amp"],
                        alpha=cfg["alpha"],
                        k=cfg["neighborhood_k"],
                    )

                    n_gamma = len(gamma)
                    n_use = min(s["n_resp_tokens"], n_gamma)
                    if n_use > 0:
                        resp_positions = s["resp_mask"].nonzero(as_tuple=True)[0]
                        gammas[s["batch_idx"], resp_positions[:n_use]] = gamma[:n_use].to(gammas.device)

                    all_stats.append(stats)

                except Exception as e:
                    if log_fn:
                        log_fn(f"Gamma failed for sample {s['batch_idx']}: {e}")

    # Log aggregate stats
    if log_fn and all_stats:
        avg_frac = sum(s.get("frac_amplified", 0) for s in all_stats) / len(all_stats)
        avg_anchors = sum(s.get("n_regular_anchor", 0) for s in all_stats) / len(all_stats)
        avg_dominated = sum(s.get("n_dominated_anchor", 0) for s in all_stats) / len(all_stats)
        log_fn(
            f"Batch gamma stats: bs={bs}, n_valid={len(all_stats)}, "
            f"n_mini_batches={(len(samples) + MINI_BATCH - 1) // MINI_BATCH}, "
            f"frac_amplified={avg_frac:.3f}, "
            f"avg_anchors={avg_anchors:.1f}, avg_dominated={avg_dominated:.1f}, "
            f"gamma_mean={gammas.mean():.4f}, gamma_max={gammas.max():.4f}"
        )

    return gammas


def reclassify_heads_on_worker(eager_model, tokenizer, step, log_fn=None):
    """Re-run head classification on the worker (Run C only).

    Following the paper: generate responses with T=0.7 using the current
    model weights, then classify heads on full prompt+response text.

    Updates H_loc/H_glob in the config file and saves turnover stats
    so the driver can log them to wandb.
    """
    from attention_sparks_thinking.src.head_classifier import classify_heads as _classify

    cfg = torch.load(_CONFIG_PATH, map_location="cpu", weights_only=False)
    prompts = cfg.get("classification_prompts")
    if not prompts:
        if log_fn:
            log_fn("No classification prompts for reclassification")
        return

    # Generate responses with current model weights (paper's method)
    if log_fn:
        log_fn(f"Generating {len(prompts)} responses for reclassification...")
    device = next(eager_model.parameters()).device
    full_texts = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = eager_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=1.0,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_texts.append(full_text)

    if log_fn:
        log_fn(f"Generated {len(full_texts)} responses. Classifying heads...")

    old_loc = set(tuple(h) for h in cfg["H_loc"])
    old_glob = set(tuple(h) for h in cfg["H_glob"])

    H_loc, H_glob, _ = _classify(
        eager_model, tokenizer, full_texts, head_quantile=cfg["head_quantile"]
    )

    new_loc = set(tuple(h) for h in H_loc)
    new_glob = set(tuple(h) for h in H_glob)

    # Compute turnover: fraction of heads that changed
    loc_intersection = old_loc & new_loc
    glob_intersection = old_glob & new_glob
    loc_union = old_loc | new_loc
    glob_union = old_glob | new_glob
    loc_turnover = 1.0 - len(loc_intersection) / max(len(loc_union), 1)
    glob_turnover = 1.0 - len(glob_intersection) / max(len(glob_union), 1)
    loc_jaccard = len(loc_intersection) / max(len(loc_union), 1)
    glob_jaccard = len(glob_intersection) / max(len(glob_union), 1)

    cfg["H_loc"] = H_loc
    cfg["H_glob"] = H_glob
    # Save reclassification stats for driver to pick up
    cfg["_last_reclassification"] = {
        "step": step,
        "loc_turnover": loc_turnover,
        "glob_turnover": glob_turnover,
        "loc_jaccard": loc_jaccard,
        "glob_jaccard": glob_jaccard,
        "n_loc": len(H_loc),
        "n_glob": len(H_glob),
        "loc_added": [list(h) for h in new_loc - old_loc],
        "loc_removed": [list(h) for h in old_loc - new_loc],
        "glob_added": [list(h) for h in new_glob - old_glob],
        "glob_removed": [list(h) for h in old_glob - new_glob],
    }
    torch.save(cfg, _CONFIG_PATH)

    if log_fn:
        log_fn(
            f"Reclassified: {len(H_loc)} local, {len(H_glob)} global heads | "
            f"loc_turnover={loc_turnover:.3f} glob_turnover={glob_turnover:.3f} | "
            f"loc_jaccard={loc_jaccard:.3f} glob_jaccard={glob_jaccard:.3f}"
        )
