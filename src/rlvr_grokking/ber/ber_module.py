"""Bidirectional Experience Replay (BER) core logic.

BER enhances 1-shot RLVR training by injecting cached responses into
homogeneous rollout groups:
- Phase 1 (all-incorrect): inject a cached correct response → positive signal
- Phase 2 (mixed): normal GRPO + cache the latest incorrect rollout
- Phase 3 (all-correct): inject a cached incorrect response → negative signal
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class BERCache:
    """Holds cached correct and incorrect responses for BER injection."""
    correct_cache: Optional[dict] = None  # {"response_tokens": 1D tensor, "reward": 1.0}
    error_cache: Optional[dict] = None    # {"response_tokens": 1D tensor, "reward": 0.0, "step_cached": int}

    @classmethod
    def from_disk(cls, correct_cache_path: Optional[str] = None) -> "BERCache":
        """Load pre-generated correct cache from disk. Error cache starts empty."""
        cache = cls()
        if correct_cache_path is not None:
            data = torch.load(correct_cache_path, map_location="cpu", weights_only=True)
            cache.correct_cache = {
                "response_tokens": data["response_tokens"],  # 1D tensor of token IDs
                "reward": 1.0,
            }
        return cache


def classify_and_inject(
    batch,
    reward_tensor: torch.Tensor,
    n_rollouts: int,
    correct_cache: Optional[dict],
    error_cache: Optional[dict],
    global_step: int,
    pad_token_id: int,
    max_error_cache_age: int = 500,
) -> tuple:
    """Classify rollout groups and inject cached responses.

    Args:
        batch: DataProto with batch tensors (prompts, responses, input_ids,
               attention_mask, position_ids).
        reward_tensor: [B*n, response_len] token-level reward scores.
                       Score is placed at last valid token position.
        n_rollouts: Number of rollouts per prompt group (n=8 typically).
        correct_cache: Dict with "response_tokens" (1D) and "reward" (float), or None.
        error_cache: Dict with "response_tokens" (1D), "reward" (0.0),
                     "step_cached" (int), or None.
        global_step: Current training step.
        pad_token_id: Tokenizer pad token ID.
        max_error_cache_age: Max steps before error cache is too stale.

    Returns:
        (batch, reward_tensor, error_cache, metrics_dict)
    """
    # Per-rollout scores: sum token-level rewards to get scalar per rollout
    scores = reward_tensor.sum(dim=-1)  # [B*n]
    total_rollouts = scores.shape[0]
    n_groups = total_rollouts // n_rollouts
    scores_grouped = scores.view(n_groups, n_rollouts)  # [B, n]

    # Check if error cache is too stale
    if error_cache is not None and (global_step - error_cache["step_cached"]) > max_error_cache_age:
        error_cache = None

    metrics = {
        "ber/phase1_groups": 0,
        "ber/phase2_groups": 0,
        "ber/phase3_groups": 0,
        "ber/injected_positive": 0,
        "ber/injected_negative": 0,
        "ber/error_cache_age": 0 if error_cache is None else (global_step - error_cache["step_cached"]),
        "ber/error_cache_available": 1.0 if error_cache is not None else 0.0,
    }

    response_len = batch.batch["responses"].shape[1]

    for i in range(n_groups):
        n_correct = (scores_grouped[i] > 0.5).sum().item()

        if n_correct == 0:
            # Phase 1: all-incorrect group
            metrics["ber/phase1_groups"] += 1
            if correct_cache is not None:
                replace_idx = i * n_rollouts + (n_rollouts - 1)
                _replace_response(batch, replace_idx, correct_cache, pad_token_id, response_len)
                _set_reward(reward_tensor, replace_idx, 1.0, batch, response_len)
                metrics["ber/injected_positive"] += 1

        elif n_correct == n_rollouts:
            # Phase 3: all-correct group
            metrics["ber/phase3_groups"] += 1
            if error_cache is not None:
                replace_idx = i * n_rollouts + (n_rollouts - 1)
                _replace_response(batch, replace_idx, error_cache, pad_token_id, response_len)
                _set_reward(reward_tensor, replace_idx, 0.0, batch, response_len)
                metrics["ber/injected_negative"] += 1

        else:
            # Phase 2: mixed group — cache last wrong rollout
            metrics["ber/phase2_groups"] += 1
            wrong_mask = scores_grouped[i] <= 0.5
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
            if len(wrong_indices) > 0:
                cache_local_idx = wrong_indices[-1].item()
                cache_global_idx = i * n_rollouts + cache_local_idx
                error_cache = _extract_response(batch, cache_global_idx, global_step)

    # Recompute response_mask after modifications
    batch.batch["response_mask"] = _compute_response_mask(batch)

    return batch, reward_tensor, error_cache, metrics


def _replace_response(batch, idx: int, cache: dict, pad_token_id: int, response_len: int):
    """Replace a single rollout's response in the batch with cached tokens."""
    cached_tokens = cache["response_tokens"]
    device = batch.batch["responses"].device

    # Truncate or pad cached tokens to match response_len
    cached_tokens = cached_tokens.to(device)
    if cached_tokens.shape[0] >= response_len:
        padded = cached_tokens[:response_len]
    else:
        padding = torch.full(
            (response_len - cached_tokens.shape[0],),
            pad_token_id,
            dtype=cached_tokens.dtype,
            device=device,
        )
        padded = torch.cat([cached_tokens, padding])

    # 1. Replace response tokens
    batch.batch["responses"][idx] = padded

    # 2. Reconstruct input_ids = prompt + new response
    prompt_len = batch.batch["prompts"].shape[1]
    prompt = batch.batch["prompts"][idx]
    batch.batch["input_ids"][idx] = torch.cat([prompt, padded])

    # 3. Update attention_mask
    # Count real tokens in prompt (non-padding, prompt is left-padded)
    prompt_real_len = (batch.batch["attention_mask"][idx, :prompt_len] != 0).sum().item()
    response_real_len = (padded != pad_token_id).sum().item()

    attn_mask = torch.zeros(prompt_len + response_len, dtype=batch.batch["attention_mask"].dtype, device=device)
    # Left-padded prompt: real tokens at the end
    if prompt_real_len > 0:
        attn_mask[prompt_len - prompt_real_len:prompt_len] = 1
    # Response: real tokens at the start (right-padded)
    if response_real_len > 0:
        attn_mask[prompt_len:prompt_len + response_real_len] = 1
    batch.batch["attention_mask"][idx] = attn_mask

    # 4. Update position_ids: cumulative sum of attention_mask - 1, clamped to 0
    cumsum = attn_mask.cumsum(dim=0)
    pos_ids = (cumsum - 1).clamp(min=0)
    batch.batch["position_ids"][idx] = pos_ids


def _extract_response(batch, idx: int, global_step: int) -> dict:
    """Extract and clone response tokens from a batch slot."""
    response_tokens = batch.batch["responses"][idx].clone().cpu()
    return {
        "response_tokens": response_tokens,
        "reward": 0.0,
        "step_cached": global_step,
    }


def _set_reward(reward_tensor: torch.Tensor, idx: int, reward_value: float, batch, response_len: int):
    """Set the reward for a replaced rollout.

    The reward tensor has scores placed at the last valid response token position.
    We zero out the row and place the new reward at the correct position.
    """
    prompt_len = batch.batch["prompts"].shape[1]
    # Count valid response tokens from attention mask
    valid_response_len = batch.batch["attention_mask"][idx, prompt_len:].sum().item()
    valid_response_len = max(int(valid_response_len), 1)  # at least 1

    reward_tensor[idx] = 0.0
    reward_tensor[idx, valid_response_len - 1] = reward_value


def _compute_response_mask(batch) -> torch.Tensor:
    """Recompute response_mask from attention_mask (last response_len positions)."""
    response_length = batch.batch["responses"].shape[1]
    attention_mask = batch.batch["attention_mask"]
    return attention_mask[:, -response_length:]
