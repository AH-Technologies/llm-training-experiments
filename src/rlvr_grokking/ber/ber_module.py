"""Bidirectional Experience Replay (BER) core logic.

BER enhances 1-shot RLVR training by injecting cached responses into
homogeneous rollout groups:
- Phase 1 (all-incorrect): inject a cached correct response → positive signal
- Phase 2 (mixed): normal GRPO + cache the latest incorrect rollout
- Phase 3 (all-correct): inject a cached incorrect response → negative signal

v2: Ring buffer for error cache + injection fraction to control signal strength.
"""

import random
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BERCache:
    """Holds cached correct response and a ring buffer of incorrect responses."""
    correct_cache: Optional[dict] = None  # {"response_tokens": 1D tensor, "reward": 1.0}
    error_buffer: list = field(default_factory=list)  # FIFO ring buffer of error dicts
    buffer_size: int = 32

    @classmethod
    def from_disk(cls, correct_cache_path: Optional[str] = None, buffer_size: int = 32) -> "BERCache":
        """Load pre-generated correct cache from disk. Error buffer starts empty."""
        cache = cls(buffer_size=buffer_size)
        if correct_cache_path is not None:
            data = torch.load(correct_cache_path, map_location="cpu", weights_only=True)
            cache.correct_cache = {
                "response_tokens": data["response_tokens"],  # 1D tensor of token IDs
                "reward": 1.0,
            }
        return cache

    def add_error(self, error_dict: dict):
        """Add error to ring buffer, evicting oldest if full."""
        self.error_buffer.append(error_dict)
        if len(self.error_buffer) > self.buffer_size:
            self.error_buffer.pop(0)

    def evict_stale(self, global_step: int, max_age: int):
        """Remove entries older than max_age steps."""
        self.error_buffer = [
            e for e in self.error_buffer
            if (global_step - e["step_cached"]) <= max_age
        ]

    def sample_error(self) -> Optional[dict]:
        """Sample a random error from the buffer."""
        if not self.error_buffer:
            return None
        return random.choice(self.error_buffer)


def classify_and_inject(
    batch,
    reward_tensor: torch.Tensor,
    n_rollouts: int,
    ber_cache: "BERCache",
    global_step: int,
    pad_token_id: int,
    max_error_cache_age: int = 500,
    injection_fraction: float = 0.1,
    stop_grad_injected: bool = False,
) -> tuple:
    """Classify rollout groups and inject cached responses.

    Args:
        batch: DataProto with batch tensors (prompts, responses, input_ids,
               attention_mask, position_ids).
        reward_tensor: [B*n, response_len] token-level reward scores.
                       Score is placed at last valid token position.
        n_rollouts: Number of rollouts per prompt group (n=8 typically).
        ber_cache: BERCache with correct_cache and error_buffer.
        global_step: Current training step.
        pad_token_id: Tokenizer pad token ID.
        max_error_cache_age: Max steps before error cache entries are evicted.
        injection_fraction: Fraction of Phase 3 groups to inject into (0.0-1.0).
        stop_grad_injected: If True, zero response_mask for injected slots so
                            they affect advantage computation but produce no
                            gradient (Strategy 5 / AR3PO Option II).

    Returns:
        (batch, reward_tensor, metrics_dict, injected_indices)
    """
    # Per-rollout scores: sum token-level rewards to get scalar per rollout
    scores = reward_tensor.sum(dim=-1)  # [B*n]
    total_rollouts = scores.shape[0]
    n_groups = total_rollouts // n_rollouts
    scores_grouped = scores.view(n_groups, n_rollouts)  # [B, n]

    # Evict stale entries from the error buffer
    ber_cache.evict_stale(global_step, max_error_cache_age)

    # Compute buffer age stats
    if ber_cache.error_buffer:
        oldest_age = global_step - ber_cache.error_buffer[0]["step_cached"]
        newest_age = global_step - ber_cache.error_buffer[-1]["step_cached"]
    else:
        oldest_age = 0
        newest_age = 0

    metrics = {
        "ber/phase1_groups": 0,
        "ber/phase2_groups": 0,
        "ber/phase3_groups": 0,
        "ber/injected_positive": 0,
        "ber/injected_negative": 0,
        "ber/buffer_fill": len(ber_cache.error_buffer),
        "ber/buffer_oldest_age": oldest_age,
        "ber/buffer_newest_age": newest_age,
    }

    response_len = batch.batch["responses"].shape[1]
    injected_indices = []  # Track which slots were injected (for stop_grad)

    for i in range(n_groups):
        n_correct = (scores_grouped[i] > 0.5).sum().item()

        if n_correct == 0:
            # Phase 1: all-incorrect group
            metrics["ber/phase1_groups"] += 1
            if ber_cache.correct_cache is not None:
                replace_idx = i * n_rollouts + (n_rollouts - 1)
                _replace_response(batch, replace_idx, ber_cache.correct_cache, pad_token_id, response_len)
                _set_reward(reward_tensor, replace_idx, 1.0, batch, response_len)
                metrics["ber/injected_positive"] += 1
                injected_indices.append(replace_idx)

        elif n_correct == n_rollouts:
            # Phase 3: all-correct group — inject with probability injection_fraction
            metrics["ber/phase3_groups"] += 1
            if ber_cache.error_buffer and random.random() < injection_fraction:
                error = ber_cache.sample_error()
                replace_idx = i * n_rollouts + (n_rollouts - 1)
                _replace_response(batch, replace_idx, error, pad_token_id, response_len)
                _set_reward(reward_tensor, replace_idx, 0.0, batch, response_len)
                metrics["ber/injected_negative"] += 1
                injected_indices.append(replace_idx)

        else:
            # Phase 2: mixed group — add last wrong rollout to buffer
            metrics["ber/phase2_groups"] += 1
            wrong_mask = scores_grouped[i] <= 0.5
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
            if len(wrong_indices) > 0:
                cache_local_idx = wrong_indices[-1].item()
                cache_global_idx = i * n_rollouts + cache_local_idx
                ber_cache.add_error(_extract_response(batch, cache_global_idx, global_step))

    # Recompute response_mask after modifications
    batch.batch["response_mask"] = _compute_response_mask(batch)

    # Strategy 5: zero response_mask for injected slots so their reward
    # affects advantage computation but no gradient flows through them.
    # The on-policy samples get gradient with advantages shaped by the
    # injected reward, but the injected response itself is not learned from.
    if stop_grad_injected and injected_indices:
        for idx in injected_indices:
            batch.batch["response_mask"][idx] = 0

    return batch, reward_tensor, metrics, injected_indices


def classify_and_inject_within_step(
    batch,
    reward_tensor: torch.Tensor,
    n_rollouts: int,
    ber_cache: "BERCache",
    global_step: int,
    pad_token_id: int,
    injection_fraction: float = 0.1,
    stop_grad_injected: bool = False,
) -> tuple:
    """Within-step amplification: inject current-batch errors into all-correct groups.

    Instead of replaying errors from previous steps, this variant collects
    wrong samples from the current batch's mixed (phase 2) groups and injects
    them into a fraction of all-correct (phase 3) groups. No cross-step
    error buffer is used.

    Two-pass approach:
      Pass 1: classify groups, collect wrong samples from phase 2, inject
              correct cache into phase 1 groups.
      Pass 2: inject collected same-step errors into phase 3 groups.

    Args:
        batch: DataProto with batch tensors.
        reward_tensor: [B*n, response_len] token-level reward scores.
        n_rollouts: Number of rollouts per prompt group.
        ber_cache: BERCache (only correct_cache is used; error_buffer ignored).
        global_step: Current training step (for metrics only).
        pad_token_id: Tokenizer pad token ID.
        injection_fraction: Fraction of phase 3 groups to inject into.
        stop_grad_injected: If True, zero response_mask for injected slots.

    Returns:
        (batch, reward_tensor, metrics_dict, injected_indices)
    """
    scores = reward_tensor.sum(dim=-1)  # [B*n]
    total_rollouts = scores.shape[0]
    n_groups = total_rollouts // n_rollouts
    scores_grouped = scores.view(n_groups, n_rollouts)

    response_len = batch.batch["responses"].shape[1]
    injected_indices = []

    # Classify groups and collect current-step errors
    phase1_groups = []
    phase2_groups = []
    phase3_groups = []
    current_step_errors = []

    for i in range(n_groups):
        n_correct = (scores_grouped[i] > 0.5).sum().item()

        if n_correct == 0:
            phase1_groups.append(i)
        elif n_correct == n_rollouts:
            phase3_groups.append(i)
        else:
            phase2_groups.append(i)
            # Collect wrong samples from this mixed group
            wrong_mask = scores_grouped[i] <= 0.5
            wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]
            for wi in wrong_indices:
                global_idx = i * n_rollouts + wi.item()
                current_step_errors.append(
                    _extract_response(batch, global_idx, global_step)
                )

    metrics = {
        "ber/phase1_groups": len(phase1_groups),
        "ber/phase2_groups": len(phase2_groups),
        "ber/phase3_groups": len(phase3_groups),
        "ber/injected_positive": 0,
        "ber/injected_negative": 0,
        "ber/current_step_errors_found": len(current_step_errors),
    }

    # Pass 1: inject correct cache into phase 1 (all-incorrect) groups
    for i in phase1_groups:
        if ber_cache.correct_cache is not None:
            replace_idx = i * n_rollouts + (n_rollouts - 1)
            _replace_response(batch, replace_idx, ber_cache.correct_cache, pad_token_id, response_len)
            _set_reward(reward_tensor, replace_idx, 1.0, batch, response_len)
            metrics["ber/injected_positive"] += 1
            injected_indices.append(replace_idx)

    # Pass 2: inject current-step errors into a fraction of phase 3 groups
    if current_step_errors:
        for i in phase3_groups:
            if random.random() < injection_fraction:
                error = random.choice(current_step_errors)
                replace_idx = i * n_rollouts + (n_rollouts - 1)
                _replace_response(batch, replace_idx, error, pad_token_id, response_len)
                _set_reward(reward_tensor, replace_idx, 0.0, batch, response_len)
                metrics["ber/injected_negative"] += 1
                injected_indices.append(replace_idx)

    # Recompute response_mask after modifications
    batch.batch["response_mask"] = _compute_response_mask(batch)

    if stop_grad_injected and injected_indices:
        for idx in injected_indices:
            batch.batch["response_mask"][idx] = 0

    return batch, reward_tensor, metrics, injected_indices


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
