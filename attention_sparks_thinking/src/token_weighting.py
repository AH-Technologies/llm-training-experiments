"""EAP-IG-based per-token weighting functions for circuit-guided GRPO.

Five conditions that use EAP-IG-discovered reasoning heads:
  D. attention:       weight by reasoning head attention received
  E. fai:             Future Attention Influence on reasoning heads
  F. fai_allheads:    FAI on all heads equally (control)
  G. fai_asymmetric:  FAI for correct, inverted FAI for incorrect
  H. anchor_credit:   discrete anchor/dependent weights

All functions return weights of shape (batch_size, response_length) with mean=1
per sequence (so the overall advantage magnitude is preserved).

Ported from attention_based_rewards/scripts/token_weighting.py with Qwen3-4B
defaults (36 layers, 32 heads).

No verl dependency — pure PyTorch.
"""

import torch
from torch import Tensor


def _normalize_per_sequence(weights: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize weights to mean=1 per sequence, respecting the response mask."""
    masked_weights = weights * mask
    seq_means = masked_weights.sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
    normalized = masked_weights / (seq_means + eps)
    return normalized * mask


def compute_attention_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
) -> Tensor:
    """Method D: weight by reasoning head attention received.

    Runs a forward pass with output_attentions=True and computes how much each
    response token is *attended to* by reasoning heads.
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device

    with torch.no_grad():
        outputs = attn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    attentions = outputs.attentions
    token_importance = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    for layer, head in reasoning_heads:
        attn_pattern = attentions[layer][:, head, :, :]  # (bs, seq_len, seq_len)
        received = attn_pattern.sum(dim=1)  # (bs, seq_len)
        importance = head_scores[layer, head].item()
        token_importance += received.float() * importance

    del attentions, outputs

    resp_len = response_mask.shape[1]
    token_importance = token_importance[:, -resp_len:]
    return _normalize_per_sequence(token_importance, response_mask)


def _compute_fai_with_hooks(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    heads_by_layer: dict[int, list[tuple[int, float]]],
) -> Tensor:
    """Core FAI computation using forward hooks for memory efficiency.

    Registers hooks on attention modules to capture and immediately process
    attention weights during the forward pass. Each layer's attention is
    processed and freed before the next layer runs.
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    token_importance = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)

    handles = []

    def _make_hook(layer_idx, head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, weight in head_list:
                    attn_pattern = attn_weights[:, head_idx, :, :]
                    future_attn = attn_pattern * future_mask.unsqueeze(0)
                    received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
                    token_importance.add_(received.float() * weight)
                return (output[0], None) + output[2:]
        return hook_fn

    for layer_idx, head_list in heads_by_layer.items():
        attn_module = attn_model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(_make_hook(layer_idx, head_list))
        handles.append(handle)

    try:
        with torch.no_grad():
            attn_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    del future_mask, future_count
    gc.collect()

    token_importance = token_importance[:, -resp_len:]
    return _normalize_per_sequence(token_importance, response_mask)


def compute_fai_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
) -> Tensor:
    """Method E: Future Attention Influence restricted to reasoning heads."""
    heads_by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, head in reasoning_heads:
        weight = head_scores[layer, head].item()
        heads_by_layer.setdefault(layer, []).append((head, weight))

    return _compute_fai_with_hooks(
        input_ids, attention_mask, response_mask, attn_model, heads_by_layer
    )


def compute_fai_weights_allheads(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    n_layers: int = 36,
    n_heads: int = 32,
) -> Tensor:
    """Method F: FAI across ALL attention heads with equal weight (control).

    Defaults to Qwen3-4B architecture (36 layers, 32 heads = 1152 total).
    """
    n_total = n_layers * n_heads
    weight_per_head = 1.0 / n_total

    heads_by_layer = {
        layer: [(head, weight_per_head) for head in range(n_heads)]
        for layer in range(n_layers)
    }

    return _compute_fai_with_hooks(
        input_ids, attention_mask, response_mask, attn_model, heads_by_layer
    )


def compute_asymmetric_weights(
    fai_weights: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
) -> Tensor:
    """Method G: amplify reasoning on correct, protect on incorrect.

    For correct responses (advantage > 0): use FAI weights as-is.
    For incorrect responses (advantage <= 0): invert weights so reasoning
    tokens get lower penalty.

    Called on the driver side where advantages are available.
    """
    is_correct = (advantages[:, 0] > 0).unsqueeze(1)  # (bs, 1)

    correct_weights = fai_weights

    w_max = (fai_weights * response_mask).max(dim=-1, keepdim=True).values
    inverted = (w_max - fai_weights + 1e-6) * response_mask
    inverted_weights = _normalize_per_sequence(inverted, response_mask)

    weights = torch.where(is_correct, correct_weights, inverted_weights)
    return weights


def _discretize_weights(
    weights: Tensor,
    mask: Tensor,
    top_fraction: float = 0.2,
    boost: float = 1.5,
) -> Tensor:
    """Convert smooth weights to discrete: top_fraction of tokens get `boost`, rest get 1.0.

    Per-sequence: selects top `top_fraction` tokens by weight value and assigns
    them the boost factor. All other tokens get weight 1.0.
    """
    bs, resp_len = weights.shape
    device = weights.device
    discrete = torch.ones(bs, resp_len, device=device, dtype=torch.float32)

    for b in range(bs):
        valid = mask[b].bool()
        n_valid = valid.sum().item()
        if n_valid < 2:
            continue
        n_top = max(1, int(n_valid * top_fraction))
        valid_weights = weights[b][valid]
        _, top_idx = valid_weights.topk(n_top)
        # Map back to full sequence positions
        valid_positions = valid.nonzero(as_tuple=True)[0]
        discrete[b, valid_positions[top_idx]] = boost

    return discrete * mask.float()


def compute_fai_discrete_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
    top_fraction: float = 0.2,
    boost: float = 1.5,
) -> Tensor:
    """Method I: Discrete FAI on reasoning heads.

    Computes smooth FAI, then selects top `top_fraction` tokens and assigns
    them `boost` weight. Rest get 1.0.
    """
    smooth = compute_fai_weights(
        input_ids, attention_mask, response_mask, attn_model,
        reasoning_heads, head_scores,
    )
    return _discretize_weights(smooth, response_mask, top_fraction, boost)


def compute_fai_discrete_allheads_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    n_layers: int = 36,
    n_heads: int = 32,
    top_fraction: float = 0.2,
    boost: float = 1.5,
) -> Tensor:
    """Method J: Discrete FAI across ALL heads.

    Computes smooth FAI over all heads, then selects top `top_fraction` tokens
    and assigns them `boost` weight. Rest get 1.0.
    """
    smooth = compute_fai_weights_allheads(
        input_ids, attention_mask, response_mask, attn_model,
        n_layers, n_heads,
    )
    return _discretize_weights(smooth, response_mask, top_fraction, boost)


def compute_anchor_credit_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
    anchor_percentile: float = 90.0,
    dependent_percentile: float = 75.0,
    anchor_boost: float = 2.0,
    dependent_boost: float = 1.5,
    max_weight: float = 3.0,
) -> Tensor:
    """Method H: Anchor-based circuit credit assignment.

    Two-pass computation using hooks:
      Pass 1: FAI on reasoning heads to identify anchor tokens.
      Pass 2: Dependency scores — how much non-anchors attend to anchors.

    Returns DISCRETE weights: anchor_boost for anchors, dependent_boost for
    dependents, 1.0 for rest. No token ever gets weight < 1.0.
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]
    prompt_len = seq_len - resp_len

    heads_by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, head in reasoning_heads:
        weight = head_scores[layer, head].item()
        heads_by_layer.setdefault(layer, []).append((head, weight))

    # ── Pass 1: Compute raw FAI scores ──
    fai_scores_full = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)

    handles = []

    def _make_fai_hook(head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, w in head_list:
                    attn_pattern = attn_weights[:, head_idx, :, :]
                    future_attn = attn_pattern * future_mask.unsqueeze(0)
                    received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
                    fai_scores_full.add_(received.float() * w)
                return (output[0], None) + output[2:]
        return hook_fn

    for layer_idx, head_list in heads_by_layer.items():
        attn_module = attn_model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(_make_fai_hook(head_list))
        handles.append(handle)

    with torch.no_grad():
        attn_model(input_ids=input_ids, attention_mask=attention_mask,
                   output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()
    handles.clear()
    del future_mask, future_count
    gc.collect()

    fai_scores = fai_scores_full[:, -resp_len:]
    del fai_scores_full

    # ── Identify anchors ──
    resp_mask_bool = response_mask.bool()
    anchor_mask = torch.zeros(bs, resp_len, device=device, dtype=torch.bool)

    for b in range(bs):
        valid_fai = fai_scores[b][resp_mask_bool[b]]
        if valid_fai.numel() < 2:
            continue
        threshold = torch.quantile(valid_fai.float(), anchor_percentile / 100.0)
        anchor_mask[b] = (fai_scores[b] >= threshold) & resp_mask_bool[b]

    # ── Pass 2: Dependency scores via reasoning-head attention to anchors ──
    max_n_anchors = int(max(anchor_mask.sum(dim=-1).max().item(), 1))

    anchor_indices_full = torch.zeros(bs, max_n_anchors, device=device, dtype=torch.long)
    anchor_valid = torch.zeros(bs, max_n_anchors, device=device, dtype=torch.bool)

    for b in range(bs):
        resp_idx = anchor_mask[b].nonzero(as_tuple=True)[0]
        n = resp_idx.shape[0]
        if n > 0:
            anchor_indices_full[b, :n] = resp_idx + prompt_len
            anchor_valid[b, :n] = True

    dep_scores = torch.zeros(bs, seq_len, max_n_anchors, device=device, dtype=torch.float32)
    expanded_idx = anchor_indices_full.unsqueeze(1).expand(-1, seq_len, -1)

    def _make_dep_hook(head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, w in head_list:
                    attn = attn_weights[:, head_idx, :, :]
                    anchor_attn = attn.gather(dim=-1, index=expanded_idx)
                    dep_scores.add_(anchor_attn.float() * w)
                return (output[0], None) + output[2:]
        return hook_fn

    for layer_idx, head_list in heads_by_layer.items():
        attn_module = attn_model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(_make_dep_hook(head_list))
        handles.append(handle)

    with torch.no_grad():
        attn_model(input_ids=input_ids, attention_mask=attention_mask,
                   output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()
    del handles, expanded_idx
    gc.collect()

    dep_scores.masked_fill_(~anchor_valid.unsqueeze(1), float("-inf"))
    dependency_full = dep_scores.max(dim=-1).values
    dependency_full[dependency_full == float("-inf")] = 0.0
    del dep_scores

    dependency = dependency_full[:, -resp_len:]
    del dependency_full

    # ── Identify dependent tokens ──
    dependent_mask = torch.zeros(bs, resp_len, device=device, dtype=torch.bool)
    non_anchor_resp = (~anchor_mask) & resp_mask_bool

    for b in range(bs):
        valid_dep = dependency[b][non_anchor_resp[b]]
        if valid_dep.numel() < 2:
            continue
        threshold = torch.quantile(valid_dep.float(), dependent_percentile / 100.0)
        dependent_mask[b] = (dependency[b] >= threshold) & non_anchor_resp[b]

    # ── Assign discrete weights ──
    weights = torch.ones(bs, resp_len, device=device, dtype=torch.float32)
    weights[anchor_mask] = anchor_boost
    weights[dependent_mask] = dependent_boost
    weights = weights.clamp(max=max_weight) * response_mask.float()

    return weights
