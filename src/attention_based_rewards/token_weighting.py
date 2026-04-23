"""Per-token weighting functions for circuit-guided GRPO.

Four conditions that differ ONLY in how per-token advantage weights are computed:
  1. uniform:  all ones (standard GRPO baseline)
  2. attention: weighted by reasoning head attention (our method)
  3. entropy:  weighted by policy entropy at each token (GTPO-style)
  4. combined: alpha * attention + (1-alpha) * entropy

All functions return weights of shape (batch_size, response_length) with mean=1
per sequence (so the overall advantage magnitude is preserved).

No verl dependency — pure PyTorch.
"""

import torch
from torch import Tensor


def _normalize_per_sequence(weights: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize weights to mean=1 per sequence, respecting the response mask.

    Args:
        weights: (bs, seq_len) raw importance weights
        mask: (bs, seq_len) binary mask (1 for response tokens, 0 for padding)

    Returns:
        (bs, seq_len) weights with mean=1 over masked positions per sequence
    """
    masked_weights = weights * mask
    seq_means = masked_weights.sum(dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + eps)
    normalized = masked_weights / (seq_means + eps)
    # Ensure padding positions remain 0
    return normalized * mask


def compute_uniform_weights(response_mask: Tensor) -> Tensor:
    """Condition 1: uniform weights (standard GRPO).

    Args:
        response_mask: (bs, seq_len)

    Returns:
        (bs, seq_len) all ones where mask is 1
    """
    return response_mask.float()


def compute_attention_weights(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
) -> Tensor:
    """Condition 2: weight by reasoning head attention received.

    Runs a forward pass on attn_model (with output_attentions=True) and computes
    how much each response token is *attended to* by reasoning heads.

    Args:
        input_ids: (bs, seq_len) full sequence (prompt + response)
        attention_mask: (bs, seq_len) 1 for real tokens, 0 for padding
        response_mask: (bs, seq_len) 1 for response tokens only
        attn_model: HF model loaded with attn_implementation="eager"
        reasoning_heads: list of (layer, head) tuples
        head_scores: (n_layers, n_heads) importance scores from EAP-IG

    Returns:
        (bs, seq_len) attention-based weights, mean=1 per sequence
    """
    bs, seq_len = input_ids.shape
    device = input_ids.device

    # Determine which layers we need attention from
    layers_needed = sorted(set(layer for layer, _ in reasoning_heads))

    # Forward pass with attention output
    with torch.no_grad():
        outputs = attn_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )

    # outputs.attentions is a tuple of (bs, n_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions

    # Accumulate weighted "received attention" for each token
    token_importance = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    for layer, head in reasoning_heads:
        # attn_pattern[b, i, j] = how much position i attends to position j
        attn_pattern = attentions[layer][:, head, :, :]  # (bs, seq_len, seq_len)

        # Sum attention received by each token (sum over attending positions)
        received = attn_pattern.sum(dim=1)  # (bs, seq_len)

        # Weight by head importance
        importance = head_scores[layer, head].item()
        token_importance += received.float() * importance

    # Free attention tensors
    del attentions, outputs

    return _normalize_per_sequence(token_importance, response_mask)


def compute_entropy_weights(
    entropy: Tensor,
    response_mask: Tensor,
) -> Tensor:
    """Condition 3: GTPO-style entropy weighting.

    High entropy = important decision point = higher weight.
    verl stores per-token entropy in batch["entropys"] during old_log_prob computation.

    Args:
        entropy: (bs, seq_len) per-token entropy from the policy
        response_mask: (bs, seq_len)

    Returns:
        (bs, seq_len) entropy-based weights, mean=1 per sequence
    """
    # Clamp to avoid negative values from numerical issues
    weights = entropy.float().clamp(min=0)
    return _normalize_per_sequence(weights, response_mask)


def _compute_fai_with_hooks(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    heads_by_layer: dict[int, list[tuple[int, float]]],
) -> Tensor:
    """Core FAI computation using forward hooks for memory efficiency.

    Registers hooks on attention modules to capture and IMMEDIATELY process
    attention weights during the forward pass. Each layer's attention is
    processed and freed before the next layer runs, keeping peak memory
    to ~600MB instead of ~18GB.

    Works on CPU (bfloat16) within the TaskRunner which has no GPU access.

    Args:
        input_ids: (bs, seq_len)
        attention_mask: (bs, seq_len)
        response_mask: (bs, response_len)
        attn_model: HF model with attn_implementation="eager"
        heads_by_layer: {layer_idx: [(head_idx, weight), ...]}

    Returns:
        (bs, response_len) FAI-based weights, mean=1 per sequence
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    token_importance = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    # future_mask[i, t] = 1 if i > t
    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)

    # Hook that captures attention, computes FAI contribution, and discards
    handles = []

    def _make_hook(layer_idx, head_list):
        def hook_fn(module, args, output):
            # self_attn returns (attn_output, attn_weights, past_kv)
            # attn_weights is (bs, n_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, weight in head_list:
                    attn_pattern = attn_weights[:, head_idx, :, :]
                    future_attn = attn_pattern * future_mask.unsqueeze(0)
                    received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
                    token_importance.add_(received.float() * weight)
                # Return modified output WITHOUT attention weights to free memory
                return (output[0], None) + output[2:]
        return hook_fn

    # Register hooks on needed layers
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
    """Future Attention Influence (FAI) restricted to reasoning heads.

    For each token t, computes how much LATER positions attend to t through the
    reasoning heads, weighted by head importance.
    """
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
    n_layers: int = 28,
    n_heads: int = 12,
) -> Tensor:
    """FAI across ALL attention heads with equal weight (1/n_total each).

    Replicates the Attention Illuminates approach — uses all heads without
    any circuit information. Each head contributes equally.
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
    attn_weights: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
) -> Tensor:
    """Asymmetric weighting: amplify reasoning on correct, protect on incorrect.

    For correct responses (advantage > 0):
        w_t = attention_weight_t  (amplify reasoning tokens)
    For incorrect responses (advantage <= 0):
        w_t = inverted attention weights  (protect reasoning from penalty)

    Both are normalized to mean=1 per sequence.

    The idea: when the model gets the answer wrong but was reasoning hard,
    we don't want to penalize reasoning. The penalty falls on non-reasoning
    tokens instead, preserving the model's willingness to engage reasoning circuits.

    Args:
        attn_weights: (bs, seq_len) attention-based weights, mean=1 per sequence
        advantages: (bs, seq_len) per-token advantages (same scalar broadcast per seq)
        response_mask: (bs, seq_len)

    Returns:
        (bs, seq_len) asymmetric weights, mean=1 per sequence
    """
    # Determine correct/incorrect per sequence from advantage sign
    # In GRPO with binary rewards, positive advantage = correct, negative = incorrect
    is_correct = (advantages[:, 0] > 0).unsqueeze(1)  # (bs, 1)

    # For correct: use attention weights as-is (already mean=1)
    correct_weights = attn_weights

    # For incorrect: invert the ranking
    # max(w) - w ensures highest attention → lowest weight, preserving reasoning
    w_max = (attn_weights * response_mask).max(dim=-1, keepdim=True).values
    inverted = (w_max - attn_weights + 1e-6) * response_mask
    inverted_weights = _normalize_per_sequence(inverted, response_mask)

    # Select based on correctness
    weights = torch.where(is_correct, correct_weights, inverted_weights)
    return weights


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
) -> tuple[Tensor, Tensor, Tensor]:
    """Anchor-based circuit credit assignment.

    Two-pass computation using hooks for memory efficiency:
      Pass 1: Compute FAI on reasoning heads to identify anchor tokens
              (tokens that future reasoning heavily references).
      Pass 2: Compute dependency scores — how much each non-anchor token
              attends to anchors via reasoning heads (max over anchors).

    Returns DISCRETE weights: anchor_boost for anchors, dependent_boost for
    dependents, 1.0 for all other tokens. No token ever gets weight < 1.0.

    The caller is responsible for applying these only to correct responses.

    Args:
        input_ids: (bs, seq_len)
        attention_mask: (bs, seq_len)
        response_mask: (bs, seq_len) 1 for response tokens
        attn_model: HF model with attn_implementation="eager"
        reasoning_heads: list of (layer, head) tuples
        head_scores: (n_layers, n_heads) importance scores from EAP-IG
        anchor_percentile: FAI percentile for anchors (default 90 = top 10%)
        dependent_percentile: dependency percentile for dependents (default 75)
        anchor_boost: weight for anchor tokens (default 2.0)
        dependent_boost: weight for dependent tokens (default 1.5)
        max_weight: clip ceiling (default 3.0)

    Returns:
        weights: (bs, seq_len) discrete token weights (>= 1.0)
        anchor_mask: (bs, seq_len) bool mask for anchor tokens
        dependent_mask: (bs, seq_len) bool mask for dependent tokens
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]  # may be < seq_len (response portion only)
    prompt_len = seq_len - resp_len

    # Build heads_by_layer mapping
    heads_by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, head in reasoning_heads:
        weight = head_scores[layer, head].item()
        heads_by_layer.setdefault(layer, []).append((head, weight))

    # ── Pass 1: Compute raw FAI scores (NOT normalized) ──
    # FAI_t = sum over reasoning heads: weight * mean(attn[i, t] for i > t)
    # Uses tril (lower triangle) because i > t means row > column.
    fai_scores_full = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    # future_mask[i, t] = 1 if i > t (lower triangle, excluding diagonal)
    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)  # how many future tokens per position

    handles = []

    def _make_fai_hook(head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]  # (bs, n_heads, seq_len, seq_len)
                for head_idx, w in head_list:
                    attn_pattern = attn_weights[:, head_idx, :, :]  # (bs, seq_len, seq_len)
                    # future_attn[i, t] = attn[i, t] only where i > t
                    future_attn = attn_pattern * future_mask.unsqueeze(0)
                    # Mean future attention received by each token t
                    received = future_attn.sum(dim=1) / future_count.unsqueeze(0)  # (bs, seq_len)
                    fai_scores_full.add_(received.float() * w)
                return (output[0], None) + output[2:]  # free attention memory
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

    # Trim FAI to response portion (response_mask may be shorter than full seq)
    fai_scores = fai_scores_full[:, -resp_len:]
    del fai_scores_full

    # ── Identify anchors: tokens with FAI above anchor_percentile within each sequence ──
    resp_mask_bool = response_mask.bool()
    anchor_mask = torch.zeros(bs, resp_len, device=device, dtype=torch.bool)

    for b in range(bs):
        valid_fai = fai_scores[b][resp_mask_bool[b]]
        if valid_fai.numel() < 2:
            continue
        threshold = torch.quantile(valid_fai.float(), anchor_percentile / 100.0)
        anchor_mask[b] = (fai_scores[b] >= threshold) & resp_mask_bool[b]

    # ── Pass 2: Compute dependency scores via reasoning-head attention to anchors ──
    # For each token i: dependency = max over anchors a of
    #   (sum over reasoning heads: weight * attn[l,h,i,a])
    # Causal mask guarantees attn[i,a]=0 for a>i, so only anchors before i count.
    #
    # Anchor positions are in resp-space [0, resp_len). Convert to full-seq-space
    # for the gather on attention matrices which are (bs, seq_len, seq_len).

    max_n_anchors = int(max(anchor_mask.sum(dim=-1).max().item(), 1))

    # Padded anchor indices in FULL seq space for gather
    anchor_indices_full = torch.zeros(bs, max_n_anchors, device=device, dtype=torch.long)
    anchor_valid = torch.zeros(bs, max_n_anchors, device=device, dtype=torch.bool)

    for b in range(bs):
        resp_idx = anchor_mask[b].nonzero(as_tuple=True)[0]  # positions in resp space
        n = resp_idx.shape[0]
        if n > 0:
            anchor_indices_full[b, :n] = resp_idx + prompt_len  # convert to full-seq space
            anchor_valid[b, :n] = True

    # Accumulator: weighted attention from each token to each anchor
    # Shape (bs, seq_len, max_n_anchors) — typically ~40MB, very manageable
    dep_scores = torch.zeros(bs, seq_len, max_n_anchors, device=device, dtype=torch.float32)
    expanded_idx = anchor_indices_full.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, max_n_anchors)

    def _make_dep_hook(head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, w in head_list:
                    attn = attn_weights[:, head_idx, :, :]  # (bs, seq_len, seq_len)
                    # Gather attention to anchor positions (in full-seq space)
                    anchor_attn = attn.gather(dim=-1, index=expanded_idx)  # (bs, seq_len, max_n_anchors)
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

    # Max over anchors: for each token, the strongest dependency on any single anchor
    dep_scores.masked_fill_(~anchor_valid.unsqueeze(1), float("-inf"))
    dependency_full = dep_scores.max(dim=-1).values  # (bs, seq_len)
    dependency_full[dependency_full == float("-inf")] = 0.0
    del dep_scores

    # Trim dependency to response portion
    dependency = dependency_full[:, -resp_len:]
    del dependency_full

    # ── Identify dependent tokens: high-dependency non-anchor response tokens ──
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

    return weights, anchor_mask, dependent_mask


def compute_circuit_score(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
) -> Tensor:
    """Compute mean FAI across response tokens for each sequence.

    Returns a per-response scalar measuring how much the response activated
    reasoning circuits. Higher = more reasoning head engagement.

    Uses the same hook-based FAI computation as other methods but returns
    mean(FAI_t) over response tokens instead of per-token weights.

    Args:
        input_ids: (bs, seq_len)
        attention_mask: (bs, seq_len)
        response_mask: (bs, resp_len) — may be shorter than seq_len
        attn_model: HF model with attn_implementation="eager"
        reasoning_heads: list of (layer, head) tuples
        head_scores: (n_layers, n_heads) importance scores

    Returns:
        circuit_scores: (bs,) mean FAI per response (raw, unnormalized)
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    heads_by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, head in reasoning_heads:
        weight = head_scores[layer, head].item()
        heads_by_layer.setdefault(layer, []).append((head, weight))

    fai_scores = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)

    # future_mask[i, t] = 1 if i > t (lower triangle)
    future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
    future_count = future_mask.sum(dim=0).clamp(min=1)

    handles = []

    def _make_hook(head_list):
        def hook_fn(module, args, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                for head_idx, w in head_list:
                    attn_pattern = attn_weights[:, head_idx, :, :]
                    future_attn = attn_pattern * future_mask.unsqueeze(0)
                    received = future_attn.sum(dim=1) / future_count.unsqueeze(0)
                    fai_scores.add_(received.float() * w)
                return (output[0], None) + output[2:]
        return hook_fn

    for layer_idx, head_list in heads_by_layer.items():
        attn_module = attn_model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(_make_hook(head_list))
        handles.append(handle)

    with torch.no_grad():
        attn_model(input_ids=input_ids, attention_mask=attention_mask,
                   output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()
    del future_mask, future_count, handles
    gc.collect()

    # Trim to response portion and compute mean
    fai_resp = fai_scores[:, -resp_len:]
    masked_fai = fai_resp * response_mask
    mean_fai = masked_fai.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)

    return mean_fai  # (bs,)


def compute_activation_entropy_score(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    n_layers: int = 28,
) -> Tensor:
    """Activation entropy: how uniformly the model distributes computation.

    Hooks all self_attn + mlp modules, captures output L2 norms on response
    tokens, builds a 56-component vector, normalizes to probabilities, and
    computes Shannon entropy / log(56).

    Returns:
        (bs,) in [0, 1]. Higher = more uniform activation across components.
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    # Collect L2 norms per component: (n_components, bs)
    component_norms = []
    handles = []

    def _make_norm_hook(storage_list):
        def hook_fn(module, args, output):
            # output is a tuple for self_attn, tensor for mlp
            out = output[0] if isinstance(output, tuple) else output
            # out: (bs, seq_len, hidden_dim)
            # Compute L2 norm over hidden dim, mean over response tokens
            norms = out.float().norm(dim=-1)  # (bs, seq_len)
            resp_norms = norms[:, -resp_len:]  # (bs, resp_len)
            masked = resp_norms * response_mask
            mean_norm = masked.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # (bs,)
            storage_list.append(mean_norm.detach())
        return hook_fn

    for layer_idx in range(n_layers):
        layer = attn_model.model.layers[layer_idx]
        # Hook self_attn
        h = layer.self_attn.register_forward_hook(_make_norm_hook(component_norms))
        handles.append(h)
        # Hook MLP
        h = layer.mlp.register_forward_hook(_make_norm_hook(component_norms))
        handles.append(h)

    try:
        with torch.no_grad():
            attn_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    # component_norms: list of (bs,) tensors, length = 2 * n_layers
    norms_matrix = torch.stack(component_norms, dim=1)  # (bs, 56)
    del component_norms
    gc.collect()

    # Normalize to probabilities
    probs = norms_matrix / norms_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
    # Shannon entropy / log(n_components)
    n_components = norms_matrix.shape[1]
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=1)  # (bs,)
    normalized_entropy = entropy / torch.log(torch.tensor(float(n_components), device=device))

    return normalized_entropy.clamp(0, 1)  # (bs,)


def compute_mlp_circuit_score(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    reasoning_heads: list[tuple[int, int]],
    head_scores: Tensor,
    divergence_weights: dict,
) -> Tensor:
    """MLP-aware circuit score: FAI on attention heads + L2 norms on MLPs.

    Combines:
      - Attention component: FAI weighted by divergence (not EAP-IG)
      - MLP component: L2 norm of MLP outputs weighted by divergence

    Only includes components with positive signed divergence (more active on correct).

    Args:
        divergence_weights: dict with keys:
            "attn_divergence": (n_layers, n_heads) signed divergence per head
            "mlp_divergence": (n_layers,) signed divergence per MLP layer

    Returns:
        (bs,) combined circuit score
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    attn_div = divergence_weights["attn_divergence"].to(device)  # (n_layers, n_heads)
    mlp_div = divergence_weights["mlp_divergence"].to(device)    # (n_layers,)

    # ── Attention component: FAI weighted by divergence ──
    # Only include heads with positive divergence
    heads_by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, head in reasoning_heads:
        div_val = attn_div[layer, head].item()
        if div_val > 0:
            heads_by_layer.setdefault(layer, []).append((head, div_val))

    fai_scores = torch.zeros(bs, seq_len, device=device, dtype=torch.float32)
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
                    fai_scores.add_(received.float() * w)
                return (output[0], None) + output[2:]
        return hook_fn

    for layer_idx, head_list in heads_by_layer.items():
        attn_module = attn_model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(_make_fai_hook(head_list))
        handles.append(handle)

    # ── MLP component: L2 norm weighted by divergence ──
    mlp_scores = torch.zeros(bs, device=device, dtype=torch.float32)

    def _make_mlp_hook(layer_idx):
        def hook_fn(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            norms = out.float().norm(dim=-1)  # (bs, seq_len)
            resp_norms = norms[:, -resp_len:]
            masked = resp_norms * response_mask
            mean_norm = masked.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # (bs,)
            div_val = mlp_div[layer_idx].item()
            mlp_scores.add_(mean_norm * div_val)
        return hook_fn

    n_layers = mlp_div.shape[0]
    for layer_idx in range(n_layers):
        if mlp_div[layer_idx].item() > 0:
            handle = attn_model.model.layers[layer_idx].mlp.register_forward_hook(
                _make_mlp_hook(layer_idx)
            )
            handles.append(handle)

    try:
        with torch.no_grad():
            attn_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True if heads_by_layer else False,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    del future_mask, future_count
    gc.collect()

    # Combine: mean FAI over response tokens + MLP scores
    fai_resp = fai_scores[:, -resp_len:]
    masked_fai = fai_resp * response_mask
    mean_fai = masked_fai.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # (bs,)

    return mean_fai + mlp_scores  # (bs,)


def compute_layerwise_slope_score(
    input_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
    attn_model: "torch.nn.Module",
    n_layers: int = 28,
) -> Tensor:
    """Layerwise slope: whether activation magnitude increases across layers.

    Hooks all attn + MLP outputs, sums norms per layer to get a (bs, n_layers)
    contribution vector, then fits OLS regression (norm ~ layer_index).
    Score = R² if slope > 0, else 0.

    Returns:
        (bs,) in [0, 1]
    """
    import gc

    bs, seq_len = input_ids.shape
    device = input_ids.device
    resp_len = response_mask.shape[1]

    # Collect per-layer total norm: (bs, n_layers)
    layer_norms = torch.zeros(bs, n_layers, device=device, dtype=torch.float32)
    handles = []

    def _make_layer_hook(layer_idx):
        def hook_fn(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            norms = out.float().norm(dim=-1)  # (bs, seq_len)
            resp_norms = norms[:, -resp_len:]
            masked = resp_norms * response_mask
            mean_norm = masked.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # (bs,)
            layer_norms[:, layer_idx] += mean_norm.detach()
        return hook_fn

    for layer_idx in range(n_layers):
        layer = attn_model.model.layers[layer_idx]
        h1 = layer.self_attn.register_forward_hook(_make_layer_hook(layer_idx))
        h2 = layer.mlp.register_forward_hook(_make_layer_hook(layer_idx))
        handles.extend([h1, h2])

    try:
        with torch.no_grad():
            attn_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    gc.collect()

    # ── Vectorized OLS: layer_norms ~ layer_index ──
    # x = layer indices (0, 1, ..., n_layers-1)
    x = torch.arange(n_layers, device=device, dtype=torch.float32)
    x_mean = x.mean()
    y = layer_norms  # (bs, n_layers)
    y_mean = y.mean(dim=1, keepdim=True)  # (bs, 1)

    x_centered = x - x_mean  # (n_layers,)
    y_centered = y - y_mean  # (bs, n_layers)

    # slope = sum(x_c * y_c) / sum(x_c^2)
    ss_xy = (x_centered.unsqueeze(0) * y_centered).sum(dim=1)  # (bs,)
    ss_xx = (x_centered ** 2).sum()  # scalar

    slope = ss_xy / ss_xx.clamp(min=1e-8)  # (bs,)

    # R² = 1 - SS_res / SS_tot
    y_pred = slope.unsqueeze(1) * x_centered.unsqueeze(0) + y_mean  # (bs, n_layers)
    ss_res = ((y - y_pred) ** 2).sum(dim=1)  # (bs,)
    ss_tot = (y_centered ** 2).sum(dim=1)  # (bs,)
    r_squared = 1 - ss_res / ss_tot.clamp(min=1e-8)  # (bs,)

    # Score = R² if slope > 0, else 0
    score = torch.where(slope > 0, r_squared.clamp(0, 1), torch.zeros_like(r_squared))

    return score  # (bs,)


def compute_combined_weights(
    attn_weights: Tensor,
    entropy_weights: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """Condition 4: linear combination of attention and entropy weights.

    Args:
        attn_weights: (bs, seq_len) already normalized to mean=1
        entropy_weights: (bs, seq_len) already normalized to mean=1
        alpha: mixing coefficient (1.0 = pure attention, 0.0 = pure entropy)

    Returns:
        (bs, seq_len) combined weights (already mean~1 since inputs are mean=1)
    """
    return alpha * attn_weights + (1 - alpha) * entropy_weights
