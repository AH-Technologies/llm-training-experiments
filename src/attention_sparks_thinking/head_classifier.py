"""Head classification: local vs global attention heads.

Computes average backward distance d[l,h] for each head, then classifies
the bottom 30% as local (H_loc) and top 30% as global (H_glob).

Also provides hook-based attention aggregation for memory-efficient
extraction of A_bar_loc and A_bar_glob.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_middle_third_layers(n_layers: int, n_sample: int = 5) -> list[int]:
    """Return n_sample evenly spaced layer indices from the middle third.

    Paper: "five evenly spaced layers within the middle third of the network
    (i.e., from layers floor(L/3) to floor(2L/3))"
    """
    lo = n_layers // 3
    hi = 2 * n_layers // 3
    if n_sample <= 1:
        return [(lo + hi) // 2]
    if hi - lo + 1 <= n_sample:
        return list(range(lo, hi + 1))
    step = (hi - lo) / (n_sample - 1)
    return [lo + round(i * step) for i in range(n_sample)]


def classify_heads(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    head_quantile: float = 0.3,
    max_seq_len: int = 512,
    n_sample_layers: int = 5,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], torch.Tensor]:
    """Classify attention heads as local or global based on average backward distance.

    Only considers heads from n_sample_layers evenly spaced layers in the
    middle third of the network, following the Attention Illuminates paper.

    Args:
        model: HuggingFace model loaded with attn_implementation="eager"
        tokenizer: corresponding tokenizer
        prompts: list of prompt strings for classification
        head_quantile: fraction for bottom/top classification (default 0.3)
        max_seq_len: max sequence length for classification prompts
        n_sample_layers: number of layers to sample from middle third (default 5)

    Returns:
        H_loc: list of (layer, head) tuples — local-focused heads
        H_glob: list of (layer, head) tuples — global-focused heads
        d_matrix: (n_sample_layers, n_heads) tensor of average backward distances
    """
    device = next(model.parameters()).device
    model.eval()

    # Tokenize all prompts
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    sampled_layers = get_middle_third_layers(n_layers, n_sample_layers)
    logger.info(f"Sampling {len(sampled_layers)} layers from middle third of {n_layers}: {sampled_layers}")

    # Accumulate d[l,h] only for sampled layers
    d_accum = torch.zeros(len(sampled_layers), n_heads, device=device)
    total_response_tokens = 0

    # Process in smaller batches to save memory
    batch_size = min(4, len(prompts))
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_ids = encodings["input_ids"][batch_start:batch_end]
        batch_mask = encodings["attention_mask"][batch_start:batch_end]

        with torch.no_grad():
            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                output_attentions=True,
            )

        # outputs.attentions: tuple of (batch, n_heads, seq_len, seq_len) per layer
        seq_len = batch_ids.shape[1]
        positions = torch.arange(seq_len, device=device)
        dist = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = dist.clamp(min=0).float()

        for si, layer_idx in enumerate(sampled_layers):
            attn_layer = outputs.attentions[layer_idx]
            weighted_dist = (attn_layer * dist.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            mask = batch_mask.unsqueeze(1).float()
            weighted_dist = weighted_dist * mask
            d_accum[si] += weighted_dist.sum(dim=-1).sum(dim=0)
            if si == 0:  # count tokens once
                total_response_tokens += mask.sum().item() / n_heads

        del outputs

    # Normalize
    if total_response_tokens > 0:
        d_accum /= total_response_tokens

    # Classify: bottom head_quantile by d -> H_loc, top head_quantile -> H_glob
    # Only across the sampled layers
    d_flat = d_accum.flatten()
    n_total = d_flat.numel()
    n_select = max(1, int(n_total * head_quantile))

    sorted_indices = d_flat.argsort()
    loc_flat_indices = sorted_indices[:n_select]
    glob_flat_indices = sorted_indices[-n_select:]

    H_loc = []
    for idx in loc_flat_indices.tolist():
        si, h = divmod(idx, n_heads)
        H_loc.append((sampled_layers[si], h))

    H_glob = []
    for idx in glob_flat_indices.tolist():
        si, h = divmod(idx, n_heads)
        H_glob.append((sampled_layers[si], h))

    logger.info(f"Head classification: {n_total} heads across {len(sampled_layers)} sampled layers, "
                f"{len(H_loc)} local, {len(H_glob)} global")
    logger.info(f"H_loc layers: {sorted(set(l for l, h in H_loc))}")
    logger.info(f"H_glob layers: {sorted(set(l for l, h in H_glob))}")

    return H_loc, H_glob, d_accum


def aggregate_attention_hooks(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    H_loc: list[tuple[int, int]],
    H_glob: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract aggregated attention maps using forward hooks (memory-efficient).

    Supports batch > 1 for efficient batched gamma computation.
    Hooks on self_attn capture attention weights that Qwen3's eager attention
    always computes (returned unconditionally by eager_attention_forward).
    We pass output_attentions=False so the model doesn't store attention from
    all 36 layers — only the ~5 target layers' weights are transiently captured
    by hooks, keeping peak memory to one layer at a time.

    Args:
        model: HuggingFace model with attn_implementation="eager"
        input_ids: (batch, seq_len) input token ids
        attention_mask: (batch, seq_len) attention mask
        H_loc: list of (layer, head) local heads
        H_glob: list of (layer, head) global heads

    Returns:
        If batch == 1:
            A_bar_loc: (seq_len, seq_len) aggregated local attention
            A_bar_glob: (seq_len, seq_len) aggregated global attention
        If batch > 1:
            A_bar_loc: (batch, seq_len, seq_len) per-sample aggregated local attention
            A_bar_glob: (batch, seq_len, seq_len) per-sample aggregated global attention
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    # Build lookup: layer -> list of head indices for each group
    loc_by_layer: dict[int, list[int]] = {}
    glob_by_layer: dict[int, list[int]] = {}
    for l, h in H_loc:
        loc_by_layer.setdefault(l, []).append(h)
    for l, h in H_glob:
        glob_by_layer.setdefault(l, []).append(h)

    all_layers = set(loc_by_layer.keys()) | set(glob_by_layer.keys())

    # Accumulators — per-sample for batch support
    A_bar_loc = torch.zeros(batch_size, seq_len, seq_len, device=device)
    A_bar_glob = torch.zeros(batch_size, seq_len, seq_len, device=device)
    n_loc = len(H_loc)
    n_glob = len(H_glob)

    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Qwen3 eager_attention_forward always returns (attn_output, attn_weights)
            # attn_weights: (batch, n_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]  # (batch, n_heads, seq_len, seq_len)

                if layer_idx in loc_by_layer:
                    heads = loc_by_layer[layer_idx]
                    A_bar_loc.add_(attn_weights[:, heads].sum(dim=1))

                if layer_idx in glob_by_layer:
                    heads = glob_by_layer[layer_idx]
                    A_bar_glob.add_(attn_weights[:, heads].sum(dim=1))

        return hook_fn

    # Register hooks on attention layers that we care about
    for layer_idx in all_layers:
        attn_module = model.model.layers[layer_idx].self_attn
        handle = attn_module.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    # Forward pass — output_attentions=False since eager attention always
    # returns weights from self_attn anyway. This avoids the model storing
    # attention tensors from all 36 layers (only ~5 target layers are captured
    # by hooks, one at a time).
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Normalize
    if n_loc > 0:
        A_bar_loc /= n_loc
    if n_glob > 0:
        A_bar_glob /= n_glob

    # Backward compatible: return 2D tensors when batch=1
    if batch_size == 1:
        return A_bar_loc[0], A_bar_glob[0]
    return A_bar_loc, A_bar_glob
