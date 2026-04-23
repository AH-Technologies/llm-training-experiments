"""
GEM loss (Generative Entropy-regularized Matching) for SFT.

Ported verbatim from liziniu/GEM `ReferenceGEMLoss` in tests/test_gem_loss_triton.py.
Only h="linear" variant — the only one verified by the reference test suite.

Reference: Li et al., "GEM: Generative Entropy-regularized Matching for Training
Generative Models", ICLR 2025. https://github.com/liziniu/GEM
"""

import torch
import torch.nn.functional as F


def gem_loss(logits, labels, beta=0.7, ignore_index=-100, reduction="mean"):
    """GEM loss (h=linear), ported from liziniu/GEM ReferenceGEMLoss.

    Equivalent to: CE(y, p) + E_q[log p] where q = softmax(logits/beta) is detached.

    Args:
        logits: (N, V) unnormalized logits
        labels: (N,) target token ids
        beta: temperature for the matching distribution q
        ignore_index: label value to ignore (padding)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss (or per-token losses if reduction="none")
    """
    mask = labels != ignore_index
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    with torch.no_grad():
        q_probs = F.softmax(masked_logits / beta, dim=-1)

    gene_log_probs = F.log_softmax(masked_logits, dim=-1)
    real_log_probs = torch.gather(gene_log_probs, dim=-1, index=masked_labels.unsqueeze(-1))

    loss = -torch.sum(q_probs * (real_log_probs - gene_log_probs), dim=-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def gem_loss_causal_lm(logits, labels, beta=0.7, ignore_index=-100, reduction="mean"):
    """GEM loss with causal LM shift applied.

    Handles the standard shift: logits[..., :-1, :] predict labels[..., 1:].

    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) token ids with -100 for positions to ignore
        beta: temperature for the matching distribution q
        ignore_index: label value to ignore
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss (or per-token losses if reduction="none")
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    return gem_loss(shift_logits, shift_labels, beta=beta, ignore_index=ignore_index, reduction=reduction)


def ce_loss_causal_lm(logits, labels, ignore_index=-100):
    """Standard CE loss with causal LM shift. For logging alongside GEM."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
