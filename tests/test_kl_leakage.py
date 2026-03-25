"""Tests for KL leakage penalty computation."""
import torch
from src.self_teach.rewards import compute_kl_leakage_penalty


class TestKLLeakagePenalty:
    """Test the KL leakage penalty aggregation logic."""

    def test_zero_kl_when_distributions_match(self):
        """If informed and student log-probs are identical, penalty is 0."""
        informed_logprobs = torch.tensor([-1.0, -2.0, -1.5])
        student_logprobs = torch.tensor([-1.0, -2.0, -1.5])
        penalty = compute_kl_leakage_penalty(
            informed_logprobs, student_logprobs, alpha=0.01
        )
        assert abs(penalty) < 1e-6

    def test_positive_penalty_when_distributions_differ(self):
        """Higher informed log-probs (teacher knows more) should yield positive KL."""
        informed_logprobs = torch.tensor([-0.5, -0.3, -0.1])
        student_logprobs = torch.tensor([-2.0, -3.0, -4.0])
        penalty = compute_kl_leakage_penalty(
            informed_logprobs, student_logprobs, alpha=0.01
        )
        assert penalty > 0.0

    def test_alpha_scales_max_term(self):
        """Higher alpha should increase penalty when there's a spike."""
        informed_logprobs = torch.tensor([-0.5, -0.5, -0.1])
        student_logprobs = torch.tensor([-0.5, -0.5, -4.0])
        low_alpha = compute_kl_leakage_penalty(
            informed_logprobs, student_logprobs, alpha=0.01
        )
        high_alpha = compute_kl_leakage_penalty(
            informed_logprobs, student_logprobs, alpha=1.0
        )
        assert high_alpha > low_alpha

    def test_single_token(self):
        """Should work with a single token."""
        informed_logprobs = torch.tensor([-0.5])
        student_logprobs = torch.tensor([-2.0])
        penalty = compute_kl_leakage_penalty(
            informed_logprobs, student_logprobs, alpha=0.01
        )
        assert penalty > 0.0
