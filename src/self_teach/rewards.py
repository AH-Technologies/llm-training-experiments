"""Reward computation for the self-teach student-teacher pipeline."""
import torch


def compute_kl_leakage_penalty(
    informed_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    alpha: float = 0.01,
) -> float:
    """Compute KL-based leakage penalty for teacher feedback tokens.

    Measures how much the feedback tokens depend on information only
    available in the teacher's context (i.e., the ground truth answer).

    Args:
        informed_logprobs: Log-probs of feedback tokens under teacher-informed
            context (question + student_attempt + ground_truth + feedback).
        student_logprobs: Log-probs of feedback tokens under student-only
            context (question + student_attempt + feedback).
        alpha: Weight for the max-KL term (catches single-token spikes).

    Returns:
        Scalar penalty >= 0. Higher means more leakage.
    """
    # Per-token KL: log p_informed - log p_student
    # When informed > student (teacher context helps predict token),
    # this is positive — indicating the token carries GT information.
    per_token_kl = informed_logprobs - student_logprobs

    # Clamp to >= 0: we only penalize tokens where teacher context helps
    per_token_kl = torch.clamp(per_token_kl, min=0.0)

    mean_kl = per_token_kl.mean().item()
    max_kl = per_token_kl.max().item()

    return mean_kl + alpha * max_kl


def compute_solution_understanding_reward(
    solution_logprobs: torch.Tensor,
    alpha: float = 0.01,
) -> float:
    """Compute r^SS: how well the student understands the solution after reading feedback.

    Measures the student model's log-probability of producing the correct
    solution/answer tokens given the question, student attempt, and feedback
    as context. Higher means the feedback made the solution more predictable
    to the student — i.e., better teaching.

    Uses mean + alpha * min (following RLT): mean captures overall understanding,
    min catches the hardest token in the solution — if any part of the answer
    is very unlikely, the explanation didn't fully bridge the gap.

    Args:
        solution_logprobs: Per-token log-probs of ground truth answer tokens
            under context [question + student_attempt + feedback].
        alpha: Weight for the min term (catches worst-case tokens).

    Returns:
        Scalar reward. Higher = student better understands the solution.
    """
    mean_lp = solution_logprobs.mean().item()
    min_lp = solution_logprobs.min().item()
    return mean_lp + alpha * min_lp


def compute_self_teach_rewards(a1_correct: bool, a2_correct: bool) -> tuple[float, float]:
    """Compute (teacher_reward, student2_reward) from Student₁ and Student₂ correctness.

    Reward table:
        A₁ wrong  → A₂ correct:  teacher=1.0  student₂=1.0  (successful teaching)
        A₁ wrong  → A₂ wrong:    teacher=0.0  student₂=0.0  (failed to help)
        A₁ correct → A₂ correct: teacher=1.0  student₂=1.0  (didn't confuse student)
        A₁ correct → A₂ wrong:   teacher=-1.0 student₂=0.0  (confused the student)
    """
    if a2_correct:
        return 1.0, 1.0
    elif a1_correct and not a2_correct:
        return -1.0, 0.0
    else:
        return 0.0, 0.0
