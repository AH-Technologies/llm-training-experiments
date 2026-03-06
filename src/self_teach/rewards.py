"""Reward computation for the self-teach student-teacher pipeline."""


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
