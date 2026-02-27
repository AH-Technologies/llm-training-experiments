"""SelfTeachInteraction: 3-turn student-teacher self-play for VERL multi-turn system."""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from ..rewards.deepscaler_reward import compute_score
from .prompts import TEACHER_PROMPT, STUDENT2_PROMPT
from .rewards import compute_self_teach_rewards

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SelfTeachInteraction(BaseInteraction):
    """3-turn student-teacher interaction for self-play training.

    Turn 1: Student₁ answers the question.
    Turn 2: Teacher gives feedback (without revealing the answer).
    Turn 3: Student₂ revises based on feedback.

    All 3 turns always execute — no early termination.
    Per-role rewards are stored in metadata for the trainer to consume.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instances: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {
            "ground_truth": ground_truth,
            "data_source": kwargs.get("data_source", "math"),
            "turn": 0,
            "question": None,
            "a1": None,
            "a1_correct": None,
            "feedback": None,
            "a2": None,
            "a2_correct": None,
            "teacher_reward": None,
            "student2_reward": None,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instances[instance_id]
        state["turn"] += 1

        # Extract latest assistant message
        assistant_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                break

        if state["turn"] == 1:
            return self._handle_student1(state, messages, assistant_content)
        elif state["turn"] == 2:
            return self._handle_teacher(state, assistant_content)
        elif state["turn"] == 3:
            return self._handle_student2(state, assistant_content)
        else:
            # Should not reach here with max_assistant_turns=3
            logger.warning(f"Unexpected turn {state['turn']} for instance {instance_id}")
            return True, "", 0.0, {}

    def _handle_student1(
        self, state: dict, messages: list[dict[str, Any]], assistant_content: str
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Turn 1: Student₁ answered. Grade and transition to teacher."""
        # Extract original question from first user message
        question = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break
        state["question"] = question

        state["a1"] = assistant_content
        state["a1_correct"] = self._grade(state, assistant_content)

        # Build teacher prompt — always proceed regardless of A₁ correctness
        teacher_msg = TEACHER_PROMPT.format(
            question=question,
            student_answer=assistant_content,
        )
        return False, teacher_msg, 0.0, {}

    def _handle_teacher(
        self, state: dict, assistant_content: str
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Turn 2: Teacher gave feedback. Transition to Student₂."""
        state["feedback"] = assistant_content

        student2_msg = STUDENT2_PROMPT.format(feedback=assistant_content)
        return False, student2_msg, 0.0, {}

    def _handle_student2(
        self, state: dict, assistant_content: str
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Turn 3: Student₂ revised. Compute rewards and terminate."""
        state["a2"] = assistant_content
        state["a2_correct"] = self._grade(state, assistant_content)

        teacher_reward, student2_reward = compute_self_teach_rewards(
            a1_correct=state["a1_correct"],
            a2_correct=state["a2_correct"],
        )
        state["teacher_reward"] = teacher_reward
        state["student2_reward"] = student2_reward

        metadata = {
            "teacher_reward": teacher_reward,
            "student2_reward": student2_reward,
            "a1_correct": state["a1_correct"],
            "a2_correct": state["a2_correct"],
        }

        # Always terminate after turn 3
        return True, "", 0.0, metadata

    def _grade(self, state: dict, solution_str: str) -> bool:
        """Grade a solution using the deepscaler reward function."""
        score = compute_score(
            data_source=state["data_source"],
            solution_str=solution_str,
            ground_truth=state["ground_truth"],
        )
        return score >= 0.5

    async def calculate_score(self, instance_id: str = None, **kwargs) -> float:
        # Real rewards are in metadata, consumed by the trainer during trajectory splitting
        return 0.0

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        if instance_id and instance_id in self._instances:
            del self._instances[instance_id]
