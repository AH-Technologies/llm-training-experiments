"""Self-Play Teaching (SPT) core logic.

SPT trains a single model to alternate between student and teacher roles:
- Student steps: model learns to solve math problems using teacher feedback
- Teacher steps: model learns to give useful corrective feedback

Uses GRPO for optimization within the verl framework.
"""

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto


# ---------------------------------------------------------------------------
# System prompts (configurable defaults)
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = (
    "You are a math student. Solve the problem step by step "
    "and provide your final answer within \\boxed{}."
)

TEACHER_SYSTEM_PROMPT = (
    "You are a math teacher reviewing a student's work. "
    "You know the correct answer. Provide constructive feedback "
    "pointing out errors in their reasoning and guiding them "
    "toward the right approach. Do NOT state the final answer directly."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_student_turn1_messages(question: str, student_system_prompt: str) -> list[dict]:
    """Build message list for a student's first attempt at a problem."""
    return [
        {"role": "system", "content": student_system_prompt},
        {"role": "user", "content": question},
    ]


def build_teacher_messages(
    question: str,
    student_answer: str,
    ground_truth: str,
    teacher_system_prompt: str,
) -> list[dict]:
    """Build message list for a teacher reviewing student work."""
    return [
        {"role": "system", "content": teacher_system_prompt},
        {"role": "user", "content": (
            f"Question:\n{question}\n\n"
            f"Correct answer: {ground_truth}\n\n"
            f"Student's answer:\n{student_answer}\n\n"
            "Provide feedback to help the student improve their reasoning."
        )},
    ]


def build_student_turn2_messages(
    question: str,
    student_answer: str,
    teacher_feedback: str,
    student_system_prompt: str,
) -> list[dict]:
    """Build message list for a student's revised attempt after feedback."""
    return [
        {"role": "system", "content": student_system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": student_answer},
        {"role": "user", "content": (
            f"Your teacher provided feedback:\n{teacher_feedback}\n\n"
            "Please revise your answer based on this feedback."
        )},
    ]


# ---------------------------------------------------------------------------
# DataProto construction for generate_sequences
# ---------------------------------------------------------------------------

def build_gen_batch_from_messages(
    messages_list: list[list[dict]],
    tokenizer,
    max_prompt_length: int,
    temperature: float = 0.6,
) -> DataProto:
    """Tokenize message lists and build a DataProto for generate_sequences.

    1. Apply chat_template to each message list (add_generation_prompt=True)
    2. Tokenize -> input_ids per sample
    3. Left-pad to max_prompt_length
    4. Build attention_mask (1 for real tokens, 0 for padding)
    5. Build position_ids (cumsum of attention_mask - 1, clamped to 0)
    6. Return DataProto with batch tensors + meta_info
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    all_input_ids = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Truncate from the left if too long
        if len(tokens) > max_prompt_length:
            tokens = tokens[-max_prompt_length:]
        all_input_ids.append(torch.tensor(tokens, dtype=torch.long))

    # Left-pad to max_prompt_length
    batch_size = len(all_input_ids)
    padded_input_ids = torch.full(
        (batch_size, max_prompt_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros(batch_size, max_prompt_length, dtype=torch.long)

    for i, ids in enumerate(all_input_ids):
        seq_len = ids.shape[0]
        padded_input_ids[i, max_prompt_length - seq_len:] = ids
        attention_mask[i, max_prompt_length - seq_len:] = 1

    # Position IDs: cumulative sum of attention_mask - 1, clamped to 0
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    batch = TensorDict(
        {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    return DataProto(
        batch=batch,
        non_tensor_batch={},
        meta_info={
            "temperature": temperature,
            "do_sample": True,
        },
    )


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

def decode_responses(gen_output: DataProto, tokenizer) -> list[str]:
    """Decode response tokens from generation output to text strings.

    For each sample in gen_output, extract response tokens, mask padding,
    and decode with skip_special_tokens=True.
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    responses = gen_output.batch["responses"]  # [B, response_len]
    texts = []
    for i in range(responses.shape[0]):
        resp_tokens = responses[i]
        # Mask out padding tokens
        valid_mask = resp_tokens != pad_token_id
        valid_tokens = resp_tokens[valid_mask]
        text = tokenizer.decode(valid_tokens.tolist(), skip_special_tokens=True)
        texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Teacher reward computation
# ---------------------------------------------------------------------------

def compute_teacher_reward(turn1_correct: bool, turn2_correct: bool) -> float:
    """Compute reward for teacher feedback based on student score transitions.

    Returns:
        1.0  if wrong -> right  (teacher helped fix the error)
        0.5  if correct -> correct  (teacher didn't break anything)
       -1.0  if correct -> wrong  (teacher caused regression)
        0.0  if wrong -> wrong  (teacher didn't help)
    """
    if not turn1_correct and turn2_correct:
        return 1.0
    elif turn1_correct and turn2_correct:
        return 0.5
    elif turn1_correct and not turn2_correct:
        return -1.0
    else:
        return 0.0


# ---------------------------------------------------------------------------
# GRPO batch construction
# ---------------------------------------------------------------------------

def build_grpo_batch(
    prompt_token_ids: list[torch.Tensor],
    response_token_ids: list[torch.Tensor],
    rewards: list[float],
    uids: list[str],
    pad_token_id: int,
    max_prompt_length: int,
    max_response_length: int,
) -> tuple[DataProto, torch.Tensor]:
    """Build a complete DataProto batch + reward tensor for GRPO training.

    1. Left-pad prompt tokens to max_prompt_length
    2. Right-pad response tokens to max_response_length
    3. Concatenate -> input_ids [B, prompt_len + response_len]
    4. Build attention_mask, position_ids, response_mask
    5. Build reward_tensor: [B, response_len] with reward at last valid token
    6. Set uid in non_tensor_batch for GRPO grouping
    7. Return (DataProto, reward_tensor)
    """
    batch_size = len(prompt_token_ids)
    total_len = max_prompt_length + max_response_length

    # Initialize tensors
    input_ids = torch.full((batch_size, total_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long)
    prompts = torch.full((batch_size, max_prompt_length), pad_token_id, dtype=torch.long)
    responses = torch.full((batch_size, max_response_length), pad_token_id, dtype=torch.long)
    reward_tensor = torch.zeros(batch_size, max_response_length, dtype=torch.float32)

    for i in range(batch_size):
        p_ids = prompt_token_ids[i]
        r_ids = response_token_ids[i]

        # Truncate if needed
        if p_ids.shape[0] > max_prompt_length:
            p_ids = p_ids[-max_prompt_length:]
        if r_ids.shape[0] > max_response_length:
            r_ids = r_ids[:max_response_length]

        p_len = p_ids.shape[0]
        r_len = r_ids.shape[0]

        # Left-pad prompt
        prompts[i, max_prompt_length - p_len:] = p_ids
        # Right-pad response
        responses[i, :r_len] = r_ids

        # Concatenate into input_ids
        input_ids[i, max_prompt_length - p_len:max_prompt_length] = p_ids
        input_ids[i, max_prompt_length:max_prompt_length + r_len] = r_ids

        # Attention mask
        attention_mask[i, max_prompt_length - p_len:max_prompt_length + r_len] = 1

        # Reward at last valid response token
        if r_len > 0:
            reward_tensor[i, r_len - 1] = rewards[i]

    # Position IDs
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    # Response mask (attention mask for response portion)
    response_mask = attention_mask[:, max_prompt_length:]

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
        },
        batch_size=batch_size,
    )

    non_tensor_batch = {
        "uid": np.array(uids, dtype=object),
    }

    data_proto = DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={},
    )

    return data_proto, reward_tensor
