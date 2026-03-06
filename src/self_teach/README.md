# self_teach

Student-teacher self-play GRPO training. A single model plays both roles in a 3-turn interaction, and both roles are trained via GRPO with separate reward signals.

## How it works

Each training step runs a 3-turn conversation:

1. **Student₁** answers a math question
2. **Teacher** receives the question, the student's attempt, and the ground truth — gives feedback without revealing the answer
3. **Student₂** revises based on the feedback

## 2-phase generation

Credit assignment is handled by branching at the right point for each role:

- **Phase 1 (teacher training):** Generate one A₁ per prompt, then branch `n` different feedback samples (all see the same A₁). Teacher reward depends on whether A₂ improves.
- **Phase 2 (student₂ training):** Fix one feedback per prompt, then branch `n` different A₂ samples (all see the same A₁ + F). Student₂ reward depends on A₂ correctness.

## Reward table

| A₁ | A₂ | Teacher | Student₂ |
|----|-----|---------|----------|
| wrong | correct | +1.0 | +1.0 |
| wrong | wrong | 0.0 | 0.0 |
| correct | correct | +1.0 | +1.0 |
| correct | wrong | -1.0 | 0.0 |

## Files

- `main.py` — Entry point (`python -m src.self_teach.main`)
- `trainer.py` — 2-phase trainer and task runner (subclasses verl's `RayPPOTrainer`)
- `interaction.py` — 3-turn interaction for verl's multi-turn system
- `prompts.py` — Prompt templates with XML-tagged sections
- `rewards.py` — Reward logic per role

Run via: `scripts/run_grpo_self_teach.sh`
