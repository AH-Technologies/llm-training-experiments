#!/bin/bash
# Submit all 5 DAPO conditions for the BASE model with v2 hyperparameters:
#   Model: Qwen2.5-Math-1.5B (base, NOT Instruct)
#   LR=1e-5, 500 steps, batch=64, temp=0.7
#   Reasoning heads from base model circuit discovery
#
# Experiment names: dapo_base_{condition}_v2

set -e

echo "Submitting 5 DAPO-Math-17k GRPO conditions (BASE model, v2 params)..."
echo "  Model: Qwen/Qwen2.5-Math-1.5B"
echo "  LR=1e-5, batch=64, temp=0.7, 500 steps"
echo ""

BASE_MODEL="/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
HEADS_PATH="attention_based_rewards/data/base_model_reasoning_heads.pt"
OVERRIDES="actor_rollout_ref.actor.optim.lr=1e-5 data.train_batch_size=64 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.rollout.temperature=0.7"

for COND in uniform entropy fai_allheads fai fai_asymmetric; do
    echo "Submitting condition: ${COND}"
    sbatch --export=ALL,CONDITION=${COND},TOTAL_STEPS=500,MODEL=${BASE_MODEL},REASONING_HEADS_PATH=${HEADS_PATH},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_base_v2 \
        attention_based_rewards/slurm/train_dapo.slurm
done

echo ""
echo "All 5 base model v2 conditions submitted. Monitor with: squeue -u \$USER"
