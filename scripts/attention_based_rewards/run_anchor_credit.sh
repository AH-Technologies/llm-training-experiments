#!/bin/bash
# Submit anchor_credit experiments: 2 models × 2 conditions = 4 runs.
#
# Run A1: Instruct + uniform (baseline)
# Run A2: Instruct + anchor_credit
# Run B1: Base + uniform (baseline)
# Run B2: Base + anchor_credit
#
# All use DAPO-Math-17k with v2 hyperparameters (lr=1e-5, bs=64, temp=0.7, 500 steps).

set -e

echo "Submitting anchor_credit experiments (4 runs)..."
echo ""

INSTRUCT_MODEL="/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
BASE_MODEL="/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"

INSTRUCT_HEADS="results/attention_based_rewards/reasoning_heads.pt"
BASE_HEADS="attention_based_rewards/data/base_model_reasoning_heads.pt"

OVERRIDES="actor_rollout_ref.actor.optim.lr=1e-5 data.train_batch_size=64 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.rollout.temperature=0.7"

echo "Run A1: Instruct + uniform"
sbatch --export=ALL,CONDITION=uniform,TOTAL_STEPS=500,MODEL=${INSTRUCT_MODEL},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_anchor_exp \
    scripts/attention_based_rewards/slurm/train_dapo.slurm

echo "Run A2: Instruct + anchor_credit"
sbatch --export=ALL,CONDITION=anchor_credit,TOTAL_STEPS=500,MODEL=${INSTRUCT_MODEL},REASONING_HEADS_PATH=${INSTRUCT_HEADS},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_anchor_exp \
    scripts/attention_based_rewards/slurm/train_dapo.slurm

echo "Run B1: Base + uniform"
sbatch --export=ALL,CONDITION=uniform,TOTAL_STEPS=500,MODEL=${BASE_MODEL},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_base_anchor_exp \
    scripts/attention_based_rewards/slurm/train_dapo.slurm

echo "Run B2: Base + anchor_credit"
sbatch --export=ALL,CONDITION=anchor_credit,TOTAL_STEPS=500,MODEL=${BASE_MODEL},REASONING_HEADS_PATH=${BASE_HEADS},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_base_anchor_exp \
    scripts/attention_based_rewards/slurm/train_dapo.slurm

echo ""
echo "All 4 runs submitted. Monitor with: squeue -u \$USER"
