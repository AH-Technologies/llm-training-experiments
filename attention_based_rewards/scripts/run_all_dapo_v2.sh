#!/bin/bash
# Submit all 5 DAPO conditions with v2 hyperparameters:
#   LR=1e-5, 500 steps, batch=64, temp=0.7
#
# Experiment names suffixed with _v2 to avoid checkpoint conflicts.

set -e

echo "Submitting 5 DAPO-Math-17k GRPO conditions (v2 params: lr=1e-5, bs=64, temp=0.7, 500 steps)..."
echo ""

OVERRIDES="actor_rollout_ref.actor.optim.lr=1e-5 data.train_batch_size=64 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.rollout.temperature=0.7"

for COND in uniform entropy fai_allheads fai fai_asymmetric; do
    echo "Submitting condition: ${COND}"
    sbatch --export=ALL,CONDITION=${COND},TOTAL_STEPS=500,EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_v2 attention_based_rewards/slurm/train_dapo.slurm
done

echo ""
echo "All 5 v2 conditions submitted. Monitor with: squeue -u \$USER"
