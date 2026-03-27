#!/bin/bash
# Submit 4 Qwen3-4B-Base runs to SLURM.
# 1. Base GRPO (entropy=0)
# 2. Base GRPO + entropy 0.001
# 3. Attention Illuminates (static rhythm)
# 4. Attention Illuminates + rediscovery (adaptive rhythm)

set -e

SLURM_SCRIPT=attention_sparks_thinking/slurm/train.slurm
EXCLUDE=""
MODEL="Qwen/Qwen3-4B-Base"

# Common overrides (applied AFTER defaults in run_condition.sh)
# 64 prompts × 8 rollouts = 512 train_batch_size (ROLL default group size = 8)
COMMON="actor_rollout_ref.actor.optim.lr=1e-6 data.train_batch_size=512 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 actor_rollout_ref.actor.optim.weight_decay=0 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean data.max_response_length=1024 +algorithm.advantage_clip=2.0 +algorithm.reward_clip=10 +algorithm.max_len_mask=True trainer.project_name=attention-illuminates-qwen3"

echo "Submitting 4 Qwen3-4B-Base runs (v6: ROLL-matched loss_agg + adv_clip)..."

# 1. Base GRPO (entropy=0)
echo "  1. Base GRPO (entropy=0)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=A,MODEL=$MODEL,EXP_SUFFIX=_qwen3_base_v6,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# 2. Base GRPO + entropy 0.001
echo "  2. Base GRPO + entropy 0.001"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=A,MODEL=$MODEL,EXP_SUFFIX=_qwen3_entropy_v6,EXTRA_OVERRIDES="$COMMON actor_rollout_ref.actor.entropy_coeff=0.001" \
  $SLURM_SCRIPT

# 3. Attention Illuminates (static rhythm, entropy=0)
echo "  3. Attention Illuminates (static)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=B,MODEL=$MODEL,EXP_SUFFIX=_qwen3_illuminates_v6,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# 4. Attention Illuminates + rediscovery (adaptive, entropy=0)
echo "  4. Attention Illuminates + rediscovery"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=C,MODEL=$MODEL,EXP_SUFFIX=_qwen3_rediscovery_v6,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

echo ""
echo "All 4 runs submitted. Monitor with: squeue -u \$USER"
echo "Wandb project: attention-illuminates-qwen3"
