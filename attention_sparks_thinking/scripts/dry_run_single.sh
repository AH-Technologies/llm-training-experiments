#!/bin/bash
# Dry-run a SINGLE condition (A, B, or C) for 5 training steps.
# Must be run in its own SLURM job to avoid GPU resource conflicts.
#
# Usage:
#   Called by dry_run.slurm with RUN_TYPE env var set.

set -uo pipefail

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd "$PROJECT_DIR"

RUN_TYPE=${RUN_TYPE:?ERROR: RUN_TYPE not set}
MODEL="Qwen/Qwen2.5-Math-1.5B"
TRAIN_DATA="attention_based_rewards/data/dapo_math_17k.parquet"
REWARD_FN_PATH="src/rlvr_grokking/rewards/verl_reward.py"
REWARD_FN_NAME="compute_score"

DRY_RUN_STEPS=5
DRY_BATCH_SIZE=32
DRY_MAX_RESP_LEN=512
DRY_MAX_PROMPT_LEN=512
DRY_SAVE_FREQ=5

# Prepare data symlink
python attention_sparks_thinking/scripts/prepare_data.py

echo "======================================================"
echo "  DRY RUN: Condition $RUN_TYPE ($DRY_RUN_STEPS steps)"
echo "======================================================"

# For Run C, use reclassify_K=3 so reclassification fires at step 3
EXTRA_CUSTOM=""
if [ "$RUN_TYPE" = "C" ]; then
    EXTRA_CUSTOM="--reclassify_K 3"
fi

python3 -m attention_sparks_thinking.scripts.train \
    --run_type "$RUN_TYPE" \
    --model_name "$MODEL" \
    --num_class_prompts 20 \
    $EXTRA_CUSTOM \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files='[data/math500.parquet]' \
    data.train_batch_size="$DRY_BATCH_SIZE" \
    data.val_batch_size=50 \
    data.max_prompt_length="$DRY_MAX_PROMPT_LEN" \
    data.max_response_length="$DRY_MAX_RESP_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$DRY_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=True \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name="$REWARD_FN_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='attention-sparks-thinking' \
    trainer.experiment_name="dryrun_${RUN_TYPE}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq="$DRY_SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.val_before_train=False \
    trainer.test_freq=0 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps="$DRY_RUN_STEPS"

EXIT_CODE=$?

echo ""
echo "--- Condition $RUN_TYPE finished with exit code: $EXIT_CODE ---"

# Print rhythm logs
if [ "$RUN_TYPE" != "A" ]; then
    JOB_ID=${SLURM_JOB_ID:-default}
    echo ""
    for LOG_TYPE in driver worker trainer; do
        LOG_FILE="attention_sparks_thinking/logs/rhythm_${LOG_TYPE}_${JOB_ID}.log"
        if [ -f "$LOG_FILE" ]; then
            echo "=== Rhythm $LOG_TYPE log ==="
            cat "$LOG_FILE"
        fi
    done
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "PASS: Condition $RUN_TYPE"
else
    echo "FAIL: Condition $RUN_TYPE"
fi
exit $EXIT_CODE
