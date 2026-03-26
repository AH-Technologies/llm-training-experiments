#!/bin/bash
# Dry-run: 5 training steps for each condition (A, B, C) to verify
# the full pipeline works end-to-end.
#
# Checks:
#   - Training doesn't crash
#   - Run B/C: gamma stats printed (mean, std, frac > 1.0)
#   - Run C: reclassification code path executes without error
#   - Loss values are reasonable (not NaN, not zero)
#   - Checkpoint can be saved and loaded
#
# Usage (on compute node with 4 GPUs):
#   bash attention_sparks_thinking/scripts/dry_run.sh
#
# Or via SLURM:
#   sbatch attention_sparks_thinking/slurm/dry_run.slurm

set -uo pipefail

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd "$PROJECT_DIR"

MODEL="Qwen/Qwen2.5-Math-1.5B"
TRAIN_DATA="attention_based_rewards/data/dapo_math_17k.parquet"
REWARD_FN_PATH="src/rlvr_grokking/rewards/verl_reward.py"
REWARD_FN_NAME="compute_score"

# Prepare data symlink
python attention_sparks_thinking/scripts/prepare_data.py

DRY_RUN_STEPS=5
# Use smaller batch size and shorter sequences for speed
DRY_BATCH_SIZE=32
DRY_MAX_RESP_LEN=512
DRY_MAX_PROMPT_LEN=512
# Save a checkpoint at step 5 to verify it works
DRY_SAVE_FREQ=5

echo "============================================"
echo "ATTENTION SPARKS DRY RUN — $DRY_RUN_STEPS steps each"
echo "============================================"
echo "Batch size: $DRY_BATCH_SIZE"
echo "Max response length: $DRY_MAX_RESP_LEN"
echo "Steps per condition: $DRY_RUN_STEPS"
echo ""

cleanup_ray() {
    echo "Cleaning up Ray and GPU processes..."
    ray stop --force 2>/dev/null || true
    # Kill any leftover Python/Ray processes holding GPUs
    pkill -f "ray::" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    sleep 5
    # Reset CUDA context
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo "Cleanup done."
}

run_condition() {
    local RUN_TYPE=$1
    echo ""
    echo "======================================================"
    echo "  DRY RUN: Condition $RUN_TYPE"
    echo "======================================================"

    # For Run C, use reclassify_K=3 so the reclassification fires at step 3
    local EXTRA_CUSTOM=""
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

    local EXIT_CODE=$?

    echo ""
    echo "--- Condition $RUN_TYPE finished with exit code: $EXIT_CODE ---"

    # Check rhythm logs if B or C
    if [ "$RUN_TYPE" != "A" ]; then
        echo ""
        echo "--- Rhythm driver log (Condition $RUN_TYPE) ---"
        local JOB_ID=${SLURM_JOB_ID:-default}
        local DRIVER_LOG="attention_sparks_thinking/logs/rhythm_driver_${JOB_ID}.log"
        local WORKER_LOG="attention_sparks_thinking/logs/rhythm_worker_${JOB_ID}.log"
        local TRAINER_LOG="attention_sparks_thinking/logs/rhythm_trainer_${JOB_ID}.log"
        if [ -f "$DRIVER_LOG" ]; then
            echo "=== Driver log ==="
            cat "$DRIVER_LOG"
        else
            echo "WARNING: No driver log at $DRIVER_LOG"
        fi
        if [ -f "$WORKER_LOG" ]; then
            echo "=== Worker log ==="
            cat "$WORKER_LOG"
        else
            echo "WARNING: No worker log at $WORKER_LOG"
        fi
        if [ -f "$TRAINER_LOG" ]; then
            echo "=== Trainer log ==="
            cat "$TRAINER_LOG"
        else
            echo "WARNING: No trainer log at $TRAINER_LOG"
        fi
        echo "--- End rhythm logs ---"
    fi

    return $EXIT_CODE
}

# Run all 3 conditions sequentially, with cleanup between each
FAILED=0
for RUN in A B C; do
    if run_condition "$RUN"; then
        echo "PASS: Condition $RUN completed successfully"
    else
        echo "FAIL: Condition $RUN crashed!"
        FAILED=1
    fi
    echo ""
    # Critical: clean up Ray and GPU processes before next condition
    cleanup_ray
done

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "ALL DRY RUNS PASSED"
else
    echo "SOME DRY RUNS FAILED — check logs above"
fi
echo "============================================"
