#!/bin/bash
# GRPO training with Bidirectional Experience Replay (BER)
#
# Based on run_grpo_pi13_math500_v2.sh with BER injection:
# - Phase 1 (all-incorrect): inject cached correct response
# - Phase 2 (mixed): normal GRPO + cache incorrect rollout
# - Phase 3 (all-correct): inject cached incorrect response
#
# Prerequisites:
#   python scripts/rlvr_grokking/generate_correct_cache.py \
#       --model Qwen/Qwen2.5-Math-1.5B \
#       --train_file data/pi13_r128.parquet \
#       --output data/ber_correct_cache_pi13.pt
set -x


MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE="${DATA_DIR}/pi13_r128.parquet"

# BER config
BER_CORRECT_CACHE=${BER_CORRECT_CACHE:-"${DATA_DIR}/ber_correct_cache_pi13.pt"}
BER_MAX_ERROR_CACHE_AGE=${BER_MAX_ERROR_CACHE_AGE:-500}
BER_BUFFER_SIZE=${BER_BUFFER_SIZE:-32}
BER_INJECTION_FRACTION=${BER_INJECTION_FRACTION:-0.1}

# Tighter PPO clip ratio to prevent large policy steps with BER injection
PPO_CLIP_RATIO=${PPO_CLIP_RATIO:-0.1}

# Advantage clamping (mitigates policy collapse from extreme BER advantages)
BER_ADV_CLAMP_ENABLED=${BER_ADV_CLAMP_ENABLED:-False}
BER_ADV_CLAMP_MIN=${BER_ADV_CLAMP_MIN:--1.0}
BER_ADV_CLAMP_MAX=${BER_ADV_CLAMP_MAX:-1.0}

# Multi-prompt validation
MULTI_PROMPT_VAL=${MULTI_PROMPT_VAL:-0}

if [ "${MULTI_PROMPT_VAL}" = "1" ]; then
    VAL_FILE="${DATA_DIR}/math500_multi_prompt.parquet"
    VAL_BATCH_SIZE=1500
    echo "Generating multi-prompt validation file..."
    python3 scripts/rlvr_grokking/prepare_multi_prompt_val.py \
        --input "${DATA_DIR}/math500.parquet" \
        --output "${VAL_FILE}"
else
    VAL_FILE="${DATA_DIR}/math500.parquet"
    VAL_BATCH_SIZE=500
fi

REWARD_FN_PATH="src/rlvr_grokking/rewards/deepscaler_reward.py"
REWARD_FN_NAME="compute_score"

python3 -m src.rlvr_grokking.ber.main_ber \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.clip_ratio=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_low=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_RATIO} \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=${REWARD_FN_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rlvr-grokking' \
    trainer.experiment_name='grpo_pi13_ber' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    +ber.enabled=True \
    +ber.correct_cache_path=${BER_CORRECT_CACHE} \
    +ber.max_error_cache_age=${BER_MAX_ERROR_CACHE_AGE} \
    +ber.buffer_size=${BER_BUFFER_SIZE} \
    +ber.injection_fraction=${BER_INJECTION_FRACTION} \
    +ber.adv_clamp_enabled=${BER_ADV_CLAMP_ENABLED} \
    +ber.adv_clamp_min=${BER_ADV_CLAMP_MIN} \
    +ber.adv_clamp_max=${BER_ADV_CLAMP_MAX} \
    "$@"
