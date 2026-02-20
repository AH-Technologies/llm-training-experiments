#!/bin/bash
# GRPO training - matching One-Shot-RLVR paper config (verl v0.2)
# Adapted for 4x GH200 (96GB GPU, 120GB CPU RAM each)
#
# Paper: "Reinforcement Learning for Reasoning in LLMs with One Training Example"
# Repo: https://github.com/ypwang61/One-Shot-RLVR
#
# Key params from paper's wandb config:
#   train_batch_size=128, n=8 → 1024 rollouts per step
#   mini_batch=128 → 8 gradient updates per step
#   use_dynamic_bsz=True, ppo_max_token_len_per_gpu=24000
#   entropy_coeff=0.001 (enabled by default in paper's verl)
#   lr=1e-6, kl_loss_coef=0.001, temperature=0.6
set -x

unset ROCR_VISIBLE_DEVICES

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE="${DATA_DIR}/pi1_r128.parquet"

# Multi-prompt validation: evaluate with 3 prompting strategies (no CoT, train CoT, Qwen CoT)
# Set MULTI_PROMPT_VAL=1 to enable. Logs separate wandb metrics per prompt style.
MULTI_PROMPT_VAL=${MULTI_PROMPT_VAL:-0}

if [ "${MULTI_PROMPT_VAL}" = "1" ]; then
    VAL_FILE="${DATA_DIR}/math500_multi_prompt.parquet"
    VAL_BATCH_SIZE=1500  # 500 questions × 3 prompt styles
    # Always regenerate to pick up script changes
    echo "Generating multi-prompt validation file..."
    python3 scripts/prepare_multi_prompt_val.py \
        --input "${DATA_DIR}/math500.parquet" \
        --output "${VAL_FILE}"
else
    VAL_FILE="${DATA_DIR}/math500.parquet"
    VAL_BATCH_SIZE=500
fi

REWARD_FN_PATH="src/rlvr_grokking/rewards/deepscaler_reward.py"
REWARD_FN_NAME="compute_score"

python3 -m verl.trainer.main_ppo \
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
    trainer.experiment_name='grpo_pi1_math500_lr_increased_multi_val' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    "$@"
