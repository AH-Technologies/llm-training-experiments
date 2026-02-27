#!/bin/bash
# GRPO training with Self-Play Teaching (SPT)
#
# SPT alternates between student and teacher steps:
# - Student steps (odd): model learns to solve math problems using teacher feedback
# - Teacher steps (even): model learns to give useful corrective feedback
#
# Each step involves 3 sequential generation passes.
# rollout.n=1 because we handle repetition ourselves in _spt_training_step.
set -x

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE="${DATA_DIR}/pi13_r128.parquet"

# SPT config
SPT_N_ROLLOUTS=${SPT_N_ROLLOUTS:-8}
SPT_TEMPERATURE=${SPT_TEMPERATURE:-0.6}

# Multi-prompt validation
MULTI_PROMPT_VAL=${MULTI_PROMPT_VAL:-0}

if [ "${MULTI_PROMPT_VAL}" = "1" ]; then
    VAL_FILE="${DATA_DIR}/math500_multi_prompt.parquet"
    VAL_BATCH_SIZE=1500
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

python3 -m src.rlvr_grokking.spt.main_spt \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=4096 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${SPT_TEMPERATURE} \
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
    trainer.experiment_name='grpo_pi13_spt' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    +spt.enabled=True \
    +spt.n_rollouts=${SPT_N_ROLLOUTS} \
    +spt.temperature=${SPT_TEMPERATURE} \
    "$@"
