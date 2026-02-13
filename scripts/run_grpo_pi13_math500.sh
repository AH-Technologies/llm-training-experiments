#!/bin/bash
# GRPO training - adapted for 4 GH200s (3 train + 1 vLLM)
# Matching paper's effective batch size of 1024 rollouts per step
set -x

unset ROCR_VISIBLE_DEVICES

MODEL=${MODEL:-"Qwen/Qwen2.5-32B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE="${DATA_DIR}/pi13_r128.parquet"
VAL_FILE="${DATA_DIR}/math500.parquet"

REWARD_FN_PATH="src/rlvr_grokking/rewards/deepscaler_reward.py"
REWARD_FN_NAME="compute_score"

# Batch size calculation (matching paper):
# train_batch_size=128 prompts, n=8 rollouts each → 1024 rollouts per step
# mini_batch=128 → processes 128 rollouts per gradient update (1024/128 = 8 updates)
# use_dynamic_bsz=True for dynamic batching based on token length

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=${REWARD_FN_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rlvr-grokking' \
    trainer.experiment_name='grpo_pi13_math500_32B' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    "$@"