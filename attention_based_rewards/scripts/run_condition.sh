#!/bin/bash
# Run a single GRPO training condition with circuit-guided token weighting.
#
# Usage:
#   bash attention_based_rewards/scripts/run_condition.sh uniform
#   bash attention_based_rewards/scripts/run_condition.sh attention
#   bash attention_based_rewards/scripts/run_condition.sh entropy
#   bash attention_based_rewards/scripts/run_condition.sh combined
#
# Environment variables:
#   MODEL       - HF model path (default: Qwen/Qwen2.5-Math-1.5B)
#   TOTAL_STEPS - training steps (default: 500)
#   DRY_RUN     - if "1", run only 2 steps
set -x

CONDITION=${1:?Usage: run_condition.sh <uniform|attention|entropy|combined>}
shift  # Remove condition from $@ so remaining args pass through to verl

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
TOTAL_STEPS=${TOTAL_STEPS:-500}
REWARD_FN_PATH="src/rlvr_grokking/rewards/verl_reward.py"
REWARD_FN_NAME="compute_score"

# Dry run override
if [ "${DRY_RUN}" = "1" ]; then
    TOTAL_STEPS=2
    echo "=== DRY RUN: 2 steps only ==="
fi

# Custom args for our wrapper
CUSTOM_ARGS="--condition ${CONDITION} --model_name ${MODEL}"

# Conditions that need reasoning heads
case "${CONDITION}" in
    attention|combined|fai|asymmetric)
        CUSTOM_ARGS="${CUSTOM_ARGS} --reasoning_heads attention_based_rewards/results/reasoning_heads.pt --top_k_heads 10"
        ;;
    attention_top5)
        CUSTOM_ARGS="${CUSTOM_ARGS} --reasoning_heads attention_based_rewards/results/reasoning_heads.pt --top_k_heads 5"
        ;;
esac

# verl expects to be run from project root
cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

python3 -m attention_based_rewards.scripts.train \
    ${CUSTOM_ARGS} \
    algorithm.adv_estimator=grpo \
    data.train_files=attention_based_rewards/data/gsm8k_train.parquet \
    data.val_files=data/math500.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
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
    trainer.project_name='attention-grpo' \
    trainer.experiment_name="grpo_${CONDITION}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
