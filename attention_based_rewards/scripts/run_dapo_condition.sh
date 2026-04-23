#!/bin/bash
# Run a single GRPO training condition on DAPO-Math-17k.
#
# Usage:
#   bash attention_based_rewards/scripts/run_dapo_condition.sh uniform
#   bash attention_based_rewards/scripts/run_dapo_condition.sh entropy
#   bash attention_based_rewards/scripts/run_dapo_condition.sh fai_allheads
#   bash attention_based_rewards/scripts/run_dapo_condition.sh fai
#   bash attention_based_rewards/scripts/run_dapo_condition.sh fai_asymmetric
#
# Environment variables:
#   MODEL       - HF model path (default: Qwen/Qwen2.5-Math-1.5B-Instruct)
#   TOTAL_STEPS - training steps (default: 250)
#   DRY_RUN     - if "1", run only 2 steps
set -x

CONDITION=${1:?Usage: run_dapo_condition.sh <uniform|entropy|fai_allheads|fai|fai_asymmetric|anchor_credit>}
shift  # Remove condition from $@ so remaining args pass through to verl

MODEL=${MODEL:-"/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"}
TOTAL_STEPS=${TOTAL_STEPS:-250}
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
REASONING_HEADS_PATH=${REASONING_HEADS_PATH:-"attention_based_rewards/results/reasoning_heads.pt"}
case "${CONDITION}" in
    fai|fai_asymmetric|attention|combined|asymmetric|anchor_credit|circuit_reward|mlp_circuit_reward)
        CUSTOM_ARGS="${CUSTOM_ARGS} --reasoning_heads ${REASONING_HEADS_PATH} --top_k_heads ${TOP_K_HEADS:-10}"
        ;;
    attention_top5)
        CUSTOM_ARGS="${CUSTOM_ARGS} --reasoning_heads ${REASONING_HEADS_PATH} --top_k_heads 5"
        ;;
esac

# Anchor percentile (only relevant for anchor_credit)
if [ -n "${ANCHOR_PERCENTILE:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --anchor_percentile ${ANCHOR_PERCENTILE}"
fi

# Alpha for blending (circuit_reward or combined)
if [ -n "${ALPHA:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --alpha ${ALPHA}"
fi

# Likelihood bonus params
if [ -n "${LIKELIHOOD_LAMBDA:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --likelihood_lambda ${LIKELIHOOD_LAMBDA}"
fi
if [ -n "${LIKELIHOOD_BETA:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --likelihood_beta ${LIKELIHOOD_BETA}"
fi

# Suffix mode (none/step/random)
if [ -n "${SUFFIX_MODE:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --suffix_mode ${SUFFIX_MODE}"
fi

# verl expects to be run from project root
cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

python3 -m attention_based_rewards.scripts.train \
    ${CUSTOM_ARGS} \
    algorithm.adv_estimator=grpo \
    data.train_files=attention_based_rewards/data/dapo_math_17k.parquet \
    data.val_files='[data/math500.parquet,attention_based_rewards/data/aime_2025.parquet,attention_based_rewards/data/amc_2023.parquet]' \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
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
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=${REWARD_FN_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='dapo-grpo' \
    trainer.experiment_name="dapo_${CONDITION}${EXP_SUFFIX:-}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    ${EXTRA_OVERRIDES:-} \
    "$@"
