#!/bin/bash
# Run a single attention-rhythm GRPO training condition.
#
# Usage:
#   bash attention_sparks_thinking/scripts/run_condition.sh A   # baseline
#   bash attention_sparks_thinking/scripts/run_condition.sh B   # static rhythm
#   bash attention_sparks_thinking/scripts/run_condition.sh C   # adaptive rhythm
#
# Environment variables:
#   MODEL       - HF model path (default: Qwen/Qwen2.5-Math-1.5B)
#   TOTAL_STEPS - training steps (default: 500)
#   DRY_RUN     - if "1", run only 2 steps with diagnostics
set -x

RUN_TYPE=${1:?Usage: run_condition.sh <A|B|C|D|E|F|G|H|I|J>}
shift

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
CUSTOM_ARGS="--run_type ${RUN_TYPE} --model_name ${MODEL}"

if [ "${DRY_RUN}" = "1" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --dry_run"
fi

# EAP-IG reasoning heads path for methods D/E/G/H/I
REASONING_HEADS_PATH=${REASONING_HEADS_PATH:-"attention_sparks_thinking/logs/head_importance_qwen3.pt"}
NUM_REASONING_HEADS=${NUM_REASONING_HEADS:-200}
if [[ "${RUN_TYPE}" =~ ^[DEGHI]$ ]]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --reasoning_heads_path ${REASONING_HEADS_PATH} --num_reasoning_heads ${NUM_REASONING_HEADS}"
fi

# Positive rollouts only (apply gamma weighting only to correct responses)
if [ "${POSITIVE_ROLLOUTS_ONLY:-}" = "1" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --positive_rollouts_only"
fi
# EAP-IG rediscovery (optional, set EAPIG_REDISCOVERY_K>0 to enable)
if [ -n "${EAPIG_REDISCOVERY_K:-}" ] && [ "${EAPIG_REDISCOVERY_K}" != "0" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --eapig_rediscovery_K ${EAPIG_REDISCOVERY_K}"
    if [ -n "${EAPIG_REDISCOVERY_PROBLEMS:-}" ]; then
        CUSTOM_ARGS="${CUSTOM_ARGS} --eapig_rediscovery_problems ${EAPIG_REDISCOVERY_PROBLEMS}"
    fi
fi

# Rhythm hyperparams (override via env)
if [ -n "${GAMMA_AMP:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --gamma_amp ${GAMMA_AMP}"
fi
if [ -n "${GAMMA_MODE:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --gamma_mode ${GAMMA_MODE}"
fi
if [ -n "${ALPHA:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --alpha ${ALPHA}"
fi
if [ -n "${RECLASSIFY_K:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --reclassify_K ${RECLASSIFY_K}"
fi
if [ -n "${HEAD_QUANTILE:-}" ]; then
    CUSTOM_ARGS="${CUSTOM_ARGS} --head_quantile ${HEAD_QUANTILE}"
fi

# Data paths — use existing DAPO data or symlinked copy
TRAIN_DATA="attention_based_rewards/data/dapo_math_17k.parquet"
if [ -f "attention_sparks_thinking/data/dapo_math_17k.parquet" ]; then
    TRAIN_DATA="attention_sparks_thinking/data/dapo_math_17k.parquet"
fi

cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

python3 -m attention_sparks_thinking.scripts.train \
    ${CUSTOM_ARGS} \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files='[data/math500.parquet,attention_based_rewards/data/aime_2025.parquet,attention_based_rewards/data/amc_2023.parquet]' \
    data.train_batch_size=64 \
    data.val_batch_size=500 \
    data.max_prompt_length=256 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=False \
    custom_reward_function.path=${REWARD_FN_PATH} \
    custom_reward_function.name=${REWARD_FN_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='attention-sparks-thinking' \
    trainer.experiment_name="rhythm_run${RUN_TYPE}${EXP_SUFFIX:-}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=${NNODES:-2} \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    ${EXTRA_OVERRIDES:-} \
    "$@"
