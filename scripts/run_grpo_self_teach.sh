#!/bin/bash
# Self-Teach: tree-structured student-teacher self-play GRPO training
#
# 3-step generation per training step:
#   1. Generate A₁ (student answers question)
#   2. Generate F (k teacher feedbacks per prompt — all see same A₁)
#   3. Generate A₂ (m responses per feedback — dual purpose: grade teacher + train student₂)
#
# Teacher GRPO groups: k feedbacks per prompt, reward = mean improvement rate
# Student₂ GRPO groups: m A₂s per feedback, reward = binary correctness
set -x

unset ROCR_VISIBLE_DEVICES

MODEL=${MODEL:-"Qwen/Qwen3-8B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo/dapo_all_verl.parquet"}
VAL_FILE=${VAL_FILE:-"${DATA_DIR}/val/val_combined_verl.parquet"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"self_teach_dapo_qwen3"}

PROJECT_DIR=$(cd "$(dirname "$0")/.."; pwd)

python3 -m src.self_teach.main \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=src.self_teach.agent_loop_overrides.FlexPromptAgentLoopManager \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=src/rlvr_grokking/rewards/deepscaler_reward.py \
    custom_reward_function.name=compute_score \
    +self_teach.enabled=True \
    +self_teach.num_feedbacks=6 \
    +self_teach.num_a2_per_feedback=6 \
    +self_teach.max_a2_prompt_length=4096 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rlvr-grokking' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE:-4} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=100 \
    "$@"
