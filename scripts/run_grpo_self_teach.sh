#!/bin/bash
# Self-Teach: 2-phase student-teacher self-play GRPO training
#
# 4-step generation per training step:
#   1. Generate A₁ (student answers question)
#   2. Generate F (teacher feedback, n copies per prompt — all see same A₁)
#   3. Generate A₂ for teacher grading (one per F)
#   4. Generate A₂ for student₂ training (n copies per prompt — all see same A₁+F)
#
# Teacher and Student₂ get separate GRPO advantage groups via distinct UIDs.
set -x

unset ROCR_VISIBLE_DEVICES

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_FILE="${DATA_DIR}/pi13_r128_self_teach.parquet"
VAL_FILE="${DATA_DIR}/math500.parquet"

PROJECT_DIR=$(cd "$(dirname "$0")/.."; pwd)

python3 -m src.rlvr_grokking.self_teach.main \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
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
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=src/rlvr_grokking/rewards/deepscaler_reward.py \
    custom_reward_function.name=compute_score \
    +self_teach.enabled=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='rlvr-grokking' \
    trainer.experiment_name='self_teach_pi13' \
    trainer.n_gpus_per_node=${SLURM_GPUS_ON_NODE:-4} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.test_freq=2 \
    trainer.total_epochs=2000 \
    trainer.total_training_steps=2000 \
    "$@"
