#!/bin/bash
# SFT training on A* search traces for grokking experiments
# Uses custom trainer with generation-based evaluation (optimal path accuracy)
set -x

unset ROCR_VISIBLE_DEVICES

MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B"}
DATA_DIR=${DATA_DIR:-"./data/astar_grokking_dataset"}
TRAIN_FILE="${DATA_DIR}/astar_train.parquet"
VAL_FILE="${DATA_DIR}/astar_val.parquet"

PROJECT_DIR=${PROJECT_DIR:-"/cluster/projects/nn12068k/haaklau/llm-training-experiments"}

N_GPUS=${N_GPUS:-4}

PYTHONPATH=${PROJECT_DIR}/src:$PYTHONPATH \
torchrun --nproc_per_node=${N_GPUS} -m astar_dataset.custom_sft_trainer \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.multiturn.enable=True \
    data.multiturn.messages_key=messages \
    data.max_length=5120 \
    data.truncation='error' \
    model.partial_pretrain=${MODEL} \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=False \
    optim.lr=2e-5 \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.project_name='astar-grokking' \
    trainer.experiment_name='sft_astar_7x7_9x9' \
    trainer.total_epochs=100 \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.seed=42 \
    +gen_eval.val_parquet=${VAL_FILE} \
    +gen_eval.train_parquet=${TRAIN_FILE} \
    +gen_eval.samples=10 \
    +gen_eval.max_tokens=2048 \
    +gen_eval.freq=200 \
    "$@"
