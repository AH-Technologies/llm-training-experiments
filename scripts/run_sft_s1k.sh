#!/bin/bash
# S1 SFT: Qwen2.5-32B-Instruct on s1K dataset
# Uses torchtune FSDP2+TP (same infrastructure as benchmarks/)
#
# Hyperparameters from the s1 paper (§D, Appendix D):
#   - 5 epochs (315 total steps), lr=1e-5, warmup=5%, wd=1e-4, cosine schedule
#   - batch_size=16 (global), micro_batch=1, grad_accum inferred
#   - seq_length=32768 (long enough to not truncate any samples), bf16
#   - Adam betas=(0.9, 0.95)
#
# Hardware: 4 nodes x 4 H200 GPUs = 16 GPUs total
#   - FSDP2 across all 16 GPUs (DP=4), TP=4 within each node
#   - No CPU offload needed (~20 GB/GPU for 32B with 16 GPUs)
#
# Usage:
#   Called from submit_training.slurm with TASK=s1_sft
#   Or standalone: bash scripts/run_sft_s1k.sh

set -x

MODEL=${MODEL:-"Qwen/Qwen2.5-32B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"data/s1K/s1k_sft.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/s1_sft_qwen32b"}

# Training hyperparameters (from s1 paper §D)
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SEQ_LENGTH=${SEQ_LENGTH:-16384}
NUM_EPOCHS=${NUM_EPOCHS:-5}
TP_DEGREE=${TP_DEGREE:-4}
SAVE_EVERY=${SAVE_EVERY:-1}
HF_REPO=${HF_REPO:-""}

PROJECT_DIR=${PROJECT_DIR:-"/cluster/projects/nn12068k/alexaau/llm-training-experiments"}

# Ensure src/ is on PYTHONPATH for s1.sft_trainer
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

# Number of GPUs (from SLURM or default)
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
N_NODES=${SLURM_NNODES:-4}

# Multi-node rendezvous
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST 2>/dev/null | head -n 1)}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

unset ROCR_VISIBLE_DEVICES

mkdir -p ${OUTPUT_DIR}

TRAINER_ARGS="\
    --model ${MODEL} \
    --train-file ${TRAIN_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --sequence-length ${SEQ_LENGTH} \
    --lr 1e-5 \
    --warmup-ratio 0.05 \
    --weight-decay ${WEIGHT_DECAY:-1e-4} \
    ${CPU_OFFLOAD:+--cpu-offload} \
    ${FSDP_OFFLOAD:+--fsdp-offload} \
    --tp-degree ${TP_DEGREE} \
    --save-every-n-epochs ${SAVE_EVERY} \
    --log-every-n-steps 1 \
    --seed 42 \
    ${HF_REPO:+--hf-repo ${HF_REPO} --delete-local-checkpoints} \
    $@"

# Launch trainer
# For multi-node: srun launches torchrun on each node
if [ "${N_NODES}" -gt 1 ]; then
    TORCHRUN_CMD="torchrun \
        --nproc_per_node=${N_GPUS} \
        --nnodes=${N_NODES} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        -m s1.sft_trainer \
        ${TRAINER_ARGS}"

    srun \
        --nodes=${N_NODES} \
        --ntasks-per-node=1 \
        --export=ALL \
        bash -c "unset ROCR_VISIBLE_DEVICES; ${TORCHRUN_CMD}"
else
    torchrun \
        --nproc_per_node=${N_GPUS} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        -m s1.sft_trainer \
        ${TRAINER_ARGS}
fi
