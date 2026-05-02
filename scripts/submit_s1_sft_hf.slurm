#!/bin/bash
#SBATCH --job-name=s1-hf-32b
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=logs/s1-hf-%j.out
#SBATCH --error=logs/s1-hf-%j.err

# S1 Grokking via HuggingFace Trainer + accelerate FSDP.
# Replaces the custom torchtune trainer which had memory-scaling issues at long seq.
#
# Defaults: 50 epochs (10x paper), wd=1e-3 (10x paper), seq=16384 (~1 MATH500 pt
# below paper). Override per-run by setting env vars before sbatch, e.g.:
#
#   WEIGHT_DECAY=1e-2 \
#   HF_REPO=alexauren/s1-grokking-qwen32b-hf-wd1e2 \
#   OUTPUT_DIR=checkpoints/s1_grokking_qwen32b_hf_wd1e2 \
#   WANDB_RUN_NAME=s1_hf_50ep_wd1e2_seq16k \
#   sbatch scripts/submit_s1_sft_hf.slurm

set -e

# --- Per-run knobs (override via env before sbatch) ---
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
HF_REPO=${HF_REPO:-alexauren/s1-grokking-qwen32b-hf}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/s1_grokking_qwen32b_hf}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-s1_hf_50ep_wd1e3_seq16k}

echo "=========================================="
echo "S1 Grokking HF Trainer: Qwen2.5-32B on s1K"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Weight decay: $WEIGHT_DECAY"
echo "HF repo:      $HF_REPO"
echo "Output dir:   $OUTPUT_DIR"
echo "WandB run:    $WANDB_RUN_NAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs checkpoints data

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
cd $PROJECT_DIR

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# --- HuggingFace cache ---
MODEL_CACHE_KEY="Qwen_Qwen2.5-32B-Instruct"
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

# Enable Rust-based parallel hub uploader (~5-10x faster than default Python client)
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Slingshot / NCCL / libfabric networking ---
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}
export NCCL_DEBUG=WARN

# --- CUDA paths ---
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

# --- Multi-node setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE (${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs)"
echo "HF cache: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- Background GPU/memory monitor ----
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo "CPU mem: $(free -g | awk '/^Mem:/{print $3"/"$2" GB"}')"
    sleep 120
done) &
MONITOR_PID=$!

if [ ! -f "data/s1K/s1k_sft.parquet" ]; then
    echo "Preparing s1K SFT data..."
    python scripts/prepare_s1k_sft.py
fi

# ---- Launch via accelerate (srun per node) ----
ACCEL_CMD="accelerate launch \
    --config_file ${PROJECT_DIR}/configs/accelerate_fsdp.yaml \
    --num_processes=${WORLD_SIZE} \
    --num_machines=${NUM_NODES} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --machine_rank=\${SLURM_PROCID} \
    -m s1.sft_hf \
    --model Qwen/Qwen2.5-32B-Instruct \
    --train-file data/s1K/s1k_sft.parquet \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs 50 \
    --batch-size 16 \
    --per-device-batch-size 1 \
    --sequence-length 16384 \
    --lr 1e-5 \
    --warmup-ratio 0.05 \
    --weight-decay ${WEIGHT_DECAY} \
    --max-grad-norm 1.0 \
    --save-every-n-epochs 10 \
    --wandb-project s1-qwen32b-sft \
    --wandb-run-name ${WANDB_RUN_NAME} \
    --hf-repo ${HF_REPO} \
    --seed 42"

srun --nodes=${NUM_NODES} --ntasks-per-node=1 --export=ALL \
    bash -c "unset ROCR_VISIBLE_DEVICES; ${ACCEL_CMD}"

kill $MONITOR_PID 2>/dev/null || true
wait

echo "End: $(date)"
echo "=== Done ==="
