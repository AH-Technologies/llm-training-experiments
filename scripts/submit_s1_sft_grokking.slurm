#!/bin/bash
#SBATCH --job-name=s1-grok-32b
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=logs/s1-grok-%j.out
#SBATCH --error=logs/s1-grok-%j.err

# S1 Grokking: Qwen2.5-32B-Instruct on s1K — 150 epochs (30x paper run)
# Tests whether extended training produces a grokking-like effect.
#
# Saves 10 checkpoints (every 15 epochs) for post-hoc eval.
#
# Usage:
#   sbatch scripts/submit_s1_sft_grokking.slurm

set -e

echo "=========================================="
echo "S1 Grokking: Qwen2.5-32B on s1K (150 epochs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs checkpoints data

# Load modules
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
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

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

# --- Other ---
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Multi-node setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "HF cache: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- Background GPU/memory monitor (every 120s) ----
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo "CPU mem: $(free -g | awk '/^Mem:/{print $3"/"$2" GB"}')"
    sleep 120
done) &
MONITOR_PID=$!

# ---- Prepare data if needed ----
if [ ! -f "data/s1K/s1k_sft.parquet" ]; then
    echo "Preparing s1K SFT data..."
    python scripts/prepare_s1k_sft.py
fi

# ---- Run training ----
echo "Starting S1 grokking training (150 epochs)..."
export NUM_EPOCHS=150
export SAVE_EVERY=15
export OUTPUT_DIR="checkpoints/s1_grokking_qwen32b"
export HF_REPO="alexaau/s1-grokking-qwen32b"
bash scripts/run_sft_s1k.sh --wandb-run-name "s1_grokking_150ep"

# Cleanup
kill $MONITOR_PID 2>/dev/null || true
wait

echo "End: $(date)"
echo "=== Done ==="
