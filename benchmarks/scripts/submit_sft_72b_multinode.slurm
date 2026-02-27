#!/bin/bash
#SBATCH --job-name=sft-72b
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/sft-72b-%j.out
#SBATCH --error=benchmarks/results/sft-72b-%j.err

# SFT benchmark for 72B model with DeepSpeed ZeRO-3 + full CPU offload
# 2 nodes (8 GPUs total), all GPUs in DP (no TP needed with ZeRO-3)

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
export LIBRARY_PATH=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64:${LD_LIBRARY_PATH}

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

echo "=== SFT 72B Multi-Node Benchmark ==="
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
srun --nodes=2 --ntasks-per-node=1 nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

METRICS_FILE=$PROJECT_DIR/benchmarks/results/metrics_sft_72b_${SLURM_JOB_ID}.json

echo ">>> SFT 72B (DeepSpeed ZeRO-3, 2 nodes, full CPU offload)..."
export MASTER_ADDR MASTER_PORT
srun --nodes=2 --ntasks-per-node=1 \
    torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    benchmarks/sft_trl_worker.py \
    --model Qwen/Qwen2.5-72B \
    --deepspeed benchmarks/configs/ds_zero3_offload.json \
    --train-file data/benchmark/openmath_sft_train.parquet \
    --batch-size 8 \
    --micro-batch-size 1 \
    --sequence-length 2048 \
    --num-steps 10 \
    --num-gpus 8 \
    --metrics-file $METRICS_FILE \
    --output-dir /tmp/sft_benchmark_${SLURM_JOB_ID}

echo ""
echo "Exit code: $?"

if [ -f "$METRICS_FILE" ]; then
    echo "=== Metrics ==="
    cat $METRICS_FILE
fi

echo ""
echo "End time: $(date)"
echo "=== SFT 72B Benchmark Complete ==="
