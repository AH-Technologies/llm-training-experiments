#!/bin/bash
#SBATCH --job-name=sft-tp-multinode
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/sft-tp-multinode-%j.out
#SBATCH --error=benchmarks/results/sft-tp-multinode-%j.err

# Multi-node SFT benchmark with FSDP2+TP for 72B+ models
# Uses TRL SFTTrainer with Accelerate ParallelismConfig (TP=4)
# 2 nodes x 4 GPUs = 8 GPUs total, TP=4 within node, DP=2 across nodes

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

# Get master node info for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== SFT TP Multi-Node Benchmark ==="
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: 4"
echo ""

python -m benchmarks.benchmark_runner \
    --models 72B \
    --training-types sft \
    --num-gpus 4 \
    --num-nodes 2 \
    --sequence-length 2048 \
    --num-steps 10 \
    --find-max-batch \
    --cleanup-cache \
    --output-dir benchmarks/results

# Clean up the per-job cache directory
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up job cache at $HF_HOME"
    rm -rf "$HF_HOME"
fi

echo ""
echo "End time: $(date)"
echo "=== SFT TP Multi-Node Benchmark Complete ==="
