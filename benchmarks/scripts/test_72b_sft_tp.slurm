#!/bin/bash
#SBATCH --job-name=test-72b-sft-tp
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/test-72b-sft-tp-%j.out
#SBATCH --error=benchmarks/results/test-72b-sft-tp-%j.err

# Smoke test: SFT 72B via benchmark_runner on 2 nodes (8 GPUs total)
# Uses DeepSpeed ZeRO-3 + CPU offload (auto-resolved by benchmark_runner)
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get master node info for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== 72B SFT TP Smoke Test ==="
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

srun --nodes=2 --ntasks-per-node=1 nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

echo ">>> Testing SFT 72B (DeepSpeed ZeRO-3, 2 nodes, full CPU offload)..."
python -m benchmarks.benchmark_runner \
    --models 72B \
    --training-types sft \
    --num-gpus 4 \
    --num-nodes 2 \
    --sequence-length 2048 \
    --num-steps 2 \
    --batch-size 8 \
    --output-dir benchmarks/results

# Clean up
if [ -d "$HF_HOME" ]; then
    echo "Cleaning up job cache at $HF_HOME"
    rm -rf "$HF_HOME"
fi

echo ""
echo "End time: $(date)"
echo "=== 72B SFT TP Smoke Test Complete ==="
