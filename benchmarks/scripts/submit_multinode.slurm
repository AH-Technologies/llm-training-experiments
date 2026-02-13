#!/bin/bash
#SBATCH --job-name=multinode-bench
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=06:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --output=benchmarks/results/multinode-%j.out
#SBATCH --error=benchmarks/results/multinode-%j.err

# Multi-node benchmark for larger models (32B, 72B)
# Uses 2 nodes x 4 GPUs = 8 GPUs total

set -x

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
# Note: WANDB_MODE is set to "offline" inside the benchmark scripts

# Get master node info for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== Multi-Node Benchmark ==="
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node: 4"
echo ""

# Run on each node using srun
srun --nodes=2 --ntasks-per-node=1 bash -c '
    cd /cluster/projects/nn12068k/haaklau/llm-training-experiments
    source venv/bin/activate

    python -m benchmarks.benchmark_runner \
        --models 32B 72B \
        --training-types sft \
        --num-gpus 4 \
        --num-nodes 2 \
        --sequence-length 2048 \
        --num-steps 30 \
        --find-max-batch \
        --output-dir benchmarks/results
'

echo "=== Multi-Node Benchmark Complete ==="
