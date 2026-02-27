#!/bin/bash
#SBATCH --job-name=grpo-32b
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=benchmarks/results/grpo-32b-%j.out
#SBATCH --error=benchmarks/results/grpo-32b-%j.err

# GRPO benchmark for 32B model on 2 nodes (8 GPUs)
# Uses verl GRPO trainer with FSDP (no TP for actor, vLLM TP=4)

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
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=== GRPO 32B Multi-Node Benchmark ==="
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
srun --nodes=2 --ntasks-per-node=1 nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

python -m benchmarks.benchmark_runner \
    --models 32B \
    --training-types grpo \
    --num-gpus 4 \
    --num-nodes 2 \
    --sequence-length 2048 \
    --num-steps 10 \
    --find-max-batch \
    --output-dir benchmarks/results

echo ""
echo "End time: $(date)"
echo "=== GRPO 32B Benchmark Complete ==="
