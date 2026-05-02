#!/bin/bash
#SBATCH --job-name=sft1shot-gem
#SBATCH --account=nn12068k
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/sft1shot-gem-%j.out
#SBATCH --error=logs/sft1shot-gem-%j.err

# 1-Shot SFT with GEM loss (4x GH200)
# Each step = 1 full pass over dataset (128 examples)
#
# Usage:
#   sbatch scripts/gem/submit_gem.slurm
#   DATASET=data/sft_1shot_datasets/standard_pi13/problem_0000.parquet sbatch scripts/gem/submit_gem.slurm

set -e

# Configurable via environment variables
MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATASET=${DATASET:-"data/sft_1shot_datasets/standard_pi1/problem_0000.parquet"}
GEM_BETA=${GEM_BETA:-0.7}
NUM_STEPS=${NUM_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-20}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
OUTPUT_DIR=${OUTPUT_DIR:-"models/sft_1shot_gem"}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-"sft-1shot"}
WANDB_RUN=${WANDB_RUN:-"gem_beta${GEM_BETA}_lr${LR}_${NUM_STEPS}steps"}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-0}
EVAL_DATASET=${EVAL_DATASET:-"data/math500.parquet"}
EVAL_SAMPLES=${EVAL_SAMPLES:-50}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-2048}
SEED=${SEED:-42}

echo "=========================================="
echo "1-Shot SFT — GEM Loss"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Loss: GEM (beta=$GEM_BETA)"
echo "Steps: $NUM_STEPS (1 step = full dataset pass)"
echo "Batch: ${BATCH_SIZE}x${GRAD_ACCUM}x4 = $((BATCH_SIZE * GRAD_ACCUM * 4)) effective"
echo "LR: $LR (warmup: $WARMUP_STEPS steps)"
echo "Save: every $SAVE_EVERY steps"
echo "Eval: every $EVAL_EVERY steps ($EVAL_SAMPLES MATH500 samples)"
echo "=========================================="

mkdir -p logs checkpoints

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Project directory
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments

# Activate venv
source $PROJECT_DIR/venv/bin/activate

# Load .env (WANDB_API_KEY etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Environment
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8

# Memory
export PYTORCH_ALLOC_CONF=expandable_segments:True

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Verify environment
echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

cd $PROJECT_DIR

# Verify dataset exists
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

echo "Starting GEM training..."
torchrun --nproc_per_node=4 \
    src/gem/train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --loss gem \
    --gem_beta $GEM_BETA \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_grad_norm $MAX_GRAD_NORM \
    --save_every_n_steps $SAVE_EVERY \
    --eval_every_n_steps $EVAL_EVERY \
    --eval_dataset "$EVAL_DATASET" \
    --eval_samples $EVAL_SAMPLES \
    --eval_max_new_tokens $EVAL_MAX_TOKENS \
    --wandb_project "$WANDB_PROJECT_NAME" \
    --wandb_run_name "$WANDB_RUN" \
    --seed $SEED

echo "Training complete!"
