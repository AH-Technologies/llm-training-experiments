#!/bin/bash
#SBATCH --job-name=gem-sft
#SBATCH --account=nn12068k
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/gem-sft-%j.out
#SBATCH --error=logs/gem-sft-%j.err

# GEM / CE SFT Training (4x GH200)
#
# Usage:
#   sbatch scripts/gem/submit_train.slurm
#   LOSS=ce sbatch scripts/gem/submit_train.slurm
#   DATASET=data/my_data.parquet GEM_BETA=0.5 sbatch scripts/gem/submit_train.slurm

set -e

# Configurable via environment variables
MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATASET=${DATASET:-"data/sft.parquet"}
LOSS=${LOSS:-"gem"}
GEM_BETA=${GEM_BETA:-0.7}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/gem_sft"}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-"gem-sft"}
WANDB_RUN=${WANDB_RUN:-"${LOSS}_beta${GEM_BETA}_lr${LR}_ep${NUM_EPOCHS}"}
SAVE_EVERY=${SAVE_EVERY:-1}
SEED=${SEED:-42}

echo "=========================================="
echo "GEM SFT Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Loss: $LOSS (beta=$GEM_BETA)"
echo "Epochs: $NUM_EPOCHS"
echo "Batch: ${BATCH_SIZE}x${GRAD_ACCUM}x4 = $((BATCH_SIZE * GRAD_ACCUM * 4)) effective"
echo "LR: $LR"
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

echo "Starting training..."
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    src/gem/train.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --loss "$LOSS" \
    --gem_beta $GEM_BETA \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr $LR \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_grad_norm $MAX_GRAD_NORM \
    --wandb_project "$WANDB_PROJECT_NAME" \
    --wandb_run_name "$WANDB_RUN" \
    --save_every_n_epochs $SAVE_EVERY \
    --seed $SEED

echo "Training complete!"
