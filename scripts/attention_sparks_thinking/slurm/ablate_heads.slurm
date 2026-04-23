#!/bin/bash
#SBATCH --job-name=ablate-heads
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
#SBATCH --output=attention_sparks_thinking/logs/ablate_%j.log
#SBATCH --error=attention_sparks_thinking/logs/ablate_%j.err

# Parallel head ablation eval: 3 conditions on 3 GPUs simultaneously.
#
# GPU 0: top-k curve (k=0..MAX_K)
# GPU 1: random-k control curve (same schedule, 3 seeds)
# GPU 2: bar chart (baseline vs top-10 vs 3× random-10)
# Then merge + plot.
#
# Usage:
#   sbatch scripts/attention_sparks_thinking/slurm/ablate_heads.slurm
#   sbatch --export=ALL,NUM_PROBLEMS=100,MAX_K=20 scripts/attention_sparks_thinking/slurm/ablate_heads.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen3-4B-Base"}
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"attention_sparks_thinking/logs/eapig_Qwen_Qwen3-4B-Base_317699/head_importance_qwen3.pt"}
DATA_PATH=${DATA_PATH:-"data/math500.parquet"}
NUM_PROBLEMS=${NUM_PROBLEMS:-50}
MAX_K=${MAX_K:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "==========================================="
echo "Parallel Head Ablation Eval (3 GPUs)"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Num problems: $NUM_PROBLEMS"
echo "Max K: $MAX_K"
echo "Batch size: $BATCH_SIZE"
echo "==========================================="

mkdir -p attention_sparks_thinking/logs

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH

# CUDA
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}
unset ROCR_VISIBLE_DEVICES

# Output directory
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUTPUT_DIR="$PROJECT_DIR/attention_sparks_thinking/logs/ablation_${SAFE_MODEL}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

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

COMMON_ARGS="--model_name $MODEL --importance_path $IMPORTANCE_PATH --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR --num_problems $NUM_PROBLEMS --max_k $MAX_K \
    --max_new_tokens $MAX_NEW_TOKENS --batch_size $BATCH_SIZE"

# ── Launch 3 conditions in parallel ──
echo "Launching 3 conditions in parallel..."

CUDA_VISIBLE_DEVICES=0 python scripts/attention_sparks_thinking/ablate_heads_eval.py \
    $COMMON_ARGS --condition topk_curve \
    > "$OUTPUT_DIR/topk_curve.log" 2>&1 &
PID_TOPK=$!

CUDA_VISIBLE_DEVICES=1 python scripts/attention_sparks_thinking/ablate_heads_eval.py \
    $COMMON_ARGS --condition random_curve \
    > "$OUTPUT_DIR/random_curve.log" 2>&1 &
PID_RANDOM=$!

CUDA_VISIBLE_DEVICES=2 python scripts/attention_sparks_thinking/ablate_heads_eval.py \
    $COMMON_ARGS --condition bar_chart \
    > "$OUTPUT_DIR/bar_chart.log" 2>&1 &
PID_BAR=$!

echo "  topk_curve  PID=$PID_TOPK  (GPU 0)"
echo "  random_curve PID=$PID_RANDOM (GPU 1)"
echo "  bar_chart   PID=$PID_BAR   (GPU 2)"

# ── Wait for all to finish ──
FAILED=0

wait $PID_TOPK
RC_TOPK=$?
echo "topk_curve finished with exit code $RC_TOPK"
[ $RC_TOPK -ne 0 ] && FAILED=1

wait $PID_RANDOM
RC_RANDOM=$?
echo "random_curve finished with exit code $RC_RANDOM"
[ $RC_RANDOM -ne 0 ] && FAILED=1

wait $PID_BAR
RC_BAR=$?
echo "bar_chart finished with exit code $RC_BAR"
[ $RC_BAR -ne 0 ] && FAILED=1

if [ $FAILED -ne 0 ]; then
    echo "ERROR: One or more conditions failed. Check individual logs:"
    echo "  $OUTPUT_DIR/topk_curve.log"
    echo "  $OUTPUT_DIR/random_curve.log"
    echo "  $OUTPUT_DIR/bar_chart.log"
    # Still try to plot whatever succeeded
fi

# ── Merge + plot ──
echo ""
echo "Merging results and generating plots..."
python scripts/attention_sparks_thinking/ablate_heads_eval.py \
    $COMMON_ARGS --condition plot

echo ""
echo "==========================================="
echo "Done! Output saved to: $OUTPUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "==========================================="
ls -la "$OUTPUT_DIR"
