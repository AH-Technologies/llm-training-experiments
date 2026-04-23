#!/bin/bash
#SBATCH --job-name=s1-eval-grok-all
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/s1-eval-grok-all-%j.out
#SBATCH --error=logs/s1-eval-grok-all-%j.err

# Evaluate every checkpoint saved by the s1 grokking HF-Trainer job.
# Iterates checkpoints/$CKPT_DIR/checkpoint-* in numerical step order and
# runs s1.eval (MATH500 / AIME24 / GPQA Diamond) on each, writing one JSON
# per checkpoint to $OUTPUT_DIR.
#
# Resumable: re-submitting skips checkpoints whose output JSON already exists,
# so it's safe to run this while new checkpoints are still being written.
#
# Usage:
#   sbatch scripts/submit_eval_grokking_all.slurm
#   CKPT_DIR=checkpoints/s1_grokking_qwen32b_hf BENCHMARKS="aime24 gpqa" \
#       sbatch scripts/submit_eval_grokking_all.slurm
#   # Run only after the grokking job finishes:
#   sbatch --dependency=afterok:<grokking_job_id> scripts/submit_eval_grokking_all.slurm

set -e

CKPT_DIR=${CKPT_DIR:-"checkpoints/s1_grokking_qwen32b_hf"}
BENCHMARKS=${BENCHMARKS:-"math500 aime24 gpqa"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/s1_grokking_eval"}

echo "=========================================="
echo "S1 Grokking: Evaluate All Checkpoints"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Ckpt dir:  $CKPT_DIR"
echo "Benchmarks: $BENCHMARKS"
echo "Output:    $OUTPUT_DIR"
echo "Start:     $(date)"
echo "=========================================="

mkdir -p logs "$OUTPUT_DIR"

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

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

# Shared HF cache (no downloads needed; checkpoints are local)
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_eval
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}
unset ROCR_VISIBLE_DEVICES

nvidia-smi --query-gpu=name,memory.total --format=csv

# Discover checkpoints in numerical step order (checkpoint-312 < checkpoint-1000).
mapfile -t CKPTS < <(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "ERROR: no checkpoints found in $CKPT_DIR (expected checkpoint-* subdirs)."
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints:"
for c in "${CKPTS[@]}"; do echo "  - $c"; done
echo ""

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT")
    OUT_FILE="${OUTPUT_DIR}/eval_${CKPT_NAME}.json"

    if [ -f "$OUT_FILE" ]; then
        echo "[skip] $CKPT_NAME — $OUT_FILE already exists"
        continue
    fi

    echo "=========================================="
    echo "[eval] $CKPT_NAME  ($(date))"
    echo "=========================================="

    python -m s1.eval \
        --model "$CKPT" \
        --benchmarks ${BENCHMARKS} \
        --tp 4 \
        --max-tokens 32768 \
        --output-dir "$OUTPUT_DIR"

    echo "[done] $CKPT_NAME  ($(date))"
done

echo "End: $(date)"
echo "=== All checkpoints evaluated ==="
