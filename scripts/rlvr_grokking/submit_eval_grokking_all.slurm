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
# Iterates checkpoint-* in numerical step order and runs s1.eval
# (MATH500 / AIME24 / GPQA Diamond) on each, writing one JSON per checkpoint
# to $OUTPUT_DIR.
#
# Two source modes:
#   - Local  (default):   reads checkpoints from $CKPT_DIR/checkpoint-*
#   - Hub    ($HF_REPO):  lists Hub repo, downloads each checkpoint to
#                         $HF_LOCAL_ROOT, evals, deletes it before the next.
#
# Resumable: re-submitting skips checkpoints whose output JSON already exists.
#
# Usage:
#   # Local mode
#   sbatch scripts/rlvr_grokking/submit_eval_grokking_all.slurm
#   CKPT_DIR=checkpoints/s1_grokking_qwen32b_hf BENCHMARKS="aime24 gpqa" \
#       sbatch scripts/rlvr_grokking/submit_eval_grokking_all.slurm
#
#   # Hub mode (one-checkpoint-at-a-time download/eval/delete)
#   HF_REPO=alexauren/s1-grokking-qwen32b-hf \
#       sbatch scripts/rlvr_grokking/submit_eval_grokking_all.slurm
#
#   # Run only after the grokking job finishes:
#   sbatch --dependency=afterok:<grokking_job_id> \
#       scripts/rlvr_grokking/submit_eval_grokking_all.slurm

set -e

CKPT_DIR=${CKPT_DIR:-"checkpoints/s1_grokking_qwen32b_hf"}
HF_REPO=${HF_REPO:-}
HF_LOCAL_ROOT=${HF_LOCAL_ROOT:-"checkpoints/_eval_hub_tmp"}
BENCHMARKS=${BENCHMARKS:-"math500 aime24 gpqa"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/s1_grokking_eval"}

echo "=========================================="
echo "S1 Grokking: Evaluate All Checkpoints"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
if [ -n "$HF_REPO" ]; then
    echo "Source:    Hub repo $HF_REPO  (tmp dir: $HF_LOCAL_ROOT)"
else
    echo "Source:    local $CKPT_DIR"
fi
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
mkdir -p $HF_HOME

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}
unset ROCR_VISIBLE_DEVICES

nvidia-smi --query-gpu=name,memory.total --format=csv

# Discover checkpoints in numerical step order (checkpoint-312 < checkpoint-1000).
if [ -n "$HF_REPO" ]; then
    mapfile -t CKPTS < <(python -c "
from huggingface_hub import HfApi
import re
tree = HfApi().list_repo_tree('${HF_REPO}', recursive=False)
names = sorted(
    {e.path for e in tree if re.fullmatch(r'checkpoint-\d+', e.path)},
    key=lambda s: int(s.split('-')[1]),
)
print('\n'.join(names))
")
else
    mapfile -t CKPTS < <(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V)
fi

if [ ${#CKPTS[@]} -eq 0 ]; then
    if [ -n "$HF_REPO" ]; then
        echo "ERROR: no checkpoint-* subfolders found in Hub repo $HF_REPO."
    else
        echo "ERROR: no checkpoints found in $CKPT_DIR (expected checkpoint-* subdirs)."
    fi
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoints:"
for c in "${CKPTS[@]}"; do echo "  - $c"; done
echo ""

if [ -n "$HF_REPO" ]; then
    mkdir -p "$HF_LOCAL_ROOT"
fi

for CKPT in "${CKPTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT")
    OUT_FILE="${OUTPUT_DIR}/eval_${CKPT_NAME}.json"

    if [ -f "$OUT_FILE" ]; then
        echo "[skip] $CKPT_NAME — $OUT_FILE already exists"
        continue
    fi

    if [ -n "$HF_REPO" ]; then
        LOCAL_PATH="$HF_LOCAL_ROOT/$CKPT_NAME"
        echo "=========================================="
        echo "[download] $HF_REPO :: $CKPT_NAME -> $LOCAL_PATH  ($(date))"
        echo "=========================================="
        rm -rf "$LOCAL_PATH"
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_REPO}',
    allow_patterns=['${CKPT_NAME}/*'],
    local_dir='${HF_LOCAL_ROOT}',
)
"
        MODEL_PATH="$LOCAL_PATH"
    else
        MODEL_PATH="$CKPT"
    fi

    echo "=========================================="
    echo "[eval] $CKPT_NAME  ($(date))"
    echo "=========================================="

    python -m s1.eval \
        --model "$MODEL_PATH" \
        --benchmarks ${BENCHMARKS} \
        --tp 4 \
        --max-tokens 32768 \
        --output-dir "$OUTPUT_DIR"

    if [ -n "$HF_REPO" ]; then
        echo "[cleanup] $LOCAL_PATH"
        rm -rf "$LOCAL_PATH"
    fi

    echo "[done] $CKPT_NAME  ($(date))"
done

echo "End: $(date)"
echo "=== All checkpoints evaluated ==="
