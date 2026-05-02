#!/bin/bash
# Master script to submit all 18 systematic head identification jobs.
#
# Runs each method (EAP-IG, neurosurgery, retrieval) with 3 seeds x 2 datasets.
# Each job uses 1 GPU and outputs to:
#   results/reasoning_head_analysis/identification/systematic_base/<method>/<dataset>_seed<N>/
#
# Usage:
#   bash scripts/reasoning_head_analysis/run_systematic_identification.slurm

set -e

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
BASE_OUTPUT="results/reasoning_head_analysis/identification/systematic_base"

SEEDS=(42 123 456)

mkdir -p logs/systematic_identification

# ─── Helper to submit a single job ────────────────────────────────────
submit_job() {
    local JOB_NAME=$1
    local SCRIPT=$2
    local TIME_LIMIT=$3
    local MEM=$4
    shift 4
    local ARGS="$@"

    sbatch \
        --job-name="$JOB_NAME" \
        --account=nn12068k \
        --partition=accel \
        --gpus-per-node=1 \
        --cpus-per-task=8 \
        --mem="$MEM" \
        --time="$TIME_LIMIT" \
        --output="logs/systematic_identification/${JOB_NAME}_%j.log" \
        --error="logs/systematic_identification/${JOB_NAME}_%j.err" \
        --wrap="
set -e
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate
if [ -f \"$PROJECT_DIR/.env\" ]; then set -a; source \"$PROJECT_DIR/.env\"; set +a; fi
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_DATASETS_CACHE=\$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH
mkdir -p \$HF_DATASETS_CACHE
echo \"Job: $JOB_NAME, Node: \$SLURM_NODELIST, GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\"
$SCRIPT $ARGS
echo \"Done: $JOB_NAME\"
"
    echo "  Submitted: $JOB_NAME"
}

echo "=============================================="
echo "Systematic Head Identification (18 jobs)"
echo "=============================================="
echo "Model: $MODEL"
echo "Seeds: ${SEEDS[*]}"
echo "Output: $BASE_OUTPUT"
echo ""

# ─── EAP-IG: 6 jobs (aime x 3 seeds + math x 3 seeds) ───────────────
echo "--- EAP-IG (6 jobs, ~30 min each) ---"
for SEED in "${SEEDS[@]}"; do
    # AIME dataset
    OUTPUT_DIR="$BASE_OUTPUT/eap_ig/aime_seed${SEED}"
    submit_job "eapig-aime-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads" \
        "01:00:00" "64G" \
        --model "$MODEL" --dataset aime --seed "$SEED" --output_dir "$OUTPUT_DIR"

    # MATH dataset
    OUTPUT_DIR="$BASE_OUTPUT/eap_ig/math_seed${SEED}"
    submit_job "eapig-math-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads" \
        "01:00:00" "64G" \
        --model "$MODEL" --dataset math --seed "$SEED" --output_dir "$OUTPUT_DIR"
done

echo ""

# ─── Neurosurgery: 6 jobs (gsm8k x 3 seeds + math x 3 seeds) ────────
echo "--- Neurosurgery (6 jobs, ~20 min each) ---"
for SEED in "${SEEDS[@]}"; do
    # GSM8K (cot_vs_direct)
    OUTPUT_DIR="$BASE_OUTPUT/neurosurgery/gsm8k_seed${SEED}"
    submit_job "neuro-gsm8k-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_mathneuro" \
        "00:40:00" "64G" \
        --model "$MODEL" --contrast cot_vs_direct --seed "$SEED" \
        --device cuda --output_dir "$OUTPUT_DIR"

    # MATH (math_cot_vs_direct)
    OUTPUT_DIR="$BASE_OUTPUT/neurosurgery/math_seed${SEED}"
    submit_job "neuro-math-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_mathneuro" \
        "00:40:00" "64G" \
        --model "$MODEL" --contrast math_cot_vs_direct --seed "$SEED" \
        --device cuda --output_dir "$OUTPUT_DIR"
done

echo ""

# ─── Retrieval: 6 jobs (wikitext x 3 seeds + pg19 x 3 seeds) ─────────
echo "--- Retrieval (6 jobs, ~5 min each) ---"
for SEED in "${SEEDS[@]}"; do
    # Wikitext haystack
    OUTPUT_DIR="$BASE_OUTPUT/retrieval/wikitext_seed${SEED}"
    submit_job "retr-wiki-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_retrieval" \
        "00:20:00" "64G" \
        --model "$MODEL" --haystack_source wikitext --seed "$SEED" --output_dir "$OUTPUT_DIR"

    # PG19 haystack
    OUTPUT_DIR="$BASE_OUTPUT/retrieval/pg19_seed${SEED}"
    submit_job "retr-pg19-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_retrieval" \
        "00:20:00" "64G" \
        --model "$MODEL" --haystack_source pg19 --seed "$SEED" --output_dir "$OUTPUT_DIR"
done

echo ""
echo "=============================================="
echo "All 18 jobs submitted. Monitor with: squeue -u \$USER"
echo "=============================================="
