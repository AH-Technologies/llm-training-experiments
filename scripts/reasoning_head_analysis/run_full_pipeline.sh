#!/bin/bash
# Full pipeline: identification → aggregation → ablation
#
# Uses SLURM --dependency=afterok to chain stages.
# Submit and leave — everything runs in order.
#
# Usage:
#   bash scripts/reasoning_head_analysis/run_full_pipeline.sh

set -e

PROJECT_DIR="/cluster/projects/nn12068k/haaklau/llm-training-experiments"
MODEL="Qwen/Qwen2.5-Math-1.5B"
BASE_OUTPUT="results/reasoning_head_analysis/identification/systematic_base"
ABLATION_OUTPUT="results/reasoning_head_analysis/ablation/systematic"
SCALES="0.0|0.5|1.5|2.0"
BENCHMARKS="math500|amc"
MATH500_N=100
AMC_N=40
N_SAMPLES=4
TEMPERATURE=0.6

SEEDS=(42 123 456)

mkdir -p "$PROJECT_DIR/logs/systematic_identification"
mkdir -p "$PROJECT_DIR/logs"

# Common SLURM preamble for --wrap jobs
PREAMBLE="set -e
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate
if [ -f \"$PROJECT_DIR/.env\" ]; then set -a; source \"$PROJECT_DIR/.env\"; set +a; fi
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_DATASETS_CACHE=\$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:\$PYTHONPATH
mkdir -p \$HF_DATASETS_CACHE"

echo "============================================================"
echo "  FULL PIPELINE: identify → aggregate → ablate"
echo "============================================================"
echo "Model:     $MODEL"
echo "Math500 N: $MATH500_N"
echo "AMC N:     $AMC_N"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: Head identification (18 jobs, 1 GPU each)
# ═══════════════════════════════════════════════════════════════════
echo "--- STAGE 1: Head Identification (18 jobs) ---"

IDENT_JOB_IDS=()

submit_ident() {
    local JOB_NAME=$1
    local CMD=$2
    local TIME=$3
    local MEM=$4

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --account=nn12068k \
        --partition=accel \
        --gpus-per-node=1 \
        --cpus-per-task=8 \
        --mem="$MEM" \
        --time="$TIME" \
        --output="$PROJECT_DIR/logs/systematic_identification/${JOB_NAME}_%j.log" \
        --error="$PROJECT_DIR/logs/systematic_identification/${JOB_NAME}_%j.err" \
        --wrap="$PREAMBLE
$CMD")
    IDENT_JOB_IDS+=("$JOB_ID")
    echo "  $JOB_NAME → job $JOB_ID"
}

# EAP-IG: 6 jobs
for SEED in "${SEEDS[@]}"; do
    submit_ident "eapig-aime-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads --model $MODEL --dataset aime --seed $SEED --output_dir $BASE_OUTPUT/eap_ig/aime_seed${SEED}" \
        "01:00:00" "64G"
    submit_ident "eapig-math-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads --model $MODEL --dataset math --seed $SEED --output_dir $BASE_OUTPUT/eap_ig/math_seed${SEED}" \
        "01:00:00" "64G"
done

# Neurosurgery: 6 jobs
for SEED in "${SEEDS[@]}"; do
    submit_ident "neuro-gsm8k-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_mathneuro --model $MODEL --contrast cot_vs_direct --seed $SEED --device cuda --output_dir $BASE_OUTPUT/neurosurgery/gsm8k_seed${SEED}" \
        "00:40:00" "64G"
    submit_ident "neuro-math-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_mathneuro --model $MODEL --contrast math_cot_vs_direct --seed $SEED --device cuda --output_dir $BASE_OUTPUT/neurosurgery/math_seed${SEED}" \
        "00:40:00" "64G"
done

# Retrieval: 6 jobs
for SEED in "${SEEDS[@]}"; do
    submit_ident "retr-wiki-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_retrieval --model $MODEL --haystack_source wikitext --seed $SEED --output_dir $BASE_OUTPUT/retrieval/wikitext_seed${SEED}" \
        "00:20:00" "64G"
    submit_ident "retr-pg19-s${SEED}" \
        "python -m reasoning_head_analysis.identify_heads_retrieval --model $MODEL --haystack_source pg19 --seed $SEED --output_dir $BASE_OUTPUT/retrieval/pg19_seed${SEED}" \
        "00:20:00" "64G"
done

# Build dependency string: afterok:id1:id2:...
IDENT_DEP=$(IFS=:; echo "${IDENT_JOB_IDS[*]}")
echo ""
echo "  ${#IDENT_JOB_IDS[@]} identification jobs submitted"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: Aggregation (1 job, CPU only, depends on all stage 1)
# ═══════════════════════════════════════════════════════════════════
echo "--- STAGE 2: Aggregation (waits for identification) ---"

AGG_JOB_ID=$(sbatch --parsable \
    --job-name="agg-heads" \
    --account=nn12068k \
    --partition=accel \
    --gpus-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --dependency="afterok:${IDENT_DEP}" \
    --output="$PROJECT_DIR/logs/agg_heads_%j.log" \
    --error="$PROJECT_DIR/logs/agg_heads_%j.err" \
    --wrap="$PREAMBLE
echo 'Running aggregation + analysis...'
python scripts/reasoning_head_analysis/analyze_systematic_identification.py --skip-attention
echo 'Aggregation done.'")

echo "  agg-heads → job $AGG_JOB_ID (depends on all identification)"
echo ""

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: Ablation (4 jobs, 4 GPUs each, depends on aggregation)
# ═══════════════════════════════════════════════════════════════════
echo "--- STAGE 3: Ablation (waits for aggregation) ---"

ABLATION_JOB_IDS=()
FLAGS="--do_incremental --do_individual --do_bottom"

submit_ablation() {
    local JOB_NAME=$1
    local IMPORTANCE=$2
    local OUTPUT=$3
    local TIME=$4
    local EXTRA_FLAGS=$5

    # Build the per-shard worker loop + analyze pass
    local IMP_ARG=""
    if [ -n "$IMPORTANCE" ]; then
        IMP_ARG="--importance_path $IMPORTANCE"
    fi

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --account=nn12068k \
        --partition=accel \
        --gpus-per-node=4 \
        --cpus-per-task=16 \
        --mem=128G \
        --time="$TIME" \
        --dependency="afterok:${AGG_JOB_ID}" \
        --output="$PROJECT_DIR/logs/${JOB_NAME}_%j.log" \
        --error="$PROJECT_DIR/logs/${JOB_NAME}_%j.err" \
        --wrap="$PREAMBLE

mkdir -p $OUTPUT
N_GPUS=\$(python -c 'import torch; print(torch.cuda.device_count())')
echo \"GPUs: \$N_GPUS\"

pids=()
for SHARD in \$(seq 0 \$((N_GPUS - 1))); do
    LOG=$OUTPUT/shard\${SHARD}.log
    echo \"Shard \$SHARD on GPU \$SHARD\"
    CUDA_VISIBLE_DEVICES=\$SHARD python -m reasoning_head_analysis.ablate_systematic \
        $IMP_ARG \
        --model $MODEL \
        --output_dir $OUTPUT \
        --benchmarks '$BENCHMARKS' \
        --scales '$SCALES' \
        --math500_n $MATH500_N \
        --amc_n $AMC_N \
        --n_samples $N_SAMPLES \
        --temperature $TEMPERATURE \
        --shard_idx \$SHARD \
        --n_shards \$N_GPUS \
        $EXTRA_FLAGS \
        > \$LOG 2>&1 &
    pids+=(\$!)
done

failed=0
for pid in \"\${pids[@]}\"; do
    wait \$pid || failed=1
done
[ \$failed -eq 1 ] && { echo 'Shard(s) failed'; exit 1; }

echo 'Shards done, analyzing...'
python -m reasoning_head_analysis.ablate_systematic \
    $IMP_ARG \
    --model $MODEL \
    --output_dir $OUTPUT \
    --benchmarks '$BENCHMARKS' \
    --scales '$SCALES' \
    --math500_n $MATH500_N \
    --amc_n $AMC_N \
    --n_samples $N_SAMPLES \
    --temperature $TEMPERATURE \
    --analyze_only \
    $EXTRA_FLAGS
echo 'Done.'")

    ABLATION_JOB_IDS+=("$JOB_ID")
    echo "  $JOB_NAME → job $JOB_ID"
}

# 3 method jobs
for METHOD in eap_ig neurosurgery retrieval; do
    submit_ablation "ablsys-${METHOD}" \
        "${BASE_OUTPUT}/${METHOD}/aggregated/head_importance.pt" \
        "${ABLATION_OUTPUT}/${METHOD}" \
        "06:00:00" \
        "$FLAGS"
done

# Random control
submit_ablation "ablsys-random" \
    "" \
    "${ABLATION_OUTPUT}/random_control" \
    "02:00:00" \
    "--do_random"

echo ""

# ═══════════════════════════════════════════════════════════════════
# STAGE 4: Cross-method analysis (depends on all ablation)
# ═══════════════════════════════════════════════════════════════════
echo "--- STAGE 4: Cross-method analysis (waits for ablation) ---"

ABL_DEP=$(IFS=:; echo "${ABLATION_JOB_IDS[*]}")

FINAL_JOB_ID=$(sbatch --parsable \
    --job-name="ablsys-analyze" \
    --account=nn12068k \
    --partition=accel \
    --gpus-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --dependency="afterok:${ABL_DEP}" \
    --output="$PROJECT_DIR/logs/ablsys_analyze_%j.log" \
    --error="$PROJECT_DIR/logs/ablsys_analyze_%j.err" \
    --wrap="$PREAMBLE
echo 'Running cross-method analysis...'
python scripts/reasoning_head_analysis/analyze_systematic_ablation.py
echo 'All done!'")

echo "  ablsys-analyze → job $FINAL_JOB_ID (depends on all ablation)"
echo ""

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
echo "============================================================"
echo "  PIPELINE SUBMITTED"
echo "============================================================"
echo ""
echo "  Stage 1: ${#IDENT_JOB_IDS[@]} identification jobs"
echo "  Stage 2: 1 aggregation job ($AGG_JOB_ID)"
echo "  Stage 3: ${#ABLATION_JOB_IDS[@]} ablation jobs"
echo "  Stage 4: 1 final analysis job ($FINAL_JOB_ID)"
echo ""
echo "  Total: $((${#IDENT_JOB_IDS[@]} + 1 + ${#ABLATION_JOB_IDS[@]} + 1)) jobs"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Cancel all: scancel ${IDENT_JOB_IDS[*]} $AGG_JOB_ID ${ABLATION_JOB_IDS[*]} $FINAL_JOB_ID"
echo ""
echo "  Expected timeline:"
echo "    Identification: ~1h"
echo "    Aggregation:    ~5 min"
echo "    Ablation:       ~2-6h (100 math500 + 40 amc problems)"
echo "    Final analysis: ~5 min"
echo "============================================================"
