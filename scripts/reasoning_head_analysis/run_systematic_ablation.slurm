#!/bin/bash
# Orchestrator: submit 4 SLURM jobs for systematic ablation study.
#
# Usage:
#   bash scripts/reasoning_head_analysis/run_systematic_ablation.slurm
#
# Submits:
#   Job 1: eap_ig       (incremental + individual + bottom)  ~6h
#   Job 2: neurosurgery (incremental + individual + bottom)  ~6h
#   Job 3: retrieval    (incremental + individual + bottom)  ~6h
#   Job 4: random control (random draws only)                ~2h

set -e

PROJECT_DIR="/cluster/projects/nn12068k/haaklau/llm-training-experiments"
BASE_DIR="results/reasoning_head_analysis/identification/systematic_base"
OUTPUT_BASE="results/reasoning_head_analysis/ablation/systematic"
MODEL="Qwen/Qwen2.5-Math-1.5B"
SCALES="0.0|0.5|1.5|2.0"
BENCHMARKS="math500|amc"
MATH500_N=100
AMC_N=40
N_SAMPLES=4
TEMPERATURE=0.6

echo "==========================================="
echo "Systematic Ablation Study — Orchestrator"
echo "==========================================="

submit_method_job() {
    local METHOD=$1
    local IMPORTANCE=$2
    local OUTPUT=$3
    local TIME=$4
    local FLAGS=$5

    if [ ! -f "$PROJECT_DIR/$IMPORTANCE" ]; then
        echo "WARNING: $IMPORTANCE not found, skipping $METHOD"
        echo "  Run: srun ... python scripts/reasoning_head_analysis/analyze_systematic_identification.py --skip-attention"
        return
    fi

    local JOB_NAME="ablsys-${METHOD}"
    local SCRIPT="/tmp/${JOB_NAME}_$$.sh"

    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=${TIME}
#SBATCH --output=${PROJECT_DIR}/logs/${JOB_NAME}_%j.log
#SBATCH --error=${PROJECT_DIR}/logs/${JOB_NAME}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

set -e
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/venv/bin/activate"
[ -f "${PROJECT_DIR}/.env" ] && { set -a; source "${PROJECT_DIR}/.env"; set +a; }

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=\$HF_HOME
export HF_TOKEN=\${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:\$PYTHONPATH

N_GPUS=\$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs visible: \$N_GPUS"

mkdir -p "${OUTPUT}"
pids=()
for SHARD in \$(seq 0 \$((N_GPUS - 1))); do
    LOG="${OUTPUT}/shard\${SHARD}.log"
    echo "Launching shard \$SHARD on GPU \$SHARD"
    CUDA_VISIBLE_DEVICES=\$SHARD python -m reasoning_head_analysis.ablate_systematic \\
        --importance_path "${IMPORTANCE}" \\
        --model "${MODEL}" \\
        --output_dir "${OUTPUT}" \\
        --benchmarks "${BENCHMARKS}" \\
        --scales "${SCALES}" \\
        --math500_n ${MATH500_N} \\
        --amc_n ${AMC_N} \\
        --n_samples ${N_SAMPLES} \\
        --temperature ${TEMPERATURE} \\
        --shard_idx \$SHARD \\
        --n_shards \$N_GPUS \\
        ${FLAGS} \\
        > "\$LOG" 2>&1 &
    pids+=(\$!)
done

echo "Waiting for \${#pids[@]} workers..."
failed=0
for pid in "\${pids[@]}"; do
    if ! wait "\$pid"; then
        failed=1
        echo "Worker pid=\$pid failed"
    fi
done
[ \$failed -eq 1 ] && { echo "Shard(s) failed; see ${OUTPUT}/shard*.log"; exit 1; }

echo "All shards done. Running analyze-only pass..."
python -m reasoning_head_analysis.ablate_systematic \\
    --importance_path "${IMPORTANCE}" \\
    --model "${MODEL}" \\
    --output_dir "${OUTPUT}" \\
    --benchmarks "${BENCHMARKS}" \\
    --scales "${SCALES}" \\
    --math500_n ${MATH500_N} \\
    --amc_n ${AMC_N} \\
    --n_samples ${N_SAMPLES} \\
    --temperature ${TEMPERATURE} \\
    --analyze_only \\
    ${FLAGS}

echo "Done. Job \$SLURM_JOB_ID"
EOF
    chmod +x "$SCRIPT"
    local JOB_ID
    JOB_ID=$(sbatch "$SCRIPT" | awk '{print $NF}')
    echo "  Submitted $METHOD: job $JOB_ID ($TIME, 4 GPUs)"
    rm -f "$SCRIPT"
}

submit_random_job() {
    local OUTPUT="${OUTPUT_BASE}/random_control"
    local JOB_NAME="ablsys-random"
    local SCRIPT="/tmp/${JOB_NAME}_$$.sh"

    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=${PROJECT_DIR}/logs/${JOB_NAME}_%j.log
#SBATCH --error=${PROJECT_DIR}/logs/${JOB_NAME}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

set -e
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/venv/bin/activate"
[ -f "${PROJECT_DIR}/.env" ] && { set -a; source "${PROJECT_DIR}/.env"; set +a; }

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=\$HF_HOME
export HF_TOKEN=\${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:\$PYTHONPATH

N_GPUS=\$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs visible: \$N_GPUS"

mkdir -p "${OUTPUT}"
pids=()
for SHARD in \$(seq 0 \$((N_GPUS - 1))); do
    LOG="${OUTPUT}/shard\${SHARD}.log"
    echo "Launching shard \$SHARD on GPU \$SHARD"
    CUDA_VISIBLE_DEVICES=\$SHARD python -m reasoning_head_analysis.ablate_systematic \\
        --model "${MODEL}" \\
        --output_dir "${OUTPUT}" \\
        --benchmarks "${BENCHMARKS}" \\
        --scales "${SCALES}" \\
        --math500_n ${MATH500_N} \\
        --amc_n ${AMC_N} \\
        --n_samples ${N_SAMPLES} \\
        --temperature ${TEMPERATURE} \\
        --shard_idx \$SHARD \\
        --n_shards \$N_GPUS \\
        --do_random \\
        > "\$LOG" 2>&1 &
    pids+=(\$!)
done

echo "Waiting for \${#pids[@]} workers..."
failed=0
for pid in "\${pids[@]}"; do
    if ! wait "\$pid"; then
        failed=1
        echo "Worker pid=\$pid failed"
    fi
done
[ \$failed -eq 1 ] && { echo "Shard(s) failed; see ${OUTPUT}/shard*.log"; exit 1; }

echo "All shards done. Running analyze-only pass..."
python -m reasoning_head_analysis.ablate_systematic \\
    --model "${MODEL}" \\
    --output_dir "${OUTPUT}" \\
    --benchmarks "${BENCHMARKS}" \\
    --scales "${SCALES}" \\
    --math500_n ${MATH500_N} \\
    --amc_n ${AMC_N} \\
    --n_samples ${N_SAMPLES} \\
    --temperature ${TEMPERATURE} \\
    --analyze_only \\
    --do_random

echo "Done. Job \$SLURM_JOB_ID"
EOF
    chmod +x "$SCRIPT"
    local JOB_ID
    JOB_ID=$(sbatch "$SCRIPT" | awk '{print $NF}')
    echo "  Submitted random control: job $JOB_ID (2h, 4 GPUs)"
    rm -f "$SCRIPT"
}

mkdir -p "$PROJECT_DIR/logs"

# Submit method jobs
FLAGS="--do_incremental --do_individual --do_bottom"

submit_method_job "eap_ig" \
    "${BASE_DIR}/eap_ig/aggregated/head_importance.pt" \
    "${OUTPUT_BASE}/eap_ig" \
    "06:00:00" \
    "$FLAGS"

submit_method_job "neurosurgery" \
    "${BASE_DIR}/neurosurgery/aggregated/head_importance.pt" \
    "${OUTPUT_BASE}/neurosurgery" \
    "06:00:00" \
    "$FLAGS"

submit_method_job "retrieval" \
    "${BASE_DIR}/retrieval/aggregated/head_importance.pt" \
    "${OUTPUT_BASE}/retrieval" \
    "06:00:00" \
    "$FLAGS"

# Submit random control job
submit_random_job

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "After completion, run cross-method analysis:"
echo "  python scripts/reasoning_head_analysis/analyze_systematic_ablation.py"
