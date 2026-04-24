#!/bin/bash
#SBATCH --job-name=s1-prune-sweep
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=logs/s1-prune-sweep-%j.out
#SBATCH --error=logs/s1-prune-sweep-%j.err

# S1K pruning sweep: random vs skill_abundance across N ∈ {500, 250, 100}.
# Fixed optimizer-step budget (~315) per cell — num_epochs scaled with N.
# Skills parquet is tagged once (if missing); each cell then: prune → format
# → SFT (HF Trainer + accelerate FSDP via s1.sft_hf) → eval → aggregate.
#
# Uses the HF Trainer pipeline (s1.sft_hf), not the custom FSDP2 trainer
# (s1.sft_trainer) — the latter OOMs on Qwen-32B at seq_length=16384 even
# with FSDP offload on 16 H200s.
#
# Usage:
#   sbatch scripts/submit_prune_sweep.slurm

set -e

echo "=========================================="
echo "S1K Pruning Sweep (6 cells)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

mkdir -p logs checkpoints data/s1K/pruned eval_results results

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0
module load aws-ofi-nccl/1.14.1-GCCcore-13.3.0-CUDA-12.6.0

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

# --- HuggingFace cache ---
MODEL_CACHE_KEY="Qwen_Qwen2.5-32B-Instruct"
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

# Enable Rust-based parallel hub uploader (faster hub pushes when --hf-repo is set)
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Slingshot / NCCL / libfabric networking ---
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CROSS_NIC=1
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
export CPATH=/opt/cray/libfabric/1.22.0/include:${CPATH}
export NCCL_DEBUG=WARN

# --- CUDA paths ---
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LIBRARY_PATH=${MATH_LIBS}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${MATH_LIBS}:${LD_LIBRARY_PATH}

# --- Other ---
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Multi-node setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE (${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs)"

# ---- Step 0: ensure baseline data exists ----
if [ ! -f "data/s1K/s1k.parquet" ]; then
    echo "ERROR: data/s1K/s1k.parquet missing. Run the dataset download first."
    exit 1
fi

export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

# ---- Step 1: tag skills (once) ----
SKILLS_PATH="data/s1K/s1k_skills.parquet"
if [ ! -f "$SKILLS_PATH" ]; then
    echo "Tagging skills via Gemini API..."
    python -m s1.pruning.tag_skills \
        --input data/s1K/s1k.parquet \
        --output "$SKILLS_PATH"
else
    echo "Skills parquet already exists at $SKILLS_PATH — skipping tag pass."
fi

# ---- Step 2: iterate cells ----
# (strategy, N, num_epochs) triples. Epochs chosen so
# num_epochs * ceil(N/16) ≈ 315 steps.
CELLS=(
    "random 1000 5"
    "random 500 10"
    "skill_abundance 500 10"
)

for cell in "${CELLS[@]}"; do
    read -r STRATEGY N EPOCHS <<< "$cell"
    CELL_TAG="${STRATEGY}_n${N}"
    PRUNED_PARQUET="data/s1K/pruned/${CELL_TAG}.parquet"
    SFT_PARQUET="data/s1K/pruned/${CELL_TAG}_sft.parquet"
    OUTPUT_DIR="checkpoints/s1_prune_${CELL_TAG}"
    EVAL_OUT="eval_results/s1_prune_${CELL_TAG}.json"

    echo ""
    echo "=========================================="
    echo "Cell: $CELL_TAG (epochs=$EPOCHS)"
    echo "=========================================="

    # Skip if checkpoint already exists (resumability)
    if [ -d "$OUTPUT_DIR" ] && [ -f "$EVAL_OUT" ]; then
        echo "Already complete; skipping."
        continue
    fi

    # Per-cell subshell: a failure here won't kill other cells.
    if ! (
        set -e

        # Step 2a: prune
        if [ ! -f "$PRUNED_PARQUET" ]; then
            python -m s1.pruning.prune \
                --strategy "$STRATEGY" \
                --n "$N" \
                --seed 42 \
                --input data/s1K/s1k.parquet \
                --skills "$SKILLS_PATH" \
                --output "$PRUNED_PARQUET"
        fi

        # Step 2b: format for SFT
        if [ ! -f "$SFT_PARQUET" ]; then
            python scripts/prepare_s1k_sft.py \
                --input "$PRUNED_PARQUET" \
                --output "$SFT_PARQUET"
        fi

        # Step 2c: SFT via HF Trainer + accelerate FSDP.
        # With --hf-repo unset, sft_hf.py skips the duplicate top-level save
        # and leaves only the last checkpoint-N/ subdir. Accept either marker
        # so previously-completed cells (with legacy top-level files) aren't
        # retrained.
        if [ ! -f "${OUTPUT_DIR}/config.json" ] && \
           [ -z "$(ls ${OUTPUT_DIR}/checkpoint-*/config.json 2>/dev/null)" ]; then
            ACCEL_CMD="accelerate launch \
                --config_file ${PROJECT_DIR}/configs/accelerate_fsdp.yaml \
                --num_processes=${WORLD_SIZE} \
                --num_machines=${NUM_NODES} \
                --main_process_ip=${MASTER_ADDR} \
                --main_process_port=${MASTER_PORT} \
                --machine_rank=\${SLURM_PROCID} \
                -m s1.sft_hf \
                --model Qwen/Qwen2.5-32B-Instruct \
                --train-file ${SFT_PARQUET} \
                --output-dir ${OUTPUT_DIR} \
                --num-epochs ${EPOCHS} \
                --batch-size 16 \
                --per-device-batch-size 1 \
                --sequence-length 16384 \
                --lr 1e-5 \
                --warmup-ratio 0.05 \
                --weight-decay 1e-4 \
                --max-grad-norm 1.0 \
                --save-every-n-epochs ${EPOCHS} \
                --wandb-project s1-qwen32b-sft \
                --wandb-run-name s1_prune_${CELL_TAG}_steps315 \
                --seed 42"
            srun --nodes=${NUM_NODES} --ntasks-per-node=1 --export=ALL \
                bash -c "unset ROCR_VISIBLE_DEVICES; ${ACCEL_CMD}"
        fi

        # Step 2d: eval on MATH500 + AIME24 + GPQA-Diamond.
        # Prefer the top-level save (legacy / hf_repo-enabled runs); fall back
        # to the latest checkpoint-N subdir (default for this sweep).
        if [ ! -f "$EVAL_OUT" ]; then
            if [ -f "${OUTPUT_DIR}/config.json" ]; then
                MODEL_PATH="$OUTPUT_DIR"
            else
                MODEL_PATH=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
            fi
            if [ -z "$MODEL_PATH" ] || [ ! -f "${MODEL_PATH}/config.json" ]; then
                echo "ERROR: no trained model under $OUTPUT_DIR"
                exit 1
            fi
            # Per-cell tmp dir: eval.py names its output eval_<basename>.json,
            # and checkpoint-N basenames collide across cells with identical
            # step counts (e.g. random_n500 and skill_abundance_n500 both hit
            # step 320). Isolating the eval output dir avoids the clash.
            EVAL_TMP="eval_results/${CELL_TAG}_tmp"
            mkdir -p "$EVAL_TMP"
            python -m s1.eval \
                --model "$MODEL_PATH" \
                --benchmarks math500 aime24 gpqa \
                --output-dir "$EVAL_TMP"
            produced=$(ls ${EVAL_TMP}/eval_*.json 2>/dev/null | head -1)
            if [ -n "$produced" ]; then
                mv "$produced" "$EVAL_OUT"
                rmdir "$EVAL_TMP" 2>/dev/null || true
            else
                echo "ERROR: eval did not produce any output in ${EVAL_TMP}/"
                exit 1
            fi
        fi
    ); then
        echo "Cell $CELL_TAG FAILED — continuing to next cell"
    fi
done

# ---- Step 3: aggregate results ----
python - <<'PY'
import json, glob, os
rows = []
for path in sorted(glob.glob("eval_results/s1_prune_*.json")):
    with open(path) as f:
        data = json.load(f)
    cell = os.path.basename(path).replace("s1_prune_", "").replace(".json", "")
    # cell = "<strategy>_n<N>"
    strategy, n_part = cell.rsplit("_n", 1)
    results = data.get("results", {})
    row = {
        "strategy": strategy,
        "n": int(n_part),
        "math500": results.get("math500", {}).get("accuracy"),
        "aime24": results.get("aime24", {}).get("accuracy"),
        "gpqa": results.get("gpqa", {}).get("accuracy"),
    }
    scores = [v for v in (row["math500"], row["aime24"], row["gpqa"]) if v is not None]
    row["avg"] = sum(scores) / len(scores) if scores else None
    rows.append(row)
with open("results/pruning_sweep.json", "w") as f:
    json.dump(rows, f, indent=2)
print("Aggregated to results/pruning_sweep.json")
for r in rows:
    print(r)
PY

echo ""
echo "End: $(date)"
echo "=== Done ==="
