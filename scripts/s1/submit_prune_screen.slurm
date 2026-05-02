#!/bin/bash
#SBATCH --job-name=s1-prune-screen
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=800G
#SBATCH --output=logs/s1-prune-screen-%j.out
#SBATCH --error=logs/s1-prune-screen-%j.err

# S1K pruning *screening* sweep: thirds-based proxy for ranking strategies.
#
# For each registered ranking strategy (see s1.pruning.strategies), we sort
# all 1000 s1K examples by the strategy's per-sample score and split into
# top / middle / bottom thirds (333 each). Each third gets its own short SFT
# run, then we eval on AMC + AIME 2025 (113 problems total) — enough signal
# to ordinal-rank the strategy buckets, fast enough to iterate.
#
# Departures from the full sweep (submit_prune_sweep.slurm):
#   - 3 epochs (~63 steps), constant LR 1e-5 (no decay), N=333
#   - eval on amc + aime25 only
#   - hub subfolder is {strategy}/{position} under alexauren/s1-pruning
#
# Usage:
#   sbatch scripts/s1/submit_prune_screen.slurm

set -e

echo "=========================================="
echo "S1K Pruning Screening Sweep (5 strategies × 3 thirds)"
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

# ---- Step 0: ensure baseline data + skills exist ----
if [ ! -f "data/s1K/s1k.parquet" ]; then
    echo "ERROR: data/s1K/s1k.parquet missing."
    exit 1
fi
SKILLS_PATH="data/s1K/s1k_skills.parquet"
if [ ! -f "$SKILLS_PATH" ]; then
    echo "ERROR: $SKILLS_PATH missing — run submit_prune_sweep.slurm once to tag, or invoke s1.pruning.tag_skills directly."
    exit 1
fi
if [ ! -f "data/val/amc_verl.parquet" ] || [ ! -f "data/val/aime_2025_verl.parquet" ]; then
    echo "ERROR: validation parquets missing under data/val/. Run scripts/rlvr_grokking/prepare_val_datasets.py first."
    exit 1
fi

export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}

# ---- Step 0.5: precompute base-model NLL (one-off, ~10 min on H200×4) ----
# Used by base_loss and base_logprob_mean screening strategies. Idempotent:
# skipped if the parquet already exists. Runs on a single node to leave the
# rest of the allocation idle just briefly; bf16 fits comfortably at TP=4.
NLL_PATH="data/s1K/s1k_base_nll.parquet"
if [ ! -f "$NLL_PATH" ]; then
    echo ""
    echo "Computing base-model NLL on s1K → $NLL_PATH"
    srun --nodes=1 --ntasks-per-node=1 --gpus-per-node=4 --export=ALL \
        bash -c "unset ROCR_VISIBLE_DEVICES; python -m s1.pruning.compute_base_nll \
            --input data/s1K/s1k.parquet \
            --output $NLL_PATH \
            --tp 4"
else
    echo "Base NLL parquet exists at $NLL_PATH — skipping precomputation."
fi

# ---- Cell list ----
# (strategy, position) pairs. To add a new ranking strategy: register a
# scoring function in src/s1/pruning/strategies.py, then add three lines
# below — top, middle, bottom.
SCREEN_CELLS=(
    "response_length top"
    "response_length middle"
    "response_length bottom"
    "skill_count top"
    "skill_count middle"
    "skill_count bottom"
    "base_loss top"
    "base_loss middle"
    "base_loss bottom"
    "base_logprob_mean top"
    "base_logprob_mean middle"
    "base_logprob_mean bottom"
)

# Hyperparameters held constant across all screening cells.
N=333
EPOCHS=3
HF_REPO="alexauren/s1-pruning"

for cell in "${SCREEN_CELLS[@]}"; do
    read -r STRATEGY POSITION <<< "$cell"
    CELL_TAG="screen_${STRATEGY}_${POSITION}"
    PRUNED_PARQUET="data/s1K/pruned/${CELL_TAG}.parquet"
    SFT_PARQUET="data/s1K/pruned/${CELL_TAG}_sft.parquet"
    OUTPUT_DIR="checkpoints/s1_${CELL_TAG}"
    EVAL_OUT="eval_results/${CELL_TAG}.json"
    HF_SUBFOLDER="screen/${STRATEGY}/${POSITION}"

    echo ""
    echo "=========================================="
    echo "Cell: $CELL_TAG (n=$N, $EPOCHS epochs, lr=const)"
    echo "=========================================="

    # Resumability: eval JSON is the durable completion marker. Local checkpoint
    # is removed after upload+eval succeed, so directory presence isn't reliable.
    if [ -f "$EVAL_OUT" ]; then
        echo "Already complete; skipping."
        continue
    fi

    # Per-cell subshell: a failure here won't kill remaining cells.
    if ! (
        set -e

        # Step 1: prune (thirds-based)
        if [ ! -f "$PRUNED_PARQUET" ]; then
            python -m s1.pruning.prune \
                --strategy "$STRATEGY" \
                --position "$POSITION" \
                --n "$N" \
                --seed 42 \
                --input data/s1K/s1k.parquet \
                --skills "$SKILLS_PATH" \
                --output "$PRUNED_PARQUET"
        fi

        # Step 2: format for SFT
        if [ ! -f "$SFT_PARQUET" ]; then
            python scripts/s1/prepare_s1k_sft.py \
                --input "$PRUNED_PARQUET" \
                --output "$SFT_PARQUET"
        fi

        # Step 3: SFT (constant LR, no decay; --save-final-only writes one
        # top-level checkpoint that gets bf16-cast and uploaded in-process).
        if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
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
                --lr-scheduler-type constant_with_warmup \
                --warmup-ratio 0.05 \
                --weight-decay 1e-4 \
                --max-grad-norm 1.0 \
                --save-every-n-epochs ${EPOCHS} \
                --save-final-only \
                --hf-repo ${HF_REPO} \
                --hf-subfolder ${HF_SUBFOLDER} \
                --wandb-project s1-qwen32b-sft \
                --wandb-run-name ${CELL_TAG} \
                --seed 42"
            srun --nodes=${NUM_NODES} --ntasks-per-node=1 --export=ALL \
                bash -c "unset ROCR_VISIBLE_DEVICES; ${ACCEL_CMD}"
        fi

        # Step 4: eval on AMC + AIME 2025 (113 problems total, ~5 min on H200×4).
        if [ ! -f "$EVAL_OUT" ]; then
            if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
                echo "ERROR: no trained model at $OUTPUT_DIR"
                exit 1
            fi
            EVAL_TMP="eval_results/${CELL_TAG}_tmp"
            mkdir -p "$EVAL_TMP"
            python -m s1.eval \
                --model "$OUTPUT_DIR" \
                --benchmarks amc aime25 \
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

        # Step 5: reclaim local disk. Hub copy at $HF_REPO/$HF_SUBFOLDER remains.
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Reclaiming $OUTPUT_DIR (model archived at $HF_REPO/$HF_SUBFOLDER)"
            rm -rf "$OUTPUT_DIR"
        fi
    ); then
        echo "Cell $CELL_TAG FAILED — continuing to next cell"
    fi
done

# ---- Baseline cells ----
# Sanity-check baselines that don't fit the thirds framework:
#   - random_n333: random 333-row subset, 3 epochs (same compute as a screening
#     cell). Acts as the noise floor — strategies whose middle bucket sits at
#     this level aren't actually carrying signal.
#   - full_n1000: full s1K, 1 epoch. Same total step budget as a screening
#     cell (1000/16 × 1 = 63 steps; 333/16 × 3 = 63 steps). Tells us whether
#     thirds-with-3-epochs beats just running over everything once.
#
# Hub layout: alexauren/s1-pruning/baseline/{random_n333,full_n1000}/
# Eval JSONs:  eval_results/baseline_{random_n333,full_n1000}.json
BASELINE_CELLS=(
    "random_n333"
    "full_n1000"
)

for tag in "${BASELINE_CELLS[@]}"; do
    case "$tag" in
        random_n333)  STRATEGY=random; B_N=333;  B_EPOCHS=3 ;;
        full_n1000)   STRATEGY=random; B_N=1000; B_EPOCHS=1 ;;
        *) echo "Unknown baseline tag: $tag — skipping"; continue ;;
    esac

    CELL_TAG="baseline_${tag}"
    PRUNED_PARQUET="data/s1K/pruned/${CELL_TAG}.parquet"
    SFT_PARQUET="data/s1K/pruned/${CELL_TAG}_sft.parquet"
    OUTPUT_DIR="checkpoints/s1_${CELL_TAG}"
    EVAL_OUT="eval_results/${CELL_TAG}.json"
    HF_SUBFOLDER="baseline/${tag}"

    echo ""
    echo "=========================================="
    echo "Baseline: $CELL_TAG (n=$B_N, $B_EPOCHS epochs, lr=const)"
    echo "=========================================="

    if [ -f "$EVAL_OUT" ]; then
        echo "Already complete; skipping."
        continue
    fi

    if ! (
        set -e

        # Step 1: prune (random selection; for full_n1000 this returns all
        # 1000 rows in seeded random.sample order, equivalent to a shuffle).
        if [ ! -f "$PRUNED_PARQUET" ]; then
            python -m s1.pruning.prune \
                --strategy "$STRATEGY" \
                --n "$B_N" \
                --seed 42 \
                --input data/s1K/s1k.parquet \
                --output "$PRUNED_PARQUET"
        fi

        # Step 2: format for SFT
        if [ ! -f "$SFT_PARQUET" ]; then
            python scripts/s1/prepare_s1k_sft.py \
                --input "$PRUNED_PARQUET" \
                --output "$SFT_PARQUET"
        fi

        # Step 3: SFT
        if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
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
                --num-epochs ${B_EPOCHS} \
                --batch-size 16 \
                --per-device-batch-size 1 \
                --sequence-length 16384 \
                --lr 1e-5 \
                --lr-scheduler-type constant_with_warmup \
                --warmup-ratio 0.05 \
                --weight-decay 1e-4 \
                --max-grad-norm 1.0 \
                --save-every-n-epochs ${B_EPOCHS} \
                --save-final-only \
                --hf-repo ${HF_REPO} \
                --hf-subfolder ${HF_SUBFOLDER} \
                --wandb-project s1-qwen32b-sft \
                --wandb-run-name ${CELL_TAG} \
                --seed 42"
            srun --nodes=${NUM_NODES} --ntasks-per-node=1 --export=ALL \
                bash -c "unset ROCR_VISIBLE_DEVICES; ${ACCEL_CMD}"
        fi

        # Step 4: eval (same AMC + AIME25 set as the screening cells).
        if [ ! -f "$EVAL_OUT" ]; then
            if [ ! -f "${OUTPUT_DIR}/config.json" ]; then
                echo "ERROR: no trained model at $OUTPUT_DIR"
                exit 1
            fi
            EVAL_TMP="eval_results/${CELL_TAG}_tmp"
            mkdir -p "$EVAL_TMP"
            python -m s1.eval \
                --model "$OUTPUT_DIR" \
                --benchmarks amc aime25 \
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

        # Step 5: reclaim local disk.
        if [ -d "$OUTPUT_DIR" ]; then
            echo "Reclaiming $OUTPUT_DIR (model archived at $HF_REPO/$HF_SUBFOLDER)"
            rm -rf "$OUTPUT_DIR"
        fi
    ); then
        echo "Cell $CELL_TAG FAILED — continuing to next cell"
    fi
done

# ---- Aggregation ----
python - <<'PY'
import json, glob, os

def _row_from_eval(path, strategy, position):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", {})
    row = {
        "strategy": strategy,
        "position": position,
        "amc": results.get("amc", {}).get("accuracy"),
        "aime25": results.get("aime25", {}).get("accuracy"),
    }
    scores = [v for v in (row["amc"], row["aime25"]) if v is not None]
    row["avg"] = sum(scores) / len(scores) if scores else None
    return row

rows = []
# Screening cells: strategy = ranking strategy, position = top/middle/bottom
for path in sorted(glob.glob("eval_results/screen_*.json")):
    name = os.path.basename(path).replace("screen_", "").replace(".json", "")
    parts = name.rsplit("_", 1)
    strategy, position = (parts[0], parts[1]) if len(parts) == 2 else (name, "?")
    rows.append(_row_from_eval(path, strategy, position))

# Baseline cells: strategy = "baseline", position = the baseline tag
for path in sorted(glob.glob("eval_results/baseline_*.json")):
    name = os.path.basename(path).replace("baseline_", "").replace(".json", "")
    rows.append(_row_from_eval(path, "baseline", name))

with open("results/pruning_screen.json", "w") as f:
    json.dump(rows, f, indent=2)
print("Aggregated to results/pruning_screen.json")
for r in rows:
    print(r)
PY

echo ""
echo "End: $(date)"
echo "=== Done ==="
