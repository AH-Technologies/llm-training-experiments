#!/bin/bash
#SBATCH --job-name=ablate-v2
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/projects/nn12068k/haaklau/llm-training-experiments/logs/ablate_v2_%j.log
#SBATCH --error=/cluster/projects/nn12068k/haaklau/llm-training-experiments/logs/ablate_v2_%j.err
#SBATCH --chdir=/cluster/projects/nn12068k/haaklau/llm-training-experiments

# Thesis-quality ablation study with multi-benchmark + statistics.
#
# IMPORTANT: all list-valued vars must use '|' as separator, not ',',
# because sbatch --export parses commas as variable separators.
#
# Usage examples:
#
#   # pi13 ckpt 700 on full MATH-500:
#   sbatch --export=ALL,\
#     MODEL="reasoning_head_analysis/checkpoints/.../global_step_700",\
#     IMPORTANCE_PATH=".../head_importance.pt",\
#     OUTPUT_DIR="reasoning_head_analysis/results/ablate_v2_pi13_step700_math500",\
#     USE_ACTIVE_HEADS=1,EXCLUDE_LAYER0=1,TOP_K=8,\
#     BENCHMARK=math500,NUM_PROBLEMS=500,N_SAMPLES=4 \
#     reasoning_head_analysis/slurm/ablate_heads_v2.slurm
#
#   # base model with 6 overlap heads scaled up:
#   sbatch --export=ALL,\
#     MODEL="Qwen/Qwen2.5-Math-1.5B",\
#     IMPORTANCE_PATH="reasoning_head_analysis/results/Qwen_Qwen2.5-Math-1.5B/head_importance.pt",\
#     OUTPUT_DIR="reasoning_head_analysis/results/ablate_v2_base_overlap_math500",\
#     HEADS="11.8|15.7|18.8|19.6|19.11|23.9",TOP_K=6,\
#     BENCHMARK=math500,NUM_PROBLEMS=500,N_SAMPLES=4 \
#     reasoning_head_analysis/slurm/ablate_heads_v2.slurm

set -e

MODEL=${MODEL:?"MODEL is required"}
IMPORTANCE_PATH=${IMPORTANCE_PATH:?"IMPORTANCE_PATH is required"}
OUTPUT_DIR=${OUTPUT_DIR:?"OUTPUT_DIR is required"}
BENCHMARK=${BENCHMARK:-"math500"}
BENCHMARKS=${BENCHMARKS:-""}   # pipe-separated list, e.g. "math500|aime24|amc"
NUM_PROBLEMS=${NUM_PROBLEMS:-500}
MATH500_N=${MATH500_N:-""}
AIME24_N=${AIME24_N:-30}
AMC_N=${AMC_N:-40}
N_SAMPLES=${N_SAMPLES:-4}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
TOP_P=${TOP_P:-0.95}
BATCH_SIZE=${BATCH_SIZE:-8}
SHARD_IDX=${SHARD_IDX:-0}
# N_SHARDS: leave UNSET by default so the orchestrator below falls back to
# $(torch.cuda.device_count()). Setting it to 1 here was the bug that left
# allocated GPUs idle.
N_SHARDS=${N_SHARDS:-}
ANALYZE_ONLY=${ANALYZE_ONLY:-0}
TOP_K=${TOP_K:-8}
SCALES=${SCALES:-"0.0|0.25|0.5|0.75|1.5|2.0|3.0|4.0"}
N_RANDOM=${N_RANDOM:-10}
USE_ACTIVE_HEADS=${USE_ACTIVE_HEADS:-0}
EXCLUDE_LAYER0=${EXCLUDE_LAYER0:-0}
HEADS=${HEADS:-""}
SKIP_INDIVIDUAL=${SKIP_INDIVIDUAL:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
MATH500_PATH=${MATH500_PATH:-"data/math500.parquet"}
EXTRA_CONDITIONS_FILE=${EXTRA_CONDITIONS_FILE:-""}

echo "==========================================="
echo "Ablation v2 — $BENCHMARK ($NUM_PROBLEMS × $N_SAMPLES samples)"
echo "==========================================="
echo "Model:       $MODEL"
echo "Importance:  $IMPORTANCE_PATH"
echo "Output:      $OUTPUT_DIR"
echo "Top-K:       $TOP_K"
echo "Scales:      $SCALES"
echo "N random:    $N_RANDOM"
echo "==========================================="

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR="/cluster/projects/nn12068k/haaklau/llm-training-experiments"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_INSECURE_SERIALIZATION=1  # needed for apply_model with function args
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

mkdir -p logs "$HF_HOME"

EXTRA=""
[ "$USE_ACTIVE_HEADS" = "1" ] && EXTRA="$EXTRA --use_active_heads"
[ "$EXCLUDE_LAYER0" = "1" ]   && EXTRA="$EXTRA --exclude_layer0"
[ "$SKIP_INDIVIDUAL" = "1" ]  && EXTRA="$EXTRA --skip_individual"
[ -n "$HEADS" ]               && EXTRA="$EXTRA --heads $HEADS"
[ -n "$EXTRA_CONDITIONS_FILE" ] && EXTRA="$EXTRA --extra_conditions_file $EXTRA_CONDITIONS_FILE"

# How many GPUs did SLURM give us? Shard condition list across them, run one
# Python worker per GPU in parallel; then one --analyze_only pass at the end.
N_GPUS_DETECTED=$(python -c "import torch; print(torch.cuda.device_count())")
N_SHARDS_EFFECTIVE=${N_SHARDS:-$N_GPUS_DETECTED}
echo "GPUs visible: $N_GPUS_DETECTED  |  using $N_SHARDS_EFFECTIVE shards"

if [ "$ANALYZE_ONLY" = "1" ]; then
    # Explicit analyze-only mode — no generation.
    python -m reasoning_head_analysis.ablate_heads_v2_vllm \
        --model "$MODEL" \
        --importance_path "$IMPORTANCE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --benchmark "$BENCHMARK" \
        --math500_path "$MATH500_PATH" \
        --num_problems "$NUM_PROBLEMS" \
        --aime24_n "$AIME24_N" \
        --amc_n "$AMC_N" \
        ${BENCHMARKS:+--benchmarks "$BENCHMARKS"} \
        ${MATH500_N:+--math500_n "$MATH500_N"} \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --top_k "$TOP_K" \
        --scales "$SCALES" \
        --n_random "$N_RANDOM" \
        --analyze_only \
        $EXTRA
else
    # Fork one worker per GPU. Each sees only its own GPU via CUDA_VISIBLE_DEVICES.
    mkdir -p "$OUTPUT_DIR"
    pids=()
    for SHARD in $(seq 0 $((N_SHARDS_EFFECTIVE - 1))); do
        GPU_IDX=$((SHARD % N_GPUS_DETECTED))
        LOG="$OUTPUT_DIR/shard${SHARD}.log"
        echo "Launching shard $SHARD on GPU $GPU_IDX → $LOG"
        CUDA_VISIBLE_DEVICES=$GPU_IDX python -m reasoning_head_analysis.ablate_heads_v2_vllm \
            --model "$MODEL" \
            --importance_path "$IMPORTANCE_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --benchmark "$BENCHMARK" \
            --math500_path "$MATH500_PATH" \
            --num_problems "$NUM_PROBLEMS" \
            --n_samples "$N_SAMPLES" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --gpu_memory_utilization "$GPU_MEM_UTIL" \
            --top_k "$TOP_K" \
            --scales "$SCALES" \
            --n_random "$N_RANDOM" \
            --shard_idx "$SHARD" \
            --n_shards "$N_SHARDS_EFFECTIVE" \
            $EXTRA \
            > "$LOG" 2>&1 &
        pids+=($!)
    done

    echo "Waiting for ${#pids[@]} worker(s)..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
            echo "Worker pid=$pid failed"
        fi
    done
    if [ $failed -eq 1 ]; then
        echo "At least one shard failed; logs are in $OUTPUT_DIR/shard*.log"
        exit 1
    fi

    echo "All shards done. Running analyze-only pass..."
    python -m reasoning_head_analysis.ablate_heads_v2_vllm \
        --model "$MODEL" \
        --importance_path "$IMPORTANCE_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --benchmark "$BENCHMARK" \
        --math500_path "$MATH500_PATH" \
        --num_problems "$NUM_PROBLEMS" \
        --aime24_n "$AIME24_N" \
        --amc_n "$AMC_N" \
        ${BENCHMARKS:+--benchmarks "$BENCHMARKS"} \
        ${MATH500_N:+--math500_n "$MATH500_N"} \
        --n_samples "$N_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --top_k "$TOP_K" \
        --scales "$SCALES" \
        --n_random "$N_RANDOM" \
        --analyze_only \
        $EXTRA
fi

echo "Done. Job $SLURM_JOB_ID"
