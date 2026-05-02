#!/bin/bash
#SBATCH --job-name=claude-gen
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/claude_gen_%j.out
#SBATCH --error=logs/claude_gen_%j.err
#SBATCH --export=ALL

MODEL=${MODEL:-"claude-sonnet-4-20250514"}
DATASET=${DATASET:-"data/one_example_datasets/pi1_r128.parquet"}
NUM_CORRECT=${NUM_CORRECT:-128}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"data/sft_1shot_datasets/claude_pi1"}
CONCURRENCY=${CONCURRENCY:-3}

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:/cluster/projects/nn12068k/haaklau/llm-training-experiments/src"
export PYTHONUNBUFFERED=1

mkdir -p logs "$OUTPUT_DIR"

echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Num correct: $NUM_CORRECT"
echo "Temperature: $TEMPERATURE"
echo "Concurrency: $CONCURRENCY"

python src/sft_data_generation/generate_correct_claude.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_correct "$NUM_CORRECT" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --concurrency "$CONCURRENCY" \
    --problem_idx 0
