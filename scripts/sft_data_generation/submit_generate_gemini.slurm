#!/bin/bash
#SBATCH --job-name=gemini-gen
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gemini_gen_%j.out
#SBATCH --error=logs/gemini_gen_%j.err
#SBATCH --export=ALL

MODEL=${MODEL:-"gemini-3.1-flash-lite-preview"}
DATASET=${DATASET:-"data/one_example_datasets/pi1_r128.parquet"}
NUM_CORRECT=${NUM_CORRECT:-128}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
OUTPUT_DIR=${OUTPUT_DIR:-"data/sft_1shot_datasets/gemini_pi1"}
RPM=${RPM:-14}

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
echo "RPM: $RPM"

python src/sft_data_generation/generate_correct_gemini.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_correct "$NUM_CORRECT" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --rpm "$RPM" \
    --problem_idx 0
