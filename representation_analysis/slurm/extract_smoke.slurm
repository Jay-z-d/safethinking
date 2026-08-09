#!/usr/bin/env bash
#SBATCH -J st-repr-smoke
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${MODEL:?Export MODEL as the target model directory}"
: "${INPUT:?Export INPUT as an IA boundary_generation.jsonl}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR as a new/empty output directory}"

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Work/safethinking}
CONDA_ENV=${CONDA_ENV:-safethinking}
PYTHON_BIN=${PYTHON_BIN:-python}
LIMIT=${LIMIT:-40}
BATCH_SIZE=${BATCH_SIZE:-2}
LAYERS=${LAYERS:--1}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
MAX_LENGTH=${MAX_LENGTH:-4096}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "model=$MODEL"
echo "input=$INPUT"
echo "output_dir=$OUTPUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PYTHON_BIN" -m representation_analysis.extract_hidden_states \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --layers="$LAYERS" \
  --pooling last_token \
  --batch-size "$BATCH_SIZE" \
  --shard-size 128 \
  --dtype "$MODEL_DTYPE" \
  --output-dtype float16 \
  --max-length "$MAX_LENGTH" \
  --limit "$LIMIT"

