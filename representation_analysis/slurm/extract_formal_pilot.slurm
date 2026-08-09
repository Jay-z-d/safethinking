#!/usr/bin/env bash
#SBATCH -J st-repr-pilot
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${MODEL:?Export MODEL as the exact generation model directory}"
: "${INPUT:?Export INPUT as the controlled IA generation JSONL}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR as a new/empty representation directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
BATCH_SIZE=${BATCH_SIZE:-2}
LAYERS=${LAYERS:-"0,4,8,12,16,20,24,28,32"}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
MAX_LENGTH=${MAX_LENGTH:-4096}
export TOKENIZERS_PARALLELISM=false

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "model=$MODEL"
echo "input=$INPUT"
echo "output_dir=$OUTPUT_DIR"
echo "layers=$LAYERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python -m representation_analysis.extract_hidden_states \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --layers="$LAYERS" \
  --pooling last_token \
  --batch-size "$BATCH_SIZE" \
  --shard-size 128 \
  --dtype "$MODEL_DTYPE" \
  --output-dtype float16 \
  --max-length "$MAX_LENGTH"
