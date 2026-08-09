#!/usr/bin/env bash
#SBATCH -J st-repr-full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

: "${MODEL:?Export MODEL as the exact generation model directory}"
: "${CONTROLLED_SHARDS_DIR:?Export CONTROLLED_SHARDS_DIR}"
: "${REPRESENTATION_ROOT:?Export REPRESENTATION_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

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

printf -v SHARD_ID "%05d" "$SLURM_ARRAY_TASK_ID"
INPUT="$CONTROLLED_SHARDS_DIR/controlled-$SHARD_ID.jsonl"
OUTPUT_DIR="$REPRESENTATION_ROOT/repr-$SHARD_ID"
if [[ ! -f "$CONTROLLED_SHARDS_DIR/_SUCCESS" || ! -f "$INPUT" ]]; then
  echo "Controlled shard is incomplete or missing: $INPUT" >&2
  exit 2
fi
if [[ -f "$OUTPUT_DIR/_SUCCESS" ]]; then
  echo "Representation shard already complete: $OUTPUT_DIR"
  exit 0
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR")" ]]; then
  echo "Representation shard is partial; move it aside before retrying: $OUTPUT_DIR" >&2
  exit 3
fi

echo "job_id=${SLURM_JOB_ID:-unknown} array_task=$SLURM_ARRAY_TASK_ID"
echo "node=${SLURMD_NODENAME:-unknown} input=$INPUT output_dir=$OUTPUT_DIR"
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
