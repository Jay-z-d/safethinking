#!/usr/bin/env bash
#SBATCH -J st-probe-full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

: "${INPUT_DIR:?Export INPUT_DIR as the completed merged representations}"
: "${FOLD_MANIFEST:?Export FOLD_MANIFEST as formal_folds.jsonl}"
: "${METHOD:?Export METHOD}"
: "${BEFORE:?Export BEFORE checkpoint}"
: "${AFTER:?Export AFTER checkpoint}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR for probe chunks}"
: "${OUTPUT_PREFIX:?Export OUTPUT_PREFIX, e.g. guided-to-true}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
BOOTSTRAP_CHUNK_SIZE=${BOOTSTRAP_CHUNK_SIZE:-50}
LAYER_COLUMN=${LAYER_COLUMN:--1}
WEIGHTING=${WEIGHTING:-pair}
FOLDS=${FOLDS:-5}
BOOTSTRAP_START=$((SLURM_ARRAY_TASK_ID * BOOTSTRAP_CHUNK_SIZE))
printf -v CHUNK_ID "%05d" "$SLURM_ARRAY_TASK_ID"
OUTPUT="$OUTPUT_DIR/$OUTPUT_PREFIX.chunk-$CHUNK_ID.json"
PREDICTION_ARGS=(--skip-predictions)
if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
  PREDICTION_ARGS=()
fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

mkdir -p "$OUTPUT_DIR"
srun python -m representation_analysis.probe_separability \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT" \
  --method "$METHOD" \
  --before "$BEFORE" \
  --after "$AFTER" \
  --fold-manifest "$FOLD_MANIFEST" \
  --group-field source_group \
  --weighting "$WEIGHTING" \
  --layer-column "$LAYER_COLUMN" \
  --folds "$FOLDS" \
  --bootstrap-samples "$BOOTSTRAP_CHUNK_SIZE" \
  --bootstrap-start-index "$BOOTSTRAP_START" \
  --store-bootstrap-draws \
  "${PREDICTION_ARGS[@]}"
