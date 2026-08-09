#!/usr/bin/env bash
#SBATCH -J st-probe-formal
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT_DIR:?Export INPUT_DIR as a completed representation directory}"
: "${OUTPUT:?Export OUTPUT as the probe JSON path}"
: "${METHOD:?Export METHOD as the stored method name}"
: "${FOLD_MANIFEST:?Export FOLD_MANIFEST as formal_folds.jsonl}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
BEFORE=${BEFORE:-h_guided}
AFTER=${AFTER:-h_analysis_boundary_true}
FOLDS=${FOLDS:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-200}
LAYER_COLUMN=${LAYER_COLUMN:--1}
WEIGHTING=${WEIGHTING:-pair}
RUN_ID=${RUN_ID:-42}
RUN_ARGS=()
if [[ "$RUN_ID" != "all" ]]; then
  RUN_ARGS=(--run-id "$RUN_ID")
fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.probe_separability \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT" \
  --method "$METHOD" \
  --before "$BEFORE" \
  --after "$AFTER" \
  --fold-manifest "$FOLD_MANIFEST" \
  --group-field source_group \
  --weighting "$WEIGHTING" \
  "${RUN_ARGS[@]}" \
  --layer-column "$LAYER_COLUMN" \
  --folds "$FOLDS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
