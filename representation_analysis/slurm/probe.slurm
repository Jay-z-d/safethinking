#!/usr/bin/env bash
#SBATCH -J st-linear-probe
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT_DIR:?Export INPUT_DIR as an extracted representation directory}"
: "${OUTPUT:?Export OUTPUT as the probe JSON path}"
: "${METHOD:?Export METHOD as the method name stored in metadata}"

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Work/safethinking}
CONDA_ENV=${CONDA_ENV:-safethinking}
PYTHON_BIN=${PYTHON_BIN:-python}
BEFORE=${BEFORE:-h_query}
AFTER=${AFTER:-h_reasoned}
FOLDS=${FOLDS:-5}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun "$PYTHON_BIN" -m representation_analysis.probe_separability \
  --input-dir "$INPUT_DIR" \
  --output "$OUTPUT" \
  --method "$METHOD" \
  --before "$BEFORE" \
  --after "$AFTER" \
  --folds "$FOLDS" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"

