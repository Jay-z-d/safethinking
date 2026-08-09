#!/usr/bin/env bash
#SBATCH -J st-formal-data
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
INPUT=${INPUT:-$PROJECT_ROOT/data/boundary/wildjailbreak_pairs.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs/formal-data/llama31-ia-v1}
PILOT_PAIRS=${PILOT_PAIRS:-100}
FOLDS=${FOLDS:-5}
SEED=${SEED:-42}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

python -m representation_analysis.prepare_formal_data \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --pilot-pairs "$PILOT_PAIRS" \
  --folds "$FOLDS" \
  --seed "$SEED"
