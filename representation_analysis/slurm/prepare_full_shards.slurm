#!/usr/bin/env bash
#SBATCH -J st-full-shards
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT:?Export INPUT as the frozen 3,343-pair JSONL}"
: "${FOLD_MANIFEST:?Export FOLD_MANIFEST as formal_folds.jsonl}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR as a new pair-shard directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-250}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.formal_shards prepare-pairs \
  --input "$INPUT" \
  --fold-manifest "$FOLD_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --pairs-per-shard "$PAIRS_PER_SHARD"
