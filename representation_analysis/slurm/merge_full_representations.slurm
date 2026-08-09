#!/usr/bin/env bash
#SBATCH -J st-merge-repr
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${CONTROLLED_SHARDS_DIR:?Export CONTROLLED_SHARDS_DIR}"
: "${REPRESENTATION_ROOT:?Export REPRESENTATION_ROOT}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR as a new combined representation directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.formal_shards merge-representations \
  --controlled-shards-dir "$CONTROLLED_SHARDS_DIR" \
  --representation-root "$REPRESENTATION_ROOT" \
  --output-dir "$OUTPUT_DIR"
