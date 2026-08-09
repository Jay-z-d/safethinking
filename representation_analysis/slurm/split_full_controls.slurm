#!/usr/bin/env bash
#SBATCH -J st-split-controls
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT:?Export INPUT as the complete controlled JSONL}"
: "${PAIR_SHARDS_DIR:?Export PAIR_SHARDS_DIR}"
: "${OUTPUT_DIR:?Export OUTPUT_DIR as a new controlled-shard directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
SEEDS_CSV=${SEEDS_CSV:-"42,43,44"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.formal_shards split-controlled \
  --input "$INPUT" \
  --pair-shards-dir "$PAIR_SHARDS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seeds "$SEEDS_CSV"
