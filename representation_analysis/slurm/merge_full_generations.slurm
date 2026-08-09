#!/usr/bin/env bash
#SBATCH -J st-merge-gen
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${PAIR_SHARDS_DIR:?Export PAIR_SHARDS_DIR}"
: "${GENERATION_DIR:?Export GENERATION_DIR}"
: "${OUTPUT:?Export OUTPUT as a new merged JSONL}"
: "${MODEL:?Export MODEL as the exact generation model}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
SEEDS_CSV=${SEEDS_CSV:-"42,43,44"}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.formal_shards merge-generations \
  --pair-shards-dir "$PAIR_SHARDS_DIR" \
  --generation-dir "$GENERATION_DIR" \
  --output "$OUTPUT" \
  --seeds "$SEEDS_CSV" \
  --method safe_llm_intention_analysis \
  --model "$MODEL"
