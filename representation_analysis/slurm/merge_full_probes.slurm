#!/usr/bin/env bash
#SBATCH -J st-merge-probe
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT_DIR:?Export INPUT_DIR as the probe-chunk directory}"
: "${INPUT_PREFIX:?Export INPUT_PREFIX}"
: "${OUTPUT:?Export OUTPUT as a new final probe JSON}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
CHUNKS=${CHUNKS:-20}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.formal_shards merge-probes \
  --input-dir "$INPUT_DIR" \
  --prefix "$INPUT_PREFIX" \
  --chunks "$CHUNKS" \
  --output "$OUTPUT"
