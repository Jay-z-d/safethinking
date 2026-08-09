#!/usr/bin/env bash
#SBATCH -J st-controls-full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT:?Export INPUT as the complete merged generation JSONL}"
: "${OUTPUT:?Export OUTPUT as a new controlled JSONL}"
: "${FOLD_MANIFEST:?Export FOLD_MANIFEST as formal_folds.jsonl}"
: "${TOKENIZER:?Export TOKENIZER as the exact model/tokenizer directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
export TOKENIZERS_PARALLELISM=false

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

srun python -m representation_analysis.build_controls \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --fold-manifest "$FOLD_MANIFEST" \
  --tokenizer "$TOKENIZER"
