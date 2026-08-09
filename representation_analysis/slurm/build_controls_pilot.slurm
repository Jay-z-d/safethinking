#!/usr/bin/env bash
#SBATCH -J st-ia-controls
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${INPUT:?Export INPUT as the completed IA generation JSONL}"
: "${OUTPUT:?Export OUTPUT as a new controlled-generation JSONL}"
: "${FOLD_MANIFEST:?Export FOLD_MANIFEST as formal_folds.jsonl}"
: "${TOKENIZER:?Export TOKENIZER as the target tokenizer/model directory}"

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
