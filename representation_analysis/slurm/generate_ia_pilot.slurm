#!/usr/bin/env bash
#SBATCH -J st-ia-pilot
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${MODEL:?Export MODEL as the target model directory}"
: "${PAIRS:?Export PAIRS as the frozen 100-pair pilot JSONL}"
: "${OUTPUT:?Export OUTPUT as a new/resumable generation JSONL}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ENV=${CONDA_ENV:-safethinking_jayzd}
METHODS_ROOT=${METHODS_ROOT:-$PROJECT_ROOT/methods}
BATCH_SIZE=${BATCH_SIZE:-16}
SEEDS=${SEEDS:-"42 43 44"}
read -r -a SEED_ARGS <<< "$SEEDS"
export TOKENIZERS_PARALLELISM=false

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

if [[ ! -f "$METHODS_ROOT/SafeLLM_with_IntentionAnalysis/demo/IA_demo.py" ]]; then
  echo "Missing official IA repository under $METHODS_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "model=$MODEL"
echo "pairs=$PAIRS"
echo "output=$OUTPUT"
echo "seeds=${SEED_ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python generate_wildjailbreak_responses_vllm.py \
  --input "$PAIRS" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --method safe_llm_intention_analysis \
  --method-name llama31_8b_ia_formal_pilot \
  --methods-root "$METHODS_ROOT" \
  --seeds "${SEED_ARGS[@]}" \
  --stage1-max-new-tokens 256 \
  --max-new-tokens 512 \
  --max-model-len 4096 \
  --batch-size "$BATCH_SIZE" \
  --temperature 0.6 \
  --top-p 0.9 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --dtype auto \
  --enable-thinking false \
  --store-prompts
