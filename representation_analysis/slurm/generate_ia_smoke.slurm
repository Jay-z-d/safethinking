#!/usr/bin/env bash
#SBATCH -J st-ia-generate
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

: "${MODEL:?Export MODEL as the target model directory}"
: "${OUTPUT:?Export OUTPUT as the boundary_generation.jsonl path}"

PROJECT_ROOT=${PROJECT_ROOT:-$HOME/Work/safethinking}
CONDA_ENV=${CONDA_ENV:-safethinking}
PYTHON_BIN=${PYTHON_BIN:-python}
PAIRS=${PAIRS:-$PROJECT_ROOT/data/boundary/wildjailbreak_pairs.jsonl}
METHODS_ROOT=${METHODS_ROOT:-$PROJECT_ROOT/methods}
LIMIT=${LIMIT:-40}
BATCH_SIZE=${BATCH_SIZE:-16}

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$PYTHON_BIN" generate_wildjailbreak_responses_vllm.py \
  --input "$PAIRS" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --method safe_llm_intention_analysis \
  --method-name llama31_8b_ia_repr_smoke \
  --methods-root "$METHODS_ROOT" \
  --seeds 42 \
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
  --store-prompts \
  --limit "$LIMIT"

