#!/usr/bin/env bash
#SBATCH -J st-ia-full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err

set -euo pipefail

: "${MODEL:?Export MODEL as the target model directory}"
: "${PAIR_SHARDS_DIR:?Export PAIR_SHARDS_DIR as the completed pair shards}"
: "${GENERATION_DIR:?Export GENERATION_DIR as the generation output directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

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

printf -v SHARD_ID "%05d" "$SLURM_ARRAY_TASK_ID"
PAIRS="$PAIR_SHARDS_DIR/pairs-$SHARD_ID.jsonl"
OUTPUT="$GENERATION_DIR/generation-$SHARD_ID.jsonl"
if [[ ! -f "$PAIR_SHARDS_DIR/_SUCCESS" || ! -f "$PAIRS" ]]; then
  echo "Pair shard is incomplete or missing: $PAIRS" >&2
  exit 2
fi
mkdir -p "$GENERATION_DIR"

echo "job_id=${SLURM_JOB_ID:-unknown} array_task=$SLURM_ARRAY_TASK_ID"
echo "node=${SLURMD_NODENAME:-unknown} pairs=$PAIRS output=$OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python generate_wildjailbreak_responses_vllm.py \
  --input "$PAIRS" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  --method safe_llm_intention_analysis \
  --method-name llama31_8b_ia_formal_full \
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
