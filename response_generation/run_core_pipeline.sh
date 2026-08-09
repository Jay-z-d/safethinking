#!/usr/bin/env bash
# Run response generation and the three-metric evaluation pipeline for one model.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is not set: ${name}" >&2
    exit 2
  fi
}

for name in MODEL MODEL_TAG PAIRS REFUSAL_INDUCTION QWEN_GUARD_MODEL LLAMA_GUARD_MODEL; do
  require_env "${name}"
done

PYTHON=${PYTHON:-python}
METHODS_ROOT=${METHODS_ROOT:-${ROOT_DIR}/methods}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT_DIR}/outputs/response_generation}
RUN_TAG=${RUN_TAG:-core_v1}
METHODS=${METHODS:-"direct safe_llm_intention_analysis goal_prioritization sage"}
SEEDS=${SEEDS:-"42 43 44"}
CALIBRATION_SEED=${CALIBRATION_SEED:-42}
BENIGN_LIMIT=${BENIGN_LIMIT:-500}
HARMFUL_LIMIT=${HARMFUL_LIMIT:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
STAGE1_MAX_NEW_TOKENS=${STAGE1_MAX_NEW_TOKENS:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-128}
SCORING_BATCH_SIZE=${SCORING_BATCH_SIZE:-128}
GUARD_BATCH_SIZE=${GUARD_BATCH_SIZE:-128}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
DTYPE=${DTYPE:-auto}
ENABLE_THINKING=${ENABLE_THINKING:-auto}
STORE_PROMPTS=${STORE_PROMPTS:-false}

mkdir -p "${OUTPUT_ROOT}/refusal_patterns"

read -r -a METHOD_LIST <<< "${METHODS}"
read -r -a SEED_LIST <<< "${SEEDS}"

THINKING_ARGS=(--enable-thinking "${ENABLE_THINKING}")
STORE_PROMPTS_ARGS=()
if [[ "${STORE_PROMPTS}" == "true" ]]; then
  STORE_PROMPTS_ARGS=(--store-prompts)
fi

COMMON_GENERATION_ARGS=(
  --model "${MODEL}"
  --methods-root "${METHODS_ROOT}"
  --stage1-max-new-tokens "${STAGE1_MAX_NEW_TOKENS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --batch-size "${GENERATION_BATCH_SIZE}"
  --temperature 0.6
  --top-p 0.9
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  "${THINKING_ARGS[@]}"
  "${STORE_PROMPTS_ARGS[@]}"
)

COMMON_REFUSAL_ARGS=(
  --model "${MODEL}"
  --methods-root "${METHODS_ROOT}"
  --scoring-batch-size "${SCORING_BATCH_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  "${THINKING_ARGS[@]}"
)

COMMON_GUARD_ARGS=(
  --batch-size "${GUARD_BATCH_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-new-tokens 16
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
)

PREFIX=${OUTPUT_ROOT}/${MODEL_TAG}_${RUN_TAG}
BASE_REFUSAL_GENERATION=${PREFIX}_direct.refusal_induction.generation.jsonl
PATTERNS=${OUTPUT_ROOT}/refusal_patterns/${MODEL_TAG}_${RUN_TAG}_base_direct.json

if [[ -z "${MIXED_CALIBRATION_INPUT:-}" ]]; then
  MIXED_CALIBRATION_INPUT=${OUTPUT_ROOT}/${MODEL_TAG}_${RUN_TAG}.mixed_calibration_input.jsonl
  if [[ ! -f "${MIXED_CALIBRATION_INPUT}" ]]; then
    "${PYTHON}" -m response_generation.prepare_mixed_calibration \
      --input-pairs "${PAIRS}" \
      --output "${MIXED_CALIBRATION_INPUT}" \
      --benign-limit "${BENIGN_LIMIT}" \
      --harmful-limit "${HARMFUL_LIMIT}"
  fi
fi

BASE_MIXED_GENERATION=${PREFIX}_direct.mixed_calibration.generation.jsonl
BASE_MIXED_RAW=${PREFIX}_direct.mixed_calibration.refusal_raw.jsonl

echo "[1/4] Preparing Base refusal templates for ${MODEL_TAG}"
"${PYTHON}" generate_wildjailbreak_responses_vllm.py \
  --input "${REFUSAL_INDUCTION}" \
  --output "${BASE_REFUSAL_GENERATION}" \
  --method direct \
  --method-name "${MODEL_TAG}_direct_refusal_induction" \
  --seeds "${CALIBRATION_SEED}" \
  "${COMMON_GENERATION_ARGS[@]}"

"${PYTHON}" -m response_generation.collect_refusal_patterns \
  --input "${BASE_REFUSAL_GENERATION}" \
  --output "${PATTERNS}" \
  --method-name "${MODEL_TAG}_base_direct" \
  --max-patterns 50

echo "[2/4] Preparing the Base mixed calibration distribution"
"${PYTHON}" generate_wildjailbreak_responses_vllm.py \
  --input "${MIXED_CALIBRATION_INPUT}" \
  --output "${BASE_MIXED_GENERATION}" \
  --method direct \
  --method-name "${MODEL_TAG}_direct_mixed_calibration" \
  --seeds "${CALIBRATION_SEED}" \
  "${COMMON_GENERATION_ARGS[@]}"

"${PYTHON}" -m response_generation.score_refusal_vllm \
  --input "${BASE_MIXED_GENERATION}" \
  --output "${BASE_MIXED_RAW}" \
  --patterns "${PATTERNS}" \
  --method direct \
  --raw-only \
  "${COMMON_REFUSAL_ARGS[@]}"

echo "[3/4] Generating and scoring methods: ${METHODS}"
for method in "${METHOD_LIST[@]}"; do
  METHOD_PREFIX=${PREFIX}_${method}
  BOUNDARY_GENERATION=${METHOD_PREFIX}.boundary_generation.jsonl
  QWEN_SCORED=${METHOD_PREFIX}.qwen_guard_scored.jsonl
  GUARD_SCORED=${METHOD_PREFIX}.guard_scored.jsonl
  REFUSAL_RAW=${METHOD_PREFIX}.refusal_raw.jsonl
  MERGED_RAW=${METHOD_PREFIX}.merged_raw.jsonl
  FINAL_SCORED=${METHOD_PREFIX}.final_scored.jsonl
  CALIBRATION_STATS=${METHOD_PREFIX}.refusal_calibration_stats.json
  CORE_METRICS=${METHOD_PREFIX}.core_metrics.json
  CORE_PER_PAIR=${METHOD_PREFIX}.core_metrics.per_pair.jsonl

  echo "Generating ${MODEL_TAG}/${method}"
  "${PYTHON}" generate_wildjailbreak_responses_vllm.py \
    --input "${PAIRS}" \
    --output "${BOUNDARY_GENERATION}" \
    --method "${method}" \
    --method-name "${MODEL_TAG}_${method}" \
    --seeds "${SEED_LIST[@]}" \
    "${COMMON_GENERATION_ARGS[@]}"

  echo "Scoring harmful outcomes for ${MODEL_TAG}/${method}"
  "${PYTHON}" -m response_generation.score_harmful_guard_vllm \
    --input "${BOUNDARY_GENERATION}" \
    --output "${QWEN_SCORED}" \
    --guard-model "${QWEN_GUARD_MODEL}" \
    --guard qwen3guard \
    "${COMMON_GUARD_ARGS[@]}"

  "${PYTHON}" -m response_generation.score_harmful_guard_vllm \
    --input "${QWEN_SCORED}" \
    --output "${GUARD_SCORED}" \
    --guard-model "${LLAMA_GUARD_MODEL}" \
    --guard llamaguard \
    "${COMMON_GUARD_ARGS[@]}"

  echo "Scoring refusal tendency for ${MODEL_TAG}/${method}"
  "${PYTHON}" -m response_generation.score_refusal_vllm \
    --input "${BOUNDARY_GENERATION}" \
    --output "${REFUSAL_RAW}" \
    --patterns "${PATTERNS}" \
    --method "${method}" \
    --raw-only \
    "${COMMON_REFUSAL_ARGS[@]}"

  "${PYTHON}" -m response_generation.merge_refusal_scores \
    --guard-input "${GUARD_SCORED}" \
    --refusal-input "${REFUSAL_RAW}" \
    --output "${MERGED_RAW}"

  "${PYTHON}" -m response_generation.recalibrate_refusal_scores \
    --input "${MERGED_RAW}" \
    --output "${FINAL_SCORED}" \
    --calibration "${BASE_MIXED_RAW}" \
    --stats-output "${CALIBRATION_STATS}" \
    --calibration-name "${MODEL_TAG}_base_mixed_shared_templates_top5_mean_avg_logprob" \
    --raw-field refusal_logprob_sum

  "${PYTHON}" -m response_generation.compute_core_metrics \
    --input "${FINAL_SCORED}" \
    --output "${CORE_METRICS}" \
    --per-pair-output "${CORE_PER_PAIR}"
done

echo "[4/4] Complete"
echo "Results: ${OUTPUT_ROOT}/${MODEL_TAG}_${RUN_TAG}_*.core_metrics.json"
