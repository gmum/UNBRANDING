#!/bin/bash
#SBATCH --job-name=eval_brand_benchmark
#SBATCH --time=02:00:00
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --qos=batch
#SBATCH --partition=rtx4090_batch
#SBATCH --output=logs/eval_brand_benchmark_%A.out
#SBATCH --error=logs/eval_brand_benchmark_%A.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load CUDA
else
  echo "module command not available; skipping 'module load CUDA'"
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SBATCH_DIR="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
  REPO_ROOT="$(cd -- "${SBATCH_DIR}/.." && pwd)"
fi

# Required
MODEL_ID="${MODEL_ID:-}"

# Readable hardcoded defaults. Override through sbatch --export only when needed.
WORKDIR="${WORKDIR:-${REPO_ROOT}}"
ENV_PATH="${ENV_PATH:-.venv/bin/activate}"
VLM_CONFIG="${VLM_CONFIG:-${CONFIG_FILE:-configs/vlm_brand_benchmark.json}}"
DATA_DIR="${DATA_DIR:-data/eval_halucination}"
REFERENCE_DIR="${REFERENCE_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-results/eval_brand_benchmark}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
CLIP_COSINE_THRESHOLD="${CLIP_COSINE_THRESHOLD:-}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set."
  exit 1
fi

if ! [[ "${SHARD_COUNT}" =~ ^[0-9]+$ ]] || ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "SHARD_COUNT and SHARD_INDEX must be non-negative integers."
  exit 1
fi
if [[ "${SHARD_COUNT}" -lt 1 ]]; then
  echo "SHARD_COUNT must be >= 1."
  exit 1
fi
if [[ "${SHARD_INDEX}" -ge "${SHARD_COUNT}" ]]; then
  echo "SHARD_INDEX must be < SHARD_COUNT."
  exit 1
fi

if [[ -z "${MAX_MODEL_LEN}" ]]; then
  case "${MODEL_ID}" in
    llava-hf/llava-1.5-7b-hf|llava-hf/llava-1.5-13b-hf)
      MAX_MODEL_LEN="4096"
      ;;
    *)
      MAX_MODEL_LEN="8192"
      ;;
  esac
fi

API_BASE="http://${HOST}:${PORT}"
MODEL_ID_SAFE="${MODEL_ID//\//__}"
MODEL_ID_SAFE="${MODEL_ID_SAFE//:/_}"

mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"
source "${ENV_PATH}"

if [[ ! -f "${VLM_CONFIG}" ]]; then
  echo "Config file not found: ${VLM_CONFIG}"
  exit 1
fi

TEMPLATE_TYPE="$(
  VLM_CONFIG_PATH="${VLM_CONFIG}" python - <<'PY'
import json
import os

path = os.environ["VLM_CONFIG_PATH"]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
print(str(data.get("template_type", "brand-multiclass-benchmark")).lower())
PY
)"
IMAGE_LIMIT=1
if [[ "${TEMPLATE_TYPE}" == "comparison" ]]; then
  IMAGE_LIMIT=2
  if [[ -z "${REFERENCE_DIR}" ]]; then
    echo "REFERENCE_DIR must be set when template_type is 'comparison'."
    exit 1
  fi
fi

echo "============================================================"
echo "eval_brand_benchmark"
echo "model_id              ${MODEL_ID}"
echo "template_type         ${TEMPLATE_TYPE}"
echo "data_dir              ${DATA_DIR}"
echo "reference_dir         ${REFERENCE_DIR:-<none>}"
echo "vlm_config            ${VLM_CONFIG}"
echo "results_dir           ${RESULTS_DIR}"
echo "port                  ${PORT}"
echo "tensor_parallel_size  ${TENSOR_PARALLEL_SIZE}"
echo "max_model_len         ${MAX_MODEL_LEN}"
echo "gpu_memory_util       ${GPU_MEMORY_UTILIZATION}"
echo "max_num_seqs          ${MAX_NUM_SEQS}"
echo "image_limit_per_prompt ${IMAGE_LIMIT}"
echo "shard_index           ${SHARD_INDEX}"
echo "shard_count           ${SHARD_COUNT}"
if [[ -n "${CLIP_COSINE_THRESHOLD}" ]]; then
  echo "clip_cosine_threshold ${CLIP_COSINE_THRESHOLD}"
fi
if [[ -n "${CLIP_MODEL_ID}" ]]; then
  echo "clip_model_id         ${CLIP_MODEL_ID}"
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "max_samples           ${MAX_SAMPLES}"
fi
echo "============================================================"

LIMIT_MM_PER_PROMPT="{\"image\":${IMAGE_LIMIT}}"

VLLM_ARGS=(
  "${MODEL_ID}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)

if [[ "${MODEL_ID}" == mistralai/* ]]; then
  VLLM_ARGS+=(--tokenizer_mode mistral --config_format mistral --load_format mistral)
fi

echo "Starting vLLM on ${API_BASE}"
vllm serve "${VLLM_ARGS[@]}" > "logs/vllm_eval_brand_${MODEL_ID_SAFE}.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for server..."
for _ in {1..150}; do
  if curl -sS --max-time 2 --fail "${API_BASE}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server exited early. See logs/vllm_eval_brand_${MODEL_ID_SAFE}.log"
    exit 1
  fi
  sleep 2
done

if ! curl -sS --max-time 2 --fail "${API_BASE}/v1/models" >/dev/null; then
  echo "Server failed to start within timeout."
  exit 1
fi

EVAL_ARGS=(
  python eval_brand_benchmark.py
  --model-id "${MODEL_ID}"
  --data-dir "${DATA_DIR}"
  --vlm-config "${VLM_CONFIG}"
  --results-dir "${RESULTS_DIR}"
  --server-url "${API_BASE}"
)

if [[ "${TEMPLATE_TYPE}" == "comparison" ]]; then
  EVAL_ARGS+=(--reference-dir "${REFERENCE_DIR}")
fi
if [[ -n "${CLIP_COSINE_THRESHOLD}" ]]; then
  EVAL_ARGS+=(--clip-cosine-threshold "${CLIP_COSINE_THRESHOLD}")
fi
if [[ -n "${CLIP_MODEL_ID}" ]]; then
  EVAL_ARGS+=(--clip-model-id "${CLIP_MODEL_ID}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  EVAL_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Running eval_brand_benchmark.py"
WORLD_SIZE="${SHARD_COUNT}" RANK="${SHARD_INDEX}" "${EVAL_ARGS[@]}"

echo "Done."
