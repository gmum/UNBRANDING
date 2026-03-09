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

# Required
MODEL_ID="${MODEL_ID:-}"

# Readable hardcoded defaults. Override through sbatch --export only when needed.
WORKDIR="${WORKDIR:-/exmaple/path/to/workdir}"
ENV_PATH="${ENV_PATH:-.venv/bin/activate}"
CONFIG_FILE="${CONFIG_FILE:-configs/vlm_brand_benchmark.json}"
DATA_DIR="${DATA_DIR:-data/eval_halucination}"
RESULTS_DIR="${RESULTS_DIR:-results/eval_brand_benchmark}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID must be set."
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

echo "============================================================"
echo "eval_brand_benchmark"
echo "model_id              ${MODEL_ID}"
echo "data_dir              ${DATA_DIR}"
echo "results_dir           ${RESULTS_DIR}"
echo "port                  ${PORT}"
echo "tensor_parallel_size  ${TENSOR_PARALLEL_SIZE}"
echo "max_model_len         ${MAX_MODEL_LEN}"
echo "gpu_memory_util       ${GPU_MEMORY_UTILIZATION}"
echo "max_num_seqs          ${MAX_NUM_SEQS}"
echo "============================================================"

VLLM_ARGS=(
  "${MODEL_ID}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --limit-mm-per-prompt '{"image":1}'
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

echo "Running eval_brand_benchmark.py"
CUDA_VISIBLE_DEVICES='' python eval_brand_benchmark.py \
  --model-id "${MODEL_ID}" \
  --data-dir "${DATA_DIR}" \
  --prompts-file "${CONFIG_FILE}" \
  --results-dir "${RESULTS_DIR}" \
  --server-url "${API_BASE}"

echo "Done."
