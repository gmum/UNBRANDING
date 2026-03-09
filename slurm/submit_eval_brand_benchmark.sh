#!/bin/bash
set -euo pipefail

SBATCH_SCRIPT="slurm/eval_brand_benchmark.sbatch"
CONFIG_FILE="${CONFIG_FILE:-configs/vlm_brand_benchmark.json}"
DATA_DIR="${DATA_DIR:-data/eval_halucination}"
RESULTS_DIR="${RESULTS_DIR:-results/eval_brand_benchmark}"
BASE_PORT="${BASE_PORT:-8000}"
PORT_STRIDE="${PORT_STRIDE:-100}"
GPUS_PER_JOB="${GPUS_PER_JOB:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPUS_PER_JOB}}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 MODEL_ID [MODEL_ID ...]"
  echo "Example: GPUS_PER_JOB=2 $0 Qwen/Qwen3-VL-8B-Thinking"
  exit 1
fi

model_index=0
for model_id in "$@"; do
  port=$((BASE_PORT + model_index * PORT_STRIDE))
  echo "Submitting ${model_id}: gpus=${GPUS_PER_JOB} port=${port}"
  sbatch \
    --gres="gpu:${GPUS_PER_JOB}" \
    --export=ALL,MODEL_ID="${model_id}",CONFIG_FILE="${CONFIG_FILE}",DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",PORT="${port}",TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    "${SBATCH_SCRIPT}"
  model_index=$((model_index + 1))
done
