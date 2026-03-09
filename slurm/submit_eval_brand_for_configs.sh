#!/bin/bash
set -euo pipefail

# User-facing wrapper for eval_brand_benchmark.py.
#
# Usage:
#   bash slurm/submit_eval_brand_for_configs.sh MODEL_ID CONFIG_FILE DATA_DIR
#
# Example:
#   bash slurm/submit_eval_brand_for_configs.sh \
#     Qwen/Qwen3-VL-8B-Thinking \
#     configs/vlm_brand_prompt3.json \
#     data/bmw_ablation

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 MODEL_ID CONFIG_FILE DATA_DIR"
  exit 1
fi

MODEL_ID="$1"
CONFIG_FILE="$2"
DATA_DIR="$3"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config not found: ${CONFIG_FILE}"
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Data dir not found: ${DATA_DIR}"
  exit 1
fi

config_base="$(basename "${CONFIG_FILE}" .json)"
data_base="$(basename "${DATA_DIR}")"

export CONFIG_FILE
export DATA_DIR
export RESULTS_DIR="${RESULTS_DIR:-results/eval_brand_benchmark/${data_base}/${config_base}}"

echo "Submitting:"
echo "  model_id    ${MODEL_ID}"
echo "  config      ${CONFIG_FILE}"
echo "  data_dir    ${DATA_DIR}"
echo "  results_dir ${RESULTS_DIR}"

bash slurm/submit_eval_brand_benchmark.sh "${MODEL_ID}"
