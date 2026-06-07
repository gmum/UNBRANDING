#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Activate virtual environment if it exists (for local runs), 
# otherwise rely on system/container path (for Docker runs)
if [ -d ".venv" ]; then
  echo "--> Activating local virtual environment (.venv)..."
  source .venv/bin/activate
fi

# Configuration parameters with professional defaults (can be overridden via ENV)
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"
DATA_DIR="${DATA_DIR:-data/example/fluxdev-eraseanything-unbrand}"
REFERENCE_DIR="${REFERENCE_DIR:-data/example/fluxdev-eraseanything-base}"
RESULTS_DIR="${RESULTS_DIR:-results/example}"

echo "========================================================"
echo " Starting UNBRANDING Evaluation Pipeline"
echo "  - Model ID:   ${MODEL_ID}"
echo "  - Server URL: ${SERVER_URL}"
echo "========================================================"

echo "--> [1/3] Running Brand Prompt Score (BPS) Evaluation..."
python eval_brand_benchmark.py \
  --model-id "${MODEL_ID}" \
  --data-dir "${DATA_DIR}" \
  --vlm-config configs/vlm_bps.json \
  --results-dir "${RESULTS_DIR}/bps" \
  --server-url "${SERVER_URL}"

echo "--> [2/3] Running Visual Similarity Score (VSS) Evaluation..."
python eval_brand_benchmark.py \
  --model-id "${MODEL_ID}" \
  --data-dir "${DATA_DIR}" \
  --reference-dir "${REFERENCE_DIR}" \
  --vlm-config configs/vlm_vss.json \
  --results-dir "${RESULTS_DIR}/vss" \
  --server-url "${SERVER_URL}"

echo "--> [3/3] Computing unified B, S, U Metrics..."
python metrics.py \
  --bps "${RESULTS_DIR}/bps" \
  --vss "${RESULTS_DIR}/vss"

echo "========================================================"
echo " Evaluation Pipeline Completed Successfully!"
echo "========================================================"
