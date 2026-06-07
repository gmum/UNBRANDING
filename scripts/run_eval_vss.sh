#!/bin/bash
# ==============================================================================
# run_eval_vss.sh
# Run VSS evaluation (image comparison between unbranded and base datasets)
# ==============================================================================

# Get the root directory of the project (one level up from this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run VSS evaluation using Docker container
docker run --rm \
  --gpus all \
  --net=host \
  -v "${ROOT_DIR}":/workspace \
  unbranding-eval \
  uv run python eval_brand_benchmark.py \
  --model-id Qwen/Qwen3-VL-8B-Thinking \
  --data-dir data/example/fluxdev-eraseanything-unbrand \
  --reference-dir data/example/fluxdev-eraseanything-base \
  --vlm-config configs/vlm_vss.json \
  --results-dir results/example/vss \
  --server-url http://127.0.0.1:8000
