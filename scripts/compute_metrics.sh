#!/bin/bash
# ==============================================================================
# compute_metrics.sh
# Compute final B, S, U metrics based on results from BPS and VSS directories
# ==============================================================================

# Get the root directory of the project (one level up from this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run metrics computation using Docker container
docker run --rm \
  -v "${ROOT_DIR}":/workspace \
  unbranding-eval \
  uv run python metrics.py \
  --bps results/example/bps \
  --vss results/example/vss
