#!/bin/bash
# ==============================================================================
# run_vllm.sh
# Run vLLM server with GPU support
# ==============================================================================

# Remove any existing container with the same name
docker rm -f vllm-server 2>/dev/null

# Run vLLM server with GPU support, bypassing the CUDA driver version requirement check
# We set limit-mm-per-prompt to image=2 to support the VSS comparison mode
docker run -d \
  --name vllm-server \
  --gpus all \
  -e NVIDIA_DISABLE_REQUIRE=true \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  vllm/vllm-openai:latest \
  Qwen/Qwen3-VL-8B-Thinking \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":2}' \
  --max-model-len 8192
