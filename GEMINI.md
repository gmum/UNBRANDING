# UNBRANDING Project Context

UNBRANDING is a benchmark and task framework for trademark-safe text-to-image generation. It defines the unbranding task — the fine-grained removal of both explicit logos and implicit trade-dress (distinctive styling, shape cues, and front grilles) while preserving general object and scene semantics. The project provides evaluation datasets, custom logo/trade-dress object detector models (YOLO), a VLM-QA evaluation protocol, and automated scripts for computing unified unbranding scores.

## Project Overview

- **Purpose:** Benchmark the trade-off between brand removal and image fidelity in text-to-image models.
- **Main Technologies:**
  - **Language:** Python (`>=3.10`, typically `<3.11`)
  - **Environment & Dependency Manager:** `uv`
  - **Core ML Frameworks:** PyTorch, Diffusers, Hugging Face Transformers, vLLM (for Vision-Language Model serving)
  - **Object Detection (Logo / Trade-dress):** Ultralytics YOLOv11 / YOLOv26
  - **Distributed Processing & UI:** Ray, Streamlit (for annotation UI)
  - **Infrastructure:** SLURM cluster scheduling
- **Architecture:**
  - `detector_unbranding/`: Custom logo and trade-dress object detection utilizing YOLO models (contains training scripts, annotation converters, and model checkpoints).
  - `eval_brand/`: Brand evaluation framework code including pipeline execution, VLM client wrapper, prompting heuristics, open-clip gates, and Pydantic output schemas.
  - `image_generation/`: Core text-to-image generation wrapper and utility scripts.
  - `configs/`: VLM evaluation configurations (e.g., `vlm_bps.json` for Brand Prompt Score, `vlm_vss.json` for Visual Similarity Score).
  - `slurm/`: Job scheduling wrappers and `.sbatch` scripts for running the benchmark pipelines on cluster GPUs.
  - `generate_images.py`: Entry point for generating benchmark images using diffusion models (SDXL, FLUX, SD3.5, etc.).
  - `eval_brand_benchmark.py`: CLI client to run Brand Prompt Score (BPS) or Visual Similarity Score (VSS) benchmarks against a running vLLM server.
  - `metrics.py`: Computes B (Brand detection score), S (Visual similarity score), and U (Unbranding score, `U = alpha * S + beta * (1 - B)`).
  - `metrics_all_methods.py`: Bulk evaluation of B/S/U across multiple folders.
  - `generate_table.py`: Tool to run metrics and generate a comparative table (CSV) for all evaluated methods.

## Building and Running

- **Install Dependencies:** `uv sync` (creates a `.venv` directory and installs project dependencies)
- **Activate Virtual Environment:** `source .venv/bin/activate` (or prepend command with `uv run`)
- **Download Prompts Dataset:** Download prompt files using Hugging Face CLI:
  ```bash
  uv run huggingface-cli download MalarzDawid/UNBRANDING \
    --repo-type dataset \
    --local-dir data/hf_unbranding \
    --local-dir-use-symlinks False
  ```
- **Generate Benchmark Images:**
  ```bash
  uv run python generate_images.py \
    --model sdxl \
    --prompts_file data/hf_unbranding/unbranding_v1.csv \
    --prompt_set directed biased \
    --output_dir output/sdxl \
    --seed 42
  ```
- **Local Development Benchmark Run (VLM Server + Client):**
  - **Terminal A (Start vLLM Server):**
    ```bash
    uv run vllm serve Qwen/Qwen3-VL-8B-Thinking \
      --host 0.0.0.0 \
      --port 8000 \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --limit-mm-per-prompt '{"image":1}'
    ```
  - **Terminal B (Run Benchmark Client in Detection Mode):**
    ```bash
    uv run python eval_brand_benchmark.py \
      --model-id Qwen/Qwen3-VL-8B-Thinking \
      --data-dir data/example \
      --vlm-config configs/vlm_bps.json \
      --server-url http://127.0.0.1:8000 \
      --results-dir results/example
    ```
- **Run Cluster SLURM Benchmarks:**
  - **Brand Detection Score (BPS):**
    ```bash
    GPUS_PER_JOB=2 TENSOR_PARALLEL_SIZE=2 RESULTS_DIR=results/example/bps \
      bash slurm/submit_eval_brand_for_configs.sh Qwen/Qwen3-VL-8B-Thinking configs/vlm_bps.json data/example
    ```
  - **Comparison Benchmark (VSS):**
    ```bash
    VLM_CONFIG=configs/vlm_vss.json DATA_DIR=data/example/unbrand REFERENCE_DIR=data/example/base \
      RESULTS_DIR=results/example/vss GPUS_PER_JOB=2 TENSOR_PARALLEL_SIZE=2 \
      bash slurm/submit_eval_brand_benchmark.sh Qwen/Qwen3-VL-8B-Thinking
    ```

## Testing and Quality

- **Compute Aggregate Metrics (B/S/U):**
  ```bash
  uv run python metrics.py --bps results/example/bps --vss results/example/vss
  ```
- **Compute B/S/U for All Method Pairs:**
  ```bash
  uv run python metrics_all_methods.py --pairs-dir results/eval_brand_benchmark/vlm_vss_pairs --bps-dir results/eval_brand_benchmark/vlm_brand_prompt3
  ```
- **Generate Comparison Table:**
  ```bash
  uv run python generate_table.py --bps_dir results/eval_brand_benchmark/vlm_brand_prompt3_thinking_on --vss_dir results/eval_brand_benchmark/vlm_vss_pairs --output results/eval_brand_benchmark/metrics_table.csv
  ```

## Recent Changes

- **VLM Reasoning Integration:** Added prompt-level instructions to `configs/vlm_bps.json` to guide the Vision-Language Model (`Qwen3-VL-8B-Thinking`) to output reasoning strings based on visible evidence. Updated `BrandMulticlassRecognitionOutput` in `eval_brand/vlm_outputs.py` with a `reasoning` field to parse and record these descriptions.
- **Robust Metrics Fallback:** Adjusted `metrics.py` to seamlessly fallback to all dataset samples if no explicit filenames contain the substring `"unbrand"`, preventing crashes.
- **SLURM Custom Output Routing:** Extended `slurm/eval_brand_benchmark.sbatch` to allow customizing the JSONL output filename via the `OUTPUT_JSONL` environment variable.
