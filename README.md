# From Unlearning to UNBRANDING: A Benchmark for Trademark‑Safe Text‑to‑Image Generation


[![arXiv](https://img.shields.io/badge/arXiv-2512.13953-b31b1b.svg)](https://arxiv.org/abs/2512.13953)
[![Website](https://img.shields.io/badge/Website-Project%20Page-22c55e.svg)](https://gmum.github.io/UNBRANDING/)
[![Stars gmum/VeGaS](https://img.shields.io/github/stars/gmum/UNBRANDING?style=social)](https://gmum.github.io/UNBRANDING)

![Teaser](images/teaser_ub_v4.png)

> Benchmark and task for trademark‑safe text‑to‑image generation. We introduce unbranding — fine‑grained removal of both explicit logos and implicit trade‑dress while preserving object and scene semantics — together with a dataset and a VLM‑QA evaluation protocol.


## Authors
Dawid Malarz, Artur Kasymov, Filip Manjak, Maciej Zięba, Przemysław Spurek

## Abstract
Modern text‑to‑image diffusion models can faithfully reproduce trademarks. Prior unlearning works target general concepts (e.g., styles, celebrities) and miss brand‑specific identifiers. Brand recognition is multi‑dimensional, spanning explicit logos and distinctive structural features (e.g., a car’s front grille). We define unbranding as the fine‑grained removal of both trademarks and subtle trade‑dress while preserving semantic coherence. We introduce a benchmark dataset and a VLM‑based QA metric that probes for both explicit and implicit brand signals, going beyond classic logo detectors. Our analysis shows newer models tend to synthesize brands more readily, underscoring the urgency of unbranding. Results validated by our metric indicate that unbranding is a distinct, practically relevant problem requiring specialized techniques.

## Highlights
- Task: fine‑grained removal of brand identifiers while preserving object fidelity.
- Benchmark: dataset spanning explicit logos and trade‑dress cues.
- Metric: VLM‑based QA probing both explicit and implicit brand signals.
- Motivation: newer models (e.g., SDXL, FLUX) reproduce brands more readily than older ones.

## Results Snapshot
Trade‑off between fidelity and removal. Baseline preserves structure but often fails to remove brands; ESD removes brands but alters semantics. Effective unbranding must achieve both.

![Trade‑off](images/fig4.png)

## Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (for image generation and VLM inference)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup with uv

1. Clone the repository:
```bash
git clone https://github.com/gmum/UNBRANDING.git
cd UNBRANDING
```

2. Install dependencies using uv:
```bash
uv sync
```

This will create a virtual environment in `.venv` and install all required packages including:
- PyTorch 2.8.0 with CUDA support
- Diffusers 0.35.2 (Stable Diffusion, FLUX models)
- Transformers 4.57.1 (HuggingFace models)
- VLLM 0.11.0 (optimized VLM inference)
- Streamlit 1.51.0 (annotation UI)
- Ray 2.51.1 (distributed processing)

3. Activate the environment:
```bash
source .venv/bin/activate
```

Or run commands directly with uv:
```bash
uv run python <script.py>
```

## Quick Start

### 1. Download Prompt Data from Hugging Face

Dataset (CSV prompts): `MalarzDawid/UNBRANDING`

```bash
uv run huggingface-cli download MalarzDawid/UNBRANDING \
  --repo-type dataset \
  --local-dir data/hf_unbranding \
  --local-dir-use-symlinks False
```

The main prompt file used below is:

```bash
data/hf_unbranding/train.csv
```

### 2. Generate Images

`generate_images.py` expects `--prompts_file` (CSV) and saves outputs under `--output_dir`.

```bash
# SDXL
uv run python generate_images.py \
  --model sdxl \
  --prompts_file data/hf_unbranding/unbranding_v1.csv \
  --prompt_set directed biased \
  --output_dir output/sdxl \
  --seed 42

# FLUX.1 Schnell
uv run python generate_images.py \
  --model flux1-schnell \
  --prompts_file data/hf_unbranding/unbranding_v1.csv \
  --prompt_set directed biased \
  --output_dir output/flux1_schnell \
  --seed 42

# SD 3.5 Large
uv run python generate_images.py \
  --model sd35-large \
  --prompts_file data/hf_unbranding/unbranding_v1.csv \
  --prompt_set directed biased \
  --output_dir output/sd35_large \
  --seed 42
```

Supported `--model` values:
- `sd14`
- `sdxl`
- `sd35-large`
- `flux1-schnell`
- `flux1-dev`
- `qwen-image`

### 3. Benchmark (SLURM: sbatch + shell wrappers)

We provide an example directory generated with the EraseAnything method ([Download](https://drive.google.com/file/d/1qL9McjRS1GeQNIW5v9dKHm8_hm0A4JT9/view?usp=sharing)). You can download it and run our benchmark directly. If you want to evaluate a different method, please follow these rules:
1. Generate base images in the same environment where you run the UNBRANDING/unlearning method.
2. Use a directory structure matching our example.
3. Move the generated data to this repository and run the benchmark.

We use fixed seeds, but we cannot guarantee identical results across methods and environments. We aim to provide fair settings, but exact reproducibility is still challenging.

Examples below assume a 2xGPU setup per SLURM job (`GPUS_PER_JOB=2`, `TENSOR_PARALLEL_SIZE=2`).

#### 3.1 Brand Detection Score benchmark (single image mode)

```bash
GPUS_PER_JOB=2 TENSOR_PARALLEL_SIZE=2 \
RESULTS_DIR=results/example/bps \
bash slurm/submit_eval_brand_for_configs.sh \
  Qwen/Qwen3-VL-8B-Thinking \
  configs/vlm_bps.json \
  data/example
```

#### 3.2 Comparison benchmark (data + reference images)

`reference-dir` must contain files with matching relative paths (including subfolders) from `data-dir`.

```bash
VLM_CONFIG=configs/vlm_vss.json \
DATA_DIR=data/example/fluxdev-eraseanything-unbrand \
REFERENCE_DIR=data/example/fluxdev-eraseanything-base \
RESULTS_DIR=results/example/vss \
CLIP_COSINE_THRESHOLD=0.8 \
GPUS_PER_JOB=2 \
TENSOR_PARALLEL_SIZE=2 \
bash slurm/submit_eval_brand_benchmark.sh Qwen/Qwen3-VL-8B-Thinking
```

If you want to run on a single GPU instead, set:

```bash
GPUS_PER_JOB=1 TENSOR_PARALLEL_SIZE=1 \
bash slurm/submit_eval_brand_benchmark.sh Qwen/Qwen3-VL-8B-Thinking
```

Output paths (SLURM mode):
- Benchmark results (JSONL): `${RESULTS_DIR}` (e.g. `results/example/bps` or `results/example/vss`)
- SLURM stdout: `logs/eval_brand_benchmark_<JOB_ID>.out`
- SLURM stderr: `logs/eval_brand_benchmark_<JOB_ID>.err`
- vLLM server log: `logs/vllm_eval_brand_<MODEL_ID_SAFE>.log`

### 4. Development Run (manual, 2 terminals)

For 2xGPU local runs, expose two devices (for example `CUDA_VISIBLE_DEVICES=0,1`).

#### Terminal A: start vLLM server

Detection config (`configs/vlm_brand_benchmark.json`, 1 image per request):

```bash
uv run vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --limit-mm-per-prompt '{"image":1}'
```

Comparison config (`configs/vlm_vss.json`, 2 images per request):

```bash
uv run vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --limit-mm-per-prompt '{"image":2}'
```

#### Terminal B: run benchmark client

Detection mode:

```bash
uv run python eval_brand_benchmark.py \
  --model-id Qwen/Qwen3-VL-8B-Thinking \
  --data-dir data/example \
  --vlm-config configs/vlm_bps.json \
  --server-url http://127.0.0.1:8000 \
  --results-dir results/example
```

Comparison mode:

```bash
uv run python eval_brand_benchmark.py \
  --model-id Qwen/Qwen3-VL-8B-Thinking \
  --data-dir data/example/fluxdev-eraseanything-unbrand \
  --reference-dir data/example/fluxdev-eraseanything-base \
  --vlm-config configs/vlm_vss.json \
  --clip-cosine-threshold 0.8 \
  --server-url http://127.0.0.1:8000 \
  --results-dir results/example
```

Results are saved as JSONL files in `--results-dir` (or in `--output-jsonl` if provided).


## Citation
Please cite our work if you find it useful:

```
@misc{malarz2025unlearningunbrandingbenchmarktrademarksafe,
  title={From Unlearning to UNBRANDING: A Benchmark for Trademark-Safe Text-to-Image Generation},
  author={Dawid Malarz and Artur Kasymov and Filip Manjak and Maciej Zięba and Przemysław Spurek},
  year={2025},
  eprint={2512.13953},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.13953},
}
```

## Acknowledgments
The project page is adapted from the Academic Project Page Template (inspired by Nerfies). Images in this repository are for research/illustration.
