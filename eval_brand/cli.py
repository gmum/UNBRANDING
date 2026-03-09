from __future__ import annotations

import argparse
import re


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate brand recognition on image dataset.")
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="HF model id served by vLLM.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory with input images (top-level only, non-recursive).",
    )
    p.add_argument(
        "--reference-dir",
        default=None,
        help="Reference directory with matching filenames. If set, comparison mode is enabled.",
    )
    p.add_argument(
        "--vlm-config",
        default="configs/vlm_brand_benchmark.json",
        help="VLM config JSON.",
    )
    p.add_argument(
        "--output-jsonl",
        default=None,
        help="Path for the output JSONL. Auto-generated if omitted.",
    )
    p.add_argument(
        "--results-dir",
        default="results/eval_brand_benchmark",
        help="Directory for output files.",
    )
    p.add_argument(
        "--server-url", default="http://localhost:8000", help="vLLM server base URL."
    )
    p.add_argument(
        "--clip-model-id",
        default="openai/clip-vit-base-patch32",
        help="CLIP model used for stage-0 gating in comparison mode.",
    )
    p.add_argument(
        "--clip-cosine-threshold",
        type=float,
        default=0.8,
        help="Run VLM comparison only if cosine similarity is at least this threshold.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP timeout per request (seconds).",
    )
    p.add_argument("--retry-wait", type=int, default=5, help="Seconds between retries.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--vlm-max-workers",
        type=int,
        default=16,
        help="Number of concurrent VLM requests (aiohttp).",
    )
    p.add_argument(
        "--clip-batch-size",
        type=int,
        default=64,
        help="Number of image pairs processed per CLIP batch in comparison mode.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap sample count (for quick tests).",
    )
    return p.parse_args()


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("._-") or "model"
