#!/usr/bin/env python3
"""Evaluate brand recognition on all images under a data directory.

For every image the script queries a running vLLM server with the
brand-multiclass-benchmark prompt and writes results to a single CSV.

CSV columns
-----------
filename        – image path relative to the data root
predicted_brand – the label returned by the model
score           – confidence score (0-5) returned by the model
raw_response    – raw model output (only when parsing fails)
error           – error message (empty on success)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from pydantic import ValidationError

from vlm_outputs import BrandMulticlassRecognitionOutput

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CSV_COLUMNS = [
    "filename",
    "predicted_brand",
    "score",
    "raw_response",
    "error",
]


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    image_path: Path
    rel_path: str  # path relative to data root


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate brand recognition on image dataset.")
    p.add_argument("--model-id", required=True, help="HF model id served by vLLM.")
    p.add_argument("--data-dir", default="data", help="Root directory scanned recursively for images.")
    p.add_argument("--prompts-file", default="configs/vlm_brand_benchmark.json", help="Prompt config JSON.")
    p.add_argument("--output-csv", default=None, help="Path for the output CSV. Auto-generated if omitted.")
    p.add_argument("--results-dir", default="results/eval_brand_benchmark", help="Directory for output CSVs.")
    p.add_argument("--server-url", default="http://localhost:8000", help="vLLM server base URL.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--request-timeout", type=int, default=300, help="HTTP timeout per request (seconds).")
    p.add_argument("--retry-wait", type=int, default=5, help="Seconds between retries.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=None, help="Cap sample count (for quick tests).")
    return p.parse_args()


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("._-") or "model"


def setup_logging(model_tag: str, rank: int) -> logging.Logger:
    from datetime import datetime

    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(f"eval_brand_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"logs/eval_brand_{model_tag}_rank{rank}_{ts}.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def distributed_params() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world


def discover_samples(data_dir: Path) -> list[Sample]:
    """Walk the data directory recursively and return all image samples."""
    samples: list[Sample] = []
    for img in sorted(data_dir.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
            rel = img.relative_to(data_dir)
            samples.append(
                Sample(
                    image_path=img.resolve(),
                    rel_path=str(rel),
                )
            )
    return samples


def encode_image(image_path: Path, quality: int = 95) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_messages(
    task: str,
    system_desc: str,
    fmt_instructions: list[str],
    question: str,
    image_b64: str,
) -> list[dict[str, Any]]:
    instructions = "\n".join(fmt_instructions)
    system_content = f"{system_desc}\n\n{task}\n\n{instructions}"
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]


def query_vllm(
    messages: list[dict],
    model_id: str,
    model_cfg: dict[str, Any],
    url: str,
    seed: int,
    timeout: int,
    retry_wait: int,
    max_retries: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": model_cfg.get("max_tokens", 96),
        "temperature": model_cfg.get("temperature", 0.0),
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "brand_multiclass_recognition_output",
                "schema": BrandMulticlassRecognitionOutput.model_json_schema(),
            },
        },
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except requests.exceptions.RequestException as exc:
            last_err = exc
        if attempt < max_retries:
            time.sleep(retry_wait)

    raise last_err or RuntimeError("Unknown vLLM error")


def parse_response(text: str) -> dict[str, Any]:
    data = json.loads(text.strip())
    return BrandMulticlassRecognitionOutput(**data).model_dump()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    rank, world_size = distributed_params()
    model_tag = sanitize(args.model_id)
    logger = setup_logging(model_tag, rank)

    # Load prompt config
    prompts = json.loads(Path(args.prompts_file).read_text())
    model_cfg = prompts.get("model_configs", {}).get(args.model_id)
    if model_cfg is None:
        raise SystemExit(f"Model '{args.model_id}' not found in {args.prompts_file}")
    question = prompts["question"]

    # Build sample list
    data_dir = Path(args.data_dir)
    samples = discover_samples(data_dir)
    if not samples:
        raise SystemExit(f"No images found under {data_dir}")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    # Assign samples to this rank
    my_samples = [s for i, s in enumerate(samples) if i % world_size == rank]
    logger.info(
        "Total samples: %d | rank %d/%d handles %d samples.",
        len(samples), rank, world_size, len(my_samples),
    )

    # Determine output CSV path
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        csv_path = results_dir / f"eval_{model_tag}_rank{rank}.csv"

    url = f"{args.server_url.rstrip('/')}/v1/chat/completions"

    # Check if CSV already exists for resume support
    processed: set[str] = set()
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])
        logger.info("Resuming – %d samples already in %s", len(processed), csv_path)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        done = 0
        for sample in my_samples:
            if sample.rel_path in processed:
                done += 1
                continue

            row: dict[str, Any] = {
                "filename": sample.rel_path,
                "predicted_brand": "",
                "score": "",
                "raw_response": "",
                "error": "",
            }

            try:
                img_b64 = encode_image(sample.image_path)
                msgs = build_messages(
                    task=prompts["task"],
                    system_desc=prompts.get("system_description", ""),
                    fmt_instructions=prompts.get("format_instructions", []),
                    question=question,
                    image_b64=img_b64,
                )
                raw = query_vllm(
                    messages=msgs,
                    model_id=args.model_id,
                    model_cfg=model_cfg,
                    url=url,
                    seed=args.seed,
                    timeout=args.request_timeout,
                    retry_wait=args.retry_wait,
                    max_retries=args.max_retries,
                )
                parsed = parse_response(raw)
                row["predicted_brand"] = parsed["brand"]
                row["score"] = parsed["score"]
            except (json.JSONDecodeError, ValidationError) as exc:
                row["error"] = str(exc)
                row["raw_response"] = locals().get("raw", "")
                logger.error("Parse error for %s: %s", sample.rel_path, exc)
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)
                logger.error("Failed %s: %s", sample.rel_path, exc)

            writer.writerow(row)
            f.flush()
            done += 1
            if done % 50 == 0:
                logger.info("Progress: %d / %d", done, len(my_samples))

    logger.info("Finished. Results written to %s", csv_path)

    # If single-rank job, we're done. For multi-rank, a merge step can be added.
    if world_size == 1:
        print(f"Results saved to {csv_path}")
    else:
        print(f"Shard saved to {csv_path}  (rank {rank}/{world_size})")


if __name__ == "__main__":
    main()
