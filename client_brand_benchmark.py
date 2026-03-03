import argparse
import base64
import fcntl
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
NO_BRAND_LABEL = "NO_BRAND"
BRAND_DIR_TO_LABEL = {
    "adidas": "ADIDAS",
    "apple": "APPLE",
    "audi": "AUDI",
    "bmw": "BMW",
    "coca-cola": "COCA_COLA",
    "emirates": "EMIRATES",
    "mcdonald": "MCDONALDS",
    "mercedes": "MERCEDES",
    "monster": "MONSTER",
    "nike": "NIKE",
    "puma": "PUMA",
    "singapore_airlines": "SINGAPORE_AIRLINES",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    image_relpath: str
    split: str
    source_brand: str
    expected_label: str
    manifest_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM brand multiclass benchmark.")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name.")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model id and config key from configs/vlm_brand_benchmark.json.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="configs/vlm_brand_benchmark.json",
        help="JSON config with task/prompt/model settings.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory with images (can be directly in folder or in subfolders).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/brand_benchmark",
        help="Directory for per-rank shards and merged results.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Base VLLM server URL.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP timeout for a single VLLM request in seconds.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=5,
        help="Seconds to wait before retrying a failed request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of HTTP attempts per image.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("._-") or "model"


def setup_logging(rank: int, model_tag: str) -> logging.Logger:
    from datetime import datetime

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(f"brand_benchmark_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"brand_benchmark_rank_{rank}_{model_tag}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_distributed_params() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def load_prompts(prompts_file: str) -> dict[str, Any]:
    with open(prompts_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return os.path.relpath(path, Path.cwd())


def find_all_images(root_dir: Path) -> list[tuple[Path, str]]:
    """Find all images in root_dir and subfolders.
    
    Returns list of (image_path, source_brand) tuples.
    source_brand is the subfolder name if image is in a subfolder, else "root".
    """
    images: list[tuple[Path, str]] = []
    
    # Check for images directly in root_dir
    for item in root_dir.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((item, "root"))
    
    # Check for images in subfolders
    for item in root_dir.iterdir():
        if item.is_dir():
            for subitem in item.rglob("*"):  # Recursively search subfolders
                if subitem.is_file() and subitem.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append((subitem, item.name))
    
    return sorted(images, key=lambda x: (x[1], x[0].name))


def build_manifest(images_dir: Path) -> list[Sample]:
    """Build manifest from images directory.
    
    Images can be:
    - Directly in images_dir
    - In subfolders of images_dir (recursively)
    """
    if not images_dir.exists():
        raise SystemExit(f"Missing dataset directory: {images_dir}")

    images = find_all_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in: {images_dir}")

    manifest: list[Sample] = []
    for image_path, source_brand in images:
        # Try to map source_brand to a label, otherwise use the source_brand as-is
        if source_brand == "root":
            expected_label = "UNKNOWN"
        else:
            expected_label = BRAND_DIR_TO_LABEL.get(source_brand, source_brand.upper())
        
        sample_id = f"{source_brand}:{image_path.name}"
        manifest.append(
            Sample(
                sample_id=sample_id,
                image_path=image_path.resolve(),
                image_relpath=safe_relpath(image_path.resolve()),
                split="eval",
                source_brand=source_brand,
                expected_label=expected_label,
                manifest_index=len(manifest),
            )
        )

    return manifest


def encode_image_to_base64(image_path: Path, quality: int = 95) -> str:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        raw_bytes = BytesIO()
        rgb_image.save(raw_bytes, format="JPEG", quality=quality)
    raw_bytes.seek(0)
    return base64.b64encode(raw_bytes.read()).decode()


def create_messages(
    task: str,
    system_description: str,
    format_instructions: list[str],
    question: str,
    image_base64: str,
) -> list[dict[str, Any]]:
    instructions = "\n".join(format_instructions)
    system_content = f"{system_description}\n\n{task}\n\n{instructions}"
    user_content = [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def query_vllm(
    messages: list[dict[str, Any]],
    model_id: str,
    model_config: dict[str, Any],
    url: str,
    seed: int,
    request_timeout: int,
    retry_wait: int,
    max_retries: int,
) -> str:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": model_config.get("max_tokens", 96),
        "temperature": model_config.get("temperature", 0.2),
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

    last_error: Exception | None = None
    headers = build_headers()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            last_error = RuntimeError(
                f"Request failed with status {response.status_code}: {response.text}"
            )
        except requests.exceptions.RequestException as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(retry_wait)

    if last_error is None:
        raise RuntimeError("VLLM request failed with an unknown error.")
    raise last_error


def parse_response(response_text: str) -> dict[str, Any]:
    data = json.loads(response_text.strip())
    return BrandMulticlassRecognitionOutput(**data).model_dump()


def load_processed_sample_ids(shard_path: Path) -> set[str]:
    if not shard_path.exists():
        return set()

    processed: set[str] = set()
    with open(shard_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            processed.add(record["sample_id"])
    return processed


def count_jsonl_records(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def expected_counts_per_rank(total_samples: int, world_size: int) -> list[int]:
    counts = [0 for _ in range(world_size)]
    for index in range(total_samples):
        counts[index % world_size] += 1
    return counts


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def maybe_merge_results(
    shard_paths: list[Path],
    expected_counts: list[int],
    manifest: list[Sample],
    output_path: Path,
    lock_path: Path,
    model_id: str,
    prompts_file: str,
    exp: str,
    logger: logging.Logger,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)

        if not all(path.exists() for path in shard_paths):
            logger.info("Merge skipped: not all shard files exist yet.")
            return False

        for rank, shard_path in enumerate(shard_paths):
            actual_count = count_jsonl_records(shard_path)
            if actual_count != expected_counts[rank]:
                logger.info(
                    "Merge skipped: shard %s has %s records, expected %s.",
                    shard_path,
                    actual_count,
                    expected_counts[rank],
                )
                return False

        order = {sample.sample_id: sample.manifest_index for sample in manifest}
        records_by_id: dict[str, dict[str, Any]] = {}
        for shard_path in shard_paths:
            for record in load_jsonl_records(shard_path):
                records_by_id[record["sample_id"]] = record

        if len(records_by_id) != len(manifest):
            logger.info(
                "Merge skipped: got %s unique records, expected %s.",
                len(records_by_id),
                len(manifest),
            )
            return False

        merged_predictions = sorted(records_by_id.values(), key=lambda record: order[record["sample_id"]])
        error_count = sum(1 for record in merged_predictions if "error" in record)

        payload = {
            "exp": exp,
            "model_id": model_id,
            "prompts_file": prompts_file,
            "num_samples": len(merged_predictions),
            "num_errors": error_count,
            "predictions": merged_predictions,
        }

        tmp_path = output_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(output_path)
        logger.info("Merged final results into %s", output_path)
        return True


def main() -> None:
    args = parse_args()
    rank, world_size = get_distributed_params()
    safe_model_id = sanitize_model_id(args.model_id)
    logger = setup_logging(rank, safe_model_id)

    prompts_data = load_prompts(args.prompts_file)
    model_config = prompts_data["model_configs"].get(args.model_id)
    if model_config is None:
        raise SystemExit(f"Model '{args.model_id}' not found in {args.prompts_file}")

    question = prompts_data.get("question")
    if not question:
        raise SystemExit(f"Missing 'question' in {args.prompts_file}")

    manifest = build_manifest(Path(args.images_dir))
    if args.max_samples is not None:
        manifest = manifest[: args.max_samples]

    if not manifest:
        raise SystemExit("Manifest is empty.")

    url = f"{args.server_url.rstrip('/')}/v1/chat/completions"
    results_dir = Path(args.results_dir)
    shards_dir = results_dir / "shards"
    merge_lock_path = results_dir / ".merge.lock"
    merged_results_path = results_dir / f"{args.exp}_{safe_model_id}.json"
    shard_path = shards_dir / f"{args.exp}_{safe_model_id}_rank_{rank}.jsonl"
    shard_paths = [
        shards_dir / f"{args.exp}_{safe_model_id}_rank_{shard_rank}.jsonl"
        for shard_rank in range(world_size)
    ]

    shards_dir.mkdir(parents=True, exist_ok=True)
    processed_sample_ids = load_processed_sample_ids(shard_path)
    expected_counts = expected_counts_per_rank(len(manifest), world_size)
    assigned_count = expected_counts[rank]

    logger.info("Using VLLM endpoint %s", url)
    logger.info("Loaded %s samples. Rank %s/%s handles %s samples.", len(manifest), rank, world_size, assigned_count)
    logger.info("Resume mode: shard already contains %s samples.", len(processed_sample_ids))

    with open(shard_path, "a", encoding="utf-8") as shard_handle:
        for sample in manifest:
            if sample.manifest_index % world_size != rank:
                continue
            if sample.sample_id in processed_sample_ids:
                continue

            record: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "image_path": sample.image_relpath,
                "split": sample.split,
                "source_brand": sample.source_brand,
                "expected_label": sample.expected_label,
                "model_id": args.model_id,
                "rank": rank,
            }

            try:
                image_b64 = encode_image_to_base64(sample.image_path)
                messages = create_messages(
                    task=prompts_data["task"],
                    system_description=prompts_data.get("system_description", ""),
                    format_instructions=prompts_data.get("format_instructions", []),
                    question=question,
                    image_base64=image_b64,
                )
                response_text = query_vllm(
                    messages=messages,
                    model_id=args.model_id,
                    model_config=model_config,
                    url=url,
                    seed=args.seed,
                    request_timeout=args.request_timeout,
                    retry_wait=args.retry_wait,
                    max_retries=args.max_retries,
                )
                parsed = parse_response(response_text)
                record["prediction"] = parsed
            except (json.JSONDecodeError, ValidationError) as exc:
                record["error"] = str(exc)
                record["raw_response"] = locals().get("response_text")
                logger.error("Invalid JSON output for %s: %s", sample.sample_id, exc)
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                logger.error("Failed on %s: %s", sample.sample_id, exc)

            shard_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            shard_handle.flush()

    maybe_merge_results(
        shard_paths=shard_paths,
        expected_counts=expected_counts,
        manifest=manifest,
        output_path=merged_results_path,
        lock_path=merge_lock_path,
        model_id=args.model_id,
        prompts_file=args.prompts_file,
        exp=args.exp,
        logger=logger,
    )
    logger.info("Benchmark client finished.")


if __name__ == "__main__":
    main()
