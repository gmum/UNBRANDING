from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .types import Sample


def encode_image(image_path: Path, quality: int = 95) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_image_b64_list(sample: Sample, template_type: str) -> list[str]:
    image_b64_list = [encode_image(sample.image_path)]
    if template_type == "comparison":
        if sample.reference_image_path is None:
            raise RuntimeError(f"Missing reference image path for {sample.rel_path}")
        image_b64_list = [encode_image(sample.reference_image_path), *image_b64_list]
    return image_b64_list


def build_messages(
    task: str,
    system_desc: str,
    fmt_instructions: list[str],
    question: str,
    image_b64_list: list[str],
) -> list[dict[str, Any]]:
    instructions = "\n".join(fmt_instructions)
    system_content = f"{system_desc}\n\n{task}\n\n{instructions}"
    user_content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for image_b64 in image_b64_list:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def resolve_format_instructions(prompts: dict[str, Any]) -> list[str]:
    raw = prompts.get("format_instructions", [])
    if not isinstance(raw, list):
        raise SystemExit("'format_instructions' must be a JSON array.")
    return [str(item) for item in raw]
