from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp

from .vlm_outputs import (
    BrandMulticlassRecognitionOutput,
    VisualSimilarityEvaluationOutput,
)

from .prompting import build_image_b64_list, build_messages
from .types import EvaluationResult, Sample


def response_schema(template_type: str) -> tuple[str, dict[str, Any]]:
    if template_type == "comparison":
        return (
            "visual_similarity_evaluation",
            VisualSimilarityEvaluationOutput.model_json_schema(),
        )
    return (
        "brand_multiclass_recognition",
        BrandMulticlassRecognitionOutput.model_json_schema(),
    )


async def query_vllm_async(
    messages: list[dict[str, Any]],
    model_id: str,
    model_cfg: dict[str, Any],
    url: str,
    seed: int,
    timeout: int,
    retry_wait: int,
    max_retries: int,
    template_type: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> str:
    schema_name, json_schema = response_schema(template_type)
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
                "name": schema_name,
                "schema": json_schema,
            },
        },
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    request_timeout = aiohttp.ClientTimeout(total=timeout)
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                ) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        data = json.loads(text)
                        return data["choices"][0]["message"]["content"]
                    last_err = RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            last_err = exc
        if attempt < max_retries:
            await asyncio.sleep(retry_wait)

    raise last_err or RuntimeError("Unknown vLLM error")


def parse_response(text: str) -> dict[str, Any]:
    data = json.loads(text.strip())
    return BrandMulticlassRecognitionOutput(**data).model_dump()


def parse_comparison_response(text: str) -> dict[str, Any]:
    data = json.loads(text.strip())
    return VisualSimilarityEvaluationOutput(**data).model_dump()


def parse_vlm_result(raw: str, template_type: str) -> EvaluationResult:
    result = EvaluationResult(raw_response=raw)
    if template_type == "comparison":
        parsed = parse_comparison_response(raw)
        result.similarity_score = parsed["similarity_score"]
        result.explanation = parsed["explanation"]
    else:
        parsed = parse_response(raw)
        result.predicted_brand = parsed["brand"]
        result.score = parsed["score"]
        result.reasoning = parsed.get("reasoning", "")
    return result


async def call_vlm_api(
    sample: Sample,
    template_type: str,
    prompts: dict[str, Any],
    format_instructions: list[str],
    question: str,
    model_id: str,
    model_cfg: dict[str, Any],
    url: str,
    seed: int,
    request_timeout: int,
    retry_wait: int,
    max_retries: int,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> EvaluationResult:
    try:
        image_b64_list = build_image_b64_list(sample, template_type)
        msgs = build_messages(
            task=prompts["task"],
            system_desc=prompts.get("system_description", ""),
            fmt_instructions=format_instructions,
            question=question,
            image_b64_list=image_b64_list,
        )
        raw = await query_vllm_async(
            messages=msgs,
            model_id=model_id,
            model_cfg=model_cfg,
            url=url,
            seed=seed,
            timeout=request_timeout,
            retry_wait=retry_wait,
            max_retries=max_retries,
            template_type=template_type,
            session=session,
            semaphore=semaphore,
        )
        return parse_vlm_result(raw, template_type)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(error=str(exc))
