from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

import aiohttp

from .clip_gate import (
    compute_clip_cosine_similarities_batch,
    evaluate_clip_gate,
    init_clip_for_template,
)
from .types import EvaluationResult, Sample
from .vlm_client import call_vlm_api


def merge_evaluation_results(base: EvaluationResult, extra: EvaluationResult) -> None:
    for field in EvaluationResult.__dataclass_fields__:
        value = getattr(extra, field)
        if value is not None:
            setattr(base, field, value)


async def run_vlm_jobs_async(
    samples: list[Sample],
    template_type: str,
    prompts: dict[str, Any],
    format_instructions: list[str],
    question: str | None,
    model_id: str,
    model_cfg: dict[str, Any],
    url: str,
    seed: int,
    request_timeout: int,
    retry_wait: int,
    max_retries: int,
    vlm_max_workers: int,
    base_results: dict[str, EvaluationResult],
    done_start: int,
    total: int,
    logger: logging.Logger,
    on_result: Callable[[Sample, EvaluationResult], None] | None = None,
) -> dict[str, EvaluationResult]:
    semaphore = asyncio.Semaphore(max(1, vlm_max_workers))
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(connector=connector) as session:

        async def run_one(sample: Sample) -> tuple[Sample, EvaluationResult]:
            rel_path = sample.rel_path
            base_result = base_results[rel_path]
            try:
                vlm_result = await call_vlm_api(
                    sample=sample,
                    template_type=template_type,
                    prompts=prompts,
                    format_instructions=format_instructions,
                    question=question,
                    model_id=model_id,
                    model_cfg=model_cfg,
                    url=url,
                    seed=seed,
                    request_timeout=request_timeout,
                    retry_wait=retry_wait,
                    max_retries=max_retries,
                    session=session,
                    semaphore=semaphore,
                )
            except Exception as exc:
                vlm_result = EvaluationResult(error=str(exc))
            merge_evaluation_results(base_result, vlm_result)
            return sample, base_result

        tasks = [asyncio.create_task(run_one(sample)) for sample in samples]
        done = done_start
        completed: dict[str, EvaluationResult] = {}
        for task in asyncio.as_completed(tasks):
            sample, merged_result = await task
            rel_path = sample.rel_path
            completed[rel_path] = merged_result
            if on_result is not None:
                on_result(sample, merged_result)
            done += 1
            if done % 50 == 0:
                logger.info("Progress: %d / %d", done, total)

    return completed


def run_evaluation(
    samples: list[Sample],
    is_comparison: bool,
    clip_model_id: str,
    clip_cosine_threshold: float,
    clip_batch_size: int,
    prompts: dict[str, Any],
    format_instructions: list[str],
    question: str | None,
    model_id: str,
    model_cfg: dict[str, Any],
    url: str,
    seed: int,
    request_timeout: int,
    retry_wait: int,
    max_retries: int,
    vlm_max_workers: int,
    logger: logging.Logger,
    on_result: Callable[[Sample, EvaluationResult], None] | None = None,
) -> dict[str, EvaluationResult]:
    clip_cosine_map: dict[str, float] = {}

    if is_comparison:
        clip_model, clip_processor = init_clip_for_template(
            clip_model_id=clip_model_id,
            clip_cosine_threshold=clip_cosine_threshold,
            logger=logger,
        )
        logger.info("Computing CLIP similarities in batches of %d", clip_batch_size)
        clip_cosine_map = compute_clip_cosine_similarities_batch(
            samples=samples,
            clip_model=clip_model,
            clip_processor=clip_processor,
            batch_size=clip_batch_size,
        )

    template_type = "comparison" if is_comparison else "detect"
    done = 0
    results_by_rel_path: dict[str, EvaluationResult] = {}
    samples_for_vlm: list[Sample] = []

    for sample in samples:
        base_result = EvaluationResult()
        should_query_vlm = True

        if is_comparison:
            gate_result, should_query_vlm = evaluate_clip_gate(
                rel_path=sample.rel_path,
                cosine_similarity=clip_cosine_map.get(sample.rel_path),
                clip_cosine_threshold=clip_cosine_threshold,
            )
            merge_evaluation_results(base_result, gate_result)

        results_by_rel_path[sample.rel_path] = base_result
        if should_query_vlm:
            samples_for_vlm.append(sample)
        else:
            if on_result is not None:
                on_result(sample, base_result)
            done += 1
            if done % 50 == 0:
                logger.info("Progress: %d / %d", done, len(samples))

    if samples_for_vlm:
        logger.info(
            "Submitting %d VLM requests with concurrency=%d",
            len(samples_for_vlm),
            max(1, vlm_max_workers),
        )
        completed = asyncio.run(
            run_vlm_jobs_async(
                samples=samples_for_vlm,
                template_type=template_type,
                prompts=prompts,
                format_instructions=format_instructions,
                question=question,
                model_id=model_id,
                model_cfg=model_cfg,
                url=url,
                seed=seed,
                request_timeout=request_timeout,
                retry_wait=retry_wait,
                max_retries=max_retries,
                vlm_max_workers=vlm_max_workers,
                base_results=results_by_rel_path,
                done_start=done,
                total=len(samples),
                logger=logger,
                on_result=on_result,
            )
        )
        results_by_rel_path.update(completed)

    return results_by_rel_path


def write_results_jsonl(
    jsonl_path: Path,
    samples: list[Sample],
    results_by_rel_path: dict[str, EvaluationResult],
    logger: logging.Logger,
) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        for sample in samples:
            result = results_by_rel_path.get(sample.rel_path)
            if result is None:
                result = EvaluationResult(error=f"Missing evaluation result for {sample.rel_path}")
            if result.error:
                logger.error("Failed %s: %s", sample.rel_path, result.error)
            handle.write(json.dumps(result.to_row(sample), ensure_ascii=False) + "\n")
        handle.flush()
