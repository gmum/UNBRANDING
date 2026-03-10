from __future__ import annotations

import json
from pathlib import Path

from .cli import parse_args, sanitize
from .data_io import load_samples_for_template
from .pipeline import run_evaluation
from .prompting import resolve_format_instructions
from .runtime import distributed_params, setup_logging
from .types import EvaluationResult, Sample


def main() -> None:
    args = parse_args()
    rank, world_size = distributed_params()
    model_tag = sanitize(args.model_id)
    logger = setup_logging(model_tag, rank)

    prompts = json.loads(Path(args.vlm_config).read_text())
    model_cfg = prompts.get("model_configs", {}).get(args.model_id)
    if model_cfg is None:
        raise SystemExit(f"Model '{args.model_id}' not found in {args.vlm_config}")

    question_raw = prompts.get("question")
    if question_raw is not None and not isinstance(question_raw, str):
        raise SystemExit("'question' must be a string if provided in --vlm-config.")
    question = question_raw

    format_instructions = resolve_format_instructions(prompts)

    data_dir = Path(args.data_dir)
    is_comparison = args.reference_dir is not None
    samples = load_samples_for_template(data_dir, args.reference_dir)
    if not samples:
        raise SystemExit(f"No images found under {data_dir}")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    my_samples = [sample for idx, sample in enumerate(samples) if idx % world_size == rank]
    logger.info(
        "Total samples: %d | rank %d/%d handles %d samples.",
        len(samples),
        rank,
        world_size,
        len(my_samples),
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl)
    else:
        jsonl_path = results_dir / f"eval_{model_tag}_rank{rank}.jsonl"

    url = f"{args.server_url.rstrip('/')}/v1/chat/completions"
    written_rel_paths: set[str] = set()

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_handle:

        def append_result(sample: Sample, result: EvaluationResult) -> None:
            if sample.rel_path in written_rel_paths:
                return
            if result.error:
                logger.error("Failed %s: %s", sample.rel_path, result.error)
            jsonl_handle.write(json.dumps(result.to_row(sample), ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            written_rel_paths.add(sample.rel_path)

        results_by_rel_path = run_evaluation(
            samples=my_samples,
            is_comparison=is_comparison,
            clip_model_id=args.clip_model_id,
            clip_cosine_threshold=args.clip_cosine_threshold,
            clip_batch_size=args.clip_batch_size,
            prompts=prompts,
            format_instructions=format_instructions,
            question=question,
            model_id=args.model_id,
            model_cfg=model_cfg,
            url=url,
            seed=args.seed,
            request_timeout=args.request_timeout,
            retry_wait=args.retry_wait,
            max_retries=args.max_retries,
            vlm_max_workers=args.vlm_max_workers,
            logger=logger,
            on_result=append_result,
        )

        missing_count = 0
        for sample in my_samples:
            if sample.rel_path in written_rel_paths:
                continue
            result = results_by_rel_path.get(sample.rel_path)
            if result is None:
                result = EvaluationResult(
                    error=f"Missing evaluation result for {sample.rel_path}"
                )
            append_result(sample, result)
            missing_count += 1
        if missing_count:
            logger.warning("Backfilled %d missing results at the end.", missing_count)

    logger.info("Finished. Results written to %s", jsonl_path)
    if world_size == 1:
        print(f"Results saved to {jsonl_path}")
    else:
        print(f"Shard saved to {jsonl_path}  (rank {rank}/{world_size})")


if __name__ == "__main__":
    main()
