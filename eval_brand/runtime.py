from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging(model_tag: str, rank: int) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(f"eval_brand_{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(f"logs/eval_brand_{model_tag}_rank{rank}.log")
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def distributed_params() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world
