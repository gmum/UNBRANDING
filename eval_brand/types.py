from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    image_path: Path
    rel_path: str  # path relative to data root
    reference_image_path: Path | None = None
    reference_rel_path: str | None = None


@dataclass
class EvaluationResult:
    predicted_brand: str | None = None
    score: float | int | None = None
    reasoning: str | None = None
    clip_cosine_similarity: float | None = None
    clip_gate_passed: bool | None = None
    similarity_score: float | int | None = None
    explanation: str | None = None
    raw_response: str | None = None
    error: str | None = None

    def to_row(self, sample: Sample) -> dict[str, Any]:
        return {
            "filename": sample.rel_path,
            "reference_filename": sample.reference_rel_path or "",
            "predicted_brand": self.predicted_brand,
            "score": self.score,
            "reasoning": self.reasoning,
            "clip_cosine_similarity": self.clip_cosine_similarity,
            "clip_gate_passed": self.clip_gate_passed,
            "similarity_score": self.similarity_score,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
            "error": self.error,
        }
