from __future__ import annotations

from pathlib import Path

from .types import Sample

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_image_files(data_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)
    image_paths.sort(key=lambda p: p.relative_to(data_dir).as_posix())
    return image_paths


def discover_samples(data_dir: Path) -> list[Sample]:
    """Read image samples from data_dir recursively."""
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")
    samples: list[Sample] = []
    for img in _iter_image_files(data_dir):
        rel_path = img.relative_to(data_dir).as_posix()
        samples.append(
            Sample(
                image_path=img.resolve(),
                rel_path=rel_path,
            )
        )
    return samples


def discover_comparison_samples(data_dir: Path, reference_dir: Path) -> list[Sample]:
    if not reference_dir.is_dir():
        raise SystemExit(f"Reference directory does not exist: {reference_dir}")
    data_samples = discover_samples(data_dir)
    resolved: list[Sample] = []
    for sample in data_samples:
        reference_image_path = (reference_dir / sample.rel_path).resolve()
        if not reference_image_path.is_file():
            raise SystemExit(
                f"Missing reference image in {reference_dir} for file under {data_dir}: {sample.rel_path}"
            )
        resolved.append(
            Sample(
                image_path=sample.image_path,
                rel_path=sample.rel_path,
                reference_image_path=reference_image_path,
                reference_rel_path=sample.rel_path,
            )
        )
    return resolved


def load_samples_for_template(data_dir: Path, reference_dir: str | None) -> list[Sample]:
    if reference_dir is None:
        return discover_samples(data_dir)
    return discover_comparison_samples(data_dir, Path(reference_dir))
