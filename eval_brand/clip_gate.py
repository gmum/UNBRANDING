from __future__ import annotations

import logging

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .types import EvaluationResult, Sample


def load_clip_components(model_id: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def init_clip_for_template(
    clip_model_id: str,
    clip_cosine_threshold: float,
    logger: logging.Logger,
) -> tuple[CLIPModel, CLIPProcessor]:
    if not torch.cuda.is_available():
        raise SystemExit("Comparison mode requires CUDA for CLIP stage-0.")

    clip_device = torch.device("cuda")
    logger.info(
        "Loading CLIP stage-0 model: %s on %s (cosine similarity threshold %.3f)",
        clip_model_id,
        clip_device,
        clip_cosine_threshold,
    )
    return load_clip_components(clip_model_id, clip_device)


def compute_clip_cosine_similarities_batch(
    samples: list[Sample],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch_size: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise SystemExit("--clip-batch-size must be > 0.")

    clip_device = next(clip_model.parameters()).device
    cosine_map: dict[str, float] = {}

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        images: list[Image.Image] = []
        for sample in batch_samples:
            if sample.reference_image_path is None:
                raise RuntimeError(
                    f"Missing reference image path for comparison sample: {sample.rel_path}"
                )
            with (
                Image.open(sample.reference_image_path) as ref_img,
                Image.open(sample.image_path) as eval_img,
            ):
                images.append(ref_img.convert("RGB"))
                images.append(eval_img.convert("RGB"))

        inputs = clip_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.get_image_features(**inputs)
        image_features = image_features.float()
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        cosine_scores = (image_features[0::2] * image_features[1::2]).sum(dim=-1).tolist()

        for sample, score in zip(batch_samples, cosine_scores):
            cosine_map[sample.rel_path] = float(score)

    return cosine_map


def evaluate_clip_gate(
    rel_path: str,
    cosine_similarity: float | None,
    clip_cosine_threshold: float,
) -> tuple[EvaluationResult, bool]:
    result = EvaluationResult()
    if cosine_similarity is None:
        result.error = f"Missing CLIP score for {rel_path}"
        return result, False

    gate_passed = cosine_similarity >= clip_cosine_threshold
    result.clip_cosine_similarity = round(cosine_similarity, 6)
    result.clip_gate_passed = gate_passed
    if not gate_passed:
        result.similarity_score = 0
        result.explanation = f"CLIP: {cosine_similarity:.4f}_{clip_cosine_threshold:.4f}"
        return result, False

    return result, True
