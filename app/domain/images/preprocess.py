"""*** 将上传图片处理为可推理输入的预处理工具。 ***"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.core.config import ImagePreprocessConfig


@dataclass(frozen=True)
class InferenceImageResult:
    image_path: Path
    resized: bool
    original_size: tuple[int, int]
    inference_size: tuple[int, int]


def prepare_inference_image(
    original_image_path: Path,
    output_dir: Path,
    config: ImagePreprocessConfig,
) -> InferenceImageResult:
    with Image.open(original_image_path) as image:
        original_size = image.size

        if not config.enable_resize:
            return InferenceImageResult(
                image_path=original_image_path,
                resized=False,
                original_size=original_size,
                inference_size=original_size,
            )

        longest_side = max(original_size)
        if longest_side <= config.max_side:
            return InferenceImageResult(
                image_path=original_image_path,
                resized=False,
                original_size=original_size,
                inference_size=original_size,
            )

        resized_image = image.copy()
        resized_image.thumbnail((config.max_side, config.max_side), Image.Resampling.LANCZOS)
        inference_size = resized_image.size
        output_path = output_dir / f"{original_image_path.stem}-inference.jpg"

        # Save a compact inference copy without changing the original upload.
        if resized_image.mode not in {"RGB", "L"}:
            resized_image = resized_image.convert("RGB")

        resized_image.save(
            output_path,
            format="JPEG",
            quality=config.jpeg_quality,
            optimize=True,
        )

    return InferenceImageResult(
        image_path=output_path,
        resized=True,
        original_size=original_size,
        inference_size=inference_size,
    )
