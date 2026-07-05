"""*** 边界框坐标转换与标准化处理工具。 ***"""

from __future__ import annotations

from pathlib import Path

from app.domain.images.image_metadata import read_image_dimensions


def guess_mime_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


def normalize_optional_bbox(
    bounding_box: object,
    image_path: Path | None,
    raw_content: str,
    field_name: str,
) -> list[float] | None:
    if bounding_box is None:
        return None

    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise ValueError(f"Model returned invalid {field_name}. Raw content: {raw_content}")

    return coerce_normalized_bbox(bounding_box, image_path, raw_content)


def coerce_normalized_bbox(
    bounding_box: list[object],
    image_path: Path | None,
    raw_content: str,
) -> list[float]:
    values = [float(value) for value in bounding_box]
    if all(0 <= value <= 1 for value in values):
        return [round(value, 6) for value in values]

    if image_path is None:
        raise ValueError(
            "Model returned non-normalized bounding_box, but image size is unavailable. "
            f"Raw content: {raw_content}"
        )

    image_width, image_height = read_image_dimensions(image_path)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Could not determine image dimensions for bounding_box normalization. "
            f"Raw content: {raw_content}"
        )

    x1, y1, x2, y2 = values
    normalized = [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    ]
    if any(value < 0 or value > 1 for value in normalized):
        raise ValueError(
            "Model returned bounding_box outside image bounds after normalization. "
            f"Raw content: {raw_content}"
        )

    return [round(value, 6) for value in normalized]


def combine_bounding_boxes(first_bbox: list[float], second_bbox: list[float]) -> list[float]:
    return [
        round(min(first_bbox[0], second_bbox[0]), 6),
        round(min(first_bbox[1], second_bbox[1]), 6),
        round(max(first_bbox[2], second_bbox[2]), 6),
        round(max(first_bbox[3], second_bbox[3]), 6),
    ]
