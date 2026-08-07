"""*** 主标注结构组装与 schema 构建辅助逻辑。 ***"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.domain.images.image_metadata import build_image_metadata


def build_master_annotation(
    image_path: Path,
    raw_model_output: str,
    normalized_annotation: dict,
    validation: dict,
    prompt_mode: str,
    provider: str,
    model: str,
) -> dict:
    image_metadata = build_image_metadata(image_path)
    image_id = image_path.stem
    caption = str(normalized_annotation.get("caption") or "").strip()
    normalized_annotations = normalized_annotation.get("annotations", [])
    annotations = [
        {
            # "annotation_id": f"{image_id}_{index:03d}",
            **annotation,
        }
        for index, annotation in enumerate(normalized_annotations, start=1)
    ]

    return {
        "schema_version": "3.0.0",
        "image": {
            "image_id": image_id,
            "file_name": image_metadata["file_name"],
            "width": image_metadata["width"],
            "height": image_metadata["height"],
        },
        "caption": caption,
        "annotations": annotations,
        "task": "hoi_annotation",
        "raw_model_output": raw_model_output,
        "validation": validation,
        "review": {
            "status": "unreviewed",
            "reviewed_by": None,
            "reviewed_at": None,
        },
        "source": {
            "type": "model_generated",
        },
        "generation": {
            "prompt_mode": prompt_mode,
            "provider": provider,
            "model": model,
        },
        "timestamps": {
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
