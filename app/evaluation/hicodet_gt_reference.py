"""Utilities for converting HICO_sentences.py-style payloads into HICO-DET GT JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_hicodet_gt_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_metadata = payload.get("dataset_metadata", {})
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}

    images_payload = payload.get("images", [])
    if not isinstance(images_payload, list):
        raise ValueError("HICO sentences payload must contain an images list.")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for image_index, image_entry in enumerate(images_payload, start=1):
        if not isinstance(image_entry, dict):
            continue

        image_id = _resolve_image_id(image_entry, image_index)
        file_name = str(image_entry.get("file_name") or f"{image_id}.jpg")
        raw_image_annotations = image_entry.get("annotations", [])
        if not isinstance(raw_image_annotations, list):
            raw_image_annotations = []

        image_record = {
            "id": image_entry.get("id", image_id),
            "image_id": image_id,
            "file_name": file_name,
            "width": 1,
            "height": 1,
            "annotations": [],
        }

        for annotation_index, annotation in enumerate(raw_image_annotations, start=1):
            if not isinstance(annotation, dict):
                continue

            subject = str(annotation.get("subject") or "").strip() or "person"
            object_name = str(annotation.get("object") or "").strip()
            if not object_name:
                continue

            predicates = _normalize_predicates(annotation.get("predicate"))
            references = _clean_sentence_list(annotation.get("sentences"))
            subject_bbox = _bbox_xywh_to_xyxy(_clean_bbox(annotation.get("subject_bbox")))
            object_bbox = _bbox_xywh_to_xyxy(_clean_bbox(annotation.get("object_bbox")))
            if not references or not subject_bbox or not object_bbox:
                continue

            for predicate in predicates:
                annotation_id = f"{image_id}_{annotation_index:03d}_{predicate}"
                annotation_record = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "annotation_id": annotation_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_name,
                    "subject_bbox": subject_bbox,
                    "object_bbox": object_bbox,
                    "references": references,
                    "source_predicates": predicates,
                    "triplet_key": f"{subject}|{predicate}|{object_name}",
                }
                annotations.append(annotation_record)
                image_record["annotations"].append(annotation_record.copy())

        images.append(image_record)

    return {
        "dataset_metadata": dataset_metadata,
        "images": images,
        "annotations": annotations,
    }


def save_hicodet_gt_fixture(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_image_id(image_entry: dict[str, Any], image_index: int) -> str:
    for key in ("image_id", "id", "global_id"):
        value = image_entry.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    file_name = str(image_entry.get("file_name") or "").strip()
    if file_name:
        return Path(file_name).stem
    return str(image_index)


def _clean_sentence_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [sentence.strip() for sentence in value if isinstance(sentence, str) and sentence.strip()]


def _clean_bbox(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        return []

    bbox: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return []
        bbox.append(float(item))
    return bbox


def _bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        return []

    x, y, width, height = bbox
    x2 = x + width
    y2 = y + height
    return [
        _clamp01(x),
        _clamp01(y),
        _clamp01(x2),
        _clamp01(y2),
    ]


def _normalize_predicates(value: object) -> list[str]:
    if isinstance(value, str):
        predicates = [value]
    elif isinstance(value, list):
        predicates = [str(item) for item in value]
    else:
        predicates = []

    cleaned: list[str] = []
    for predicate in predicates:
        normalized = predicate.strip()
        if normalized:
            cleaned.append(normalized)

    return cleaned or ["unknown"]


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
