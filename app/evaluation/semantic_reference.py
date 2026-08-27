"""Utilities for HICO-style semantic reference fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HicoSemanticReference:
    image_id: str
    file_name: str
    annotation_id: str
    subject: str
    predicate: str
    object_name: str
    subject_bbox: list[float]
    object_bbox: list[float]
    references: list[str]
    source_predicates: list[str]

    @property
    def triplet_key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object_name}"


def load_hico_semantic_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Semantic fixture must be a JSON object.")
    return payload


def build_hico_semantic_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    images = payload.get("images", [])
    if not isinstance(images, list):
        raise ValueError("HICO sentences payload must contain an images list.")

    records: list[dict[str, Any]] = []
    for image_index, image_entry in enumerate(images, start=1):
        if not isinstance(image_entry, dict):
            continue

        image_id = str(image_entry.get("id") or image_entry.get("image_id") or image_index)
        file_name = str(image_entry.get("file_name") or "")
        annotations = image_entry.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        for annotation_index, annotation in enumerate(annotations, start=1):
            if not isinstance(annotation, dict):
                continue

            subject = str(annotation.get("subject") or "").strip()
            object_name = str(annotation.get("object") or "").strip()
            raw_predicates = annotation.get("predicate") or []
            references = _clean_sentence_list(annotation.get("sentences"))
            subject_bbox = _clean_bbox(annotation.get("subject_bbox"))
            object_bbox = _clean_bbox(annotation.get("object_bbox"))

            if not subject or not object_name or not references:
                continue

            predicates = _normalize_predicates(raw_predicates)
            for predicate in predicates:
                annotation_id = f"{image_id}_{annotation_index:03d}_{predicate}"
                records.append(
                    {
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
                )

    return records


def build_hico_semantic_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_metadata = payload.get("dataset_metadata", {})
    if not isinstance(dataset_metadata, dict):
        dataset_metadata = {}

    return {
        "dataset_metadata": dataset_metadata,
        "records": build_hico_semantic_records(payload),
    }


def save_hico_semantic_fixture(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _normalize_predicates(value: object) -> list[str]:
    if isinstance(value, str):
        predicates = [value]
    elif isinstance(value, list):
        predicates = [str(item) for item in value]
    else:
        predicates = []

    cleaned = []
    for predicate in predicates:
        normalized = predicate.strip()
        if normalized:
            cleaned.append(normalized)

    return cleaned or ["unknown"]
