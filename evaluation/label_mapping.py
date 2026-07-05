from __future__ import annotations

import json
from pathlib import Path


def _normalize_key(value: str) -> str:
    return value.strip().casefold()


def _normalize_mapping(mapping: dict[str, object] | None) -> dict[str, int]:
    normalized: dict[str, int] = {}
    if not isinstance(mapping, dict):
        return normalized
    for key, value in mapping.items():
        if isinstance(key, str) and isinstance(value, int):
            normalized[_normalize_key(key)] = value
    return normalized


def load_label_mapping(path: Path | None) -> dict[str, dict[str, dict[str, int]]]:
    empty_mapping = {
        "coco": {"object_to_category_id": {}},
        "hicodet": {
            "object_to_id": {},
            "predicate_to_id": {},
        },
    }
    if path is None:
        return empty_mapping

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Label mapping file must be a JSON object.")

    coco = payload.get("coco")
    hicodet = payload.get("hicodet")
    return {
        "coco": {
            "object_to_category_id": _normalize_mapping(
                coco.get("object_to_category_id") if isinstance(coco, dict) else None
            ),
        },
        "hicodet": {
            "object_to_id": _normalize_mapping(
                hicodet.get("object_to_id") if isinstance(hicodet, dict) else None
            ),
            "predicate_to_id": _normalize_mapping(
                hicodet.get("predicate_to_id") if isinstance(hicodet, dict) else None
            ),
        },
    }
