"""*** HICO-DET 真值与预测结果的评估适配逻辑。 ***"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from app.evaluation.adapters.coco_adapter import normalized_to_absolute_xyxy
from app.evaluation.types import HicoGtInstance, HicoPredictionInstance


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_id(image_entry: dict) -> str:
    for key in ("image_id", "id", "file_name", "global_id"):
        value = image_entry.get(key)
        if value is not None:
            return str(value)
    return ""


def _image_aliases(image_entry: dict) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    def add_alias(value: object) -> None:
        if value is None:
            return

        text = str(value).strip()
        if not text:
            return

        for candidate in (text, Path(text).stem):
            candidate = str(candidate).strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                aliases.append(candidate)

    for key in ("image_id", "id", "global_id", "file_name"):
        add_alias(image_entry.get(key))

    return aliases


def _annotation_image_aliases(annotation: dict) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()

    def add_alias(value: object) -> None:
        if value is None:
            return

        text = str(value).strip()
        if not text:
            return

        for candidate in (text, Path(text).stem):
            candidate = str(candidate).strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                aliases.append(candidate)

    # Prefer filename-derived stems so evaluations can align with uploaded file stems.
    file_name = annotation.get("file_name")
    if file_name is not None:
        add_alias(Path(str(file_name)).stem)
        add_alias(file_name)
    add_alias(annotation.get("image_id"))
    add_alias(annotation.get("id"))
    add_alias(annotation.get("global_id"))

    return aliases


def _prediction_image_aliases(export_payload: dict) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    image_block = export_payload.get("image", {})
    if not isinstance(image_block, dict):
        image_block = {}

    def add_alias(value: object) -> None:
        if value is None:
            return

        text = str(value).strip()
        if not text:
            return

        for candidate in (text, Path(text).stem):
            candidate = str(candidate).strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                aliases.append(candidate)

    for value in (
        export_payload.get("image_id"),
        image_block.get("image_id"),
        image_block.get("id"),
        image_block.get("global_id"),
        image_block.get("file_name"),
    ):
        add_alias(value)

    return aliases


def _build_image_info_lookup(images: list[dict]) -> dict[str, dict]:
    image_info_by_id: dict[str, dict] = {}
    for image in images:
        if not isinstance(image, dict):
            continue

        aliases = _image_aliases(image)
        if not aliases:
            continue

        image_info = {
            "image_id": aliases[0],
            "file_name": image.get("file_name"),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }
        for alias in aliases:
            image_info_by_id.setdefault(alias, image_info)

    return image_info_by_id


def _iter_hicodet_annotations(payload: dict) -> list[dict]:
    annotations = payload.get("annotations", [])
    if isinstance(annotations, list) and annotations:
        return [annotation for annotation in annotations if isinstance(annotation, dict)]

    flattened: list[dict] = []
    images = payload.get("images", [])
    if not isinstance(images, list):
        return flattened

    for image in images:
        if not isinstance(image, dict):
            continue

        image_id = _resolve_image_id(image)
        nested_annotations = image.get("annotations", [])
        if not isinstance(nested_annotations, list):
            continue

        for annotation in nested_annotations:
            if not isinstance(annotation, dict):
                continue
            merged = dict(annotation)
            if image_id and not merged.get("image_id"):
                merged["image_id"] = image_id
            if image.get("file_name") and not merged.get("file_name"):
                merged["file_name"] = image.get("file_name")
            flattened.append(merged)

    return flattened


def load_hicodet_ground_truth(annotation_path: Path) -> tuple[list[HicoGtInstance], dict[str, dict]]:
    payload = load_json(annotation_path)
    if not isinstance(payload, dict):
        raise ValueError("HICO-DET annotation file must be a JSON object.")

    images = payload.get("images", [])
    annotations = _iter_hicodet_annotations(payload)
    image_info_by_id = _build_image_info_lookup(images if isinstance(images, list) else [])

    gt_instances: list[HicoGtInstance] = []
    for annotation in annotations:
        annotation_image_aliases = _annotation_image_aliases(annotation)
        image_id = next((alias for alias in annotation_image_aliases if alias in image_info_by_id), "")
        image_info = image_info_by_id.get(image_id) if image_id else None
        if image_info is None:
            continue

        subject_bbox = annotation.get("subject_bbox")
        object_bbox = annotation.get("object_bbox")
        if not (isinstance(subject_bbox, list) and isinstance(object_bbox, list)):
            continue

        gt_instances.append(
            HicoGtInstance(
                image_id=image_id,
                annotation_id=str(annotation.get("id") or annotation.get("annotation_id") or len(gt_instances) + 1),
                subject_bbox_xyxy_abs=tuple(float(value) for value in subject_bbox),  # type: ignore[arg-type]
                object_bbox_xyxy_abs=tuple(float(value) for value in object_bbox),  # type: ignore[arg-type]
                predicate=str(annotation.get("predicate") or annotation.get("verb") or ""),
                object_name=str(annotation.get("object") or annotation.get("object_name") or ""),
                subject_name=str(annotation.get("subject") or "person"),
                predicate_id=annotation.get("predicate_id"),
                object_id=annotation.get("object_id"),
                is_rare=annotation.get("is_rare"),
            )
        )

    return gt_instances, image_info_by_id


def load_prediction_exports(prediction_dir: Path) -> list[dict]:
    exports: list[dict] = []
    for path in sorted(prediction_dir.glob("*-prediction.json")):
        exports.append(json.loads(path.read_text(encoding="utf-8")))
    return exports


def adapt_predictions_to_hicodet(
    prediction_exports: list[dict],
    image_info_by_id: dict[str, dict],
    object_mapping: dict[str, int] | None = None,
    predicate_mapping: dict[str, int] | None = None,
) -> tuple[list[HicoPredictionInstance], list[dict]]:
    predictions: list[HicoPredictionInstance] = []
    issues: list[dict] = []
    normalized_object_mapping = object_mapping or {}
    normalized_predicate_mapping = predicate_mapping or {}

    for export_payload in prediction_exports:
        image_aliases = _prediction_image_aliases(export_payload)
        image_id = next((alias for alias in image_aliases if alias in image_info_by_id), "")
        image_info = image_info_by_id.get(image_id) if image_id else None
        if image_info is None:
            issues.append({"image_id": image_aliases[0] if image_aliases else "", "reason": "image_id_not_found"})
            continue

        width = int(image_info["width"])
        height = int(image_info["height"])
        for prediction in export_payload.get("predictions", []):
            subject_bbox = prediction.get("subject_bbox")
            object_bbox = prediction.get("object_bbox")
            if not (isinstance(subject_bbox, list) and len(subject_bbox) == 4):
                issues.append(
                    {
                        "image_id": image_id,
                        "annotation_id": prediction.get("annotation_id"),
                        "reason": "missing_subject_bbox",
                    }
                )
                continue
            if not (isinstance(object_bbox, list) and len(object_bbox) == 4):
                issues.append(
                    {
                        "image_id": image_id,
                        "annotation_id": prediction.get("annotation_id"),
                        "reason": "missing_object_bbox",
                    }
                )
                continue

            dataset_labels = prediction.get("dataset_labels") or {}
            object_name = str(prediction.get("object") or "")
            predicate_name = str(prediction.get("predicate") or "")
            mapped_object_id = dataset_labels.get("hicodet_object_id")
            mapped_predicate_id = dataset_labels.get("hicodet_verb_id")
            if not isinstance(mapped_object_id, int):
                mapped_object_id = normalized_object_mapping.get(object_name.strip().casefold())
            if not isinstance(mapped_predicate_id, int):
                mapped_predicate_id = normalized_predicate_mapping.get(predicate_name.strip().casefold())
            predictions.append(
                HicoPredictionInstance(
                    image_id=image_id,
                    annotation_id=str(prediction["annotation_id"]),
                    subject_bbox_xyxy_abs=normalized_to_absolute_xyxy(subject_bbox, width, height),
                    object_bbox_xyxy_abs=normalized_to_absolute_xyxy(object_bbox, width, height),
                    predicate=predicate_name,
                    object_name=object_name,
                    subject_name=str(prediction.get("subject") or "person"),
                    score=float(prediction.get("score") or 1.0),
                    predicate_id=mapped_predicate_id if isinstance(mapped_predicate_id, int) else None,
                    object_id=mapped_object_id if isinstance(mapped_object_id, int) else None,
                )
            )

    return predictions, issues


def gt_count_by_triplet(gt_instances: list[HicoGtInstance]) -> dict[str, int]:
    return dict(Counter(_triplet_metric_key(instance) for instance in gt_instances))


def _triplet_metric_key(instance: HicoGtInstance | HicoPredictionInstance) -> str:
    if instance.predicate_id is not None and instance.object_id is not None:
        return f"{instance.subject_name.casefold()}|verb:{instance.predicate_id}|object:{instance.object_id}"
    return instance.triplet_key.casefold()
