from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.adapters.coco_adapter import normalized_to_absolute_xyxy
from evaluation.types import HicoGtInstance, HicoPredictionInstance


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_id(image_entry: dict) -> str:
    for key in ("image_id", "id", "file_name", "global_id"):
        value = image_entry.get(key)
        if value is not None:
            return str(value)
    return ""


def load_hicodet_ground_truth(annotation_path: Path) -> tuple[list[HicoGtInstance], dict[str, dict]]:
    payload = load_json(annotation_path)
    if not isinstance(payload, dict):
        raise ValueError("HICO-DET annotation file must be a JSON object.")

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_info_by_id = {
        _resolve_image_id(image): {
            "image_id": _resolve_image_id(image),
            "file_name": image.get("file_name"),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }
        for image in images
        if _resolve_image_id(image)
    }

    gt_instances: list[HicoGtInstance] = []
    for annotation in annotations:
        image_id = str(annotation.get("image_id") or annotation.get("file_name") or "")
        image_info = image_info_by_id.get(image_id)
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
        image_block = export_payload.get("image", {})
        image_id = str(export_payload.get("image_id") or image_block.get("image_id") or "")
        image_info = image_info_by_id.get(image_id)
        if image_info is None:
            issues.append({"image_id": image_id, "reason": "image_id_not_found"})
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
