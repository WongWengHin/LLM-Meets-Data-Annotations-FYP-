from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evaluation.types import CocoGtInstance, CocoPredictionInstance


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_to_absolute_xyxy(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        float(x1) * width,
        float(y1) * height,
        float(x2) * width,
        float(y2) * height,
    )


def build_coco_name_to_id_map(annotation_payload: dict) -> dict[str, int]:
    categories = annotation_payload.get("categories", [])
    name_to_id: dict[str, int] = {}
    for category in categories:
        name = str(category.get("name", "")).strip().casefold()
        category_id = category.get("id")
        if name and isinstance(category_id, int):
            name_to_id[name] = category_id
    return name_to_id


def load_coco_ground_truth(annotation_path: Path) -> tuple[list[CocoGtInstance], dict[str, dict], dict[str, int]]:
    payload = load_json(annotation_path)
    images = payload.get("images", [])
    image_info_by_id = {
        str(image["id"]): {
            "image_id": str(image["id"]),
            "file_name": image.get("file_name"),
            "width": int(image["width"]),
            "height": int(image["height"]),
        }
        for image in images
    }

    gt_instances: list[CocoGtInstance] = []
    for annotation in payload.get("annotations", []):
        image_id = str(annotation["image_id"])
        image_info = image_info_by_id.get(image_id)
        if image_info is None:
            continue
        x, y, width, height = annotation["bbox"]
        gt_instances.append(
            CocoGtInstance(
                image_id=image_id,
                category_id=int(annotation["category_id"]),
                bbox_xyxy_abs=(float(x), float(y), float(x + width), float(y + height)),
                annotation_id=str(annotation["id"]),
                area=float(annotation.get("area", width * height)),
                iscrowd=bool(annotation.get("iscrowd", 0)),
            )
        )

    return gt_instances, image_info_by_id, build_coco_name_to_id_map(payload)


def load_prediction_exports(prediction_dir: Path) -> list[dict]:
    exports: list[dict] = []
    for path in sorted(prediction_dir.glob("*-prediction.json")):
        exports.append(load_json(path))
    return exports


def adapt_predictions_to_coco(
    prediction_exports: list[dict],
    image_info_by_id: dict[str, dict],
    name_to_category_id: dict[str, int],
    label_mapping: dict[str, int] | None = None,
) -> tuple[list[CocoPredictionInstance], list[dict]]:
    predictions: list[CocoPredictionInstance] = []
    unmapped: list[dict] = []
    normalized_mapping = label_mapping or {}

    for export_payload in prediction_exports:
        image_block = export_payload.get("image", {})
        image_id = str(export_payload.get("image_id") or image_block.get("image_id") or "")
        image_info = image_info_by_id.get(image_id)
        if image_info is None:
            unmapped.append({"image_id": image_id, "reason": "image_id_not_found"})
            continue

        width = int(image_info["width"])
        height = int(image_info["height"])
        for prediction in export_payload.get("predictions", []):
            dataset_labels = prediction.get("dataset_labels") or {}
            mapped_category_id = dataset_labels.get("coco_category_id")
            if not isinstance(mapped_category_id, int):
                label_name = str(prediction.get("object") or prediction.get("category") or "").strip().casefold()
                mapped_category_id = normalized_mapping.get(label_name)
            if not isinstance(mapped_category_id, int):
                label_name = str(prediction.get("object") or prediction.get("category") or "").strip().casefold()
                mapped_category_id = name_to_category_id.get(label_name)

            bbox = prediction.get("object_bbox")
            if not isinstance(mapped_category_id, int):
                unmapped.append(
                    {
                        "image_id": image_id,
                        "annotation_id": prediction.get("annotation_id"),
                        "reason": "unmapped_category",
                        "label": prediction.get("object") or prediction.get("category"),
                    }
                )
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                unmapped.append(
                    {
                        "image_id": image_id,
                        "annotation_id": prediction.get("annotation_id"),
                        "reason": "missing_object_bbox",
                    }
                )
                continue

            predictions.append(
                CocoPredictionInstance(
                    image_id=image_id,
                    annotation_id=str(prediction["annotation_id"]),
                    category_id=mapped_category_id,
                    bbox_xyxy_abs=normalized_to_absolute_xyxy(bbox, width, height),
                    score=float(prediction.get("score") or 1.0),
                    raw_label=str(prediction.get("object") or prediction.get("category") or ""),
                )
            )

    return predictions, unmapped


def gt_count_by_category(gt_instances: list[CocoGtInstance]) -> dict[int, int]:
    return dict(Counter(instance.category_id for instance in gt_instances))
