from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def intersection_over_union(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compare_boxes(reference: Any, prediction: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    reference_annotations = reference.get("annotations") if isinstance(reference, dict) else None
    prediction_annotations = prediction.get("predictions") if isinstance(prediction, dict) else None

    if not isinstance(reference_annotations, list) or not isinstance(prediction_annotations, list):
        raise ValueError("Reference must contain 'annotations' and prediction must contain 'predictions'.")

    for index, (ref_item, pred_item) in enumerate(zip(reference_annotations, prediction_annotations), start=1):
        if not isinstance(ref_item, dict) or not isinstance(pred_item, dict):
            continue

        ref_subject_bbox = parse_bbox(ref_item.get("subject_bbox"))
        pred_subject_bbox = parse_bbox(pred_item.get("subject_bbox"))
        ref_object_bbox = parse_bbox(ref_item.get("object_bbox"))
        pred_object_bbox = parse_bbox(pred_item.get("object_bbox"))

        subject_iou = (
            intersection_over_union(ref_subject_bbox, pred_subject_bbox)
            if ref_subject_bbox is not None and pred_subject_bbox is not None
            else None
        )
        object_iou = (
            intersection_over_union(ref_object_bbox, pred_object_bbox)
            if ref_object_bbox is not None and pred_object_bbox is not None
            else None
        )

        results.append(
            {
                "index": index,
                "subject": ref_item.get("subject"),
                "object": ref_item.get("object"),
                "subject_iou": subject_iou,
                "object_iou": object_iou,
                "subject_match_pct": round((subject_iou or 0.0) * 100, 2) if subject_iou is not None else None,
                "object_match_pct": round((object_iou or 0.0) * 100, 2) if object_iou is not None else None,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reference boxes to LLM-predicted boxes using IoU.")
    parser.add_argument("reference_json", type=Path, help="Path to a reference JSON file with annotations")
    parser.add_argument("prediction_json", type=Path, help="Path to a prediction JSON file with predictions")
    args = parser.parse_args()

    reference = load_json(args.reference_json)
    prediction = load_json(args.prediction_json)
    results = compare_boxes(reference, prediction)

    if not results:
        print("No comparable boxes found.")
        return

    print(f"Compared {len(results)} annotation(s).")
    for result in results:
        print(
            f"#{result['index']} {result['subject']} / {result['object']}: "
            f"subject={result['subject_match_pct']}%, object={result['object_match_pct']}%"
        )

    subject_scores = [item["subject_iou"] for item in results if item["subject_iou"] is not None]
    object_scores = [item["object_iou"] for item in results if item["object_iou"] is not None]
    if subject_scores:
        avg_subject = sum(subject_scores) / len(subject_scores)
        print(f"Average subject IoU: {avg_subject:.3f} ({avg_subject*100:.2f}%)")
    if object_scores:
        avg_object = sum(object_scores) / len(object_scores)
        print(f"Average object IoU: {avg_object:.3f} ({avg_object*100:.2f}%)")


if __name__ == "__main__":
    main()
