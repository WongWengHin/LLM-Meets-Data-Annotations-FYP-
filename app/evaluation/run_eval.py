"""Offline COCO report evaluator.

The database-backed HICO-DET evaluator is
``evaluation_new/HICO-DET/hicodet_evaluator.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.adapters.coco_adapter import adapt_predictions_to_coco, gt_count_by_category, load_coco_ground_truth, load_prediction_exports
from app.evaluation.label_mapping import load_label_mapping
from app.evaluation.matching import coco_match_by_threshold
from app.evaluation.metrics import summarize_coco_threshold
from app.evaluation.reporting import write_csv, write_debug_reports, write_json, write_markdown


def build_coco_thresholds() -> list[float]:
    return [round(0.5 + 0.05 * index, 2) for index in range(10)]


def run_coco(predictions_dir: Path, annotations_path: Path, output_dir: Path, *, label_mapping: dict | None = None) -> None:
    truths, image_info, names = load_coco_ground_truth(annotations_path)
    predictions, unmapped = adapt_predictions_to_coco(load_prediction_exports(predictions_dir), image_info, names, label_mapping=(label_mapping or {}).get("coco", {}).get("object_to_category_id", {}))
    summaries, rows, debug = {}, [], []
    for threshold in build_coco_thresholds():
        outcomes, records = coco_match_by_threshold(predictions, truths, threshold)
        summary = summarize_coco_threshold(outcomes, gt_count_by_category(truths))
        summaries[f"{threshold:.2f}"] = summary
        debug.extend(records)
        rows.extend({"iou_threshold": f"{threshold:.2f}", "category_id": category, "ap": ap} for category, ap in summary["per_category_ap"].items())
    payload = {"dataset": "coco", "prediction_count": len(predictions), "ground_truth_count": len(truths), "unmapped_count": len(unmapped), "AP50": summaries["0.50"]["mean_ap"], "AP75": summaries["0.75"]["mean_ap"], "mAP_50_95": sum(item["mean_ap"] for item in summaries.values()) / len(summaries), "thresholds": summaries, "unmapped": unmapped}
    write_json(output_dir / "coco-summary.json", payload)
    write_csv(output_dir / "coco-per-category.csv", rows, ["iou_threshold", "category_id", "ap"])
    write_markdown(output_dir / "coco-summary.md", "COCO Evaluation Summary", [f"- Prediction count: {len(predictions)}", f"- Ground truth count: {len(truths)}", f"- AP50: {payload['AP50']:.4f}", f"- AP75: {payload['AP75']:.4f}", f"- mAP@[0.50:0.95]: {payload['mAP_50_95']:.4f}", f"- Unmapped predictions: {len(unmapped)}"])
    write_debug_reports(output_dir / "debug", debug)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("coco",), default="coco")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-mapping")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_coco(Path(args.predictions_dir), Path(args.annotations), output_dir, label_mapping=load_label_mapping(Path(args.label_mapping) if args.label_mapping else None))


if __name__ == "__main__":
    main()
