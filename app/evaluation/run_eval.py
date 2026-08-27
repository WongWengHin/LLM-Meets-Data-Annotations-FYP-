"""*** 离线 COCO 与 HICO-DET IoU 评估的命令行入口。 ***"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.adapters.coco_adapter import (
    adapt_predictions_to_coco,
    gt_count_by_category,
    load_coco_ground_truth,
    load_prediction_exports as load_coco_predictions,
)
from app.evaluation.adapters.hicodet_adapter import (
    adapt_predictions_to_hicodet,
    gt_count_by_triplet,
    load_hicodet_ground_truth,
    load_prediction_exports as load_hicodet_predictions,
)
from app.evaluation.label_mapping import load_label_mapping
from app.evaluation.matching import coco_match_by_threshold, hicodet_match_by_threshold
from app.evaluation.metrics import summarize_coco_threshold, summarize_hicodet_threshold
from app.evaluation.reporting import write_csv, write_debug_reports, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline COCO or HICO-DET IoU evaluation.")
    parser.add_argument("--dataset", choices=("coco", "hicodet"), required=True)
    parser.add_argument("--predictions-dir", required=True, help="Directory containing *-prediction.json files.")
    parser.add_argument("--annotations", required=True, help="Path to official dataset annotation JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to write evaluation reports.")
    parser.add_argument("--label-mapping", required=False, help="Optional path to dataset label mapping JSON.")
    return parser.parse_args()


def build_coco_thresholds() -> list[float]:
    return [round(0.5 + 0.05 * index, 2) for index in range(10)]


def run_coco(
    predictions_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    label_mapping: dict[str, dict[str, dict[str, int]]] | None = None,
) -> None:
    gt_instances, image_info_by_id, name_to_category_id = load_coco_ground_truth(annotations_path)
    prediction_exports = load_coco_predictions(predictions_dir)
    predictions, unmapped = adapt_predictions_to_coco(
        prediction_exports,
        image_info_by_id,
        name_to_category_id,
        label_mapping=(label_mapping or {}).get("coco", {}).get("object_to_category_id", {}),
    )

    thresholds = build_coco_thresholds()
    gt_counts = gt_count_by_category(gt_instances)
    summary_by_threshold: dict[str, dict] = {}
    per_category_rows: list[dict] = []
    all_debug_records = []

    for threshold in thresholds:
        match_outcomes, debug_records = coco_match_by_threshold(predictions, gt_instances, threshold)
        threshold_summary = summarize_coco_threshold(match_outcomes, gt_counts)
        summary_by_threshold[f"{threshold:.2f}"] = threshold_summary
        all_debug_records.extend(debug_records)
        for category_id, ap in threshold_summary["per_category_ap"].items():
            per_category_rows.append(
                {
                    "iou_threshold": f"{threshold:.2f}",
                    "category_id": category_id,
                    "ap": ap,
                }
            )

    ap50 = summary_by_threshold["0.50"]["mean_ap"]
    ap75 = summary_by_threshold["0.75"]["mean_ap"]
    map_50_95 = sum(item["mean_ap"] for item in summary_by_threshold.values()) / len(summary_by_threshold)
    summary_payload = {
        "dataset": "coco",
        "prediction_count": len(predictions),
        "ground_truth_count": len(gt_instances),
        "unmapped_count": len(unmapped),
        "label_mapping_enabled": bool((label_mapping or {}).get("coco", {}).get("object_to_category_id")),
        "AP50": ap50,
        "AP75": ap75,
        "mAP_50_95": map_50_95,
        "thresholds": summary_by_threshold,
        "unmapped": unmapped,
    }

    write_json(output_dir / "coco-summary.json", summary_payload)
    write_csv(output_dir / "coco-per-category.csv", per_category_rows, ["iou_threshold", "category_id", "ap"])
    write_markdown(
        output_dir / "coco-summary.md",
        "COCO Evaluation Summary",
        [
            f"- Prediction count: {len(predictions)}",
            f"- Ground truth count: {len(gt_instances)}",
            f"- AP50: {ap50:.4f}",
            f"- AP75: {ap75:.4f}",
            f"- mAP@[0.50:0.95]: {map_50_95:.4f}",
            f"- Unmapped predictions: {len(unmapped)}",
        ],
    )
    write_debug_reports(output_dir / "debug", all_debug_records)


def run_hicodet(
    predictions_dir: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    label_mapping: dict[str, dict[str, dict[str, int]]] | None = None,
) -> None:
    gt_instances, image_info_by_id = load_hicodet_ground_truth(annotations_path)
    prediction_exports = load_hicodet_predictions(predictions_dir)
    hicodet_mapping = (label_mapping or {}).get("hicodet", {})
    predictions, issues = adapt_predictions_to_hicodet(
        prediction_exports,
        image_info_by_id,
        object_mapping=hicodet_mapping.get("object_to_id", {}),
        predicate_mapping=hicodet_mapping.get("predicate_to_id", {}),
    )
    thresholds = [0.5, 0.75]
    gt_counts = gt_count_by_triplet(gt_instances)
    summary_by_threshold: dict[str, dict] = {}
    per_triplet_rows: list[dict] = []
    all_debug_records = []

    for threshold in thresholds:
        match_outcomes, debug_records = hicodet_match_by_threshold(predictions, gt_instances, threshold)
        threshold_summary = summarize_hicodet_threshold(match_outcomes, gt_counts)
        summary_by_threshold[f"{threshold:.2f}"] = threshold_summary
        all_debug_records.extend(debug_records)
        for triplet_key, triplet_metrics in threshold_summary["per_triplet"].items():
            per_triplet_rows.append(
                {
                    "iou_threshold": f"{threshold:.2f}",
                    "triplet_key": triplet_key,
                    "tp": triplet_metrics["tp"],
                    "fp": triplet_metrics["fp"],
                    "fn": triplet_metrics["fn"],
                    "precision": triplet_metrics["precision"],
                    "recall": triplet_metrics["recall"],
                    "ap": triplet_metrics["ap"],
                }
            )

    summary_payload = {
        "dataset": "hicodet",
        "prediction_count": len(predictions),
        "ground_truth_count": len(gt_instances),
        "label_mapping_enabled": bool(
            hicodet_mapping.get("object_to_id") or hicodet_mapping.get("predicate_to_id")
        ),
        "issues": issues,
        "thresholds": summary_by_threshold,
        "triplet_mean_ap@0.50": summary_by_threshold["0.50"]["triplet_mean_ap"],
        "F1@0.50": summary_by_threshold["0.50"]["overall"]["f1"],
    }
    write_json(output_dir / "hicodet-summary.json", summary_payload)
    write_csv(
        output_dir / "hicodet-per-triplet.csv",
        per_triplet_rows,
        ["iou_threshold", "triplet_key", "tp", "fp", "fn", "precision", "recall", "ap"],
    )
    recall_at_05 = summary_by_threshold["0.50"]["overall"]["recall"]
    precision_at_05 = summary_by_threshold["0.50"]["overall"]["precision"]
    f1_at_05 = summary_by_threshold["0.50"]["overall"]["f1"]
    map_at_05 = summary_by_threshold["0.50"]["triplet_mean_ap"]
    write_markdown(
        output_dir / "hicodet-summary.md",
        "HICO-DET Evaluation Summary",
        [
            f"- Prediction count: {len(predictions)}",
            f"- Ground truth count: {len(gt_instances)}",
            f"- Precision@IoU=0.5: {precision_at_05:.4f}",
            f"- Recall@IoU=0.5: {recall_at_05:.4f}",
            f"- F1@IoU=0.5: {f1_at_05:.4f}",
            f"- triplet mean AP@IoU=0.5: {map_at_05:.4f}",
            f"- Issues: {len(issues)}",
        ],
    )
    write_debug_reports(output_dir / "debug", all_debug_records)


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    annotations_path = Path(args.annotations)
    output_dir = Path(args.output_dir)
    label_mapping = load_label_mapping(Path(args.label_mapping) if args.label_mapping else None)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "coco":
        run_coco(predictions_dir, annotations_path, output_dir, label_mapping=label_mapping)
        return
    run_hicodet(predictions_dir, annotations_path, output_dir, label_mapping=label_mapping)


if __name__ == "__main__":
    main()
