"""Run HICO-style semantic evaluation on matched HOI pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.reporting import write_csv, write_json, write_markdown
from app.evaluation.semantic_metrics import evaluate_hico_semantic_pairs, summarize_semantic_matches
from app.evaluation.semantic_reference import load_hico_semantic_fixture
from app.evaluation.adapters.hicodet_adapter import load_prediction_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic evaluation for HICO-style HOI predictions.")
    parser.add_argument("--predictions-dir", required=True, help="Directory containing *-prediction.json files.")
    parser.add_argument("--semantic-reference", required=True, help="Path to converted HICO semantic reference JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory to write semantic evaluation reports.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for spatial matching.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    reference_path = Path(args.semantic_reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_payload = load_hico_semantic_fixture(reference_path)
    semantic_records = reference_payload.get("records", [])
    if not isinstance(semantic_records, list):
        raise RuntimeError("Semantic reference fixture does not contain a records list.")

    prediction_exports = load_prediction_exports(predictions_dir)
    matches, issues = evaluate_hico_semantic_pairs(
        prediction_exports,
        [record for record in semantic_records if isinstance(record, dict)],
        iou_threshold=args.iou_threshold,
    )
    summary = summarize_semantic_matches(matches)
    summary_payload = {
        "dataset": "hicodet_semantic",
        "pair_count": summary["pair_count"],
        "matched_pair_count": summary["matched_pair_count"],
        "match_rate": summary["match_rate"],
        "bleu": summary["bleu"],
        "meteor": summary["meteor"],
        "iou_threshold": args.iou_threshold,
        "issues": issues,
        "per_triplet": summary["per_triplet"],
    }

    write_json(output_dir / "hicodet-semantic-summary.json", summary_payload)
    write_csv(
        output_dir / "hicodet-semantic-pairs.csv",
        [
            {
                "image_id": match["image_id"],
                "prediction_id": match["prediction_id"],
                "reference_id": match["reference_id"] or "",
                "triplet_key": match["triplet_key"],
                "matched": match["matched"],
                "subject_iou": match["subject_iou"],
                "object_iou": match["object_iou"],
                "bleu": match["bleu"],
                "meteor": match["meteor"],
                "candidate_count": match["candidate_count"],
                "reference_count": match["reference_count"],
            }
            for match in matches
        ],
        [
            "image_id",
            "prediction_id",
            "reference_id",
            "triplet_key",
            "matched",
            "subject_iou",
            "object_iou",
            "bleu",
            "meteor",
            "candidate_count",
            "reference_count",
        ],
    )
    write_markdown(
        output_dir / "hicodet-semantic-summary.md",
        "HICO Semantic Evaluation Summary",
        [
            f"- Pair count: {summary['pair_count']}",
            f"- Matched pair count: {summary['matched_pair_count']}",
            f"- Match rate: {summary['match_rate']:.4f}",
            f"- BLEU: {summary['bleu']:.4f}",
            f"- METEOR: {summary['meteor']:.4f}",
            f"- IoU threshold: {args.iou_threshold:.2f}",
            f"- Issues: {len(issues)}",
        ],
    )


if __name__ == "__main__":
    main()
