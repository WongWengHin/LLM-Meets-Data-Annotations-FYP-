"""Single-image COCO evaluation using weighted-cost Hungarian matching."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.evaluation_database import (
    calculate_coco_batch_metrics,
    create_coco_batch_evaluation_run,
    save_coco_evaluation,
    set_evaluation_run_status,
)
from evaluation_new.COCO.sentence_metrics import SentenceMetricScorer, sentence_list

DEFAULT_GT = Path(__file__).parent / "ground_truth" / "official_coco_gt.json"
DEFAULT_PREDICTIONS = ROOT / "data" / "predictions"
DEFAULT_DATABASE = ROOT / "data" / "user_results.db"
DEFAULT_EVALUATION_DATABASE = ROOT / "data" / "evaluation_result.db"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid or empty JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Top level of {path} must be an object")
    return value


def annotations_in(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Support app prediction, frontend/master, evaluation-export JSON."""
    for key in ("predictions", "annotations", "bbox_export"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("Prediction JSON needs a predictions, annotations, or bbox_export list")


def load_prediction_file(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    payload = load_json(path)
    image_id = payload.get("image_id")
    if image_id is None and isinstance(payload.get("image"), dict):
        image_id = payload["image"].get("image_id")
    return annotations_in(payload), None if image_id is None else str(image_id)


def load_database_run(path: Path, run_id: int) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise ValueError(f"SQLite database does not exist: {path}")
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "user_annotation_runs" in tables:
            table = "user_annotation_runs"
        elif "annotation_runs" in tables:
            table = "annotation_runs"
        else:
            raise ValueError(
                f"{path} has neither annotation_runs nor user_annotation_runs"
            )
        row = connection.execute(
            f"SELECT image_id, annotation_result FROM {table} WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"No annotation run has id {run_id}")
    try:
        payload = json.loads(row["annotation_result"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Run {run_id} has invalid annotation_result JSON") from exc
    return annotations_in(payload), canonical_image_id(row["image_id"])


def load_latest_database_run_for_image(
    path: Path, image_id: str
) -> tuple[list[dict[str, Any]], str, int]:
    """Load the newest SQLite annotation run matching an image ID."""
    if not path.is_file():
        raise ValueError(f"SQLite database does not exist: {path}")
    wanted_id = canonical_image_id(image_id)
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "user_annotation_runs" in tables:
            table = "user_annotation_runs"
        elif "annotation_runs" in tables:
            table = "annotation_runs"
        else:
            raise ValueError(
                f"{path} has neither annotation_runs nor user_annotation_runs"
            )
        rows = connection.execute(
            f"SELECT id, image_id, annotation_result FROM {table} "
            "ORDER BY created_at DESC, id DESC"
        ).fetchall()
    row = next(
        (item for item in rows if canonical_image_id(item["image_id"]) == wanted_id),
        None,
    )
    if row is None:
        raise ValueError(f"No annotation run found for image_id {image_id!r}")
    try:
        payload = json.loads(row["annotation_result"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Run {row['id']} has invalid annotation_result JSON") from exc
    return annotations_in(payload), wanted_id, int(row["id"])


def load_database_prompt_mode(path: Path, run_id: int) -> str | None:
    """Return a run's prompt mode when the source database provides one."""
    with closing(sqlite3.connect(path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        table = "user_annotation_runs" if "user_annotation_runs" in tables else "annotation_runs"
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "prompt_mode" not in columns:
            return None
        row = connection.execute(
            f"SELECT prompt_mode FROM {table} WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None or row[0] is None:
        return None
    value = str(row[0]).strip()
    return value or None


def load_database_batch_id(path: Path, run_id: int) -> str | None:
    """Return the source batch ID when the annotation run belongs to a batch."""
    with closing(sqlite3.connect(path)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        table = "user_annotation_runs" if "user_annotation_runs" in tables else "annotation_runs"
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "batch_id" not in columns:
            return None
        row = connection.execute(
            f"SELECT batch_id FROM {table} WHERE id = ?", (run_id,)
        ).fetchone()
    if row is None or row[0] is None:
        return None
    value = str(row[0]).strip()
    return value or None


def load_batch_annotation_runs(
    path: Path, batch_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the latest completed annotation for every image in an upload batch."""
    if not path.is_file():
        raise ValueError(f"SQLite database does not exist: {path}")
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        batch = connection.execute(
            "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError(f"Batch run not found: {batch_id}")
        rows = connection.execute(
            """
            SELECT run.id, run.image_id, run.prompt_mode, run.annotation_result
            FROM user_annotation_runs AS run
            INNER JOIN (
                SELECT image_id, MAX(id) AS id
                FROM user_annotation_runs
                WHERE batch_id = ? AND task_status = 'completed'
                GROUP BY image_id
            ) AS latest ON latest.id = run.id
            ORDER BY run.id
            """,
            (batch_id,),
        ).fetchall()
    return dict(batch), [dict(row) for row in rows]


def infer_prompt_mode(predictions: list[dict[str, Any]]) -> str:
    """Infer a prompt mode from exported prediction generation metadata."""
    modes = {
        str(item.get("generation", {}).get("prompt_mode")).strip()
        for item in predictions
        if isinstance(item.get("generation"), dict)
        and item["generation"].get("prompt_mode") is not None
        and str(item["generation"]["prompt_mode"]).strip()
    }
    if len(modes) == 1:
        return modes.pop()
    return "mixed" if modes else "unknown"


def canonical_image_id(image_id: Any) -> str:
    value = str(image_id).strip()
    return str(int(value)) if value.isdecimal() else value


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def validate_annotation(item: dict[str, Any], source: str) -> None:
    bbox = item.get("bbox") if isinstance(item, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{source}.bbox must be normalized [x1, y1, x2, y2]")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in bbox):
        raise ValueError(f"{source}.bbox values must be numbers")
    x1, y1, x2, y2 = map(float, bbox)
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (x1, y1, x2, y2)):
        raise ValueError(f"{source}.bbox values must be finite and in 0..1: {bbox}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{source}.bbox must have x2 > x1 and y2 > y1: {bbox}")
    if not isinstance(item.get("class"), str) or not item["class"].strip():
        raise ValueError(f"{source}.class must be a non-empty string")


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def l2(a: list[float], b: list[float]) -> float:
    """Euclidean distance over the four normalized box coordinates."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def class_name(item: dict[str, Any]) -> str:
    return item["class"].strip().casefold()


def pair_matrices(predictions, truths, iou_weight, class_weight, l2_weight):
    costs, ious, distances = [], [], []
    for prediction in predictions:
        cost_row, iou_row, distance_row = [], [], []
        for truth in truths:
            overlap = iou(prediction["bbox"], truth["bbox"])
            distance = l2(prediction["bbox"], truth["bbox"])
            mismatch = float(class_name(prediction) != class_name(truth))
            cost_row.append(iou_weight * (1 - overlap) + class_weight * mismatch + l2_weight * distance)
            iou_row.append(overlap)
            distance_row.append(distance)
        costs.append(cost_row); ious.append(iou_row); distances.append(distance_row)
    return costs, ious, distances


def _assign_rows(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Hungarian algorithm for rows <= columns."""
    rows, columns = len(cost), len(cost[0])
    u, v, assigned, path = [0.0] * (rows + 1), [0.0] * (columns + 1), [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        assigned[0] = row
        minimum, used, column = [math.inf] * (columns + 1), [False] * (columns + 1), 0
        while True:
            used[column] = True
            current, delta, next_column = assigned[column], math.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]: continue
                reduced = cost[current - 1][candidate - 1] - u[current] - v[candidate]
                if reduced < minimum[candidate]: minimum[candidate], path[candidate] = reduced, column
                if minimum[candidate] < delta: delta, next_column = minimum[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]: u[assigned[candidate]] += delta; v[candidate] -= delta
                else: minimum[candidate] -= delta
            column = next_column
            if assigned[column] == 0: break
        while True:
            previous = path[column]
            assigned[column] = assigned[previous]
            column = previous
            if column == 0: break
    return sorted((row - 1, column - 1) for column, row in enumerate(assigned[1:], 1) if row)


def hungarian_matching(cost: list[list[float]]) -> list[tuple[int, int]]:
    if not cost or not cost[0]: return []
    if len(cost) <= len(cost[0]): return _assign_rows(cost)
    return sorted((column, row) for row, column in _assign_rows([list(x) for x in zip(*cost)]))


def ground_truth_for_image(payload: dict[str, Any], image_id: str) -> list[dict[str, Any]]:
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Ground-truth JSON needs an annotations list")
    image_id = canonical_image_id(image_id)
    matches = [
        item for item in annotations
        if canonical_image_id(item.get("image_id")) == image_id
    ]
    if not matches:
        available = sorted({canonical_image_id(item.get("image_id")) for item in annotations})
        raise ValueError(f"No ground truth for image_id {image_id!r}; sample available IDs: {', '.join(available[:10]) or 'none'}")
    return matches


def evaluate_image(predictions, truths, *, image_id, iou_threshold=0.5, iou_weight=2.0, class_weight=1.0, l2_weight=1.0):
    for index, item in enumerate(predictions): validate_annotation(item, f"prediction[{index}]")
    for index, item in enumerate(truths): validate_annotation(item, f"ground_truth[{index}]")
    costs, ious, distances = pair_matrices(predictions, truths, iou_weight, class_weight, l2_weight)
    assignments = hungarian_matching(costs)
    pairs, tp, errors = [], 0, 0
    for p, g in assignments:
        same_class, overlap = class_name(predictions[p]) == class_name(truths[g]), ious[p][g]
        accepted = overlap >= iou_threshold and same_class
        rejection_reasons = []
        if overlap < iou_threshold:
            rejection_reasons.append("low_iou")
        if not same_class:
            rejection_reasons.append("class_mismatch")
        tp += int(accepted); errors += int(overlap >= iou_threshold and not same_class)
        pairs.append({
            "prediction_index": p, "ground_truth_index": g,
            "prediction_annotation_id": predictions[p].get("annotation_id"),
            "ground_truth_annotation_id": truths[g].get("annotation_id", truths[g].get("id")),
            "predicted_class": predictions[p]["class"], "ground_truth_class": truths[g]["class"],
            "predicted_bbox": predictions[p]["bbox"], "ground_truth_bbox": truths[g]["bbox"],
            "iou": overlap, "l2_distance": distances[p][g], "matching_cost": costs[p][g],
            "class_correct": same_class,
            "status": "TP" if accepted else "REJECTED",
            "rejection_reason": ",".join(rejection_reasons) or None,
            "result": "TP" if accepted else "rejected",
        })
    used_p, used_g = {p for p, _ in assignments}, {g for _, g in assignments}
    fp = len(predictions) - tp
    fn = len(truths) - tp
    return {
        "image_id": image_id,
        "config": {"iou_threshold": iou_threshold, "cost_weights": {"iou": iou_weight, "class": class_weight, "l2": l2_weight}},
        "counts": {"predictions": len(predictions), "ground_truths": len(truths), "tp": tp,
                   "fp": fp, "fn": fn, "classification_errors": errors},
        "metrics": precision_recall_f1(tp, fp, fn),
        "cost_matrix": costs, "assigned_pairs": pairs,
        "sentence_summary": {
            "eligible_pairs": 0, "evaluated_pairs": 0,
            "mean_bleu": None, "mean_meteor": None,
        },
        "unassigned_prediction_indices": [i for i in range(len(predictions)) if i not in used_p],
        "unassigned_ground_truth_indices": [i for i in range(len(truths)) if i not in used_g],
    }


def apply_sentence_evaluation(result, predictions, truths, *, scorer=None):
    """Score captions for IoU-eligible matched pairs without changing detection metrics."""
    metric_scorer = scorer or SentenceMetricScorer()
    evaluated_scores = []
    eligible_pairs = 0
    for pair in result["assigned_pairs"]:
        prediction = predictions[pair["prediction_index"]]
        truth = truths[pair["ground_truth_index"]]
        predicted_sentences = sentence_list(prediction.get("sentences"))
        ground_truth_sentences = sentence_list(truth.get("sentences"))
        pair["predicted_sentences"] = predicted_sentences
        pair["ground_truth_sentences"] = ground_truth_sentences
        pair["bleu"] = None
        pair["meteor"] = None
        pair["sentence_eligible"] = pair["iou"] >= result["config"]["iou_threshold"]
        if pair["sentence_eligible"]:
            eligible_pairs += 1
            if not predicted_sentences:
                pair["sentence_skip_reason"] = "missing_predicted_sentences"
            elif not ground_truth_sentences:
                pair["sentence_skip_reason"] = "missing_ground_truth_sentences"
            else:
                bleu, meteor = metric_scorer.score_text(
                    predicted_sentences[0], ground_truth_sentences
                )
                pair["bleu"] = bleu
                pair["meteor"] = meteor
                pair["sentence_skip_reason"] = None
                evaluated_scores.append((bleu, meteor))
        else:
            pair["sentence_skip_reason"] = "low_iou"
    result["sentence_summary"] = {
        "eligible_pairs": eligible_pairs,
        "evaluated_pairs": len(evaluated_scores),
        "mean_bleu": (
            sum(score[0] for score in evaluated_scores) / len(evaluated_scores)
            if evaluated_scores else None
        ),
        "mean_meteor": (
            sum(score[1] for score in evaluated_scores) / len(evaluated_scores)
            if evaluated_scores else None
        ),
    }
    return result


def evaluate_batch(
    *,
    batch_id: str,
    user_database_path: Path,
    evaluation_database_path: Path,
    ground_truth_path: Path,
    iou_threshold: float = 0.5,
    iou_weight: float = 2.0,
    class_weight: float = 1.0,
    l2_weight: float = 1.0,
    sentence_scorer=None,
) -> dict[str, Any]:
    """Evaluate every available image in a batch under one evaluation run ID."""
    batch, annotation_runs = load_batch_annotation_runs(user_database_path, batch_id)
    ground_truth_payload = load_json(ground_truth_path)
    evaluation_run_id = create_coco_batch_evaluation_run(
        evaluation_database_path,
        batch_id=batch_id,
        prompt_mode=batch["prompt_mode"],
    )
    scorer = sentence_scorer or SentenceMetricScorer()

    try:
        for index, annotation_run in enumerate(annotation_runs, start=1):
            image_id = canonical_image_id(annotation_run["image_id"])
            print(
                f"Evaluating batch image {index}/{batch['total_images']}: {image_id}"
            )
            try:
                payload = json.loads(annotation_run["annotation_result"])
                predictions = annotations_in(payload)
                truths = ground_truth_for_image(ground_truth_payload, image_id)
                result = evaluate_image(
                    predictions,
                    truths,
                    image_id=image_id,
                    iou_threshold=iou_threshold,
                    iou_weight=iou_weight,
                    class_weight=class_weight,
                    l2_weight=l2_weight,
                )
                apply_sentence_evaluation(
                    result, predictions, truths, scorer=scorer
                )
                save_coco_evaluation(
                    evaluation_database_path,
                    source_batch_id=batch_id,
                    image_id=image_id,
                    prompt_mode=annotation_run["prompt_mode"] or batch["prompt_mode"],
                    predictions=predictions,
                    ground_truths=truths,
                    result=result,
                    evaluation_run_id=evaluation_run_id,
                )
            except Exception as exc:
                print(f"Failed to evaluate image {image_id}: {exc}", file=sys.stderr)

        metrics = calculate_coco_batch_metrics(
            evaluation_database_path,
            evaluation_run_id=evaluation_run_id,
            batch_id=batch_id,
            user_database_path=user_database_path,
        )
        set_evaluation_run_status(
            evaluation_database_path,
            evaluation_run_id=evaluation_run_id,
            status="completed",
        )
        return metrics
    except Exception:
        set_evaluation_run_status(
            evaluation_database_path,
            evaluation_run_id=evaluation_run_id,
            status="failed",
        )
        raise


def print_result(result):
    count = result["counts"]
    metrics = result["metrics"]
    print(f"\nCOCO single-image evaluation: {result['image_id']}")
    print(f"Predictions: {count['predictions']} | Ground truths: {count['ground_truths']}")
    print(f"TP: {count['tp']} | FP: {count['fp']} | FN: {count['fn']}")
    print(
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
    )
    print(f"Classification errors (assigned pair with passing IoU): {count['classification_errors']}")
    print("\nHungarian-selected pairs:")
    if not result["assigned_pairs"]: print("  (none)")
    for pair in result["assigned_pairs"]:
        rejection = f" ({pair['rejection_reason']})" if pair["rejection_reason"] else ""
        print(f"  P{pair['prediction_index']} {pair['predicted_class']} -> GT{pair['ground_truth_index']} "
              f"{pair['ground_truth_class']} | cost={pair['matching_cost']:.4f} IoU={pair['iou']:.4f} "
              f"L2={pair['l2_distance']:.4f} | {pair['status']}{rejection}")
    if result["unassigned_prediction_indices"]: print(f"Unassigned predictions (FP): {result['unassigned_prediction_indices']}")
    if result["unassigned_ground_truth_indices"]: print(f"Unassigned ground truths (FN): {result['unassigned_ground_truth_indices']}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--prediction", type=Path, help="JSON from data/predictions (recommended)")
    source.add_argument("--run-id", type=int, help="annotation run row ID in either supported SQLite database")
    source.add_argument(
        "--database-image-id",
        help="Load the newest SQLite annotation run for this image ID",
    )
    source.add_argument(
        "--batch-id",
        help="Calculate and store aggregate COCO metrics for this batch ID",
    )
    result.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    result.add_argument(
        "--evaluation-database", type=Path, default=DEFAULT_EVALUATION_DATABASE,
        help="SQLite destination for evaluation results",
    )
    result.add_argument(
        "--source-batch-id",
        help="batch_runs.batch_id when evaluating a prediction file",
    )
    result.add_argument(
        "--prompt-mode",
        help="Prompt mode override; otherwise inferred from the source run or prediction JSON",
    )
    result.add_argument("--no-save", action="store_true", help="Evaluate without writing to SQLite")
    result.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    result.add_argument("--image-id", help="COCO ground-truth image ID; overrides prediction image ID")
    result.add_argument("--iou-threshold", type=float, default=0.5)
    result.add_argument("--iou-weight", type=float, default=2.0)
    result.add_argument("--class-weight", type=float, default=1.0)
    result.add_argument("--l2-weight", type=float, default=1.0)
    result.add_argument("--list-predictions", action="store_true")
    result.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    return result


def main() -> int:
    arguments, cli = parser(), None
    args = arguments.parse_args()
    try:
        if args.list_predictions:
            files = sorted(args.predictions_dir.glob("*.json"))
            print("\n".join(str(path) for path in files) if files else "No prediction JSON files found.")
            return 0
        if args.batch_id is not None:
            batch_metrics = evaluate_batch(
                batch_id=args.batch_id,
                user_database_path=args.database,
                evaluation_database_path=args.evaluation_database,
                ground_truth_path=args.ground_truth,
                iou_threshold=args.iou_threshold,
                iou_weight=args.iou_weight,
                class_weight=args.class_weight,
                l2_weight=args.l2_weight,
            )
            print(json.dumps(batch_metrics, indent=2))
            return 0
        if args.prediction is None and args.run_id is None and args.database_image_id is None:
            arguments.error(
                "choose --prediction FILE, --run-id ID, --database-image-id ID, "
                "or --batch-id ID "
                "(or use --list-predictions)"
            )
        if not 0 <= args.iou_threshold <= 1: raise ValueError("IoU threshold must be in 0..1")
        if min(args.iou_weight, args.class_weight, args.l2_weight) < 0: raise ValueError("Cost weights cannot be negative")
        selected_run_id = None
        selected_batch_id = args.source_batch_id
        source_prompt_mode = None
        if args.prediction is not None:
            predictions, inferred_id = load_prediction_file(args.prediction)
        elif args.run_id is not None:
            predictions, inferred_id = load_database_run(args.database, args.run_id)
            selected_run_id = args.run_id
            source_prompt_mode = load_database_prompt_mode(args.database, selected_run_id)
            if selected_batch_id is None:
                selected_batch_id = load_database_batch_id(args.database, selected_run_id)
        else:
            predictions, inferred_id, selected_run_id = load_latest_database_run_for_image(
                args.database, args.database_image_id
            )
            source_prompt_mode = load_database_prompt_mode(args.database, selected_run_id)
            if selected_batch_id is None:
                selected_batch_id = load_database_batch_id(args.database, selected_run_id)
            print(f"Using newest annotation run for image {inferred_id}: run ID {selected_run_id}")
        image_id = canonical_image_id(args.image_id) if args.image_id is not None else inferred_id
        if image_id is None: raise ValueError("Cannot infer image ID; pass --image-id")
        image_id = canonical_image_id(image_id)
        truths = ground_truth_for_image(load_json(args.ground_truth), image_id)
        cli = evaluate_image(predictions, truths, image_id=image_id, iou_threshold=args.iou_threshold,
                             iou_weight=args.iou_weight, class_weight=args.class_weight, l2_weight=args.l2_weight)
        apply_sentence_evaluation(cli, predictions, truths)
        print_result(cli)
        if not args.no_save:
            evaluation_run_id = save_coco_evaluation(
                args.evaluation_database,
                source_batch_id=selected_batch_id,
                image_id=image_id,
                prompt_mode=args.prompt_mode or source_prompt_mode or infer_prompt_mode(predictions),
                predictions=predictions,
                ground_truths=truths,
                result=cli,
            )
            print(
                f"Saved evaluation run {evaluation_run_id} to "
                f"{args.evaluation_database}"
            )
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        arguments.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
