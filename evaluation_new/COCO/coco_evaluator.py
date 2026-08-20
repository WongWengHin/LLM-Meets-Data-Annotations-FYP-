"""Single-image COCO evaluation using weighted-cost Hungarian matching."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
DEFAULT_GT = Path(__file__).parent / "ground_truth" / "official_coco_gt.json"
DEFAULT_PREDICTIONS = ROOT / "data" / "predictions"
DEFAULT_DATABASE = ROOT / "data" / "annotations.db"


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


def canonical_image_id(image_id: Any) -> str:
    value = str(image_id).strip()
    return str(int(value)) if value.isdecimal() else value


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
        tp += int(accepted); errors += int(overlap >= iou_threshold and not same_class)
        pairs.append({
            "prediction_index": p, "ground_truth_index": g,
            "prediction_annotation_id": predictions[p].get("annotation_id"),
            "ground_truth_annotation_id": truths[g].get("annotation_id", truths[g].get("id")),
            "predicted_class": predictions[p]["class"], "ground_truth_class": truths[g]["class"],
            "iou": overlap, "l2_distance": distances[p][g], "matching_cost": costs[p][g],
            "class_correct": same_class, "result": "TP" if accepted else "rejected",
        })
    used_p, used_g = {p for p, _ in assignments}, {g for _, g in assignments}
    return {
        "image_id": image_id,
        "config": {"iou_threshold": iou_threshold, "cost_weights": {"iou": iou_weight, "class": class_weight, "l2": l2_weight}},
        "counts": {"predictions": len(predictions), "ground_truths": len(truths), "tp": tp,
                   "fp": len(predictions) - tp, "fn": len(truths) - tp, "classification_errors": errors},
        "cost_matrix": costs, "assigned_pairs": pairs,
        "unassigned_prediction_indices": [i for i in range(len(predictions)) if i not in used_p],
        "unassigned_ground_truth_indices": [i for i in range(len(truths)) if i not in used_g],
    }


def print_result(result):
    count = result["counts"]
    print(f"\nCOCO single-image evaluation: {result['image_id']}")
    print(f"Predictions: {count['predictions']} | Ground truths: {count['ground_truths']}")
    print(f"TP: {count['tp']} | FP: {count['fp']} | FN: {count['fn']}")
    print(f"Classification errors (assigned pair with passing IoU): {count['classification_errors']}")
    print("\nHungarian-selected pairs:")
    if not result["assigned_pairs"]: print("  (none)")
    for pair in result["assigned_pairs"]:
        print(f"  P{pair['prediction_index']} {pair['predicted_class']} -> GT{pair['ground_truth_index']} "
              f"{pair['ground_truth_class']} | cost={pair['matching_cost']:.4f} IoU={pair['iou']:.4f} "
              f"L2={pair['l2_distance']:.4f} | {pair['result']}")
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
    result.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
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
        if args.prediction is None and args.run_id is None and args.database_image_id is None:
            arguments.error(
                "choose --prediction FILE, --run-id ID, or --database-image-id ID "
                "(or use --list-predictions)"
            )
        if not 0 <= args.iou_threshold <= 1: raise ValueError("IoU threshold must be in 0..1")
        if min(args.iou_weight, args.class_weight, args.l2_weight) < 0: raise ValueError("Cost weights cannot be negative")
        selected_run_id = None
        if args.prediction is not None:
            predictions, inferred_id = load_prediction_file(args.prediction)
        elif args.run_id is not None:
            predictions, inferred_id = load_database_run(args.database, args.run_id)
            selected_run_id = args.run_id
        else:
            predictions, inferred_id, selected_run_id = load_latest_database_run_for_image(
                args.database, args.database_image_id
            )
            print(f"Using newest annotation run for image {inferred_id}: run ID {selected_run_id}")
        image_id = str(args.image_id) if args.image_id is not None else inferred_id
        if image_id is None: raise ValueError("Cannot infer image ID; pass --image-id")
        truths = ground_truth_for_image(load_json(args.ground_truth), image_id)
        cli = evaluate_image(predictions, truths, image_id=image_id, iou_threshold=args.iou_threshold,
                             iou_weight=args.iou_weight, class_weight=args.class_weight, l2_weight=args.l2_weight)
        print_result(cli)
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        arguments.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
