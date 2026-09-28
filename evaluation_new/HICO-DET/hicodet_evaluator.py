"""Persisted HICO-DET batch evaluator.

Ground truth is read directly from the sibling ``hico_gt.py`` file. A source
annotation is one GT interaction whose predicate list contains alternatives.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import runpy
import sqlite3
import sys
from json import JSONDecodeError
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.evaluation_database import (
    calculate_hicodet_batch_metrics,
    create_hicodet_batch_evaluation_run,
    create_hicodet_image_evaluation_run,
    reuse_existing_image_evaluation,
    save_hicodet_evaluation,
    save_hicodet_failure,
    set_evaluation_run_status,
)
from app.evaluation.hicodet_registry import normalize_hico_object_label

DEFAULT_GROUND_TRUTH = Path(__file__).with_name("hico_gt.py")
DEFAULT_USER_DATABASE = ROOT / "data" / "user_results.db"
DEFAULT_EVALUATION_DATABASE = ROOT / "data" / "evaluation_result.db"


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _key(*values: Any) -> tuple[str, ...]:
    """Canonicalize labels so `hot dog`, `hot_dog`, and `hot-dog` agree."""
    return tuple(re.sub(r"[ _-]+", " ", _text(value).casefold()).strip() for value in values)


def _box(value: Any, *, xywh: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Bounding boxes must contain four normalized numbers.")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) and 0 <= item <= 1 for item in values):
        raise ValueError(f"Bounding box values must be finite and in 0..1: {value}")
    x1, y1, x2, y2 = values
    if xywh:
        x2 += x1
        y2 += y1
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bounding box must have positive area: {value}")
    return [x1, y1, x2, y2]


def _sentences(value: Any) -> list[str]:
    return [_text(item) for item in value] if isinstance(value, list) else []


def _image_id(entry: dict[str, Any]) -> str:
    filename = _text(entry.get("file_name"))
    if not filename:
        raise ValueError("Ground-truth image has no id or file_name.")
    return Path(filename).stem


def load_ground_truth(path: Path = DEFAULT_GROUND_TRUTH) -> dict[str, list[dict[str, Any]]]:
    """Load Python HICO GT; predicate lists remain one annotation's alternatives."""
    if not path.is_file():
        raise ValueError(f"HICO-DET ground-truth file does not exist: {path}")
    namespace = runpy.run_path(str(path))
    payload = next((value for name, value in namespace.items()
                    if name.startswith("HICO_SENTENCES_") and isinstance(value, dict)), None)
    if payload is None or not isinstance(payload.get("images"), list):
        raise ValueError(f"{path} must define HICO_SENTENCES_* with an images list.")
    output: dict[str, list[dict[str, Any]]] = {}
    for image in payload["images"]:
        if not isinstance(image, dict):
            continue
        image_id = _image_id(image)
        numeric_id = _text(image.get("id"))
        records: list[dict[str, Any]] = []
        annotations = image.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"Ground-truth annotations for {image_id} must be a list.")
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                continue
            subject = _text(annotation.get("subject"))
            object_name = normalize_hico_object_label(_text(annotation.get("object")))
            predicates = annotation.get("predicate")
            predicates = predicates if isinstance(predicates, list) else [predicates]
            subject_box = _box(annotation.get("subject_bbox"), xywh=True)
            object_box = _box(annotation.get("object_bbox"), xywh=True)
            predicates = [_text(predicate) for predicate in predicates if _text(predicate)]
            if not (subject and object_name and predicates):
                continue
            records.append({
                "annotation_id": f"{image_id}:{annotation_index}",
                "subject": subject,
                # This string is kept for the database audit trail. Matching
                # uses ``predicates`` so any listed verb is accepted.
                "predicate": " | ".join(predicates),
                "predicates": predicates,
                "object": object_name,
                "subject_bbox": subject_box, "object_bbox": object_box,
                "sentences": _sentences(annotation.get("sentences")),
            })
        output[image_id] = records
        # Uploads normally use the filename stem; some older records use HICO's
        # numeric image id.  Both names refer to this one GT image.
        if numeric_id:
            output[numeric_id] = records
    return output


def load_batch_annotation_runs(path: Path, batch_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"User results database does not exist: {path}")
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        batch = connection.execute("SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)).fetchone()
        if batch is None:
            raise ValueError(f"Batch run not found: {batch_id}")
        rows = connection.execute("""
            SELECT run.id, run.image_id, run.prompt_mode, run.annotation_result
            FROM user_annotation_runs AS run
            INNER JOIN (
              SELECT image_id, MAX(id) AS id FROM user_annotation_runs
              WHERE batch_id = ? AND task_status = 'completed' GROUP BY image_id
            ) AS newest ON newest.id = run.id
            ORDER BY run.id
            """, (batch_id,)).fetchall()
    return dict(batch), [dict(row) for row in rows]


def load_latest_annotation_run(path: Path, image_id: str, batch_id: str | None = None) -> dict[str, Any]:
    """Load the newest completed source annotation for one image."""
    if not path.is_file():
        raise ValueError(f"User results database does not exist: {path}")
    query = """
        SELECT id, image_id, batch_id, prompt_mode, annotation_result
        FROM user_annotation_runs
        WHERE image_id = ? AND task_status = 'completed'
    """
    parameters: list[Any] = [image_id]
    if batch_id:
        query += " AND batch_id = ?"
        parameters.append(batch_id)
    query += " ORDER BY id DESC LIMIT 1"
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(query, parameters).fetchone()
    if row is None:
        scope = f" in batch {batch_id}" if batch_id else ""
        raise ValueError(f"No completed annotation run found for image {image_id}{scope}.")
    return dict(row)


def predictions_in(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for name in ("predictions", "annotations", "bbox_export"):
        if isinstance(payload.get(name), list):
            return payload[name]
    raise ValueError("Annotation payload needs a predictions, annotations, or bbox_export list.")


def normalize_predictions(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction {index} is not an object.")
        subject = _text(item.get("subject"))
        predicate = _text(item.get("predicate"))
        object_name = normalize_hico_object_label(_text(item.get("object")))
        if not (subject and predicate and object_name):
            raise ValueError(f"Prediction {index} needs non-empty subject, predicate, and object.")
        records.append({
            "annotation_id": _text(item.get("annotation_id")) or f"prediction:{index}",
            "subject": subject, "predicate": predicate, "object": object_name,
            "subject_bbox": _box(item.get("subject_bbox")), "object_bbox": _box(item.get("object_bbox")),
            "sentences": _sentences(item.get("sentences")),
        })
    return records


def iou(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = ((first[2] - first[0]) * (first[3] - first[1]) +
             (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / union if union else 0.0


def _pair_base(prediction: dict[str, Any] | None, truth: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "prediction_annotation_id": prediction.get("annotation_id") if prediction else None,
        "ground_truth_annotation_id": truth.get("annotation_id") if truth else None,
        "predicted_subject": prediction.get("subject") if prediction else None,
        "ground_truth_subject": truth.get("subject") if truth else None,
        "predicted_predicate": prediction.get("predicate") if prediction else None,
        "ground_truth_predicate": truth.get("predicate") if truth else None,
        "predicted_object": prediction.get("object") if prediction else None,
        "ground_truth_object": truth.get("object") if truth else None,
        "predicted_subject_bbox": prediction.get("subject_bbox") if prediction else None,
        "ground_truth_subject_bbox": truth.get("subject_bbox") if truth else None,
        "predicted_object_bbox": prediction.get("object_bbox") if prediction else None,
        "ground_truth_object_bbox": truth.get("object_bbox") if truth else None,
        "predicted_sentences": prediction.get("sentences", []) if prediction else [],
        "ground_truth_sentences": truth.get("sentences", []) if truth else [],
    }


def _matching_triplet(prediction: dict[str, Any], truth: dict[str, Any]) -> bool:
    """Return whether labels match, allowing any predicate in one GT list."""
    allowed_predicates = truth.get("predicates")
    if not isinstance(allowed_predicates, list):
        allowed_predicates = [truth.get("predicate")]
    return (
        _key(prediction["subject"]) == _key(truth["subject"])
        and _key(normalize_hico_object_label(prediction["object"])) == _key(normalize_hico_object_label(truth["object"]))
        and _key(prediction["predicate"])[0]
        in {_key(predicate)[0] for predicate in allowed_predicates if _text(predicate)}
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _sentence_scores(candidates: list[str], references: list[str]) -> tuple[float, float] | None:
    # The same smoothed BLEU-4 and unigram-alignment METEOR approximation used by COCO.
    if not candidates or not references:
        return None
    from evaluation_new.COCO.sentence_metrics import SentenceMetricScorer
    scorer = SentenceMetricScorer()
    values = [scorer.score_text(candidate, references) for candidate in candidates]
    return (sum(value[0] for value in values) / len(values), sum(value[1] for value in values) / len(values))


def evaluate_image(predictions: list[dict[str, Any]], truths: list[dict[str, Any]], *, iou_threshold: float = 0.5) -> dict[str, Any]:
    """Greedily match returned predictions to unused GT of the same triplet.

    Predictions are handled in their original LLM order. Candidate
    correspondence is the GT with largest min(subject IoU, object IoU).
    A TP requires both individual IoUs to meet the threshold.
    """
    used_truths: set[int] = set()
    pairs: list[dict[str, Any]] = []
    tp = 0
    # Preserve the LLM's returned annotation order.  Earlier predictions get
    # first claim on a GT interaction; score is not required in LLM output.
    for prediction_index, prediction in enumerate(predictions):
        candidates = [index for index, truth in enumerate(truths)
                      if index not in used_truths and _matching_triplet(prediction, truth)]
        best_index = None
        best_subject_iou = best_object_iou = 0.0
        best_score = -1.0
        for truth_index in candidates:
            truth = truths[truth_index]
            subject_iou, object_iou = iou(prediction["subject_bbox"], truth["subject_bbox"]), iou(prediction["object_bbox"], truth["object_bbox"])
            score = min(subject_iou, object_iou)
            if score > best_score:
                best_index, best_subject_iou, best_object_iou, best_score = truth_index, subject_iou, object_iou, score
        truth = truths[best_index] if best_index is not None else None
        pair = _pair_base(prediction, truth)
        pair.update({"prediction_index": prediction_index, "ground_truth_index": best_index,
                     "subject_iou": best_subject_iou if truth else None, "object_iou": best_object_iou if truth else None,
                     "match_score": best_score if truth else None,
                     "subject_class_correct": 1 if truth else None, "object_class_correct": 1 if truth else None,
                     "predicate_correct": 1 if truth else None, "class_correct": 1 if truth else None,
                     "sentence_eligible": 0, "sentence_skip_reason": None, "bleu": None, "meteor": None})
        if truth is None:
            pair.update(status="FP", rejection_reason="no_unused_ground_truth_with_same_triplet", sentence_skip_reason="unmatched_prediction")
        elif best_subject_iou >= iou_threshold and best_object_iou >= iou_threshold:
            used_truths.add(best_index)
            tp += 1
            pair.update(status="TP", rejection_reason=None)
        else:
            # Retain the prediction/GT correspondence even when it fails the
            # threshold. It is an FP, while recall uses total GT directly.
            used_truths.add(best_index)
            pair.update(status="FP", rejection_reason="subject_or_object_iou_below_threshold", sentence_skip_reason="not_a_true_positive")
        # Semantic text is valid whenever the subject, object, and predicate
        # matched a GT interaction, even if one or both boxes miss the IoU
        # threshold.  A label mismatch has no candidate truth and stays skipped.
        if truth is not None:
            scores = _sentence_scores(pair["predicted_sentences"], pair["ground_truth_sentences"])
            if scores is None:
                pair["sentence_skip_reason"] = "missing_prediction_or_ground_truth_sentences"
            else:
                pair.update(sentence_eligible=1, sentence_skip_reason=None, bleu=scores[0], meteor=scores[1])
        pairs.append(pair)
    unmatched_truths = [index for index in range(len(truths)) if index not in used_truths]
    for truth_index in unmatched_truths:
        truth = truths[truth_index]
        pair = _pair_base(None, truth)
        pair.update(prediction_index=None, ground_truth_index=truth_index, subject_iou=None, object_iou=None,
                    match_score=None, subject_class_correct=None, object_class_correct=None, predicate_correct=None,
                    class_correct=None, status="FP", rejection_reason="unmatched_ground_truth", sentence_eligible=0,
                    sentence_skip_reason="unmatched_ground_truth", bleu=None, meteor=None)
        pairs.append(pair)
    fp = len(predictions) - tp + len(unmatched_truths)
    semantic = [pair for pair in pairs if pair["sentence_eligible"]]
    matching_triplet_pairs = [
        pair for pair in pairs
        if pair["subject_iou"] is not None and pair["object_iou"] is not None
    ]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(truths) if truths else 0.0
    class_values = [pair["class_correct"] for pair in pairs if pair["class_correct"] is not None]
    return {"iou_threshold": iou_threshold, "pairs": pairs,
            "counts": {"predictions": len(predictions), "ground_truths": len(truths), "tp": tp, "fp": fp,
                       "classification_errors": len(class_values) - sum(class_values)},
            "metrics": {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                        "class_accuracy": sum(class_values) / len(class_values) if class_values else 0.0,
                        # A GT image has one annotated interaction. Keep the
                        # selected same-triplet pair's IoUs even below threshold.
                        "subject_iou": matching_triplet_pairs[0]["subject_iou"] if matching_triplet_pairs else None,
                        "object_iou": matching_triplet_pairs[0]["object_iou"] if matching_triplet_pairs else None},
            "sentence_summary": {"eligible_pairs": len(semantic), "evaluated_pairs": len(semantic),
                                  "mean_bleu": sum(pair["bleu"] for pair in semantic) / len(semantic) if semantic else None,
                                  "mean_meteor": sum(pair["meteor"] for pair in semantic) / len(semantic) if semantic else None}}


def evaluate_batch(*, batch_id: str, user_database_path: Path = DEFAULT_USER_DATABASE,
                   evaluation_database_path: Path = DEFAULT_EVALUATION_DATABASE,
                   ground_truth_path: Path = DEFAULT_GROUND_TRUTH, iou_threshold: float = 0.5) -> dict[str, Any]:
    batch, runs = load_batch_annotation_runs(user_database_path, batch_id)
    ground_truth = load_ground_truth(ground_truth_path)
    run_id = create_hicodet_batch_evaluation_run(evaluation_database_path, batch_id=batch_id, prompt_mode=batch.get("prompt_mode") or "unknown")
    try:
        for run in runs:
            image_id = _text(run["image_id"])
            try:
                if reuse_existing_image_evaluation(
                    evaluation_database_path,
                    evaluation_run_id=run_id,
                    batch_id=batch_id,
                    image_id=image_id,
                    dataset="HICO-DET",
                ):
                    print(f"Reused existing image metrics: {image_id}")
                    continue
                if image_id not in ground_truth:
                    raise ValueError(f"No HICO-DET ground truth found for image {image_id}.")
                try:
                    payload = json.loads(run["annotation_result"])
                except JSONDecodeError as exc:
                    raise HicodetEvaluationError("invalid_json", str(exc)) from exc
                try:
                    raw_predictions = predictions_in(payload)
                except ValueError as exc:
                    raise HicodetEvaluationError("missing_predictions", str(exc)) from exc
                try:
                    predictions = normalize_predictions(raw_predictions)
                except ValueError as exc:
                    raise HicodetEvaluationError("invalid_prediction", str(exc)) from exc
                try:
                    result = evaluate_image(predictions, ground_truth[image_id], iou_threshold=iou_threshold)
                except Exception as exc:
                    raise HicodetEvaluationError("evaluation", str(exc)) from exc
                save_hicodet_evaluation(evaluation_database_path, evaluation_run_id=run_id, batch_id=batch_id, image_id=image_id, result=result)
            except Exception as exc:
                # A failed source annotation deliberately has no hicodet_results row.
                # The aggregate therefore counts it as failed and does not average it.
                failure_stage = _failure_stage(exc)
                try:
                    save_hicodet_failure(
                        evaluation_database_path,
                        evaluation_run_id=run_id,
                        batch_id=batch_id,
                        image_id=image_id,
                        failure_stage=failure_stage,
                        failure_reason=str(exc),
                    )
                except Exception as failure_exc:
                    print(f"Failed to record HICO-DET failure for image {image_id}: {failure_exc}", file=sys.stderr)
                print(f"Failed to evaluate image {image_id}: {exc}", file=sys.stderr)
        metrics = calculate_hicodet_batch_metrics(evaluation_database_path, evaluation_run_id=run_id, batch_id=batch_id, user_database_path=user_database_path)
        set_evaluation_run_status(evaluation_database_path, evaluation_run_id=run_id, status="completed")
        return metrics
    except Exception:
        set_evaluation_run_status(evaluation_database_path, evaluation_run_id=run_id, status="failed")
        raise


class HicodetEvaluationError(ValueError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage


def _failure_stage(exc: Exception) -> str:
    if isinstance(exc, HicodetEvaluationError):
        return exc.stage
    message = str(exc)
    if "No HICO-DET ground truth found" in message:
        return "missing_ground_truth"
    if "SQLite" in message or isinstance(exc, sqlite3.Error):
        return "database"
    return "unknown"


def evaluate_single_image(*, image_id: str, batch_id: str | None = None,
                          user_database_path: Path = DEFAULT_USER_DATABASE,
                          evaluation_database_path: Path = DEFAULT_EVALUATION_DATABASE,
                          ground_truth_path: Path = DEFAULT_GROUND_TRUTH,
                          iou_threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate the latest completed annotation for one uploaded image."""
    source = load_latest_annotation_run(user_database_path, image_id, batch_id)
    source_batch_id = _text(source.get("batch_id")) or None
    ground_truth = load_ground_truth(ground_truth_path)
    if image_id not in ground_truth:
        raise ValueError(f"No HICO-DET ground truth found for image {image_id}.")
    run_id = create_hicodet_image_evaluation_run(
        evaluation_database_path, batch_id=source_batch_id, image_id=image_id,
        prompt_mode=_text(source.get("prompt_mode")) or "unknown",
    )
    try:
        payload = json.loads(source["annotation_result"])
        result = evaluate_image(normalize_predictions(predictions_in(payload)), ground_truth[image_id], iou_threshold=iou_threshold)
        save_hicodet_evaluation(evaluation_database_path, evaluation_run_id=run_id,
                                batch_id=source_batch_id, image_id=image_id, result=result)
        set_evaluation_run_status(evaluation_database_path, evaluation_run_id=run_id, status="completed")
        return {"evaluation_run_id": run_id, "batch_id": source_batch_id, "image_id": image_id, **result}
    except Exception:
        set_evaluation_run_status(evaluation_database_path, evaluation_run_id=run_id, status="failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch-id")
    source.add_argument("--image-id")
    parser.add_argument("--source-batch-id", help="Restrict --image-id to this uploaded batch.")
    parser.add_argument("--database", type=Path, default=DEFAULT_USER_DATABASE)
    parser.add_argument("--evaluation-database", type=Path, default=DEFAULT_EVALUATION_DATABASE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in 0..1")
    if args.image_id:
        outcome = evaluate_single_image(image_id=args.image_id, batch_id=args.source_batch_id,
                                        user_database_path=args.database, evaluation_database_path=args.evaluation_database,
                                        ground_truth_path=args.ground_truth, iou_threshold=args.iou_threshold)
    else:
        outcome = evaluate_batch(batch_id=args.batch_id, user_database_path=args.database,
                                 evaluation_database_path=args.evaluation_database,
                                 ground_truth_path=args.ground_truth, iou_threshold=args.iou_threshold)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
