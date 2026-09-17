"""SQLite persistence for dataset evaluation results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connection(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_evaluation_database(database_path: Path) -> None:
    """Create the evaluation database and its COCO tables when absent."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connection(database_path)) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = OFF")
        _migrate_legacy_schema(connection)
        _create_evaluation_tables(connection)
        _add_missing_columns(connection, "coco_results", {
            "precision": "REAL NOT NULL DEFAULT 0",
            "recall": "REAL NOT NULL DEFAULT 0",
            "f1": "REAL NOT NULL DEFAULT 0",
            "sentence_eligible_pairs": "INTEGER NOT NULL DEFAULT 0",
            "sentence_evaluated_pairs": "INTEGER NOT NULL DEFAULT 0",
            "mean_bleu": "REAL",
            "mean_meteor": "REAL",
        })
        _add_missing_columns(connection, "coco_pairs", {
            "sentence_eligible": "INTEGER",
            "sentence_skip_reason": "TEXT",
            "predicted_sentences": "TEXT",
            "ground_truth_sentences": "TEXT",
            "bleu": "REAL",
            "meteor": "REAL",
        })
        connection.execute(
            """
            UPDATE coco_results
            SET precision = CASE
                    WHEN tp + fp > 0 THEN CAST(tp AS REAL) / (tp + fp)
                    ELSE 0.0
                END,
                recall = CASE
                    WHEN tp + fn > 0 THEN CAST(tp AS REAL) / (tp + fn)
                    ELSE 0.0
                END,
                f1 = CASE
                    WHEN 2 * tp + fp + fn > 0
                    THEN CAST(2 * tp AS REAL) / (2 * tp + fp + fn)
                    ELSE 0.0
                END
            """
        )
        connection.execute(
            """
            UPDATE coco_batch_metrics
            SET mean_precision = (
                    SELECT AVG(result.precision)
                    FROM coco_results AS result
                    WHERE result.evaluation_run_id = coco_batch_metrics.evaluation_run_id
                      AND result.batch_id = coco_batch_metrics.batch_id
                ),
                mean_recall = (
                    SELECT AVG(result.recall)
                    FROM coco_results AS result
                    WHERE result.evaluation_run_id = coco_batch_metrics.evaluation_run_id
                      AND result.batch_id = coco_batch_metrics.batch_id
                ),
                mean_f1 = (
                    SELECT AVG(result.f1)
                    FROM coco_results AS result
                    WHERE result.evaluation_run_id = coco_batch_metrics.evaluation_run_id
                      AND result.batch_id = coco_batch_metrics.batch_id
                )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluation_runs_image_id "
            "ON evaluation_runs(image_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluation_runs_source_batch_id "
            "ON evaluation_runs(source_batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_coco_results_batch_id "
            "ON coco_results(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_hicodet_results_batch_id "
            "ON hicodet_results(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_coco_pairs_evaluation_run_id "
            "ON coco_pairs(evaluation_run_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_coco_pairs_batch_id "
            "ON coco_pairs(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_hicodet_pairs_evaluation_run_id "
            "ON hicodet_pairs(evaluation_run_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_hicodet_pairs_batch_id "
            "ON hicodet_pairs(batch_id)"
        )
        connection.execute(
            "DROP INDEX IF EXISTS idx_coco_batch_metrics_batch_id"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_coco_batch_metrics_batch_id "
            "ON coco_batch_metrics(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_hicodet_batch_metrics_batch_id "
            "ON hicodet_batch_metrics(batch_id)"
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")


def _create_evaluation_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            evaluation_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_batch_id TEXT,
            image_id TEXT,
            dataset TEXT NOT NULL,
            task_type TEXT NOT NULL,
            prompt_mode TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            evaluated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS coco_results (
            evaluation_run_id INTEGER NOT NULL,
            batch_id TEXT,
            image_id TEXT NOT NULL,
            num_predictions INTEGER NOT NULL,
            num_ground_truths INTEGER NOT NULL,
            tp INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            precision REAL NOT NULL DEFAULT 0,
            recall REAL NOT NULL DEFAULT 0,
            f1 REAL NOT NULL DEFAULT 0,
            classification_errors INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            class_accuracy REAL NOT NULL,
            iou_threshold REAL NOT NULL,
            iou_weight REAL NOT NULL,
            class_weight REAL NOT NULL,
            l2_weight REAL NOT NULL,
            sentence_eligible_pairs INTEGER NOT NULL DEFAULT 0,
            sentence_evaluated_pairs INTEGER NOT NULL DEFAULT 0,
            mean_bleu REAL,
            mean_meteor REAL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE,
            PRIMARY KEY (evaluation_run_id, image_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS coco_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_run_id INTEGER NOT NULL,
            batch_id TEXT,
            image_id TEXT NOT NULL,
            prediction_index INTEGER,
            ground_truth_index INTEGER,
            prediction_annotation_id TEXT,
            ground_truth_annotation_id TEXT,
            predicted_class TEXT,
            ground_truth_class TEXT,
            predicted_bbox TEXT,
            ground_truth_bbox TEXT,
            iou REAL,
            l2_distance REAL,
            matching_cost REAL,
            class_correct INTEGER,
            status TEXT NOT NULL,
            rejection_reason TEXT,
            sentence_eligible INTEGER,
            sentence_skip_reason TEXT,
            predicted_sentences TEXT,
            ground_truth_sentences TEXT,
            bleu REAL,
            meteor REAL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hicodet_results (
            evaluation_run_id INTEGER NOT NULL,
            batch_id TEXT,
            image_id TEXT NOT NULL,
            num_predictions INTEGER NOT NULL,
            num_ground_truths INTEGER NOT NULL,
            tp INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            precision REAL NOT NULL DEFAULT 0,
            recall REAL NOT NULL DEFAULT 0,
            f1 REAL NOT NULL DEFAULT 0,
            classification_errors INTEGER NOT NULL,
            class_accuracy REAL NOT NULL,
            mean_subject_iou REAL NOT NULL,
            mean_object_iou REAL NOT NULL,
            iou_threshold REAL NOT NULL,
            sentence_eligible_pairs INTEGER NOT NULL DEFAULT 0,
            sentence_evaluated_pairs INTEGER NOT NULL DEFAULT 0,
            mean_bleu REAL,
            mean_meteor REAL,
            calculated_at TEXT NOT NULL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE,
            PRIMARY KEY (evaluation_run_id, image_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS coco_batch_metrics (
            evaluation_run_id INTEGER PRIMARY KEY,
            batch_id TEXT NOT NULL,
            total_images INTEGER NOT NULL,
            successful_images INTEGER NOT NULL,
            failed_images INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            total_ground_truths INTEGER NOT NULL,
            total_tp INTEGER NOT NULL,
            total_fp INTEGER NOT NULL,
            total_fn INTEGER NOT NULL,
            mean_precision REAL,
            mean_recall REAL,
            mean_f1 REAL,
            iou_evaluated_pairs INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            class_correct_pairs INTEGER NOT NULL,
            class_evaluated_pairs INTEGER NOT NULL,
            total_classification_errors INTEGER NOT NULL,
            class_accuracy REAL NOT NULL,
            sentence_eligible_pairs INTEGER NOT NULL,
            sentence_evaluated_pairs INTEGER NOT NULL,
            mean_bleu REAL,
            mean_meteor REAL,
            calculated_at TEXT NOT NULL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hicodet_pairs (
            pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_run_id INTEGER NOT NULL,
            batch_id TEXT,
            image_id TEXT NOT NULL,
            prediction_index INTEGER,
            ground_truth_index INTEGER,
            prediction_annotation_id TEXT,
            ground_truth_annotation_id TEXT,
            predicted_subject TEXT,
            ground_truth_subject TEXT,
            predicted_predicate TEXT,
            ground_truth_predicate TEXT,
            predicted_object TEXT,
            ground_truth_object TEXT,
            predicted_subject_bbox TEXT,
            ground_truth_subject_bbox TEXT,
            predicted_object_bbox TEXT,
            ground_truth_object_bbox TEXT,
            subject_iou REAL,
            object_iou REAL,
            match_score REAL,
            subject_class_correct INTEGER,
            object_class_correct INTEGER,
            predicate_correct INTEGER,
            class_correct INTEGER,
            status TEXT NOT NULL,
            rejection_reason TEXT,
            sentence_eligible INTEGER,
            sentence_skip_reason TEXT,
            predicted_sentences TEXT,
            ground_truth_sentences TEXT,
            bleu REAL,
            meteor REAL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hicodet_batch_metrics (
            evaluation_run_id INTEGER PRIMARY KEY,
            batch_id TEXT NOT NULL,
            total_images INTEGER NOT NULL,
            successful_images INTEGER NOT NULL,
            failed_images INTEGER NOT NULL,
            total_predictions INTEGER NOT NULL,
            total_gt INTEGER NOT NULL,
            total_tp INTEGER NOT NULL,
            total_fp INTEGER NOT NULL,
            total_fn INTEGER NOT NULL,
            mean_precision REAL,
            mean_recall REAL,
            mean_f1 REAL,
            iou_evaluated_pairs INTEGER NOT NULL,
            mean_subj_iou REAL NOT NULL,
            mean_obj_iou REAL NOT NULL,
            class_correct_pairs INTEGER NOT NULL,
            class_evaluated_pairs INTEGER NOT NULL,
            total_classification_errors INTEGER NOT NULL,
            class_accuracy REAL NOT NULL,
            sentence_eligible_pairs INTEGER NOT NULL,
            sentence_evaluated_pairs INTEGER NOT NULL,
            mean_bleu REAL,
            mean_meteor REAL,
            calculated_at TEXT NOT NULL,
            FOREIGN KEY (evaluation_run_id)
                REFERENCES evaluation_runs(evaluation_run_id) ON DELETE CASCADE
        )
        """
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def _column_is_not_null(
    connection: sqlite3.Connection, table: str, column: str
) -> bool:
    return any(
        row[1] == column and bool(row[3])
        for row in connection.execute(f"PRAGMA table_info({table})")
    )


def _primary_key_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        row[1]
        for row in sorted(
            connection.execute(f"PRAGMA table_info({table})"),
            key=lambda row: row[5],
        )
        if row[5] > 0
    ]


def _migrate_legacy_schema(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "evaluation_runs"):
        return

    run_columns = _table_columns(connection, "evaluation_runs")
    result_columns = _table_columns(connection, "coco_results")
    pair_columns = _table_columns(connection, "coco_pairs")
    batch_metric_columns = _table_columns(connection, "coco_batch_metrics")
    schema_is_current = (
        "evaluation_run_id" in run_columns
        and "source_batch_id" in run_columns
        and "status" in run_columns
        and "id" not in run_columns
        and "source_run_id" not in run_columns
        and not _column_is_not_null(connection, "evaluation_runs", "image_id")
        and "batch_id" in result_columns
        and "id" not in result_columns
        and _primary_key_columns(connection, "coco_results")
        == ["evaluation_run_id", "image_id"]
        and "pair_id" in pair_columns
        and "batch_id" in pair_columns
        and "id" not in pair_columns
        and {"mean_precision", "mean_recall", "mean_f1"} <= batch_metric_columns
        and not {"micro_precision", "micro_recall", "micro_f1"} & batch_metric_columns
        and not any(
            _column_is_not_null(connection, "coco_batch_metrics", column)
            for column in ("mean_precision", "mean_recall", "mean_f1")
        )
    )
    if schema_is_current:
        return

    legacy_tables = {
        table: f"{table}_migration_legacy"
        for table in (
            "coco_pairs",
            "coco_results",
            "coco_batch_metrics",
            "evaluation_runs",
        )
        if _table_exists(connection, table)
    }
    for legacy_table in legacy_tables.values():
        if _table_exists(connection, legacy_table):
            raise RuntimeError(
                f"Cannot migrate evaluation database while {legacy_table} exists."
            )

    connection.execute("BEGIN IMMEDIATE")
    try:
        for table, legacy_table in legacy_tables.items():
            connection.execute(f"ALTER TABLE {table} RENAME TO {legacy_table}")

        _create_evaluation_tables(connection)
        legacy_runs = legacy_tables["evaluation_runs"]
        legacy_run_columns = _table_columns(connection, legacy_runs)
        run_id_column = (
            "evaluation_run_id"
            if "evaluation_run_id" in legacy_run_columns
            else "id"
        )
        source_batch_expression = (
            "source_batch_id" if "source_batch_id" in legacy_run_columns else "NULL"
        )
        status_expression = "status" if "status" in legacy_run_columns else "'completed'"
        connection.execute(
            f"""
            INSERT INTO evaluation_runs
                (evaluation_run_id, source_batch_id, image_id, dataset,
                 task_type, prompt_mode, status, evaluated_at)
            SELECT {run_id_column}, {source_batch_expression}, image_id, dataset,
                   task_type, prompt_mode, {status_expression}, evaluated_at
            FROM {legacy_runs}
            """
        )

        if "coco_results" in legacy_tables:
            legacy_results = legacy_tables["coco_results"]
            legacy_result_columns = _table_columns(connection, legacy_results)
            result_batch_expression = (
                "batch_id" if "batch_id" in legacy_result_columns else "NULL"
            )
            connection.execute(
                f"""
                INSERT INTO coco_results
                    (evaluation_run_id, batch_id, image_id, num_predictions,
                     num_ground_truths, tp, fp, fn, precision, recall, f1,
                     classification_errors,
                     mean_iou, class_accuracy, iou_threshold, iou_weight,
                     class_weight, l2_weight, sentence_eligible_pairs,
                     sentence_evaluated_pairs, mean_bleu, mean_meteor)
                SELECT evaluation_run_id, {result_batch_expression}, image_id,
                       num_predictions, num_ground_truths, tp, fp, fn,
                       CASE WHEN tp + fp > 0
                            THEN CAST(tp AS REAL) / (tp + fp) ELSE 0.0 END,
                       CASE WHEN tp + fn > 0
                            THEN CAST(tp AS REAL) / (tp + fn) ELSE 0.0 END,
                       CASE WHEN 2 * tp + fp + fn > 0
                            THEN CAST(2 * tp AS REAL) / (2 * tp + fp + fn)
                            ELSE 0.0 END,
                       classification_errors, mean_iou, class_accuracy,
                       iou_threshold, iou_weight, class_weight, l2_weight,
                       sentence_eligible_pairs, sentence_evaluated_pairs,
                       mean_bleu, mean_meteor
                FROM {legacy_results}
                """
            )

        if "coco_pairs" in legacy_tables:
            legacy_pairs = legacy_tables["coco_pairs"]
            legacy_pair_columns = _table_columns(connection, legacy_pairs)
            pair_id_column = "pair_id" if "pair_id" in legacy_pair_columns else "id"
            pair_batch_expression = (
                "batch_id" if "batch_id" in legacy_pair_columns else "NULL"
            )
            connection.execute(
                f"""
                INSERT INTO coco_pairs
                    (pair_id, evaluation_run_id, batch_id, image_id,
                     prediction_index, ground_truth_index,
                     prediction_annotation_id, ground_truth_annotation_id,
                     predicted_class, ground_truth_class, predicted_bbox,
                     ground_truth_bbox, iou, l2_distance, matching_cost,
                     class_correct, status, rejection_reason, sentence_eligible,
                     sentence_skip_reason, predicted_sentences,
                     ground_truth_sentences, bleu, meteor)
                SELECT {pair_id_column}, evaluation_run_id,
                       {pair_batch_expression}, image_id, prediction_index,
                       ground_truth_index, prediction_annotation_id,
                       ground_truth_annotation_id, predicted_class,
                       ground_truth_class, predicted_bbox, ground_truth_bbox,
                       iou, l2_distance, matching_cost, class_correct, status,
                       rejection_reason, sentence_eligible, sentence_skip_reason,
                       predicted_sentences, ground_truth_sentences, bleu, meteor
                FROM {legacy_pairs}
                """
            )

        if "coco_batch_metrics" in legacy_tables:
            legacy_batch_metrics = legacy_tables["coco_batch_metrics"]
            legacy_batch_metric_columns = _table_columns(
                connection, legacy_batch_metrics
            )
            precision_column = (
                "mean_precision"
                if "mean_precision" in legacy_batch_metric_columns
                else "micro_precision"
            )
            recall_column = (
                "mean_recall"
                if "mean_recall" in legacy_batch_metric_columns
                else "micro_recall"
            )
            f1_column = (
                "mean_f1"
                if "mean_f1" in legacy_batch_metric_columns
                else "micro_f1"
            )
            connection.execute(
                f"""
                INSERT INTO coco_batch_metrics
                    (evaluation_run_id, batch_id, total_images,
                     successful_images, failed_images, total_predictions,
                     total_ground_truths, total_tp, total_fp, total_fn,
                     mean_precision, mean_recall, mean_f1,
                     iou_evaluated_pairs, mean_iou, class_correct_pairs,
                     class_evaluated_pairs, total_classification_errors,
                     class_accuracy, sentence_eligible_pairs,
                     sentence_evaluated_pairs, mean_bleu, mean_meteor,
                     calculated_at)
                SELECT evaluation_run_id, batch_id, total_images,
                       successful_images, failed_images, total_predictions,
                       total_ground_truths, total_tp, total_fp, total_fn,
                       {precision_column}, {recall_column}, {f1_column},
                       iou_evaluated_pairs, mean_iou, class_correct_pairs,
                       class_evaluated_pairs, total_classification_errors,
                       class_accuracy, sentence_eligible_pairs,
                       sentence_evaluated_pairs, mean_bleu, mean_meteor,
                       calculated_at
                FROM {legacy_batch_metrics}
                """
            )

        for table in (
            "coco_pairs",
            "coco_results",
            "coco_batch_metrics",
            "evaluation_runs",
        ):
            legacy_table = legacy_tables.get(table)
            if legacy_table is not None:
                connection.execute(f"DROP TABLE {legacy_table}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _add_missing_columns(
    connection: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    existing = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    for name, declaration in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def _json_bbox(bbox: Any) -> str | None:
    return None if bbox is None else json.dumps(bbox, separators=(",", ":"))


def _json_sentences(sentences: Any) -> str:
    return json.dumps(sentences if isinstance(sentences, list) else [], ensure_ascii=False)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def save_coco_evaluation(
    database_path: Path,
    *,
    source_batch_id: str | None,
    image_id: str,
    prompt_mode: str,
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    result: dict[str, Any],
    evaluation_run_id: int | None = None,
) -> int:
    """Store one image result, creating a run unless a batch run ID is supplied."""
    init_evaluation_database(database_path)
    counts = result["counts"]
    config = result["config"]
    assigned_pairs = result["assigned_pairs"]
    sentence_summary = result.get("sentence_summary", {})
    sentence_eligible_pairs = sentence_summary.get("eligible_pairs", 0)
    if sentence_eligible_pairs == 0 and (
        sentence_summary.get("mean_bleu") is not None
        or sentence_summary.get("mean_meteor") is not None
    ):
        sentence_eligible_pairs = len(assigned_pairs)
    mean_iou = (
        sum(float(pair["iou"]) for pair in assigned_pairs) / len(assigned_pairs)
        if assigned_pairs
        else 0.0
    )
    class_accuracy = (
        sum(int(bool(pair["class_correct"])) for pair in assigned_pairs) / len(assigned_pairs)
        if assigned_pairs
        else 0.0
    )
    precision, recall, f1 = _precision_recall_f1(
        counts["tp"], counts["fp"], counts["fn"]
    )

    with closing(_connection(database_path)) as connection:
        if evaluation_run_id is None:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_runs
                    (source_batch_id, image_id, dataset, task_type,
                     prompt_mode, status, evaluated_at)
                VALUES (?, ?, 'COCO', 'object_detection', ?, 'completed', ?)
                """,
                (
                    source_batch_id,
                    image_id,
                    prompt_mode,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            evaluation_run_id = int(cursor.lastrowid)
        else:
            run = connection.execute(
                """
                SELECT source_batch_id, dataset, task_type, status
                FROM evaluation_runs
                WHERE evaluation_run_id = ?
                """,
                (evaluation_run_id,),
            ).fetchone()
            if run is None:
                raise ValueError(f"Evaluation run not found: {evaluation_run_id}")
            if run["dataset"] != "COCO" or run["task_type"] != "object_detection_batch":
                raise ValueError(f"Evaluation run {evaluation_run_id} is not a COCO batch run.")
            if run["status"] != "processing":
                raise ValueError(f"Evaluation run {evaluation_run_id} is not processing.")
            if run["source_batch_id"] != source_batch_id:
                raise ValueError("Evaluation run batch ID does not match the image batch ID.")
        weights = config["cost_weights"]
        connection.execute(
            """
            INSERT INTO coco_results
                (evaluation_run_id, batch_id, image_id,
                 num_predictions, num_ground_truths,
                 tp, fp, fn, precision, recall, f1,
                 classification_errors, mean_iou, class_accuracy,
                 iou_threshold, iou_weight, class_weight, l2_weight,
                 sentence_eligible_pairs, sentence_evaluated_pairs, mean_bleu, mean_meteor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_run_id,
                source_batch_id,
                image_id,
                counts["predictions"],
                counts["ground_truths"],
                counts["tp"],
                counts["fp"],
                counts["fn"],
                precision,
                recall,
                f1,
                counts["classification_errors"],
                mean_iou,
                class_accuracy,
                config["iou_threshold"],
                weights["iou"],
                weights["class"],
                weights["l2"],
                sentence_eligible_pairs,
                sentence_summary.get("evaluated_pairs", 0),
                sentence_summary.get("mean_bleu"),
                sentence_summary.get("mean_meteor"),
            ),
        )

        pair_rows: list[tuple[Any, ...]] = []
        for pair in assigned_pairs:
            pair_rows.append(
                (
                    evaluation_run_id,
                    source_batch_id,
                    image_id,
                    pair["prediction_index"],
                    pair["ground_truth_index"],
                    pair["prediction_annotation_id"],
                    pair["ground_truth_annotation_id"],
                    pair["predicted_class"],
                    pair["ground_truth_class"],
                    _json_bbox(pair["predicted_bbox"]),
                    _json_bbox(pair["ground_truth_bbox"]),
                    pair["iou"],
                    pair["l2_distance"],
                    pair["matching_cost"],
                    int(bool(pair["class_correct"])),
                    pair["status"],
                    pair["rejection_reason"],
                    int(bool(pair.get("sentence_eligible"))) if "sentence_eligible" in pair else None,
                    pair.get("sentence_skip_reason"),
                    _json_sentences(pair.get("predicted_sentences")),
                    _json_sentences(pair.get("ground_truth_sentences")),
                    pair.get("bleu"),
                    pair.get("meteor"),
                )
            )
        for prediction_index in result["unassigned_prediction_indices"]:
            prediction = predictions[prediction_index]
            pair_rows.append(
                (
                    evaluation_run_id,
                    source_batch_id,
                    image_id,
                    prediction_index,
                    None,
                    prediction.get("annotation_id"),
                    None,
                    prediction["class"],
                    None,
                    _json_bbox(prediction["bbox"]),
                    None,
                    None,
                    None,
                    None,
                    None,
                    "FP",
                    "unmatched_prediction",
                    None, None, _json_sentences(prediction.get("sentences")), None, None, None,
                )
            )
        for ground_truth_index in result["unassigned_ground_truth_indices"]:
            truth = ground_truths[ground_truth_index]
            pair_rows.append(
                (
                    evaluation_run_id,
                    source_batch_id,
                    image_id,
                    None,
                    ground_truth_index,
                    None,
                    truth.get("annotation_id", truth.get("id")),
                    None,
                    truth["class"],
                    None,
                    _json_bbox(truth["bbox"]),
                    None,
                    None,
                    None,
                    None,
                    "FN",
                    "unmatched_ground_truth",
                    None, None, None, _json_sentences(truth.get("sentences")), None, None,
                )
            )

        connection.executemany(
            """
            INSERT INTO coco_pairs
                (evaluation_run_id, batch_id, image_id,
                 prediction_index, ground_truth_index,
                 prediction_annotation_id, ground_truth_annotation_id,
                 predicted_class, ground_truth_class, predicted_bbox, ground_truth_bbox,
                  iou, l2_distance, matching_cost, class_correct, status, rejection_reason,
                  sentence_eligible, sentence_skip_reason, predicted_sentences,
                  ground_truth_sentences, bleu, meteor)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pair_rows,
        )
        connection.commit()
    return evaluation_run_id


def create_coco_batch_evaluation_run(
    database_path: Path,
    *,
    batch_id: str,
    prompt_mode: str,
) -> int:
    init_evaluation_database(database_path)
    with closing(_connection(database_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs
                (source_batch_id, image_id, dataset, task_type,
                 prompt_mode, status, evaluated_at)
            VALUES (?, NULL, 'COCO', 'object_detection_batch', ?, 'processing', ?)
            """,
            (batch_id, prompt_mode, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def create_hicodet_batch_evaluation_run(
    database_path: Path,
    *,
    batch_id: str,
    prompt_mode: str,
) -> int:
    """Create the parent run for a HICO-DET batch evaluation."""
    init_evaluation_database(database_path)
    with closing(_connection(database_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs
                (source_batch_id, image_id, dataset, task_type,
                 prompt_mode, status, evaluated_at)
            VALUES (?, NULL, 'HICO-DET', 'hoi_detection_batch', ?, 'processing', ?)
            """,
            (batch_id, prompt_mode, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def create_hicodet_image_evaluation_run(
    database_path: Path, *, batch_id: str | None, image_id: str, prompt_mode: str
) -> int:
    """Create a standalone HICO-DET evaluation run for one image."""
    init_evaluation_database(database_path)
    with closing(_connection(database_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs
                (source_batch_id, image_id, dataset, task_type, prompt_mode, status, evaluated_at)
            VALUES (?, ?, 'HICO-DET', 'hoi_detection', ?, 'processing', ?)
            """,
            (batch_id, image_id, prompt_mode, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def save_hicodet_evaluation(
    database_path: Path,
    *,
    evaluation_run_id: int,
    batch_id: str | None,
    image_id: str,
    result: dict[str, Any],
) -> None:
    """Persist one HICO-DET image result and its complete pair audit trail."""
    init_evaluation_database(database_path)
    counts = result["counts"]
    metrics = result["metrics"]
    sentence = result["sentence_summary"]
    calculated_at = datetime.now(timezone.utc).isoformat()
    with closing(_connection(database_path)) as connection:
        run = connection.execute(
            "SELECT source_batch_id, dataset, task_type, status FROM evaluation_runs "
            "WHERE evaluation_run_id = ?", (evaluation_run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(f"Evaluation run not found: {evaluation_run_id}")
        if (run["source_batch_id"], run["dataset"], run["status"]) != (
            batch_id, "HICO-DET", "processing"
        ) or run["task_type"] not in {"hoi_detection", "hoi_detection_batch"}:
            raise ValueError("Evaluation run is not an active HICO-DET run for this batch ID.")
        connection.execute(
            """
            INSERT INTO hicodet_results
                (evaluation_run_id, batch_id, image_id, num_predictions,
                 num_ground_truths, tp, fp, fn, precision, recall, f1,
                 classification_errors, class_accuracy, mean_subject_iou,
                 mean_object_iou, iou_threshold, sentence_eligible_pairs,
                 sentence_evaluated_pairs, mean_bleu, mean_meteor, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (evaluation_run_id, batch_id, image_id, counts["predictions"],
             counts["ground_truths"], counts["tp"], counts["fp"], counts["fn"],
             metrics["precision"], metrics["recall"], metrics["f1"],
             counts["classification_errors"], metrics["class_accuracy"],
             metrics["mean_subject_iou"], metrics["mean_object_iou"],
             result["iou_threshold"], sentence["eligible_pairs"],
             sentence["evaluated_pairs"], sentence["mean_bleu"],
             sentence["mean_meteor"], calculated_at),
        )
        rows = []
        for pair in result["pairs"]:
            rows.append((
                evaluation_run_id, batch_id, image_id,
                pair.get("prediction_index"), pair.get("ground_truth_index"),
                pair.get("prediction_annotation_id"), pair.get("ground_truth_annotation_id"),
                pair.get("predicted_subject"), pair.get("ground_truth_subject"),
                pair.get("predicted_predicate"), pair.get("ground_truth_predicate"),
                pair.get("predicted_object"), pair.get("ground_truth_object"),
                _json_bbox(pair.get("predicted_subject_bbox")), _json_bbox(pair.get("ground_truth_subject_bbox")),
                _json_bbox(pair.get("predicted_object_bbox")), _json_bbox(pair.get("ground_truth_object_bbox")),
                pair.get("subject_iou"), pair.get("object_iou"), pair.get("match_score"),
                pair.get("subject_class_correct"), pair.get("object_class_correct"),
                pair.get("predicate_correct"), pair.get("class_correct"), pair["status"],
                pair.get("rejection_reason"), pair.get("sentence_eligible"),
                pair.get("sentence_skip_reason"), _json_sentences(pair.get("predicted_sentences")),
                _json_sentences(pair.get("ground_truth_sentences")), pair.get("bleu"), pair.get("meteor"),
            ))
        connection.executemany(
            """
            INSERT INTO hicodet_pairs
                (evaluation_run_id, batch_id, image_id, prediction_index, ground_truth_index,
                 prediction_annotation_id, ground_truth_annotation_id, predicted_subject,
                 ground_truth_subject, predicted_predicate, ground_truth_predicate,
                 predicted_object, ground_truth_object, predicted_subject_bbox,
                 ground_truth_subject_bbox, predicted_object_bbox, ground_truth_object_bbox,
                 subject_iou, object_iou, match_score, subject_class_correct,
                 object_class_correct, predicate_correct, class_correct, status,
                 rejection_reason, sentence_eligible, sentence_skip_reason,
                 predicted_sentences, ground_truth_sentences, bleu, meteor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows,
        )
        connection.commit()


def calculate_hicodet_batch_metrics(
    database_path: Path, *, evaluation_run_id: int, batch_id: str,
    user_database_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate HICO-DET image rows. Only successfully saved images contribute."""
    normalized_batch_id = batch_id.strip()
    if not normalized_batch_id:
        raise ValueError("A batch ID is required.")
    batch = _load_source_batch(user_database_path or database_path.with_name("user_results.db"), normalized_batch_id)
    init_evaluation_database(database_path)
    with closing(_connection(database_path)) as connection:
        run = connection.execute("SELECT source_batch_id, dataset, task_type FROM evaluation_runs WHERE evaluation_run_id = ?", (evaluation_run_id,)).fetchone()
        if run is None or (run["source_batch_id"], run["dataset"], run["task_type"]) != (normalized_batch_id, "HICO-DET", "hoi_detection_batch"):
            raise ValueError("Evaluation run is not a HICO-DET batch for this batch ID.")
        images = connection.execute("SELECT * FROM hicodet_results WHERE evaluation_run_id = ? AND batch_id = ?", (evaluation_run_id, normalized_batch_id)).fetchall()
        pairs = connection.execute("SELECT * FROM hicodet_pairs WHERE evaluation_run_id = ? AND batch_id = ?", (evaluation_run_id, normalized_batch_id)).fetchall()
        successful = len(images)
        subject_ious = [float(row["subject_iou"]) for row in pairs if row["subject_iou"] is not None]
        object_ious = [float(row["object_iou"]) for row in pairs if row["object_iou"] is not None]
        class_values = [int(row["class_correct"]) for row in pairs if row["class_correct"] is not None]
        values = (
            evaluation_run_id, normalized_batch_id, int(batch["total_images"]), successful,
            max(int(batch["total_images"]) - successful, 0),
            sum(row["num_predictions"] for row in images), sum(row["num_ground_truths"] for row in images),
            sum(row["tp"] for row in images), sum(row["fp"] for row in images), sum(row["fn"] for row in images),
            _mean_scored_image_metric(images, "precision"), _mean_scored_image_metric(images, "recall"), _mean_scored_image_metric(images, "f1"),
            len(subject_ious), sum(subject_ious) / len(subject_ious) if subject_ious else 0.0,
            sum(object_ious) / len(object_ious) if object_ious else 0.0,
            sum(class_values), len(class_values), len(class_values) - sum(class_values),
            sum(class_values) / len(class_values) if class_values else 0.0,
            sum(row["sentence_eligible_pairs"] for row in images), sum(row["sentence_evaluated_pairs"] for row in images),
            _weighted_metric_mean(images, "mean_bleu", "sentence_evaluated_pairs"),
            _weighted_metric_mean(images, "mean_meteor", "sentence_evaluated_pairs"), datetime.now(timezone.utc).isoformat(),
        )
        connection.execute(
            """INSERT INTO hicodet_batch_metrics
                (evaluation_run_id, batch_id, total_images, successful_images, failed_images,
                 total_predictions, total_gt, total_tp, total_fp, total_fn, mean_precision,
                 mean_recall, mean_f1, iou_evaluated_pairs, mean_subj_iou, mean_obj_iou,
                 class_correct_pairs, class_evaluated_pairs, total_classification_errors,
                 class_accuracy, sentence_eligible_pairs, sentence_evaluated_pairs, mean_bleu,
                 mean_meteor, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evaluation_run_id) DO UPDATE SET
              batch_id=excluded.batch_id, total_images=excluded.total_images, successful_images=excluded.successful_images,
              failed_images=excluded.failed_images, total_predictions=excluded.total_predictions, total_gt=excluded.total_gt,
              total_tp=excluded.total_tp, total_fp=excluded.total_fp, total_fn=excluded.total_fn,
              mean_precision=excluded.mean_precision, mean_recall=excluded.mean_recall, mean_f1=excluded.mean_f1,
              iou_evaluated_pairs=excluded.iou_evaluated_pairs, mean_subj_iou=excluded.mean_subj_iou,
              mean_obj_iou=excluded.mean_obj_iou, class_correct_pairs=excluded.class_correct_pairs,
              class_evaluated_pairs=excluded.class_evaluated_pairs, total_classification_errors=excluded.total_classification_errors,
              class_accuracy=excluded.class_accuracy, sentence_eligible_pairs=excluded.sentence_eligible_pairs,
              sentence_evaluated_pairs=excluded.sentence_evaluated_pairs, mean_bleu=excluded.mean_bleu,
              mean_meteor=excluded.mean_meteor, calculated_at=excluded.calculated_at
            """, values)
        connection.commit()
        return dict(connection.execute("SELECT * FROM hicodet_batch_metrics WHERE evaluation_run_id = ?", (evaluation_run_id,)).fetchone())


def set_evaluation_run_status(
    database_path: Path,
    *,
    evaluation_run_id: int,
    status: str,
) -> None:
    with closing(_connection(database_path)) as connection:
        cursor = connection.execute(
            """
            UPDATE evaluation_runs
            SET status = ?, evaluated_at = ?
            WHERE evaluation_run_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), evaluation_run_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Evaluation run not found: {evaluation_run_id}")
        connection.commit()


def calculate_coco_batch_metrics(
    database_path: Path,
    *,
    evaluation_run_id: int,
    batch_id: str,
    user_database_path: Path | None = None,
) -> dict[str, Any]:
    """Calculate metrics from per-image rows belonging to one batch evaluation run."""
    normalized_batch_id = batch_id.strip()
    if not normalized_batch_id:
        raise ValueError("A batch ID is required.")

    source_database = user_database_path or database_path.with_name("user_results.db")
    batch = _load_source_batch(source_database, normalized_batch_id)
    init_evaluation_database(database_path)
    calculated_at = datetime.now(timezone.utc).isoformat()

    with closing(_connection(database_path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = connection.execute(
            """
            SELECT source_batch_id, dataset, task_type
            FROM evaluation_runs
            WHERE evaluation_run_id = ?
            """,
            (evaluation_run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"Evaluation run not found: {evaluation_run_id}")
        if run["source_batch_id"] != normalized_batch_id:
            raise ValueError("Evaluation run batch ID does not match the requested batch ID.")
        if run["dataset"] != "COCO" or run["task_type"] != "object_detection_batch":
            raise ValueError(f"Evaluation run {evaluation_run_id} is not a COCO batch run.")

        image_results = connection.execute(
            """
            SELECT *
            FROM coco_results
            WHERE evaluation_run_id = ? AND batch_id = ?
            ORDER BY image_id
            """,
            (evaluation_run_id, normalized_batch_id),
        ).fetchall()

        successful_images = len(image_results)
        failed_images = max(int(batch["total_images"]) - successful_images, 0)
        total_predictions = sum(row["num_predictions"] for row in image_results)
        total_ground_truths = sum(row["num_ground_truths"] for row in image_results)
        total_tp = sum(row["tp"] for row in image_results)
        total_fp = sum(row["fp"] for row in image_results)
        total_fn = sum(row["fn"] for row in image_results)
        mean_precision = _mean_scored_image_metric(image_results, "precision")
        mean_recall = _mean_scored_image_metric(image_results, "recall")
        mean_f1 = _mean_scored_image_metric(image_results, "f1")
        total_classification_errors = sum(
            row["classification_errors"] for row in image_results
        )
        sentence_eligible_pairs = sum(
            row["sentence_eligible_pairs"] for row in image_results
        )
        sentence_evaluated_pairs = sum(
            row["sentence_evaluated_pairs"] for row in image_results
        )

        evaluated_pairs = connection.execute(
            """
            SELECT iou, class_correct
            FROM coco_pairs
            WHERE evaluation_run_id = ?
              AND batch_id = ?
              AND (iou IS NOT NULL OR class_correct IS NOT NULL)
            """,
            (evaluation_run_id, normalized_batch_id),
        ).fetchall()

        iou_values = [float(row["iou"]) for row in evaluated_pairs if row["iou"] is not None]
        class_values = [
            int(row["class_correct"])
            for row in evaluated_pairs
            if row["class_correct"] is not None
        ]
        iou_evaluated_pairs = len(iou_values)
        mean_iou = sum(iou_values) / iou_evaluated_pairs if iou_values else 0.0
        class_evaluated_pairs = len(class_values)
        class_correct_pairs = sum(class_values)
        class_accuracy = (
            class_correct_pairs / class_evaluated_pairs
            if class_evaluated_pairs > 0
            else 0.0
        )
        mean_bleu = _weighted_metric_mean(
            image_results, "mean_bleu", "sentence_evaluated_pairs"
        )
        mean_meteor = _weighted_metric_mean(
            image_results, "mean_meteor", "sentence_evaluated_pairs"
        )

        values = (
            evaluation_run_id,
            normalized_batch_id,
            batch["total_images"],
            successful_images,
            failed_images,
            total_predictions,
            total_ground_truths,
            total_tp,
            total_fp,
            total_fn,
            mean_precision,
            mean_recall,
            mean_f1,
            iou_evaluated_pairs,
            mean_iou,
            class_correct_pairs,
            class_evaluated_pairs,
            total_classification_errors,
            class_accuracy,
            sentence_eligible_pairs,
            sentence_evaluated_pairs,
            mean_bleu,
            mean_meteor,
            calculated_at,
        )
        connection.execute(
            """
            INSERT INTO coco_batch_metrics
                (evaluation_run_id, batch_id, total_images, successful_images,
                 failed_images, total_predictions, total_ground_truths,
                 total_tp, total_fp, total_fn, mean_precision, mean_recall,
                 mean_f1, iou_evaluated_pairs, mean_iou,
                 class_correct_pairs, class_evaluated_pairs,
                 total_classification_errors, class_accuracy,
                 sentence_eligible_pairs, sentence_evaluated_pairs,
                 mean_bleu, mean_meteor, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(evaluation_run_id) DO UPDATE SET
                batch_id = excluded.batch_id,
                total_images = excluded.total_images,
                successful_images = excluded.successful_images,
                failed_images = excluded.failed_images,
                total_predictions = excluded.total_predictions,
                total_ground_truths = excluded.total_ground_truths,
                total_tp = excluded.total_tp,
                total_fp = excluded.total_fp,
                total_fn = excluded.total_fn,
                mean_precision = excluded.mean_precision,
                mean_recall = excluded.mean_recall,
                mean_f1 = excluded.mean_f1,
                iou_evaluated_pairs = excluded.iou_evaluated_pairs,
                mean_iou = excluded.mean_iou,
                class_correct_pairs = excluded.class_correct_pairs,
                class_evaluated_pairs = excluded.class_evaluated_pairs,
                total_classification_errors = excluded.total_classification_errors,
                class_accuracy = excluded.class_accuracy,
                sentence_eligible_pairs = excluded.sentence_eligible_pairs,
                sentence_evaluated_pairs = excluded.sentence_evaluated_pairs,
                mean_bleu = excluded.mean_bleu,
                mean_meteor = excluded.mean_meteor,
                calculated_at = excluded.calculated_at
            """,
            values,
        )
        connection.commit()
        return dict(
            connection.execute(
                "SELECT * FROM coco_batch_metrics WHERE evaluation_run_id = ?",
                (evaluation_run_id,),
            ).fetchone()
        )


def _load_source_batch(database_path: Path, batch_id: str) -> sqlite3.Row:
    if not database_path.is_file():
        raise ValueError(f"User results database does not exist: {database_path}")
    with closing(_connection(database_path)) as connection:
        if not _table_exists(connection, "batch_runs"):
            raise ValueError(f"{database_path} does not contain batch_runs.")
        batch = connection.execute(
            """
            SELECT total_images, successful_images, failed_images, prompt_mode
            FROM batch_runs
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
    if batch is None:
        raise ValueError(f"Batch run not found: {batch_id}")
    return batch


def _weighted_metric_mean(
    rows: list[sqlite3.Row], metric_column: str, weight_column: str
) -> float | None:
    weighted_values = [
        (float(row[metric_column]), int(row[weight_column]))
        for row in rows
        if row[metric_column] is not None and int(row[weight_column]) > 0
    ]
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight == 0:
        return None
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _mean_scored_image_metric(
    rows: list[sqlite3.Row], metric_column: str
) -> float | None:
    scores = [
        float(row[metric_column])
        for row in rows
        if row[metric_column] is not None
    ]
    return sum(scores) / len(scores) if scores else None
