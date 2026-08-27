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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_run_id INTEGER,
                image_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                task_type TEXT NOT NULL,
                prompt_mode TEXT NOT NULL,
                evaluated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS coco_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL UNIQUE,
                image_id TEXT NOT NULL,
                num_predictions INTEGER NOT NULL,
                num_ground_truths INTEGER NOT NULL,
                tp INTEGER NOT NULL,
                fp INTEGER NOT NULL,
                fn INTEGER NOT NULL,
                classification_errors INTEGER NOT NULL,
                mean_iou REAL NOT NULL,
                class_accuracy REAL NOT NULL,
                iou_threshold REAL NOT NULL,
                iou_weight REAL NOT NULL,
                class_weight REAL NOT NULL,
                l2_weight REAL NOT NULL,
                FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS coco_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id INTEGER NOT NULL,
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
                FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluation_runs_image_id "
            "ON evaluation_runs(image_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_coco_pairs_evaluation_run_id "
            "ON coco_pairs(evaluation_run_id)"
        )
        connection.commit()


def _json_bbox(bbox: Any) -> str | None:
    return None if bbox is None else json.dumps(bbox, separators=(",", ":"))


def save_coco_evaluation(
    database_path: Path,
    *,
    source_run_id: int | None,
    image_id: str,
    prompt_mode: str,
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    result: dict[str, Any],
) -> int:
    """Atomically store one COCO evaluation and return its master run ID."""
    init_evaluation_database(database_path)
    counts = result["counts"]
    config = result["config"]
    assigned_pairs = result["assigned_pairs"]
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

    with closing(_connection(database_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_runs
                (source_run_id, image_id, dataset, task_type, prompt_mode, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_run_id,
                image_id,
                "COCO",
                "object_detection",
                prompt_mode,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        evaluation_run_id = int(cursor.lastrowid)
        weights = config["cost_weights"]
        connection.execute(
            """
            INSERT INTO coco_results
                (evaluation_run_id, image_id, num_predictions, num_ground_truths,
                 tp, fp, fn, classification_errors, mean_iou, class_accuracy,
                 iou_threshold, iou_weight, class_weight, l2_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_run_id,
                image_id,
                counts["predictions"],
                counts["ground_truths"],
                counts["tp"],
                counts["fp"],
                counts["fn"],
                counts["classification_errors"],
                mean_iou,
                class_accuracy,
                config["iou_threshold"],
                weights["iou"],
                weights["class"],
                weights["l2"],
            ),
        )

        pair_rows: list[tuple[Any, ...]] = []
        for pair in assigned_pairs:
            pair_rows.append(
                (
                    evaluation_run_id,
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
                )
            )
        for prediction_index in result["unassigned_prediction_indices"]:
            prediction = predictions[prediction_index]
            pair_rows.append(
                (
                    evaluation_run_id,
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
                )
            )
        for ground_truth_index in result["unassigned_ground_truth_indices"]:
            truth = ground_truths[ground_truth_index]
            pair_rows.append(
                (
                    evaluation_run_id,
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
                )
            )

        connection.executemany(
            """
            INSERT INTO coco_pairs
                (evaluation_run_id, image_id, prediction_index, ground_truth_index,
                 prediction_annotation_id, ground_truth_annotation_id,
                 predicted_class, ground_truth_class, predicted_bbox, ground_truth_bbox,
                 iou, l2_distance, matching_cost, class_correct, status, rejection_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            pair_rows,
        )
        connection.commit()
    return evaluation_run_id
