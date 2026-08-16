"""SQLite persistence for private, Google-account-owned user results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def init_user_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                owner_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_data BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_annotation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_status TEXT NOT NULL,
                strategy TEXT,
                prompt_mode TEXT,
                attempt_count INTEGER DEFAULT 0,
                review_required INTEGER DEFAULT 0,
                review_reason TEXT,
                annotation_result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES user_images(image_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_images_owner_uploaded_at ON user_images(owner_id, uploaded_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_annotation_runs_image_id ON user_annotation_runs(image_id)")


def _connection(database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_user_image(
    database_path: Path, *, image_id: str, owner_id: str, filename: str,
    file_data: bytes, size_bytes: int, width: int | None = None, height: int | None = None,
) -> None:
    with _connection(database_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_images
            (image_id, owner_id, filename, file_data, size_bytes, width, height, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (image_id, owner_id, filename, file_data, size_bytes, width, height, datetime.now(timezone.utc).isoformat()),
        )


def save_user_annotation_result(
    database_path: Path, *, image_id: str, task_id: str, task_status: str,
    strategy: str | None = None, prompt_mode: str | None = None, attempt_count: int = 0,
    review_required: bool = False, review_reason: str | None = None, annotation_result: dict,
) -> None:
    with _connection(database_path) as conn:
        conn.execute(
            """INSERT INTO user_annotation_runs
            (image_id, task_id, task_status, strategy, prompt_mode, attempt_count,
             review_required, review_reason, annotation_result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (image_id, task_id, task_status, strategy, prompt_mode, attempt_count,
             int(review_required), review_reason, json.dumps(annotation_result, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )


def get_user_image_by_id(database_path: Path, image_id: str, owner_id: str) -> dict | None:
    with _connection(database_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_images WHERE image_id = ? AND owner_id = ?", (image_id, owner_id)
        ).fetchone()
        return dict(row) if row else None


def get_user_results(database_path: Path, owner_id: str) -> list[dict]:
    with _connection(database_path) as conn:
        rows = conn.execute(
            """
            SELECT user_images.image_id, user_images.filename, user_images.size_bytes, user_images.uploaded_at,
                   user_annotation_runs.task_status, user_annotation_runs.review_required,
                   user_annotation_runs.annotation_result, user_annotation_runs.created_at AS annotated_at
            FROM user_images
            LEFT JOIN user_annotation_runs ON user_annotation_runs.id = (
                SELECT id FROM user_annotation_runs
                WHERE user_annotation_runs.image_id = user_images.image_id
                ORDER BY user_annotation_runs.created_at DESC, user_annotation_runs.id DESC LIMIT 1
            )
            WHERE user_images.owner_id = ?
            ORDER BY user_images.uploaded_at DESC
            """,
            (owner_id,),
        ).fetchall()
        return [dict(row) for row in rows]
