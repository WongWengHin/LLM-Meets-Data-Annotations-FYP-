"""SQLite persistence for private, Google-account-owned user results."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.image_ids import canonical_image_id


def init_user_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_runs (
                batch_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total_images INTEGER NOT NULL,
                processed_images INTEGER NOT NULL DEFAULT 0,
                successful_images INTEGER NOT NULL DEFAULT 0,
                failed_images INTEGER NOT NULL DEFAULT 0,
                prompt_mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                batch_id TEXT,
                owner_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_data BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batch_runs(batch_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_annotation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                batch_id TEXT,
                task_id TEXT NOT NULL,
                task_status TEXT NOT NULL,
                strategy TEXT,
                prompt_mode TEXT,
                attempt_count INTEGER DEFAULT 0,
                review_required INTEGER DEFAULT 0,
                review_reason TEXT,
                annotation_result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES user_images(image_id),
                FOREIGN KEY (batch_id) REFERENCES batch_runs(batch_id)
            )
        """)
        _add_missing_column(
            conn,
            "user_images",
            "batch_id",
            "TEXT REFERENCES batch_runs(batch_id)",
        )
        _add_missing_column(
            conn,
            "user_annotation_runs",
            "batch_id",
            "TEXT REFERENCES batch_runs(batch_id)",
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_images_owner_uploaded_at ON user_images(owner_id, uploaded_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_images_batch_id ON user_images(batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_annotation_runs_image_id ON user_annotation_runs(image_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_annotation_runs_batch_id ON user_annotation_runs(batch_id)")
        conn.commit()


def _add_missing_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    existing_columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing_columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _connection(database_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_batch_run(
    database_path: Path,
    *,
    total_images: int,
    prompt_mode: str,
) -> dict:
    if total_images <= 0:
        raise ValueError("A batch must contain at least one image.")

    normalized_prompt_mode = prompt_mode.strip()
    if not normalized_prompt_mode:
        raise ValueError("A batch prompt mode is required.")

    batch_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connection(database_path)) as conn:
        conn.execute(
            """
            INSERT INTO batch_runs
                (batch_id, status, total_images, processed_images,
                 successful_images, failed_images, prompt_mode, created_at,
                 started_at, completed_at)
            VALUES (?, 'pending', ?, 0, 0, 0, ?, ?, NULL, NULL)
            """,
            (batch_id, total_images, normalized_prompt_mode, created_at),
        )
        conn.commit()
        return dict(
            conn.execute(
                "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        )


def get_batch_run(database_path: Path, batch_id: str) -> dict | None:
    with closing(_connection(database_path)) as conn:
        row = conn.execute(
            "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        return dict(row) if row else None


def start_batch_run(
    database_path: Path,
    *,
    batch_id: str,
    prompt_mode: str,
) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    with closing(_connection(database_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        batch = conn.execute(
            "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError("Batch run not found.")
        if batch["prompt_mode"] != prompt_mode:
            raise ValueError("Batch prompt mode does not match the upload prompt mode.")
        if batch["processed_images"] >= batch["total_images"]:
            raise ValueError("Batch run is already complete.")

        conn.execute(
            """
            UPDATE batch_runs
            SET status = 'processing', started_at = COALESCE(started_at, ?)
            WHERE batch_id = ?
            """,
            (started_at, batch_id),
        )
        conn.commit()


def record_batch_image_result(
    database_path: Path,
    *,
    batch_id: str,
    successful: bool,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connection(database_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        batch = conn.execute(
            "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            raise ValueError("Batch run not found.")
        if batch["processed_images"] >= batch["total_images"]:
            raise ValueError("Batch run is already complete.")

        processed_images = batch["processed_images"] + 1
        successful_images = batch["successful_images"] + int(successful)
        failed_images = batch["failed_images"] + int(not successful)
        completed_at = None
        status = "processing"
        if processed_images == batch["total_images"]:
            completed_at = now
            if failed_images == 0:
                status = "completed"
            elif successful_images == 0:
                status = "failed"
            else:
                status = "completed_with_errors"

        conn.execute(
            """
            UPDATE batch_runs
            SET status = ?, processed_images = ?, successful_images = ?,
                failed_images = ?, started_at = COALESCE(started_at, ?),
                completed_at = ?
            WHERE batch_id = ?
            """,
            (
                status,
                processed_images,
                successful_images,
                failed_images,
                now,
                completed_at,
                batch_id,
            ),
        )
        conn.commit()
        return dict(
            conn.execute(
                "SELECT * FROM batch_runs WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        )


def save_user_image(
    database_path: Path, *, image_id: str, owner_id: str, filename: str,
    file_data: bytes, size_bytes: int, width: int | None = None,
    height: int | None = None, batch_id: str | None = None,
) -> None:
    image_id = canonical_image_id(image_id)
    with closing(_connection(database_path)) as conn:
        conn.execute(
            """INSERT INTO user_images
            (image_id, batch_id, owner_id, filename, file_data, size_bytes, width, height, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_id) DO UPDATE SET
                batch_id = excluded.batch_id,
                owner_id = excluded.owner_id,
                filename = excluded.filename,
                file_data = excluded.file_data,
                size_bytes = excluded.size_bytes,
                width = excluded.width,
                height = excluded.height,
                uploaded_at = excluded.uploaded_at""",
            (image_id, batch_id, owner_id, filename, file_data, size_bytes, width,
             height, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def save_user_annotation_result(
    database_path: Path, *, image_id: str, task_id: str, task_status: str,
    strategy: str | None = None, prompt_mode: str | None = None, attempt_count: int = 0,
    review_required: bool = False, review_reason: str | None = None,
    annotation_result: dict, batch_id: str | None = None,
) -> None:
    image_id = canonical_image_id(image_id)
    with closing(_connection(database_path)) as conn:
        conn.execute(
            """INSERT INTO user_annotation_runs
            (image_id, batch_id, task_id, task_status, strategy, prompt_mode, attempt_count,
             review_required, review_reason, annotation_result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (image_id, batch_id, task_id, task_status, strategy, prompt_mode, attempt_count,
             int(review_required), review_reason, json.dumps(annotation_result, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_user_image_by_id(database_path: Path, image_id: str, owner_id: str) -> dict | None:
    image_id = canonical_image_id(image_id)
    with closing(_connection(database_path)) as conn:
        row = conn.execute(
            "SELECT * FROM user_images WHERE image_id = ? AND owner_id = ?", (image_id, owner_id)
        ).fetchone()
        return dict(row) if row else None


def get_user_results(database_path: Path, owner_id: str) -> list[dict]:
    with closing(_connection(database_path)) as conn:
        rows = conn.execute(
            """
            SELECT user_images.image_id, user_images.filename, user_images.size_bytes, user_images.uploaded_at,
                   user_annotation_runs.batch_id, user_annotation_runs.task_status,
                   user_annotation_runs.review_required,
                   user_annotation_runs.annotation_result, user_annotation_runs.created_at AS annotated_at
            FROM user_images
            INNER JOIN user_annotation_runs ON user_annotation_runs.id = (
                SELECT id FROM user_annotation_runs
                WHERE user_annotation_runs.image_id = user_images.image_id
                  AND user_annotation_runs.task_status = 'completed'
                ORDER BY user_annotation_runs.created_at DESC, user_annotation_runs.id DESC LIMIT 1
            )
            WHERE user_images.owner_id = ?
            ORDER BY user_images.uploaded_at DESC
            """,
            (owner_id,),
        ).fetchall()
        return [dict(row) for row in rows]
