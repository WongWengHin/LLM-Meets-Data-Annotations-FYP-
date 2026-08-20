"""*** SQLite 数据库初始化与图片、标注结果持久化逻辑。 ***"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.core.image_ids import canonical_image_id


def init_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_data BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS annotation_runs (
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
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_annotation_runs_image_id
            ON annotation_runs(image_id)
        """)
        conn.commit()


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_image(
    db_path: Path,
    *,
    image_id: str,
    filename: str,
    file_data: bytes,
    size_bytes: int,
    width: int | None = None,
    height: int | None = None,
) -> None:
    image_id = canonical_image_id(image_id)
    uploaded_at = datetime.now(timezone.utc).isoformat()
    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO images
                (image_id, filename, file_data, size_bytes, width, height, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, filename, file_data, size_bytes, width, height, uploaded_at),
        )
        conn.commit()


def save_annotation_result(
    db_path: Path,
    *,
    image_id: str,
    task_id: str,
    task_status: str,
    strategy: str | None = None,
    prompt_mode: str | None = None,
    attempt_count: int = 0,
    review_required: bool = False,
    review_reason: str | None = None,
    annotation_result: dict,
) -> None:
    image_id = canonical_image_id(image_id)
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(get_connection(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO annotation_runs
                (image_id, task_id, task_status, strategy, prompt_mode,
                 attempt_count, review_required, review_reason, annotation_result, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                task_id,
                task_status,
                strategy,
                prompt_mode,
                attempt_count,
                int(review_required),
                review_reason,
                json.dumps(annotation_result, ensure_ascii=False),
                created_at,
            ),
        )
        conn.commit()


def get_image_by_id(db_path: Path, image_id: str) -> dict | None:
    image_id = canonical_image_id(image_id)
    with closing(get_connection(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM images WHERE image_id = ?",
            (image_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def get_annotations_by_image_id(db_path: Path, image_id: str) -> list[dict]:
    image_id = canonical_image_id(image_id)
    with closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM annotation_runs WHERE image_id = ? ORDER BY created_at DESC",
            (image_id,),
        ).fetchall()
        return [dict(row) for row in rows]
