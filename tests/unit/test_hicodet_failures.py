from __future__ import annotations

import sqlite3
import shutil
import uuid
from pathlib import Path

from app.core.evaluation_database import (
    create_hicodet_batch_evaluation_run,
    save_hicodet_failure,
)


def test_save_hicodet_failure_records_stage_and_reason() -> None:
    directory = Path(".tmp_tests") / f"hicodet_failures_{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        database_path = directory / "evaluation_result.db"
        run_id = create_hicodet_batch_evaluation_run(
            database_path,
            batch_id="batch-1",
            prompt_mode="hoi",
        )

        save_hicodet_failure(
            database_path,
            evaluation_run_id=run_id,
            batch_id="batch-1",
            image_id="HICO_train2015_00000001",
            failure_stage="invalid_prediction",
            failure_reason="Prediction 0 needs non-empty subject, predicate, and object.",
        )

        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM hicodet_failures").fetchone()
        connection.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    assert row is not None
    assert row["evaluation_run_id"] == run_id
    assert row["batch_id"] == "batch-1"
    assert row["image_id"] == "HICO_train2015_00000001"
    assert row["failure_stage"] == "invalid_prediction"
    assert row["failure_reason"] == "Prediction 0 needs non-empty subject, predicate, and object."
