from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

TaskStatus = Literal["pending", "running", "completed", "failed", "review_required"]
TaskStage = Literal["created", "preprocess", "annotate", "normalize", "validate", "export"]
TaskStrategy = Literal["llm_only", "retry_llm_only"]


@dataclass
class AnnotationTaskState:
    task_id: str
    image_path: Path
    status: TaskStatus
    stage: TaskStage
    strategy: TaskStrategy
    attempt_count: int = 0
    max_attempts: int = 2
    review_required: bool = False
    review_reason: str | None = None
    last_error: str | None = None
    trace_events: list[dict] = field(default_factory=list)
    raw_model_output: str | None = None
    prompt_mode: str | None = None
    normalized_annotation: dict | None = None
    validation: dict | None = None
    master_annotation: dict | None = None
    inference_image: dict | None = None
    artifacts: dict[str, str | None] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["image_path"] = str(self.image_path)
        return payload


def create_task_id(image_path: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"task_{image_path.stem}_{timestamp}"


def create_initial_task_state(image_path: Path, *, max_attempts: int = 2) -> AnnotationTaskState:
    return AnnotationTaskState(
        task_id=create_task_id(image_path),
        image_path=image_path,
        status="pending",
        stage="created",
        strategy="llm_only",
        max_attempts=max_attempts,
    )
