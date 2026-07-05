"""*** agent 运行过程中的轨迹记录与产物持久化辅助逻辑。 ***"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.agent.state import AnnotationTaskState


def append_trace_event(
    state: AnnotationTaskState,
    *,
    stage: str,
    status: str,
    details: dict | None = None,
) -> None:
    state.trace_events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "attempt": state.attempt_count,
            "strategy": state.strategy,
            "details": details or {},
        }
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_task_state(path: Path, state: AnnotationTaskState) -> None:
    write_json(path, state.to_dict())


def build_trace_payload(state: AnnotationTaskState) -> dict:
    return {
        "task_id": state.task_id,
        "image_id": state.image_path.stem,
        "image_path": str(state.image_path),
        "status": state.status,
        "stage": state.stage,
        "attempt_count": state.attempt_count,
        "max_attempts": state.max_attempts,
        "strategy": state.strategy,
        "review_required": state.review_required,
        "review_reason": state.review_reason,
        "last_error": state.last_error,
        "created_at": state.created_at,
        "completed_at": state.completed_at,
        "events": state.trace_events,
    }


def write_trace(path: Path, state: AnnotationTaskState) -> None:
    write_json(path, build_trace_payload(state))
