from __future__ import annotations

from agent.state import AnnotationTaskState


def is_recoverable_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    recoverable_markers = (
        "json",
        "empty",
        "missing",
        "bbox",
        "validation",
        "annotation item",
        "no choices",
        "invalid message",
    )
    return any(marker in message for marker in recoverable_markers)


def should_retry(state: AnnotationTaskState, exc: Exception | None = None) -> bool:
    if state.attempt_count >= state.max_attempts:
        return False
    if exc is None:
        return True
    return is_recoverable_error(exc)


def advance_strategy_after_failure(state: AnnotationTaskState) -> None:
    state.strategy = "retry_llm_only"
