

from __future__ import annotations

from app.services.prompts.hoi_prompt import SYSTEM_PROMPT as HOI_SYSTEM_PROMPT, USER_PROMPT as HOI_USER_PROMPT
from app.services.prompts.object_detection_prompt import (
    SYSTEM_PROMPT as OBJECT_DETECTION_SYSTEM_PROMPT,
    USER_PROMPT as OBJECT_DETECTION_USER_PROMPT,
)


def get_annotation_prompts(mode: str | None) -> tuple[str, str, str]:
    normalized_mode = (mode or "object_detection").strip().lower()

    if normalized_mode in {"hoi", "human-object interaction", "human_object_interaction"}:
        return "hoi", HOI_SYSTEM_PROMPT, HOI_USER_PROMPT

    if normalized_mode in {"object_detection", "object-detection", "coco", "standard"}:
        return "object_detection", OBJECT_DETECTION_SYSTEM_PROMPT, OBJECT_DETECTION_USER_PROMPT

    return "object_detection", OBJECT_DETECTION_SYSTEM_PROMPT, OBJECT_DETECTION_USER_PROMPT
