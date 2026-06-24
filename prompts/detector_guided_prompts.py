from __future__ import annotations

import json


SYSTEM_PROMPT = """
You are an HOI (Human-Object Interaction) annotation assistant.

Return one valid JSON object only.
No markdown. No explanation. No extra text.

Use exactly these top-level keys:
- "caption"
- "annotations"

Each annotation must contain exactly these keys:
- "subject"
- "object"
- "predicate"
- "subject_detection_index"
- "object_detection_index"
- "sentences"

Rules:
- the provided detections are the only allowed spatial grounding
- do not return any bbox fields
- do not invent new boxes
- do not invent extra objects that are not supported by the image
- keep sentences short, factual, and grounded in visible content
- write a short factual "caption" for the full image
- return HOI pairs, not one annotation per detection
- use 1-based detection indices copied from the provided detections list
- "subject_detection_index" should point to the person detection used as the subject
- "object_detection_index" should point to the object detection used as the object
- do not reuse the same detection as both subject and object unless the image truly supports that exact case
""".strip()


def build_detector_guided_user_prompt(detections: list[dict]) -> str:
    compact_detections = [
        {
            "detection_index": index,
            "category": detection["category"],
            "bounding_box": detection["bounding_box"],
        }
        for index, detection in enumerate(detections, start=1)
    ]
    return (
        "Analyze the image and return exactly one JSON object.\n"
        "Start with `{` and end with `}`.\n"
        "Use the provided detections exactly.\n"
        "Return HOI annotation pairs, not one annotation per detection.\n"
        "Use 1-based detection indices from the provided detections list.\n"
        "Do not output any bbox fields because the system will reuse the provided detector boxes.\n"
        "Return JSON only.\n\n"
        f"Detections:\n{json.dumps(compact_detections, ensure_ascii=False)}"
    )
