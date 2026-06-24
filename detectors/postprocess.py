from __future__ import annotations


def filter_detections(detections: list[dict], score_threshold: float, max_detections: int) -> list[dict]:
    filtered: list[dict] = []
    for detection in detections:
        score = detection.get("score")
        bbox = detection.get("bounding_box")
        category = detection.get("category")

        if not isinstance(category, str) or not category.strip():
            continue
        if not isinstance(score, (int, float)) or float(score) < score_threshold:
            continue
        if not _is_valid_bbox(bbox):
            continue

        filtered.append(
            {
                **detection,
                "category": category.strip(),
                "score": round(float(score), 6),
            }
        )

    filtered.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return filtered[:max_detections]


def _is_valid_bbox(bbox: object) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False

    return 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
