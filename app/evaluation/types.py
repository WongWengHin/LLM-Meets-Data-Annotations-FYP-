"""Types shared by the retained offline COCO evaluator."""
from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoGtInstance:
    image_id: str
    category_id: int
    bbox_xyxy_abs: BBox
    annotation_id: str
    area: float
    iscrowd: bool = False


@dataclass(frozen=True)
class CocoPredictionInstance:
    image_id: str
    annotation_id: str
    category_id: int
    bbox_xyxy_abs: BBox
    score: float
    raw_label: str


@dataclass(frozen=True)
class MatchDebugRecord:
    image_id: str
    prediction_id: str
    gt_id: str | None
    score: float
    iou_threshold: float
    matched: bool
    iou: float
    details: dict[str, object] = field(default_factory=dict)
