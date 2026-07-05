"""*** 离线评估适配器与指标计算共用的数据类型定义。 ***"""

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
class HicoGtInstance:
    image_id: str
    annotation_id: str
    subject_bbox_xyxy_abs: BBox
    object_bbox_xyxy_abs: BBox
    predicate: str
    object_name: str
    subject_name: str = "person"
    predicate_id: int | None = None
    object_id: int | None = None
    is_rare: bool | None = None

    @property
    def triplet_key(self) -> str:
        return f"{self.subject_name}|{self.predicate}|{self.object_name}"


@dataclass(frozen=True)
class HicoPredictionInstance:
    image_id: str
    annotation_id: str
    subject_bbox_xyxy_abs: BBox
    object_bbox_xyxy_abs: BBox
    predicate: str
    object_name: str
    score: float
    subject_name: str = "person"
    predicate_id: int | None = None
    object_id: int | None = None

    @property
    def triplet_key(self) -> str:
        return f"{self.subject_name}|{self.predicate}|{self.object_name}"


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
