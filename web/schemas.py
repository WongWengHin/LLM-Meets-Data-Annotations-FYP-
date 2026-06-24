from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(examples=["ok"])


class ValidationSummary(BaseModel):
    is_valid: bool = Field(examples=[True])
    errors: list[str] = Field(examples=[[]])
    warnings: list[str] = Field(examples=[["annotations[2].sentences contains duplicated sentence content."]])


class SourceSummary(BaseModel):
    type: str = Field(examples=["model_generated"])


class DetectionMetadata(BaseModel):
    source: str | None = Field(default=None, examples=["rt_detr"])
    model: str | None = Field(default=None, examples=["PekingU/rtdetr_r50vd"])


class FrontendAnnotation(BaseModel):
    annotation_id: str = Field(examples=["image-7_001"])
    subject: str = Field(examples=["person"])
    object: str = Field(examples=["snowboard"])
    predicate: str = Field(examples=["ride"])
    subject_bbox: list[float] = Field(examples=[[0.329688, 0.095833, 0.625000, 0.847917]])
    object_bbox: list[float] = Field(examples=[[0.384375, 0.779167, 0.565625, 0.862500]])
    category: str = Field(examples=["person ride snowboard"])
    bounding_box: list[float] = Field(examples=[[0.329688, 0.095833, 0.625000, 0.862500]])
    sentences: list[str] = Field(examples=[["A person is riding a snowboard on a snowy slope."]])
    score: float | None = Field(default=None, examples=[0.9421])
    detection: DetectionMetadata | None = None


class ArtifactSummary(BaseModel):
    raw_output_file: str = Field(examples=["data/raw_outputs/image-7-raw-output.json"])
    master_annotation_file: str = Field(examples=["data/master_annotations/image-7-master-annotation.json"])
    prediction_file: str = Field(examples=["data/predictions/image-7-prediction.json"])
    evaluation_file: str = Field(examples=["data/evaluation_exports/image-7-evaluation.json"])
    detection_file: str | None = Field(default=None, examples=["data/detections/image-7-detections.json"])
    inference_image_file: str | None = Field(default=None, examples=["data/inference_images/image-7-inference.jpg"])


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(examples=["Image uploaded and annotated successfully."])
    filename: str = Field(examples=["image-7.jpg"])
    uploaded_image_url: str = Field(examples=["/uploads/image-7.jpg"])
    size_bytes: int = Field(examples=[83412])
    schema_version: str = Field(examples=["3.0.0"])
    image_id: str = Field(examples=["image-7"])
    caption: str = Field(examples=["A person is riding a snowboard on a snowy slope."])
    annotation_count: int = Field(examples=[2])
    validation: ValidationSummary
    source: SourceSummary
    annotations: list[FrontendAnnotation]
    artifacts: ArtifactSummary


class ErrorResponse(BaseModel):
    error: str = Field(examples=["No image file provided."])
