"""*** Web API 返回数据使用的 Pydantic 结构定义。 ***"""

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


class FrontendAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    annotation_id: str = Field(examples=["image-7_001"])
    subject: str | None = Field(default=None, examples=["person"])
    object: str | None = Field(default=None, examples=["snowboard"])
    predicate: str | None = Field(default=None, examples=["ride"])
    subject_bbox: list[float] | None = Field(default=None, examples=[[0.329688, 0.095833, 0.625000, 0.847917]])
    object_bbox: list[float] | None = Field(default=None, examples=[[0.384375, 0.779167, 0.565625, 0.862500]])
    category: str | None = Field(default=None, examples=["person ride snowboard"])
    bounding_box: list[float] | None = Field(default=None, examples=[[0.329688, 0.095833, 0.625000, 0.862500]])
    bbox: list[float] | None = Field(default=None, examples=[[0.245000, 0.312000, 0.624000, 0.856000]])
    sentences: list[str] = Field(examples=[["A person is riding a snowboard on a snowy slope."]])
    score: float | None = Field(default=None, examples=[0.9421])
    image_id: str | None = Field(default=None, examples=["image-7"])
    class_name: str | None = Field(default=None, alias="class", examples=["dog"])


class ArtifactSummary(BaseModel):
    raw_output_file: str | None = Field(default=None, examples=["data/raw_outputs/image-7-raw-output.json"])
    master_annotation_file: str | None = Field(default=None, examples=["data/master_annotations/image-7-master-annotation.json"])
    prediction_file: str | None = Field(default=None, examples=["data/predictions/image-7-prediction.json"])
    evaluation_file: str | None = Field(default=None, examples=["data/evaluation_exports/image-7-evaluation.json"])
    inference_image_file: str | None = Field(default=None, examples=["data/inference_images/image-7-inference.jpg"])
    normalized_output_file: str | None = Field(default=None, examples=["data/normalized_outputs/image-7-normalized.json"])
    validation_report_file: str | None = Field(default=None, examples=["data/validation_reports/image-7-validation.json"])
    orchestration_trace_file: str | None = Field(default=None, examples=["data/orchestration_traces/image-7-trace.json"])
    task_state_file: str | None = Field(default=None, examples=["data/task_states/image-7-task.json"])


class AgentSummary(BaseModel):
    task_id: str = Field(examples=["task_image-7_20260705T101530Z"])
    task_status: str = Field(examples=["completed"])
    stage: str = Field(examples=["export"])
    strategy: str = Field(examples=["llm_only"])
    attempt_count: int = Field(examples=[1])
    review_required: bool = Field(examples=[False])
    review_reason: str | None = Field(default=None, examples=["Validation failed after 2 attempts."])


class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(examples=["Image uploaded and annotated successfully."])
    filename: str = Field(examples=["image-7.jpg"])
    uploaded_image_url: str = Field(examples=["/uploads/image-7.jpg"])
    size_bytes: int = Field(examples=[83412])
    schema_version: str = Field(examples=["3.0.0"])
    image_id: str = Field(examples=["image-7"])
    caption: str | None = Field(default=None, examples=["A person is riding a snowboard on a snowy slope."])
    annotation_count: int = Field(examples=[2])
    validation: ValidationSummary
    source: SourceSummary
    annotations: list[FrontendAnnotation]
    agent: AgentSummary
    artifacts: ArtifactSummary


class ErrorResponse(BaseModel):
    error: str = Field(examples=["No image file provided."])
