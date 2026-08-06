"""*** 健康检查与图像标注上传接口的路由定义。 ***"""

from __future__ import annotations

import traceback
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.agent.orchestrator import AnnotationAgent
from app.api.responses import format_exception_message, format_exception_status
from app.api.schemas import ErrorResponse, HealthResponse, UploadResponse
from app.core.config import AppConfig
from app.core.database import save_annotation_result, save_image

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns a lightweight status response so clients can verify the API is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Upload image and run annotation agent workflow",
    description=(
        "Accepts a multipart image file under the `image` field, stores the upload, "
        "runs the single-orchestrator annotation agent, writes export and trace artifacts, and returns the "
        "frontend-oriented annotation response."
    ),
)
async def upload_image(
    request: Request,
    image: UploadFile | None = File(None, alias="image", description="Image file to annotate."),
    uploaded_image: UploadFile | None = File(
        None,
        alias="uploaded_image",
        description="Legacy image file field name.",
    ),
    prompt_mode: str | None = Form(default=None, alias="prompt_mode"),
) -> UploadResponse | JSONResponse:
    config = _get_config(request)
    try:
        selected_image = image or uploaded_image
        if selected_image is None:
            raise ValueError("No image file provided.")

        filename = Path(selected_image.filename or "").name
        if not filename:
            raise ValueError("No image file provided.")

        file_bytes = await selected_image.read()
        if not file_bytes:
            raise ValueError("No image file provided.")

        destination = _unique_destination(config.paths.uploads_dir, filename)
        destination.write_bytes(file_bytes)
        print(f"Saved upload: {destination.name}")
        print(f"[upload] Stored file at {destination}")
        print(f"[upload] Uploaded size: {len(file_bytes)} bytes")
        selected_prompt_mode = _normalize_prompt_mode(prompt_mode)
        agent = AnnotationAgent(config)
        response_payload = agent.annotate_image(destination, prompt_mode=selected_prompt_mode)

        save_image(
            config.paths.db_path,
            image_id=destination.stem,
            filename=destination.name,
            file_data=file_bytes,
            size_bytes=len(file_bytes),
        )
        agent_info = response_payload.get("agent", {})
        save_annotation_result(
            config.paths.db_path,
            image_id=destination.stem,
            task_id=agent_info.get("task_id", ""),
            task_status=agent_info.get("task_status", ""),
            strategy=agent_info.get("strategy"),
            prompt_mode=selected_prompt_mode,
            attempt_count=agent_info.get("attempt_count", 0),
            review_required=agent_info.get("review_required", False),
            review_reason=agent_info.get("review_reason"),
            annotation_result=response_payload,
        )

        return UploadResponse.model_validate(response_payload)
    except ValueError as exc:
        return _json_error_response(exc, status_code=400, log_traceback=False)
    except Exception as exc:
        return _json_error_response(exc, log_traceback=True)


def _unique_destination(upload_dir: Path, filename: str) -> Path:
    candidate = upload_dir / filename
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1

    while candidate.exists():
        candidate = upload_dir / f"{stem}-{index}{suffix}"
        index += 1

    return candidate


def _normalize_prompt_mode(prompt_mode: str | None) -> str:
    normalized_mode = (prompt_mode or "object_detection").strip().lower()
    if normalized_mode in {"hoi", "human-object interaction", "human_object_interaction"}:
        return "hoi"
    return "object_detection"


def _get_config(request: Request) -> AppConfig:
    config = getattr(request.app.state, "config", None)
    if not isinstance(config, AppConfig):
        raise RuntimeError("Application config is not available on app.state.")
    return config


def _json_error_response(
    exc: Exception,
    *,
    status_code: int | None = None,
    log_traceback: bool,
) -> JSONResponse:
    print("Internal server error:" if log_traceback else "Upload error:")
    if log_traceback:
        traceback.print_exc()
    else:
        print(str(exc))

    response_status = status_code or int(format_exception_status(exc))
    return JSONResponse(
        content={"error": format_exception_message(exc)},
        status_code=response_status,
    )
