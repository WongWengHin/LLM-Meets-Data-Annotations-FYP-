"""*** 健康检查与图像标注上传接口的路由定义。 ***"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from app.agent.orchestrator import AnnotationAgent
from app.api.auth_dependencies import require_authenticated_user, require_google_user
from app.api.responses import format_exception_message, format_exception_status
from app.api.schemas import ErrorResponse, HealthResponse, UploadResponse
from app.core.config import AppConfig
from app.core.user_database import get_user_image_by_id, get_user_results, save_user_annotation_result, save_user_image

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
    user: dict | None = Depends(require_authenticated_user),
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

        owner_id = _owner_id(user)
        # Images are served through an ownership-checked route, never from a
        # publicly browsable upload directory.
        response_payload["uploaded_image_url"] = f"/api/results/{destination.stem}/image"
        save_user_image(
            config.paths.user_db_path,
            image_id=destination.stem,
            owner_id=owner_id,
            filename=destination.name,
            file_data=file_bytes,
            size_bytes=len(file_bytes),
        )
        agent_info = response_payload.get("agent", {})
        save_user_annotation_result(
            config.paths.user_db_path,
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


@router.get("/api/results")
async def list_results(
    request: Request,
    user: dict = Depends(require_google_user),
) -> dict:
    """List only the current Google user's uploaded images and latest results."""
    config = _get_config(request)
    owner_id = _owner_id(user)
    results = get_user_results(config.paths.user_db_path, owner_id)
    items = []
    for result in results:
        payload = json.loads(result.pop("annotation_result")) if result.get("annotation_result") else None
        items.append({
            **result,
            "image_url": f"/api/results/{result['image_id']}/image",
            "result": payload,
        })
    return {"items": items}


@router.get("/api/results/{image_id}/image")
async def get_result_image(
    image_id: str,
    request: Request,
    user: dict = Depends(require_google_user),
) -> Response:
    config = _get_config(request)
    image = get_user_image_by_id(config.paths.user_db_path, image_id, _owner_id(user))
    if image is None:
        # Do not disclose whether another user's image exists.
        raise HTTPException(status_code=404, detail="Image not found.")
    return Response(content=image["file_data"], media_type=_image_media_type(image["filename"]))


def _unique_destination(upload_dir: Path, filename: str) -> Path:
    """
    Keep upload filenames stable so image_id stays aligned with GT filenames.

    We intentionally overwrite an existing file with the same name instead of
    creating a suffixed duplicate like "2149-1.jpg".
    """
    return upload_dir / filename


def _normalize_prompt_mode(prompt_mode: str | None) -> str:
    normalized_mode = (prompt_mode or "object_detection").strip().lower()
    if normalized_mode in {"hoi", "human-object interaction", "human_object_interaction"}:
        return "hoi"
    return "object_detection"


def _owner_id(user: dict | None) -> str:
    if not user:
        return "anonymous"
    google_sub = str(user.get("sub", "")).strip()
    if not google_sub:
        raise HTTPException(status_code=401, detail="Google login required.")
    return f"google:{google_sub}"


def _image_media_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(suffix, "application/octet-stream")


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
