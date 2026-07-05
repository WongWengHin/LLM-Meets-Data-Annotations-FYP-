"""*** FastAPI 应用工厂与静态文件挂载逻辑。 ***"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import AppConfig
from app.api.routes import router


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(
        title="Large Language Meets Data Annotation",
        description=(
            "FastAPI backend for image upload, annotation pipeline execution, "
            "and export-oriented annotation responses."
        ),
    )
    app.state.config = config

    @app.get("/", summary="Frontend entry page", description="Serves the static frontend application.")
    async def index() -> FileResponse:
        return FileResponse(config.paths.frontend_dir / "index.html")

    app.include_router(router)
    app.mount("/uploads", StaticFiles(directory=config.paths.uploads_dir), name="uploads")
    app.mount("/", StaticFiles(directory=config.paths.frontend_dir, html=True), name="frontend")
    return app
