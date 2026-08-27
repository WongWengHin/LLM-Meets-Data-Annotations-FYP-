"""*** FastAPI 应用工厂与静态文件挂载逻辑。 ***"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import AppConfig
from app.core.evaluation_database import init_evaluation_database
from app.core.user_database import init_user_database
from app.api.routes import router
from app.api.auth_routes import router as auth_router


def create_app(config: AppConfig) -> FastAPI:
    init_user_database(config.paths.user_db_path)
    init_evaluation_database(config.paths.data_dir / "evaluation_result.db")

    app = FastAPI(
        title="Large Language Meets Data Annotation",
        description=(
            "FastAPI backend for image upload, annotation pipeline execution, "
            "and export-oriented annotation responses."
        ),
    )
    app.state.config = config
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.auth.session_secret_key,
        same_site="lax",
        https_only=False,
    )

    @app.get("/", summary="Frontend entry page", description="Serves the static frontend application.")
    async def index() -> FileResponse:
        return FileResponse(config.paths.frontend_dir / "index.html")

    app.include_router(router)
    app.include_router(auth_router)
    app.mount("/uploads", StaticFiles(directory=config.paths.uploads_dir), name="uploads")
    app.mount("/", StaticFiles(directory=config.paths.frontend_dir, html=True), name="frontend")
    return app
