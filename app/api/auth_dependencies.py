from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.core.config import AppConfig


def get_current_user(request: Request) -> dict | None:
    user = request.session.get("user")
    return user if isinstance(user, dict) else None


def require_authenticated_user(request: Request) -> dict | None:
    config = _get_config(request)
    if not config.auth.enabled:
        return None

    user = get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google login required.",
        )
    return user


def require_google_user(request: Request) -> dict:
    """Require a verified session for endpoints containing private user data."""
    user = get_current_user(request)
    if user is None or not str(user.get("sub", "")).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google login required.",
        )
    return user


def _get_config(request: Request) -> AppConfig:
    config = getattr(request.app.state, "config", None)
    if not isinstance(config, AppConfig):
        raise RuntimeError("Application config is not available on app.state.")
    return config
