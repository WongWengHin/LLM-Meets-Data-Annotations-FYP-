from __future__ import annotations

import secrets
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import AppConfig
from app.services.auth.google_oauth import (
    GoogleOAuthError,
    build_google_authorization_url,
    exchange_code_for_tokens,
    fetch_google_userinfo,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
async def auth_me(request: Request) -> dict:
    config = _get_config(request)
    user = request.session.get("user")
    return {
        "auth_enabled": config.auth.enabled,
        "require_login": config.auth.require_login,
        "authenticated": isinstance(user, dict),
        "user": user if isinstance(user, dict) else None,
    }


@router.get("/google/login")
async def google_login(request: Request):
    config = _get_config(request)
    if not config.auth.enabled:
        return JSONResponse(
            {"error": "Google login is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."},
            status_code=503,
        )

    state = secrets.token_urlsafe(32)
    request.session["google_oauth_state"] = state
    return RedirectResponse(
        build_google_authorization_url(
            client_id=config.auth.google_client_id,
            redirect_uri=config.auth.google_redirect_uri,
            state=state,
        ),
        status_code=302,
    )


@router.get("/google/callback")
async def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None, error_description: str | None = None):
    config = _get_config(request)
    if not config.auth.enabled:
        return JSONResponse(
            {"error": "Google login is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."},
            status_code=503,
        )

    if error:
        message = error_description or error
        return RedirectResponse(_build_frontend_error_redirect(message), status_code=302)

    expected_state = request.session.get("google_oauth_state")
    if not expected_state or state != expected_state:
        return RedirectResponse(_build_frontend_error_redirect("Google login failed CSRF validation."), status_code=302)

    if not code:
        return RedirectResponse(_build_frontend_error_redirect("Google login did not return an authorization code."), status_code=302)

    try:
        token_payload = exchange_code_for_tokens(
            client_id=config.auth.google_client_id,
            client_secret=config.auth.google_client_secret,
            redirect_uri=config.auth.google_redirect_uri,
            code=code,
        )
        access_token = str(token_payload.get("access_token", "")).strip()
        if not access_token:
            raise GoogleOAuthError("Google token response did not include an access token.")

        user = fetch_google_userinfo(access_token=access_token)
    except GoogleOAuthError as exc:
        return RedirectResponse(_build_frontend_error_redirect(str(exc)), status_code=302)

    request.session["user"] = user.as_session_payload()
    request.session.pop("google_oauth_state", None)
    return RedirectResponse("/?auth=google_success", status_code=302)


@router.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    request.session.pop("google_oauth_state", None)
    return RedirectResponse("/?auth=logged_out", status_code=302)


def _get_config(request: Request) -> AppConfig:
    config = getattr(request.app.state, "config", None)
    if not isinstance(config, AppConfig):
        raise RuntimeError("Application config is not available on app.state.")
    return config


def _build_frontend_error_redirect(message: str) -> str:
    return f"/?auth_error={quote(message)}"
