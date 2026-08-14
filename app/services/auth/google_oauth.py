from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


GOOGLE_AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleOAuthError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class GoogleUser:
    sub: str
    email: str | None
    email_verified: bool
    name: str | None
    given_name: str | None
    family_name: str | None
    picture: str | None
    hd: str | None

    def as_session_payload(self) -> dict[str, object]:
        return {
            "provider": "google",
            "sub": self.sub,
            "email": self.email,
            "email_verified": self.email_verified,
            "name": self.name,
            "given_name": self.given_name,
            "family_name": self.family_name,
            "picture": self.picture,
            "hd": self.hd,
        }


def build_google_authorization_url(*, client_id: str, redirect_uri: str, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account",
            "state": state,
        }
    )
    return f"{GOOGLE_AUTHORIZATION_ENDPOINT}?{query}"


def exchange_code_for_tokens(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> dict:
    payload = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code": code,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GOOGLE_TOKEN_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    return _request_json(request)


def fetch_google_userinfo(*, access_token: str) -> GoogleUser:
    request = urllib.request.Request(
        GOOGLE_USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    payload = _request_json(request)
    sub = str(payload.get("sub", "")).strip()
    if not sub:
        raise GoogleOAuthError("Google userinfo response did not include a stable user id.")

    return GoogleUser(
        sub=sub,
        email=_optional_string(payload.get("email")),
        email_verified=bool(payload.get("email_verified", False)),
        name=_optional_string(payload.get("name")),
        given_name=_optional_string(payload.get("given_name")),
        family_name=_optional_string(payload.get("family_name")),
        picture=_optional_string(payload.get("picture")),
        hd=_optional_string(payload.get("hd")),
    )


def _request_json(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise GoogleOAuthError(_extract_error_message(error_body) or f"Google OAuth request failed with status {exc.code}.", status_code=502) from exc
    except urllib.error.URLError as exc:
        raise GoogleOAuthError(f"Google OAuth request failed: {exc.reason}.", status_code=502) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise GoogleOAuthError("Google OAuth response was not valid JSON.", status_code=502) from exc

    if not isinstance(payload, dict):
        raise GoogleOAuthError("Google OAuth response had an unexpected shape.", status_code=502)
    if "error" in payload and payload["error"]:
        raise GoogleOAuthError(_extract_error_message(payload) or "Google OAuth returned an error.", status_code=502)
    return payload


def _extract_error_message(payload: object) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            return error.strip()
        description = payload.get("error_description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    return None


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
