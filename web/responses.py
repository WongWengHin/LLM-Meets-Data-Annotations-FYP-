from __future__ import annotations

import json
from http import HTTPStatus

from core.errors import ExternalAPIError


def format_exception_message(exc: Exception) -> str:
    if isinstance(exc, ExternalAPIError):
        try:
            payload = json.loads(exc.body)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            error_block = payload.get("error")
            if isinstance(error_block, dict):
                message = error_block.get("message")
                code = error_block.get("code")
                if message and code:
                    return f"{message} (code: {code})"
                if message:
                    return str(message)

    return str(exc)


def format_exception_status(exc: Exception) -> HTTPStatus:
    status_code = exc.status_code if isinstance(exc, ExternalAPIError) else getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code <= 599:
        return HTTPStatus(status_code)
    return HTTPStatus.INTERNAL_SERVER_ERROR

