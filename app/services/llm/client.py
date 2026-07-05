"""*** 面向图像标注生成任务的 LLM 客户端请求逻辑。 ***"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import LLMConfig
from app.core.errors import ExternalAPIError
from app.domain.images.bbox_utils import guess_mime_type
from app.services.prompts.image_annotation_prompts import get_annotation_prompts


def request_image_annotation(
    image_path: Path,
    llm_config: LLMConfig,
) -> tuple[str, str]:
    if not llm_config.api_key:
        raise RuntimeError(
            "Missing LLM API key in .env. Set LLM_PROVIDER with its matching "
            "<PROVIDER>_API_KEY, or keep using the legacy ANNOTATION_API_KEY."
        )

    active_prompt_mode, system_prompt, user_prompt = get_annotation_prompts(None)

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = guess_mime_type(image_path.suffix.lower())
    print(
        "[llm] Prepared request payload: "
        f"provider={llm_config.provider}, "
        f"model={llm_config.model}, "
        f"image={image_path.name}, "
        f"bytes={len(image_bytes)}, "
        f"max_tokens={llm_config.max_tokens}, "
        f"timeout={llm_config.timeout_seconds}s"
    )

    payload = {
        "model": llm_config.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": _build_user_content(user_prompt, mime_type, image_base64),
            },
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": llm_config.max_tokens,
    }

    if llm_config.provider == "ollama":
        # Prefer direct answer content over reasoning-only output for local Qwen models.
        payload["think"] = False

    request = Request(
        url=f"{llm_config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {llm_config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        request_url = f"{llm_config.base_url.rstrip('/')}/chat/completions"
        print(f"[llm] Sending request to {request_url}")
        with urlopen(request, timeout=llm_config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        print("[llm] Response received from LLM provider.")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            "[llm] Provider returned HTTP error: "
            f"provider={llm_config.provider}, "
            f"status={exc.code}, "
            f"url={request.full_url}"
        )
        print("[llm] Raw error body:")
        print(_truncate_for_log(body))
        raise ExternalAPIError(exc.code, body) from exc
    except URLError as exc:
        print(
            "[llm] Provider connection failed: "
            f"provider={llm_config.provider}, "
            f"url={request.full_url}, "
            f"reason={exc.reason}"
        )
        raise RuntimeError(f"Annotation API connection failed: {exc.reason}") from exc
    except SocketTimeout as exc:
        print(
            "[llm] Provider request timed out: "
            f"provider={llm_config.provider}, "
            f"url={request.full_url}, "
            f"timeout={llm_config.timeout_seconds}s"
        )
        raise RuntimeError(
            f"Annotation API request timed out after {llm_config.timeout_seconds} seconds."
        ) from exc

    print("[llm] Raw provider payload:")
    print(json.dumps(response_payload, ensure_ascii=False, indent=2))

    raw_output = extract_chat_completion_text(response_payload)
    print(f"Annotation API raw response ({llm_config.provider}, prompt={active_prompt_mode}):")
    print(raw_output)
    return active_prompt_mode, raw_output


def extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            else:
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(part for part in text_parts if part).strip()

    return str(content or "")


def extract_chat_completion_text(payload: dict) -> str:
    direct_message = payload.get("message")
    if isinstance(direct_message, dict):
        direct_text = extract_message_text(direct_message.get("content"))
        if direct_text:
            return direct_text

    direct_response = payload.get("response")
    if isinstance(direct_response, str) and direct_response.strip():
        return direct_response.strip()

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Annotation API returned no choices. Raw payload: {json.dumps(payload, ensure_ascii=False)}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Annotation API returned an invalid message. Raw payload: {json.dumps(payload, ensure_ascii=False)}")

    content_text = extract_message_text(message.get("content"))
    if content_text:
        return content_text

    for fallback_key in ("text", "response", "output_text"):
        fallback_value = message.get(fallback_key)
        if isinstance(fallback_value, str) and fallback_value.strip():
            return fallback_value.strip()

    delta = choices[0].get("delta")
    if isinstance(delta, dict):
        delta_text = extract_message_text(delta.get("content"))
        if delta_text:
            return delta_text

    reasoning_text = message.get("reasoning")
    finish_reason = choices[0].get("finish_reason")
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        raise ValueError(
            "Annotation API returned empty message content while generating reasoning-only output. "
            f"finish_reason={finish_reason}. Raw payload: {json.dumps(payload, ensure_ascii=False)}"
        )

    raise ValueError(f"Annotation API returned an empty message content. Raw payload: {json.dumps(payload, ensure_ascii=False)}")


def _truncate_for_log(text: str, *, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _build_user_content(user_prompt: str, mime_type: str, image_base64: str) -> list[dict]:
    content: list[dict] = []
    if user_prompt.strip():
        content.append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            },
        }
    )
    return content
