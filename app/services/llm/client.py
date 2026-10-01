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
from app.services.prompts.few_shot_examples import (get_few_shot_examples, get_calibration_example)

def request_image_annotation(
    image_path: Path,
    llm_config: LLMConfig,
    prompt_mode: str | None = None,
) -> tuple[str, str]:
    if not llm_config.api_key:
        raise RuntimeError(
            "Missing LLM API key in .env. Set LLM_PROVIDER with its matching "
            "<PROVIDER>_API_KEY, or keep using the legacy ANNOTATION_API_KEY."
        )

    active_prompt_mode, system_prompt, user_prompt = get_annotation_prompts(prompt_mode)

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = guess_mime_type(image_path.suffix.lower())
    print(
        "[llm] Prepared request payload: "
        f"provider={llm_config.provider}, "
        f"model={llm_config.model}, "
        f"image={image_path.name}, "
        f"bytes={len(image_bytes)}, "
        f"timeout={llm_config.timeout_seconds}s, "
        f"reasoning_effort={llm_config.reasoning_effort or 'default'}"
    )

    # payload = {
    # "model": llm_config.model,
    # "messages": [
    #     {
    #         "role": "system",
    #         "content": system_prompt,
    #     },
    #     {
    #         "role": "user",
    #         "content": _build_user_content(user_prompt, mime_type, image_base64),
    #     },
    # ],
    # "stream": False,
    # "temperature": 0,
    # # "reasoning": {
    # #     "effort": "none"
    # # },
    # }


    ###New for fewshot
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Add few-shot demonstrations
    few_shot_examples = get_few_shot_examples(active_prompt_mode)

    for example_image_path, example_json_path in few_shot_examples:

        # Make sure both files exist
        if not example_image_path.exists():
            raise FileNotFoundError(
                f"Few-shot example image not found: {example_image_path}"
            )

        if not example_json_path.exists():
            raise FileNotFoundError(
                f"Few-shot example JSON not found: {example_json_path}"
            )

        # Encode example image
        example_image_bytes = example_image_path.read_bytes()
        example_image_base64 = base64.b64encode(
            example_image_bytes
        ).decode("utf-8")

        example_mime_type = guess_mime_type(
            example_image_path.suffix.lower()
        )

        # Load the correct ground-truth JSON
        with example_json_path.open("r", encoding="utf-8") as f:
            example_annotation = json.load(f)

        # Example input + correct ground truth
        messages.append(
            {
                "role": "user",
                "content": _build_reference_content(
                    example_annotation,
                    example_mime_type,
                    example_image_base64,
                ),
            }
        )

    print(
        f"[llm] Added {len(few_shot_examples)} "
        f"few-shot examples for mode={active_prompt_mode}"
    )

    # Load calibration image and ground-truth JSON
    calibration_image_path, calibration_json_path = get_calibration_example(
        active_prompt_mode
    )

    if not calibration_image_path.exists():
        raise FileNotFoundError(
            f"Calibration image not found: {calibration_image_path}"
        )

    if not calibration_json_path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {calibration_json_path}"
        )

    calibration_image_bytes = calibration_image_path.read_bytes()
    calibration_image_base64 = base64.b64encode(
        calibration_image_bytes
    ).decode("utf-8")

    calibration_mime_type = guess_mime_type(
        calibration_image_path.suffix.lower()
    )

    with calibration_json_path.open("r", encoding="utf-8") as f:
        calibration_ground_truth = json.load(f)

    # First request:
    # ask the LLM to predict the calibration image
    # WITHOUT showing the ground truth.
    calibration_messages = messages.copy()

    calibration_messages.append(
        {
            "role": "user",
            "content": _build_user_content(
                (
                    "CALIBRATION IMAGE\n"
                    "Generate the annotation for this image using the same "
                    "format and rules as the reference examples. "
                    "Return JSON only."
                ),
                calibration_mime_type,
                calibration_image_base64,
            ),
        }
    )

    print("[llm] Sending calibration image for prediction.")

    calibration_response_payload = _send_llm_request(
        calibration_messages,
        llm_config,
    )

    calibration_prediction = extract_chat_completion_text(
        calibration_response_payload
    )

    print("[llm] Calibration prediction:")
    print(calibration_prediction)

    # Second request:
    # give the model its previous prediction + real ground truth
    # before giving the final end-user image.
    final_messages = messages.copy()

    final_messages.append(
        {
            "role": "user",
            "content": _build_calibration_feedback_content(
                calibration_prediction=calibration_prediction,
                calibration_ground_truth=calibration_ground_truth,
                mime_type=calibration_mime_type,
                image_base64=calibration_image_base64,
            ),
        }
    )

    # Finally add the actual image that needs to be annotated
    final_messages.append(
        {
            "role": "user",
            "content": _build_user_content(
                (
                    f"{user_prompt}\n\n"
                    "This is the final image to annotate. "
                    "Use the reference examples and the calibration correction "
                    "above when producing the annotation."
                ),
                mime_type,
                image_base64,
            ),
        }
    )

    print("[llm] Sending final end-user image.")

    response_payload = _send_llm_request(
        final_messages,
        llm_config,
    )

    raw_output = extract_chat_completion_text(response_payload)

    print(
        f"Annotation API raw response "
        f"({llm_config.provider}, prompt={active_prompt_mode}):"
    )
    print(raw_output)

    return active_prompt_mode, raw_output

    ##Commented out to test few shot##
    # # Finally add the actual image that needs to be annotated
    # messages.append(
    #     {
    #         "role": "user",
    #         "content": _build_user_content(
    #             user_prompt,
    #             mime_type,
    #             image_base64,
    #         ),
    #     }
    # )

    # payload = {
    #     "model": llm_config.model,
    #     "messages": messages,
    #     "stream": False,
    #     "temperature": 0,
    # }
    # if llm_config.reasoning_effort:
    #     payload["reasoning"] = {"effort": llm_config.reasoning_effort}
    # ###



    

    # if llm_config.provider == "ollama":
    #     # Prefer direct answer content over reasoning-only output for local Qwen models.
    #     payload["think"] = False

    # request = Request(
    #     url=f"{llm_config.base_url.rstrip('/')}/chat/completions",
    #     data=json.dumps(payload).encode("utf-8"),
    #     headers={
    #         "Authorization": f"Bearer {llm_config.api_key}",
    #         "Content-Type": "application/json",
    #     },
    #     method="POST",
    # )

    # try:
    #     request_url = f"{llm_config.base_url.rstrip('/')}/chat/completions"
    #     print(f"[llm] Sending request to {request_url}")
    #     with urlopen(request, timeout=llm_config.timeout_seconds) as response:
    #         response_payload = json.loads(response.read().decode("utf-8"))
    #     print("[llm] Response received from LLM provider.")
    # except HTTPError as exc:
    #     body = exc.read().decode("utf-8", errors="replace")
    #     print(
    #         "[llm] Provider returned HTTP error: "
    #         f"provider={llm_config.provider}, "
    #         f"status={exc.code}, "
    #         f"url={request.full_url}"
    #     )
    #     print("[llm] Raw error body:")
    #     print(_truncate_for_log(body))
    #     raise ExternalAPIError(exc.code, body) from exc
    # except URLError as exc:
    #     print(
    #         "[llm] Provider connection failed: "
    #         f"provider={llm_config.provider}, "
    #         f"url={request.full_url}, "
    #         f"reason={exc.reason}"
    #     )
    #     raise RuntimeError(f"Annotation API connection failed: {exc.reason}") from exc
    # except SocketTimeout as exc:
    #     print(
    #         "[llm] Provider request timed out: "
    #         f"provider={llm_config.provider}, "
    #         f"url={request.full_url}, "
    #         f"timeout={llm_config.timeout_seconds}s"
    #     )
    #     raise RuntimeError(
    #         f"Annotation API request timed out after {llm_config.timeout_seconds} seconds."
    #     ) from exc

    # print("[llm] Raw provider payload:")
    # print(json.dumps(response_payload, ensure_ascii=False, indent=2))

    # raw_output = extract_chat_completion_text(response_payload)
    # print(f"Annotation API raw response ({llm_config.provider}, prompt={active_prompt_mode}):")
    # print(raw_output)
    # return active_prompt_mode, raw_output

def _send_llm_request(
    messages: list[dict],
    llm_config: LLMConfig,
) -> dict:

    payload = {
        "model": llm_config.model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
    }

    if llm_config.reasoning_effort:
        payload["reasoning"] = {
            "effort": llm_config.reasoning_effort
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

        with urlopen(
            request,
            timeout=llm_config.timeout_seconds,
        ) as response:
            response_payload = json.loads(
                response.read().decode("utf-8")
            )

        print("[llm] Response received from LLM provider.")

        return response_payload

    except HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        print(
            "[llm] Provider returned HTTP error: "
            f"provider={llm_config.provider}, "
            f"status={exc.code}, "
            f"url={request.full_url}"
        )

        print("[llm] Raw error body:")
        print(_truncate_for_log(body))

        raise ExternalAPIError(
            exc.code,
            body,
        ) from exc

    except URLError as exc:
        print(
            "[llm] Provider connection failed: "
            f"provider={llm_config.provider}, "
            f"url={request.full_url}, "
            f"reason={exc.reason}"
        )

        raise RuntimeError(
            f"Annotation API connection failed: {exc.reason}"
        ) from exc

    except SocketTimeout as exc:
        print(
            "[llm] Provider request timed out: "
            f"provider={llm_config.provider}, "
            f"url={request.full_url}, "
            f"timeout={llm_config.timeout_seconds}s"
        )

        raise RuntimeError(
            f"Annotation API request timed out after "
            f"{llm_config.timeout_seconds} seconds."
        ) from exc

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

# For few-shot reference examples
def _build_reference_content(
    example_annotation: dict,
    mime_type: str,
    image_base64: str,
) -> list[dict]:

    return [
        {
            "type": "text",
            "text": (
                "REFERENCE EXAMPLE\n"
                "Study the following image and its correct annotation."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            },
        },
        {
            "type": "text",
            "text": (
                "The correct annotation for the reference image is:\n"
                f"{json.dumps(example_annotation, ensure_ascii=False)}"
            ),
        },
    ]


def _build_calibration_feedback_content(
    calibration_prediction: str,
    calibration_ground_truth: dict,
    mime_type: str,
    image_base64: str,
) -> list[dict]:

    return [
        {
            "type": "text",
            "text": (
                "CALIBRATION FEEDBACK\n"
                "The following is the calibration image that you "
                "previously annotated."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            },
        },
        {
            "type": "text",
            "text": (
                "Your previous prediction was:\n"
                f"{calibration_prediction}\n\n"
                "The correct ground-truth annotation is:\n"
                f"{json.dumps(calibration_ground_truth, ensure_ascii=False)}\n\n"
                "Compare your previous prediction with the correct annotation. "
                "Use the correction as an additional reference when annotating "
                "the next image. Do not output anything for this calibration "
                "image again."
            ),
        },
    ]
