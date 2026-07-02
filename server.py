from __future__ import annotations

# import psycopg2
import base64
import json
import os
import posixpath
import re
import struct
import sys
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from annotation_prompts import get_annotation_prompts


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def load_env_file() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


def clear_invalid_loopback_proxies() -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    cleared_keys: list[str] = []

    for key in proxy_keys:
        value = os.environ.get(key, "").strip()
        if not value:
            continue

        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
        if hostname in {"127.0.0.1", "localhost"} and port == 9:
            os.environ.pop(key, None)
            cleared_keys.append(key)

    if cleared_keys:
        print(
            "Ignored invalid loopback proxy settings for this app: "
            + ", ".join(cleared_keys)
        )


clear_invalid_loopback_proxies()

def resolve_annotation_config() -> tuple[str, str, str, str]:
    provider = os.environ.get("LLM_PROVIDER", "").strip().upper()

    if provider:
        api_key = os.environ.get(f"{provider}_API_KEY", "").strip()
        base_url = os.environ.get(f"{provider}_BASE_URL", "").strip()
        model = os.environ.get(f"{provider}_MODEL", "").strip()

        if not api_key:
            raise RuntimeError(
                f"Missing {provider}_API_KEY for selected LLM_PROVIDER={provider.lower()}."
            )

        return (
            provider.lower(),
            api_key,
            base_url or "https://api.b.ai/v1",
            model or "gpt-5.2",
        )

    legacy_provider = "annotation"
    api_key = (
        os.environ.get("ANNOTATION_API_KEY")
        or os.environ.get("BAI_API_KEY")
        or os.environ.get("MIMO_API_KEY", "")
    ).strip()
    base_url = (
        os.environ.get("ANNOTATION_BASE_URL")
        or os.environ.get("BAI_BASE_URL")
        or os.environ.get("MIMO_BASE_URL")
        or "https://api.b.ai/v1"
    ).strip()
    model = (
        os.environ.get("ANNOTATION_MODEL")
        or os.environ.get("BAI_MODEL")
        or os.environ.get("MIMO_MODEL")
        or "gpt-5.2"
    ).strip()

    return legacy_provider, api_key, base_url, model


ANNOTATION_PROVIDER, ANNOTATION_API_KEY, ANNOTATION_BASE_URL, ANNOTATION_MODEL = resolve_annotation_config()
ANNOTATION_PROMPT_MODE, HOI_SYSTEM_PROMPT, HOI_USER_PROMPT = get_annotation_prompts(
    os.environ.get("ANNOTATION_PROMPT_MODE", "rich")
)


def get_available_llm_providers() -> list[tuple[str, bool]]:
    providers: set[str] = set()

    for key in os.environ:
        if key.endswith("_API_KEY"):
            provider = key[:-8].strip().lower()
            if provider and provider != "annotation":
                providers.add(provider)

    return [
        (provider, bool(os.environ.get(f"{provider.upper()}_API_KEY", "").strip()))
        for provider in sorted(providers)
    ]


def print_llm_provider_options() -> None:
    providers = get_available_llm_providers()
    if not providers:
        print("Selectable LLM_PROVIDER values: none found in environment")
        return

    formatted = []
    for provider, configured in providers:
        label = provider
        if provider == ANNOTATION_PROVIDER:
            label += " [current]"
        label += " [configured]" if configured else " [missing key]"
        formatted.append(label)

    print("Selectable LLM_PROVIDER values: " + ", ".join(formatted))


def choose_llm_provider_interactively() -> None:
    if not sys.stdin.isatty():
        return

    providers = get_available_llm_providers()
    configured_providers = [provider for provider, configured in providers if configured]
    if not configured_providers:
        return

    print("Quick select LLM_PROVIDER:")
    for index, provider in enumerate(configured_providers, start=1):
        current_marker = " [current]" if provider == ANNOTATION_PROVIDER else ""
        print(f"  {index}. {provider}{current_marker}")

    while True:
        choice = input("Choose provider number, or press Enter to keep current: ").strip()
        if not choice:
            return

        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(configured_providers):
                selected_provider = configured_providers[selected_index]
                os.environ["LLM_PROVIDER"] = selected_provider
                return

        print("Invalid selection. Enter a listed number, or press Enter to keep current.")


def annotate_image(image_path: Path, prompt_mode: str | None = None) -> dict:
    if not ANNOTATION_API_KEY:
        raise RuntimeError(
            "Missing LLM API key in .env. Set LLM_PROVIDER with its matching "
            "<PROVIDER>_API_KEY, or keep using the legacy ANNOTATION_API_KEY."
        )

    active_prompt_mode, system_prompt, user_prompt = get_annotation_prompts(
        prompt_mode or ANNOTATION_PROMPT_MODE
    )

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = guess_mime_type(image_path.suffix.lower())

    payload = {
        "model": ANNOTATION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    request = Request(
        url=f"{ANNOTATION_BASE_URL.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {ANNOTATION_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ExternalAPIError(exc.code, body) from exc
    except URLError as exc:
        raise RuntimeError(f"Annotation API connection failed: {exc.reason}") from exc

    content = extract_chat_completion_text(response_payload)
    print(f"Annotation API raw response ({ANNOTATION_PROVIDER}, prompt={active_prompt_mode}):")
    print(content)
    return normalize_annotation_payload(content, image_path)


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


class ExternalAPIError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"External API returned status {status_code}")
        self.status_code = status_code
        self.body = body


def guess_mime_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


def normalize_annotation_payload(content: str, image_path: Path | None = None) -> dict:
    cleaned = clean_json_text(content)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON. Raw content: {cleaned}") from exc

    if not isinstance(data, dict):
        raise ValueError("Model did not return a JSON object.")

    annotation_class = pick_first_value(
        data,
        "class",
        "interaction_label",
        "label",
        "category",
        "object_class",
        "object",
        "name",
    )
    person_class = pick_first_value(
        data,
        "person_class",
        "human_class",
        "subject_class",
    )
    object_class = pick_first_value(
        data,
        "object_class",
        "target_class",
        "item_class",
    )
    bounding_box = pick_first_value(
        data,
        "bounding_box",
        "bbox",
        "box",
        "boundingBox",
        "coordinates",
    )
    person_bounding_box = pick_first_value(
        data,
        "person_bounding_box",
        "person_bbox",
        "human_bounding_box",
        "human_bbox",
        "person_box",
        "human_box",
    )
    object_bounding_box = pick_first_value(
        data,
        "object_bounding_box",
        "object_bbox",
        "object_box",
        "target_bounding_box",
        "target_bbox",
    )
    interaction_bounding_box = pick_first_value(
        data,
        "interaction_bounding_box",
        "interaction_bbox",
        "union_bounding_box",
        "union_bbox",
    )
    sentences = pick_first_value(
        data,
        "data_annotation_sentences",
        "sentences",
        "descriptions",
        "captions",
        "annotation_sentences",
    )

    if isinstance(data.get("annotation"), dict):
        annotation_block = data["annotation"]
        annotation_class = annotation_class or pick_first_value(annotation_block, "class", "interaction_label", "label")
        person_class = person_class or pick_first_value(annotation_block, "person_class", "human_class", "subject_class")
        object_class = object_class or pick_first_value(annotation_block, "object_class", "target_class", "item_class", "object")
        bounding_box = bounding_box or pick_first_value(annotation_block, "bounding_box", "bbox", "box")
        person_bounding_box = person_bounding_box or pick_first_value(
            annotation_block,
            "person_bounding_box",
            "person_bbox",
            "human_bounding_box",
            "human_bbox",
        )
        object_bounding_box = object_bounding_box or pick_first_value(
            annotation_block,
            "object_bounding_box",
            "object_bbox",
            "target_bounding_box",
            "target_bbox",
        )
        interaction_bounding_box = interaction_bounding_box or pick_first_value(
            annotation_block,
            "interaction_bounding_box",
            "interaction_bbox",
            "union_bounding_box",
            "union_bbox",
        )
        sentences = sentences or pick_first_value(
            annotation_block,
            "data_annotation_sentences",
            "sentences",
            "descriptions",
        )

    normalized_person_bbox = normalize_optional_bbox(person_bounding_box, image_path, cleaned, "person_bounding_box")
    normalized_object_bbox = normalize_optional_bbox(object_bounding_box, image_path, cleaned, "object_bounding_box")
    normalized_interaction_bbox = normalize_optional_bbox(
        interaction_bounding_box or bounding_box,
        image_path,
        cleaned,
        "interaction_bounding_box",
    )

    if normalized_interaction_bbox is None:
        if normalized_person_bbox and normalized_object_bbox:
            normalized_interaction_bbox = combine_bounding_boxes(normalized_person_bbox, normalized_object_bbox)
        else:
            raise ValueError(f"Model returned invalid interaction_bounding_box. Raw content: {cleaned}")

    if not isinstance(sentences, list):
        if isinstance(sentences, str) and sentences.strip():
            sentences = [sentences.strip()]
        else:
            raise ValueError(f"Model returned invalid data_annotation_sentences. Raw content: {cleaned}")

    normalized_sentences = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    if not normalized_sentences:
        normalized_sentences = ["No descriptive annotation sentence was returned by the model."]
    person_class = str(person_class or "person").strip() or "person"
    object_class = str(object_class or annotation_class or "unknown").strip() or "unknown"
    if not annotation_class:
        annotation_class = f"{person_class} interacting with {object_class}"

    return {
        "class": annotation_class,
        "person_class": person_class,
        "object_class": object_class,
        "bounding_box": normalized_interaction_bbox,
        "person_bounding_box": normalized_person_bbox,
        "object_bounding_box": normalized_object_bbox,
        "interaction_bounding_box": normalized_interaction_bbox,
        "data_annotation_sentences": normalized_sentences,
    }


def normalize_optional_bbox(
    bounding_box: object,
    image_path: Path | None,
    raw_content: str,
    field_name: str,
) -> list[float] | None:
    if bounding_box is None:
        return None

    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise ValueError(f"Model returned invalid {field_name}. Raw content: {raw_content}")

    return coerce_normalized_bbox(bounding_box, image_path, raw_content)


def combine_bounding_boxes(first_bbox: list[float], second_bbox: list[float]) -> list[float]:
    return [
        round(min(first_bbox[0], second_bbox[0]), 6),
        round(min(first_bbox[1], second_bbox[1]), 6),
        round(max(first_bbox[2], second_bbox[2]), 6),
        round(max(first_bbox[3], second_bbox[3]), 6),
    ]


def coerce_normalized_bbox(
    bounding_box: list[object],
    image_path: Path | None,
    raw_content: str,
) -> list[float]:
    values = [float(value) for value in bounding_box]
    if all(0 <= value <= 1 for value in values):
        return [round(value, 6) for value in values]

    if image_path is None:
        raise ValueError(
            "Model returned non-normalized bounding_box, but image size is unavailable. "
            f"Raw content: {raw_content}"
        )

    image_width, image_height = read_png_dimensions(image_path)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Could not determine image dimensions for bounding_box normalization. "
            f"Raw content: {raw_content}"
        )

    x1, y1, x2, y2 = values
    normalized = [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height,
    ]
    if any(value < 0 or value > 1 for value in normalized):
        raise ValueError(
            "Model returned bounding_box outside image bounds after normalization. "
            f"Raw content: {raw_content}"
        )

    return [round(value, 6) for value in normalized]


def read_png_dimensions(image_path: Path) -> tuple[int, int]:
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()

    if suffix == ".png" and len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height

    if suffix in {".jpg", ".jpeg"}:
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue

            marker = data[index + 1]
            index += 2

            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue

            segment_length = int.from_bytes(data[index:index + 2], "big")
            if segment_length < 2 or index + segment_length > len(data):
                break

            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height = int.from_bytes(data[index + 3:index + 5], "big")
                width = int.from_bytes(data[index + 5:index + 7], "big")
                return width, height

            index += segment_length

    if suffix == ".gif" and len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        return width, height

    if suffix == ".bmp" and len(data) >= 26 and data[:2] == b"BM":
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = abs(int.from_bytes(data[22:26], "little", signed=True))
        return width, height

    if suffix == ".webp" and len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk_type = data[12:16]
        if chunk_type == b"VP8X" and len(data) >= 30:
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return width, height
        if chunk_type == b"VP8 " and len(data) >= 30:
            start_code_index = data.find(b"\x9d\x01\x2a")
            if start_code_index != -1 and start_code_index + 7 < len(data):
                width = struct.unpack("<H", data[start_code_index + 3:start_code_index + 5])[0] & 0x3FFF
                height = struct.unpack("<H", data[start_code_index + 5:start_code_index + 7])[0] & 0x3FFF
                return width, height
        if chunk_type == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height

    raise ValueError(f"Could not read image dimensions from: {image_path.name}")


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
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"Annotation API returned no choices. Raw payload: {json.dumps(payload, ensure_ascii=False)}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Annotation API returned an invalid message. Raw payload: {json.dumps(payload, ensure_ascii=False)}")

    return extract_message_text(message.get("content"))


def clean_json_text(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    return cleaned.strip()


def pick_first_value(data: dict, *keys: str) -> object:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


class UploadHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        normalized = posixpath.normpath(parsed_path)
        parts = [part for part in normalized.split("/") if part]

        target = BASE_DIR
        for part in parts:
            target /= part
        return str(target)

    def do_GET(self) -> None:
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path != "/upload":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return

        prompt_mode = parse_qs(parsed_url.query).get("prompt_mode", [ANNOTATION_PROMPT_MODE])[0]

        try:
            filename, file_bytes = self._extract_uploaded_file()
            safe_name = Path(filename).name
            destination = self._unique_destination(safe_name)

            with destination.open("wb") as output_file:
                output_file.write(file_bytes)

            print(f"Saved upload: {destination.name}")
            annotation = annotate_image(destination, prompt_mode=prompt_mode)
        except ValueError as exc:
            print("Upload error:")
            print(str(exc))
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            print("Internal server error:")
            traceback.print_exc()
            self._send_json(
                {"error": format_exception_message(exc)},
                status=format_exception_status(exc),
            )
            return

        response = {
            "message": "Image uploaded and annotated successfully.",
            "filename": destination.name,
            "uploaded_image_url": f"/uploads/{destination.name}",
            "size_bytes": destination.stat().st_size,
            **annotation,
        }
        self._send_json(response, status=HTTPStatus.OK)

    def _unique_destination(self, filename: str) -> Path:
        candidate = UPLOAD_DIR / filename
        stem = candidate.stem
        suffix = candidate.suffix
        index = 1

        while candidate.exists():
            candidate = UPLOAD_DIR / f"{stem}-{index}{suffix}"
            index += 1

        return candidate

    def _extract_uploaded_file(self) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        boundary_match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
        if not boundary_match:
            raise ValueError("Invalid multipart request.")

        boundary = boundary_match.group(1) or boundary_match.group(2)
        boundary_bytes = f"--{boundary}".encode("utf-8")

        for part in raw_body.split(boundary_bytes):
            if b'name="image"' not in part or b"filename=" not in part:
                continue

            header_block, separator, payload = part.partition(b"\r\n\r\n")
            if not separator:
                continue

            filename_match = re.search(rb'filename="([^"]+)"', header_block)
            if not filename_match:
                continue

            filename = filename_match.group(1).decode("utf-8", errors="ignore")
            file_bytes = payload.rstrip(b"\r\n-")
            if not filename or not file_bytes:
                break

            return filename, file_bytes

        raise ValueError("No image file provided.")

    def _send_json(self, payload: dict, status: HTTPStatus) -> None:
        response_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)
    
    # Foo Yu Jun
    def insert_annotation_to_db(file_name, annotation_data):
        """
        Inserts the file name and JSON annotation data into PostgreSQL.
        annotation_data: The dictionary returned by normalize_annotation_payload().
        """
        try:
            # Establish connection (Update with your local db.sql credentials)
            conn = psycopg2.connect(
                dbname="postgres", 
                user="postgres", 
                password="Zxcvb123!!", 
                host="localhost", 
                port="5432"
            )
            cur = conn.cursor()

            # SQL Insert matching the "Rich" Framework keys [Source 1, 133]
            insert_query = """
            INSERT INTO hoi_annotations (
                uploaded_image_url, 
                class, 
                person_class, 
                object_class, 
                person_bounding_box, 
                object_bounding_box, 
                interaction_bounding_box, 
                data_annotation_sentences
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """

            # Execute with data from the JSON results
            cur.execute(insert_query, (
                file_name, # The name of the image in the /uploads folder [7]
                annotation_data.get("class"),
                annotation_data.get("person_class"),
                annotation_data.get("object_class"),
                annotation_data.get("person_bounding_box"), # [x1, y1, x2, y2]
                annotation_data.get("object_bounding_box"),
                annotation_data.get("interaction_bounding_box"),
                annotation_data.get("data_annotation_sentences") # Exactly 3 sentences [4]
            ))

            conn.commit()
            print(f"Successfully saved {file_name} to database.")
            
            cur.close()
            conn.close()

        except Exception as e:
            print(f"Database Error: {e}")



def run() -> None:
    global ANNOTATION_PROVIDER, ANNOTATION_API_KEY, ANNOTATION_BASE_URL, ANNOTATION_MODEL

    print(f"Current LLM_PROVIDER: {ANNOTATION_PROVIDER}")
    print_llm_provider_options()
    choose_llm_provider_interactively()
    ANNOTATION_PROVIDER, ANNOTATION_API_KEY, ANNOTATION_BASE_URL, ANNOTATION_MODEL = resolve_annotation_config()

    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), UploadHandler)
    print(f"Using LLM_PROVIDER: {ANNOTATION_PROVIDER}")
    print(f"Default ANNOTATION_PROMPT_MODE: {ANNOTATION_PROMPT_MODE}")
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()

# def test_database_connection():
#     try:
#         # Connect using your local PostgreSQL credentials
#         conn = psycopg2.connect(
#             dbname="postgres", # or your specific db name
#             user="postgres", 
#             password="Zxcvb123!!", 
#             host="localhost", 
#             port="5432"
#         )
#         cur = conn.cursor()
        
#         # Test INSERT using the mock data
#         insert_query = """
#         INSERT INTO image_annotations (
#             uploaded_image_url, class, person_class, object_class, 
#             person_bounding_box, object_bounding_box, interaction_bounding_box, 
#             data_annotation_sentences
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
#         """
        
#         cur.execute(insert_query, (
#             "test_image.png",
#             mock_annotation["class"],
#             mock_annotation["person_class"],
#             mock_annotation["object_class"],
#             mock_annotation["person_bounding_box"],
#             mock_annotation["object_bounding_box"],
#             mock_annotation["interaction_bounding_box"],
#             mock_annotation["data_annotation_sentences"]
#         ))
        
#         conn.commit()
#         print("Database Test Successful: Row inserted!")
        
#         cur.close()
#         conn.close()
#     except Exception as e:
#         print(f"Database Test Failed: {e}")

# mock_annotation = {
#     "class": "person holding apple",
#     "person_class": "person",
#     "object_class": "apple",
#     "person_bounding_box": [0.2, 0.1, 0.6, 0.9],
#     "object_bounding_box": [0.4, 0.5, 0.5, 0.6],
#     "interaction_bounding_box": [0.2, 0.1, 0.6, 0.9],
#     "data_annotation_sentences": [
#         "A person is holding a red apple.",
#         "The apple is in the person's right hand.",
#         "The scene is a bright kitchen."
#     ]
# }
# if __name__ == "__main__":
#     test_database_connection()


if __name__ == "__main__":
    run()
