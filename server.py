from __future__ import annotations

import base64
import json
import os
import posixpath
import re
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI


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

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")

client = OpenAI(
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
)


def annotate_image_with_mimo(image_path: Path) -> dict:
    if not MIMO_API_KEY:
        raise RuntimeError("Missing MIMO_API_KEY in .env.")

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = guess_mime_type(image_path.suffix.lower())

    completion = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an image annotation assistant. "
                    "You must return only one valid JSON object and nothing else. "
                    "Do not include markdown, explanations, headings, or code fences. "
                    "The JSON object must use exactly these keys: "
                    "\"class\", \"bounding_box\", \"data_annotation_sentences\". "
                    "\"class\" must be a string. "
                    "\"bounding_box\" must be an array of four integers in the format [x1, y1, x2, y2]. "
                    "\"data_annotation_sentences\" must be an array of 2 to 4 short descriptive strings."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image and return exactly one JSON object. "
                            "Use this exact format and these exact keys only: "
                            "{\"class\":\"object_name\","
                            "\"bounding_box\":[12,34,220,410],"
                            "\"data_annotation_sentences\":[\"sentence 1\",\"sentence 2\"]}. "
                            "If the object is unclear, still provide your best guess. "
                            "Do not rename keys. Do not wrap the JSON in markdown. "
                            "Do not return any extra keys."
                        ),
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
        max_completion_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        stream=False,
    )

    content = extract_message_text(completion.choices[0].message.content)
    print("MiMo raw response:")
    print(content)
    return normalize_annotation_payload(content)


def guess_mime_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


def normalize_annotation_payload(content: str) -> dict:
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
        "label",
        "category",
        "object_class",
        "object",
        "name",
    )
    bounding_box = pick_first_value(
        data,
        "bounding_box",
        "bbox",
        "box",
        "boundingBox",
        "coordinates",
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
        annotation_class = annotation_class or pick_first_value(annotation_block, "class", "label", "object")
        bounding_box = bounding_box or pick_first_value(annotation_block, "bounding_box", "bbox", "box")
        sentences = sentences or pick_first_value(
            annotation_block,
            "data_annotation_sentences",
            "sentences",
            "descriptions",
        )

    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise ValueError(f"Model returned invalid bounding_box. Raw content: {cleaned}")

    normalized_bbox = [int(value) for value in bounding_box]

    if not isinstance(sentences, list):
        if isinstance(sentences, str) and sentences.strip():
            sentences = [sentences.strip()]
        else:
            raise ValueError(f"Model returned invalid data_annotation_sentences. Raw content: {cleaned}")

    normalized_sentences = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
    if not normalized_sentences:
        normalized_sentences = ["No descriptive annotation sentence was returned by the model."]
    if not annotation_class:
        annotation_class = "unknown"

    return {
        "class": annotation_class,
        "bounding_box": normalized_bbox,
        "data_annotation_sentences": normalized_sentences,
    }


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
        if self.path != "/upload":
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return

        try:
            filename, file_bytes = self._extract_uploaded_file()
            safe_name = Path(filename).name
            destination = self._unique_destination(safe_name)

            with destination.open("wb") as output_file:
                output_file.write(file_bytes)

            print(f"Saved upload: {destination.name}")
            annotation = annotate_image_with_mimo(destination)
        except ValueError as exc:
            print("Upload error:")
            print(str(exc))
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            print("Internal server error:")
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
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


def run() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), UploadHandler)
    print(f"Server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
