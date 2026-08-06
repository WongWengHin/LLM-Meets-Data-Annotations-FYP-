"""*** 应用配置加载与路径解析相关的核心辅助逻辑。 ***"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class ImagePreprocessConfig:
    enable_resize: bool
    max_side: int
    jpeg_quality: int


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    frontend_dir: Path
    data_dir: Path
    db_path: Path
    uploads_dir: Path
    inference_images_dir: Path
    raw_outputs_dir: Path
    master_annotations_dir: Path
    predictions_dir: Path
    ground_truth_dir: Path
    evaluation_exports_dir: Path
    normalized_outputs_dir: Path
    validation_reports_dir: Path
    orchestration_traces_dir: Path
    task_states_dir: Path


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    image_preprocess: ImagePreprocessConfig
    paths: AppPaths
    host: str
    port: int


def load_env_file(base_dir: Path) -> None:
    env_path = base_dir / ".env"
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


def clear_invalid_loopback_proxies() -> None:
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

    for key in proxy_keys:
        value = os.environ.get(key, "").strip()
        if not value:
            continue

        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
        if hostname in {"127.0.0.1", "localhost"} and port == 9:
            os.environ.pop(key, None)


def build_app_paths(base_dir: Path) -> AppPaths:
    data_dir = base_dir / "data"
    paths = AppPaths(
        base_dir=base_dir,
        frontend_dir=base_dir / "frontend",
        data_dir=data_dir,
        db_path=data_dir / "annotations.db",
        uploads_dir=data_dir / "uploads",
        inference_images_dir=data_dir / "inference_images",
        raw_outputs_dir=data_dir / "raw_outputs",
        master_annotations_dir=data_dir / "master_annotations",
        predictions_dir=data_dir / "predictions",
        ground_truth_dir=data_dir / "ground_truth",
        evaluation_exports_dir=data_dir / "evaluation_exports",
        normalized_outputs_dir=data_dir / "normalized_outputs",
        validation_reports_dir=data_dir / "validation_reports",
        orchestration_traces_dir=data_dir / "orchestration_traces",
        task_states_dir=data_dir / "task_states",
    )

    for path in (
        paths.data_dir,
        paths.uploads_dir,
        paths.inference_images_dir,
        paths.raw_outputs_dir,
        paths.master_annotations_dir,
        paths.predictions_dir,
        paths.ground_truth_dir,
        paths.evaluation_exports_dir,
        paths.normalized_outputs_dir,
        paths.validation_reports_dir,
        paths.orchestration_traces_dir,
        paths.task_states_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return paths


def parse_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value for {name}: {raw_value}") from exc

    if parsed_value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")

    return parsed_value


def parse_env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name, "").strip().lower()
    if not raw_value:
        return default

    if raw_value in {"1", "true", "yes", "on"}:
        return True

    if raw_value in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(f"Invalid boolean value for {name}: {raw_value}")


def resolve_llm_config() -> LLMConfig:
    provider = os.environ.get("LLM_PROVIDER", "").strip().upper()

    if provider:
        api_key = os.environ.get(f"{provider}_API_KEY", "").strip()
        base_url = os.environ.get(f"{provider}_BASE_URL", "").strip()
        model = os.environ.get(f"{provider}_MODEL", "").strip()
        max_tokens = parse_env_int(f"{provider}_MAX_TOKENS", 512)
        timeout_seconds = parse_env_int(f"{provider}_TIMEOUT_SECONDS", 300)

        if not api_key:
            raise RuntimeError(
                f"Missing {provider}_API_KEY for selected LLM_PROVIDER={provider.lower()}."
            )

        return LLMConfig(
            provider=provider.lower(),
            api_key=api_key,
            base_url=base_url or "https://api.b.ai/v1",
            model=model or "gpt-5.2",
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

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
    max_tokens = parse_env_int("ANNOTATION_MAX_TOKENS", 512)
    timeout_seconds = parse_env_int("ANNOTATION_TIMEOUT_SECONDS", 300)

    return LLMConfig(
        provider="annotation",
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def resolve_image_preprocess_config() -> ImagePreprocessConfig:
    jpeg_quality = parse_env_int("IMAGE_JPEG_QUALITY", 85)
    if jpeg_quality > 100:
        raise RuntimeError("IMAGE_JPEG_QUALITY must be between 1 and 100.")

    return ImagePreprocessConfig(
        enable_resize=parse_env_bool("ENABLE_IMAGE_RESIZE", True),
        max_side=parse_env_int("IMAGE_MAX_SIDE", 768),
        jpeg_quality=jpeg_quality,
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


def print_llm_provider_options(current_provider: str) -> None:
    providers = get_available_llm_providers()
    if not providers:
        print("Selectable LLM_PROVIDER values: none found in environment")
        return

    formatted = []
    for provider, configured in providers:
        label = provider
        if provider == current_provider:
            label += " [current]"
        label += " [configured]" if configured else " [missing key]"
        formatted.append(label)

    print("Selectable LLM_PROVIDER values: " + ", ".join(formatted))


def choose_llm_provider_interactively(current_provider: str) -> None:
    if not sys.stdin.isatty():
        return

    providers = get_available_llm_providers()
    configured_providers = [provider for provider, configured in providers if configured]
    if not configured_providers:
        return

    print("Quick select LLM_PROVIDER:")
    for index, provider in enumerate(configured_providers, start=1):
        current_marker = " [current]" if provider == current_provider else ""
        print(f"  {index}. {provider}{current_marker}")

    while True:
        choice = input("Choose provider number, or press Enter to keep current: ").strip()
        if not choice:
            return

        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(configured_providers):
                os.environ["LLM_PROVIDER"] = configured_providers[selected_index]
                return

        print("Invalid selection. Enter a listed number, or press Enter to keep current.")


def load_app_config(base_dir: Path) -> AppConfig:
    load_env_file(base_dir)
    clear_invalid_loopback_proxies()
    paths = build_app_paths(base_dir)
    llm = resolve_llm_config()
    image_preprocess = resolve_image_preprocess_config()
    return AppConfig(
        llm=llm,
        image_preprocess=image_preprocess,
        paths=paths,
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8000")),
    )
