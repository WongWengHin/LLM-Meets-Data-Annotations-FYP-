from __future__ import annotations

from unittest.mock import patch

from app.core.config import resolve_llm_config


def test_resolve_llm_config_reads_provider_reasoning_effort() -> None:
    with patch.dict(
        "os.environ",
        {
            "LLM_PROVIDER": "bai",
            "BAI_API_KEY": "test-key",
            "BAI_BASE_URL": "https://example.invalid/v1",
            "BAI_MODEL": "gpt-5.4",
            "BAI_REASONING_EFFORT": "high",
        },
        clear=True,
    ):
        config = resolve_llm_config()

    assert config.provider == "bai"
    assert config.model == "gpt-5.4"
    assert config.reasoning_effort == "high"
