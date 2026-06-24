from __future__ import annotations

from pathlib import Path

import uvicorn

from core.config import choose_llm_provider_interactively, load_app_config, print_llm_provider_options
from web.app import create_app


def run() -> None:
    base_dir = Path(__file__).resolve().parent
    config = load_app_config(base_dir)

    print(f"Current LLM_PROVIDER: {config.llm.provider}")
    print_llm_provider_options(config.llm.provider)
    choose_llm_provider_interactively(config.llm.provider)
    config = load_app_config(base_dir)
    app = create_app(config)
    print(f"Using LLM_PROVIDER: {config.llm.provider}")
    print("Prompt mode: standard")
    print(f"Server running at http://{config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    run()
