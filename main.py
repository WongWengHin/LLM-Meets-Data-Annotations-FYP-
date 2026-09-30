"""*** FastAPI 标注服务的应用启动入口。 ***"""

from __future__ import annotations

from pathlib import Path

import uvicorn

from app.core.config import choose_llm_provider_interactively, load_app_config, print_llm_provider_options
from app.api.app import create_app


def run_server() -> None:
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
    run_server()


# run COCO evaluation
# python evaluation_new/COCO/coco_evaluator.py --database-image-id IMAGE_ID
# python evaluation_new/COCO/coco_evaluator.py --batch-id 0c5cec2e-d3bf-40c6-aa7a-421d2fc05a98

# run HICODET evaluation
# python evaluation_new/HICO-DET/hicodet_evaluator.py --image-id HICO_train2015_00000076
# python evaluation_new/HICO-DET/hicodet_evaluator.py --batch-id YOUR_BATCH_ID