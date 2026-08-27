# Large Language Meets Data Annotation

## Overview

This project is my Final Year Project (FYP), exploring how large language models and multimodal models can support image-based data annotation.

The system provides a web interface where a user uploads an image, the backend runs an annotation pipeline, and the application returns structured JSON output containing:

- interaction class
- person, object, and interaction bounding boxes
- annotation sentences

The project is designed not only as a frontend demo, but also as a research-oriented prototype for schema-based annotation generation, normalization, validation, and future evaluation.

## Objectives

The main goals of this project are:

- explore how LLM/VLM systems can assist automated data annotation
- generate structured annotation outputs from uploaded images
- design a professional annotation pipeline beyond prompt-only output
- support future evaluation with prediction files, ground truth files, and evaluation exports
- support a research-grade JSON schema for evaluation-ready annotation records

## Current Features

The current prototype supports:

- image upload through a browser-based interface
- prompt-driven multimodal annotation generation
- structured JSON return format
- normalized bounding box handling
- annotation sentence generation
- normalization and validation of model outputs
- master annotation schema generation
- export-oriented backend architecture for future evaluation tasks

## System Architecture

The backend now follows a `single orchestrator agent + deterministic tools` design.

High-level flow:

User Upload Image  
-> Annotation Agent  
-> Image Preprocess Tool  
-> LLM Annotation Tool  
-> Normalizer Tool  
-> Validator Tool  
-> Master Annotation Schema  
-> Frontend Response / Prediction File / Evaluation Export / Trace Artifacts

The `Annotation Agent` is responsible for:

- selecting the active annotation strategy
- retrying bounded recoverable failures
- escalating unresolved results to review
- writing task state and orchestration trace artifacts

This design separates:

- orchestration and strategy control
- model generation
- data normalization
- data validation
- schema construction
- output export
- traceability and review escalation

## Schema v1.0

The current backend uses a research-grade `JSON schema v1.0`.

### Master annotation schema

The master annotation record is the source of truth and contains:

- `schema_version`
- `image.image_id`, `file_name`, `width`, `height`
- `annotation.annotation_id`, `task`, `class`, `person_class`, `object_class`
- `person_bounding_box`, `object_bounding_box`, `interaction_bounding_box`
- `data_annotation_sentences`
- `generation.provider`, `generation.model`, `generation.prompt_mode`
- `validation.is_valid`, `validation.errors`, `validation.warnings`
- `review.status`, `review.reviewed_by`, `review.reviewed_at`
- `source.type`
- `timestamps.created_at`

### ID rules

- `image_id` uses the uploaded image stem
- `annotation_id` uses `<image_id>_001`
- IDs are stored as strings across master, prediction, and ground truth records

### Review lifecycle

- `unreviewed`
  Default state for model-generated records

- `approved`
  Human-reviewed records that can be exported as ground truth

- `rejected`
  Records excluded from evaluation export and ground truth export

### Export purposes

- `frontend response`
  Lightweight response used by the web UI

- `prediction file`
  Model output record for experiment tracking and future evaluation

- `ground truth file`
  Export for human-reviewed approved records only

- `evaluation export`
  Two evaluation blocks for this phase:
  `bbox_export` and `text_export`

## Project Structure

```text
FIT3163/
  main.py
  app/
  frontend/
  samples/
  scripts/
  data/
```

### Key folders

- `main.py`
  Project entry point

- `frontend/`
  Frontend HTML, CSS, and JavaScript files

- `app/`
  Layered application modules:
  `app/api/` for FastAPI wiring,
  `app/agent/` for annotation orchestration, task state, policies, tool wrappers, and trace writing,
  `app/core/` for shared configuration and error types,
  `app/domain/images/` and `app/domain/annotations/` for core domain logic,
  `app/services/llm/` and `app/services/prompts/` for external model access and prompt templates,
  `app/evaluation/` for offline COCO and HICO-DET aligned evaluation

- `scripts/`
  Local helper scripts such as the PowerShell startup shortcut

- `samples/`
  Example images and XML annotations kept for manual inspection or demos

- `data/`
  Runtime output directory for uploads, generated JSON artifacts, and the active `user_results.db` SQLite database

## How to Run

### 1. Requirements

- Python 3.10+
- A valid API key for the selected provider

### 2. Install dependencies

Install the FastAPI backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file based on `.env_example`.

Example:

```env
LLM_PROVIDER=bai

BAI_API_KEY=your_bai_api_key_here
BAI_BASE_URL=https://api.b.ai/v1
BAI_MODEL=gpt-5.2

PORT=8000
```

You may also configure other providers already listed in `.env_example`, such as OpenAI, Anthropic, MINMAX, or MIMO.

For Google sign-in, add:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI` like `http://127.0.0.1:8000/auth/google/callback`
- `SESSION_SECRET_KEY`
- `REQUIRE_GOOGLE_LOGIN=true` if you want to block uploads until login succeeds

The application now persists Google-account-owned uploads and annotation runs in `data/user_results.db`. The older `annotations.db` path has been removed from the startup flow.

### 4. Start the project

```powershell
python main.py
```

The backend now runs on FastAPI and is started through `uvicorn` inside `main.py`.

Windows PowerShell shortcut:

```powershell
.\scripts\start_server.ps1
```

Offline evaluation shortcut:

```powershell
.\scripts\run_eval.ps1 --dataset coco --predictions-dir data/predictions --annotations path\to\coco_annotations.json --output-dir evaluation_reports\coco
```

Recommended first-run sequence:

```powershell
Copy-Item .env_example .env
python -m pip install -r requirements.txt
.\scripts\start_server.ps1
```

### 5. Open the application

After the server starts, open:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /`
  Frontend page
- `GET /health`
  FastAPI health check
- `POST /upload`
  Image upload and annotation endpoint with documented response schema in FastAPI OpenAPI docs
- `GET /docs`
  Swagger UI for trying the API in the browser

### 5.1 Run offline evaluation

The repository also includes an offline IoU evaluation module for COCO and HICO-DET style evaluation.
HICO-DET reports triplet-level precision, recall, F1, and `triplet mean AP@IoU=0.5` so the detection summary stays distinct from COCO mAP.

For semantic analysis, you can convert the manually annotated `HICO_sentences.py` fixture into JSON and run the separate HICO semantic evaluation script:

```powershell
python scripts/convert_hico_sentences.py path\to\HICO_sentences.py data\hico_semantic_fixture.json
python scripts/run_hico_semantic_eval.py --predictions-dir data\predictions --semantic-reference data\hico_semantic_fixture.json --output-dir evaluation_reports\hicodet_semantic
```

You can also merge semantic results into the main HICO-DET report:

```powershell
python -m app.evaluation.run_eval --dataset hicodet --predictions-dir data\predictions --annotations path\to\hicodet_annotations.json --output-dir evaluation_reports\hicodet --semantic-reference data\hico_semantic_fixture.json
```

Example COCO run:

```powershell
python -m app.evaluation.run_eval --dataset coco --predictions-dir data/predictions --annotations path\to\coco_annotations.json --output-dir evaluation_reports\coco
```

Example HICO-DET run:

```powershell
python -m app.evaluation.run_eval --dataset hicodet --predictions-dir data/predictions --annotations path\to\hicodet_annotations.json --output-dir evaluation_reports\hicodet
```

Optional label mapping:

```powershell
python -m app.evaluation.run_eval --dataset coco --predictions-dir data/predictions --annotations path\to\coco_annotations.json --output-dir evaluation_reports\coco --label-mapping app\evaluation\label_mappings.example.json
```

The evaluator writes:

- summary json
- csv breakdown files
- markdown summary
- per-image debug match reports
- mapping-aware category or verb/object id assignment when a label mapping file is provided

### 6. Use the demo

- upload an image
- click `Upload and Annotate`
- inspect the returned JSON output in the interface
- review the agent strategy, attempt count, and review escalation status in the workspace

## Input and Output

### Input

- user-uploaded image

### Output

The system returns structured JSON containing fields such as:

- `class`
- `person_class`
- `object_class`
- `person_bounding_box`
- `object_bounding_box`
- `interaction_bounding_box`
- `data_annotation_sentences`

## Output Files

Generated files are stored in the `data/` directory:

- `data/uploads/`
  Uploaded images

- `data/inference_images/`
  Resized inference copies used by the agent when preprocessing is enabled

- `data/raw_outputs/`
  Raw model responses

- `data/normalized_outputs/`
  Normalized intermediate annotation payloads

- `data/validation_reports/`
  Validation results recorded per run

- `data/master_annotations/`
  Master annotation records

- `data/predictions/`
  Prediction-oriented exports

- `data/evaluation_exports/`
  Evaluation-oriented exports

- `data/orchestration_traces/`
  Agent decision traces including attempts, strategy changes, and final outcome

- `data/task_states/`
  Serialized task state snapshots for debugging and reproducibility

Ground truth support is part of the architecture, but the full human-reviewed ground truth workflow is still under development.

The current export behavior is:

- prediction exports are written from validated master annotations
- evaluation exports are written from validated master annotations unless a record is marked `rejected`
- ground truth exports are only valid for records marked `approved`
- review-required runs keep their trace and task state artifacts even when final export quality is unresolved

## Current Limitations

This project is still a prototype, so several limitations remain:

- annotation quality still depends partly on model output quality
- no complete human-reviewed ground truth dataset is included yet
- confidence scoring is currently nullable because the model pipeline does not return a reliable score
- evaluation workflow is still being expanded
- human review still happens outside the current web interface

## Future Work

Planned future extensions include:

- improving the master schema design further
- building a human-reviewed ground truth dataset
- adding quantitative evaluation metrics
- comparing prompt settings and model output quality
- improving robustness across different image types

## Research Relevance

This project sits at the intersection of:

- large language models
- multimodal understanding
- structured data annotation
- evaluation-ready schema design

It is intended as both a demo system and an experimental foundation for studying how LLM-based systems can contribute to automated annotation workflows.
