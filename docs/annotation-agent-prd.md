# Product Requirements Document

## Title

Annotation Agent PRD for Image-Based Data Annotation System

## Document Status

Draft v1.0

## Background

The current project is a FastAPI-based image annotation prototype. A user uploads an image, the backend runs a fixed pipeline, and the system returns structured annotation JSON containing interaction labels, bounding boxes, and annotation sentences.

The current backend is strong in schema design, normalization, validation, and export generation. However, the execution flow is still a deterministic one-pass pipeline. It does not yet support adaptive decision-making such as deciding when to retry prompting, when to escalate to review, or how to recover from partial failures.

To improve controllability, extensibility, and research value, the next version of the system will introduce a single orchestration agent called the Annotation Agent. This agent will coordinate a set of deterministic tools rather than replacing them.

## Problem Statement

The existing pipeline has the following limitations:

- it follows a fixed execution path for every image
- it cannot retry intelligently when model output fails normalization or validation
- it cannot distinguish between recoverable model failures and records that should be sent for human review
- it does not maintain explicit task state across processing stages
- it is difficult to compare orchestration strategies for research experiments

## Product Vision

Build an agent-oriented annotation system in which a single Annotation Agent plans and coordinates the annotation workflow while invoking existing backend modules as deterministic tools.

The system should remain stable and evaluation-friendly, while becoming more adaptive, explainable, and easier to extend with review and experimentation workflows.

## Goals

- convert the current one-pass pipeline into a stateful orchestrated workflow
- preserve deterministic normalization, validation, schema construction, and export behavior
- allow adaptive execution decisions per image
- support automatic retry and fallback policies
- support explicit human-review escalation for uncertain or invalid cases
- make orchestration decisions traceable for research and debugging
- minimize disruption to the current codebase

## Non-Goals

- replacing all business logic with a free-form autonomous agent
- introducing multiple independent agents in v1
- building a full human review UI in the first phase
- changing the existing master schema in ways that break current export compatibility
- optimizing for open-ended conversation with end users

## Target Users

- student researcher or project owner running annotation experiments
- developer maintaining the backend pipeline
- evaluator or reviewer inspecting generated annotation outputs
- future human annotator who reviews low-confidence or invalid results

## Core Product Concept

The system will introduce one orchestration layer:

- `Annotation Agent`

The agent will decide:

- which prompt mode to use
- whether to retry after malformed or invalid model output
- whether to fall back to another strategy
- whether to mark the record for human review

The agent will call the following deterministic tools:

- image preprocess tool
- llm annotation tool
- normalization tool
- validation tool
- schema builder tool
- export tool

## Proposed User Flow

1. User uploads an image from the web interface.
2. Backend creates an annotation task.
3. Annotation Agent reads task context and image metadata.
4. Agent decides an execution plan.
5. Agent invokes tools step by step.
6. If output is invalid, the agent retries or changes strategy according to policy.
7. If the result still cannot pass quality gates, the agent marks it for human review.
8. System returns structured output plus orchestration metadata.
9. Artifacts are stored for debugging, evaluation, and future review.

## Functional Requirements

### FR1. Task-Based Execution

The system must create a task record for every uploaded image.

Each task should include:

- task id
- image path
- current stage
- current status
- attempt count
- selected strategy
- tool execution history
- final outcome

### FR2. Single Orchestration Agent

The system must provide one Annotation Agent responsible for workflow control.

The agent must not directly implement normalization, validation, or exporting logic. It must call these as tools.

### FR3. Tool Invocation

The agent must be able to invoke:

- image preprocessing
- LLM/VLM annotation generation
- output normalization
- annotation validation
- master schema construction
- frontend and evaluation exports

### FR4. Strategy Selection

The agent must choose an execution strategy for each task.

Initial supported strategies:

- `llm_only`
- `retry_llm_only`
- `review_required`

### FR5. Retry Policy

The agent must retry when failures are potentially recoverable.

Recoverable conditions include:

- invalid JSON
- missing required fields
- malformed bounding boxes
- empty or unusable annotation payload

Retry policy for v1:

- maximum 2 model-generation attempts per image
- all attempts must be logged

### FR6. Validation Gate

The agent must treat validation as a hard gate before schema export.

If validation fails after all retry attempts:

- the task must be marked as `review_required` or `failed`
- the final response must clearly indicate that the annotation is not production-ready

### FR7. Review Escalation

The system must support review escalation without requiring a full review UI in v1.

Escalated records must include:

- review status
- reason for escalation
- failed validation messages
- attempt history

### FR8. Orchestration Trace

The system must store agent decision metadata for each task.

The trace should include:

- selected strategy
- prompt mode used
- retry count
- validation outcomes per attempt
- final disposition

### FR9. Backward-Compatible Output

The frontend response must remain compatible with the current client where possible.

Additional metadata may be appended, but existing fields should not be removed in v1.

### FR10. Artifact Persistence

The system must persist per-task artifacts for reproducibility.

Artifacts should include:

- raw model outputs
- normalized intermediate data
- validation reports
- final master annotation
- prediction export
- evaluation export
- orchestration trace

## Quality Requirements

### Reliability

- deterministic tools must remain deterministic for the same input
- agent retries must follow clear, bounded policy rules
- partial failures must not crash the whole service without an error record

### Explainability

- every major agent decision must be inspectable in logs or artifacts
- failures must be understandable by developers and researchers

### Maintainability

- orchestration logic must be isolated from tool logic
- existing modules should be reused rather than rewritten

### Extensibility

- new prompt modes should be pluggable
- future detectors should be swappable
- human review workflow should be attachable later

## Success Metrics

Primary metrics:

- percentage of tasks that complete successfully without manual intervention
- percentage of invalid first-pass outputs successfully recovered by retry
- percentage of tasks escalated to review
- average processing time per image

Research metrics:

- validation pass rate on first attempt versus second attempt
- reduction in malformed output frequency after orchestration

## MVP Scope

The MVP should include:

- one Annotation Agent class or orchestration service
- task state model
- tool wrappers around existing modules
- strategy selection for bounded retries
- bounded retry policy
- orchestration trace artifact
- response metadata showing task status and decision summary

The MVP should not require:

- multi-agent architecture
- database persistence
- message queues
- live reviewer dashboard

## Proposed Architecture

### High-Level Components

- `Web API Layer`
- `Annotation Task Manager`
- `Annotation Agent`
- `Tool Adapters`
- `Artifact Store`

### Suggested Internal Modules

- `app/agent/orchestrator.py`
- `app/agent/state.py`
- `app/agent/policies.py`
- `app/agent/tools.py`
- `app/agent/trace.py`

### Tool Mapping to Current Code

- preprocess tool -> `app.domain.images.preprocess`
- llm tool -> `app.services.llm.client`
- normalize tool -> `app.domain.annotations.normalizer`
- validate tool -> `app.domain.annotations.validator`
- schema tool -> `app.domain.annotations.schema`
- export tool -> `app.domain.annotations.exporters`

## Example Decision Policy

Default v1 policy:

1. preprocess image
2. call annotation generation
3. normalize output
4. validate output
5. if validation passes, export success artifacts
6. if validation fails on attempt 1, retry once
7. if validation still fails, mark as `review_required`

## API Requirements

### Existing Endpoint

Keep:

- `POST /upload`

### Response Additions

Add optional fields such as:

- `task_id`
- `task_status`
- `agent_strategy`
- `attempt_count`
- `review_required`
- `orchestration_trace_file`

## Data Requirements

The system should introduce an orchestration trace schema.

Suggested fields:

- `task_id`
- `image_id`
- `status`
- `attempts`
- `strategy_history`
- `tool_runs`
- `errors`
- `final_decision`
- `created_at`
- `completed_at`

## Risks

- increased complexity may reduce short-term development speed
- retry logic may increase latency
- poorly designed policies may hide root-cause model issues
- mixing orchestration metadata into current responses may complicate frontend assumptions

## Open Questions

- should retry use the same model or allow provider fallback
- should review escalation return HTTP success with warning, or a separate status code
- should task state remain file-based or later move to a database

## Phased Delivery Plan

### Phase 1

- introduce task state
- introduce Annotation Agent orchestration service
- wrap existing modules as tools
- add trace artifact

### Phase 2

- add retry and fallback policies
- enrich response metadata
- add better error categories

### Phase 3

- add review queue support
- add reviewer actions and approval flow
- support comparative experimental runs

## Acceptance Criteria

- a user can upload an image and receive annotation output through the existing endpoint
- the backend executes through the Annotation Agent rather than a hard-coded single path
- existing deterministic modules are reused as tools
- invalid first-pass output can trigger bounded retry
- unresolved invalid cases are marked for review
- orchestration trace is written as an artifact
- current frontend remains functional with backward-compatible response fields

## Summary

This PRD proposes evolving the current annotation prototype into a controlled agent-oriented system. The key principle is not to replace the existing stable pipeline modules, but to add a single orchestration agent that can make bounded workflow decisions, recover from common failures, and prepare the system for future human review and research experimentation.
