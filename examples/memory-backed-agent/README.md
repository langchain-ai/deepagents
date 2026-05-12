# memory-backed-agent

A minimal one-file example that treats Context Hub as a living knowledge base:

- `/memories/` is durable in Context Hub.
- default backend stays thread-scoped (`StateBackend`).
- one script defines the graph and auto-wires a LangSmith issues board to the same Context Hub repo handle.

## Single code file

- `deploy_memory_backed_agent.py`
  - defines `agent` with `CompositeBackend` + `ContextHubBackend`
  - deploys the graph (`langgraph deploy`)
  - create-or-patch wires `/issues-agent`

## Backend pattern

```python
backend = CompositeBackend(
    default=StateBackend(),  # thread-scoped
    routes={
        "/memories/": ContextHubBackend("my-agent"),  # durable in Context Hub
    },
)

agent = create_deep_agent(
    model=init_chat_model(model="anthropic:claude-sonnet-4-6"),
    backend=backend,
)
```

## Prerequisites

Set these environment variables:

- `ANTHROPIC_API_KEY`
- `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`)

Optional:

- `MEMORIES_HUB_IDENTIFIER` (for example `my-agent` or `my-org/my-agent`)
- `LANGSMITH_ENDPOINT` / `LANGCHAIN_ENDPOINT`
- `LANGSMITH_TENANT_ID`
- `DEEPAGENT_MODEL`

## Run

```bash
cd examples/memory-backed-agent

# Deploy + auto-wire issues board
uv run python deploy_memory_backed_agent.py --project-name my-agent
```

## What the wiring does

After resolving the deployed tracing project id, the script:

1. `POST /v1/platform/sessions/{session_id}/issues-agent`
2. If `409 conflict`, `PATCH` the existing board with `context_hub_repo_handle`

This mirrors the same create-or-patch approach used in deploy auto-wiring logic.
