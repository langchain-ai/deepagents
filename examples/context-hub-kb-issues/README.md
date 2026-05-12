# context-hub-kb-issues

A minimal example that treats Context Hub as a living knowledge base:

- `/memories/` is durable in Context Hub.
- default backend stays thread-scoped (`StateBackend`).
- deployment can auto-wire a LangSmith issues board to the same Context Hub repo handle.

## Backend pattern

```python
backend = CompositeBackend(
    default=StateBackend(),  # thread-scoped
    routes={
        "/memories/": ContextHubBackend("-/my-agent"),  # durable in Context Hub
    },
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=backend,
)
```

## Files

- `agent.py` — graph definition using `CompositeBackend` + `ContextHubBackend`
- `langgraph.json` — graph entrypoint for local/dev/deploy
- `deploy_and_wire_issues.py` — deploys and POST/PATCHes `/issues-agent`

## Prerequisites

Set these environment variables:

- `ANTHROPIC_API_KEY`
- `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`)
- `MEMORIES_HUB_IDENTIFIER` (e.g. `-/my-agent` or `my-org/my-agent`)

Optional:

- `LANGSMITH_ENDPOINT` / `LANGCHAIN_ENDPOINT`
- `LANGSMITH_TENANT_ID`
- `DEEPAGENT_MODEL`

## Run

```bash
cd examples/context-hub-kb-issues
cp .env.example .env

# Deploy + auto-wire issues board
uv run python deploy_and_wire_issues.py \
  --project-name my-agent \
  --memories-identifier -/my-agent
```

If you already deployed and only want to wire/update the issues board:

```bash
uv run python deploy_and_wire_issues.py \
  --project-name my-agent \
  --memories-identifier -/my-agent \
  --skip-deploy
```

## What the wiring does

After resolving the deployed tracing project id, the script:

1. `POST /v1/platform/sessions/{session_id}/issues-agent`
2. If `409 conflict`, `PATCH` the existing board with `context_hub_repo_handle`

This mirrors the same create-or-patch approach used in deploy auto-wiring logic.
