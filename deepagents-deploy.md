---
name: deepagents-deploy
description: Minimal reference for the `deepagents` CLI — `init`, `dev`, `deploy`.
---

# deepagents deploy

## Project layout

```
src/
  AGENTS.md        # required — system prompt + read-only /memories/AGENTS.md
  skills/          # optional — seeded under /skills/
  mcp.json         # optional — HTTP/SSE MCP servers
  deepagents.toml
```

## `deepagents.toml`

```toml
[agent]
name = "my-agent"
model = "anthropic:claude-sonnet-4-6"

# [sandbox] is optional — omit to run tools in-process.
[sandbox]
provider = "langsmith"   # none | langsmith | daytona | modal | runloop
scope    = "thread"      # thread | assistant
# template = "deepagents-deploy"
# image    = "python:3"
```

That's the entire surface. Skills, MCP servers, and model deps are auto-detected.

## CLI

```bash
deepagents init                         # scaffold deepagents.toml in cwd
deepagents dev    --config src/deepagents.toml [--port 2024]
deepagents deploy --config src/deepagents.toml [--dry-run]
```

## Runtime

- **System prompt:** `src/AGENTS.md` verbatim, baked in at build time.
- **Memories:** `/memories/AGENTS.md` in the LangGraph store, namespace `(assistant_id, "memories")`. Read-only at runtime — edit the source file and redeploy.
- **Skills:** `/skills/<skill>/...` in the store, namespace `(assistant_id, "skills")`. Also read-only.
- **Sandbox:** default backend. Per-thread cache by default; set `[sandbox].scope = "assistant"` to share one sandbox across all threads of an assistant. Omit `[sandbox]` entirely to fall back to an in-process `StateBackend`.
- **MCP:** HTTP/SSE only. Stdio is rejected at bundle time.

## Gotchas

- `/memories/` and `/skills/` are read-only. Edit source files and redeploy.
- `deepagents deploy` creates a new revision on every invocation (full cloud rebuild). Use `deepagents dev` for iteration.
- The in-process sandbox cache does not survive process restarts; thread-scoped sandboxes get re-provisioned if the server recycles.
- Custom Python tools are not supported — use MCP servers.
