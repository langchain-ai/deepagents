# deepagents-repl

A [`deepagents`](../deepagents) middleware that exposes a persistent JavaScript REPL to an agent via an `eval` tool, backed by [`quickjs-rs`](../../../quickjs-wasm) (sandboxed QuickJS via PyO3 + rquickjs).

## Usage

```python
from deepagents import create_deep_agent
from deepagents_repl import REPLMiddleware

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    middleware=[REPLMiddleware()],
)
```

State persists across `eval` calls within the same LangGraph thread: `let x = 40` in one call, `x + 2` in the next, returns `42`.

## Sandbox

The REPL has no filesystem, network, real clock, or `fetch`. Only an opt-in `console.log/warn/error` bridge is exposed (buffered and returned in the tool output).

## Status

Alpha. Depends on a pre-release `quickjs-rs` (0.3.0.dev); both are resolved via local editable installs during development (see `pyproject.toml`).
