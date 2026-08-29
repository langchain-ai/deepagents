# A2A supervisor middleware

This runnable example gives a [Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview) or LangChain `create_agent` four generic tools for supervising remote agents that implement the [A2A protocol](https://a2a-protocol.org/latest/):

- `find_a2a_agents` lists static agents and searches an optional registry.
- `delegate_to_a2a_agent` sends work and supports `taskId`/`contextId` continuation.
- `get_a2a_task` polls state and artifacts.
- `cancel_a2a_task` requests cancellation.

The middleware uses the official `a2a-sdk` with JSON-RPC and HTTP+JSON bindings. Agent Cards, messages, and artifacts are explicitly marked as untrusted data and normalized into bounded JSON tool results.

## Run it

```bash
cd examples/a2a-supervisor
uv sync
export A2A_AGENT_URL=https://agent.example.com
export ANTHROPIC_API_KEY=...
uv run python supervisor.py
```

`supervisor.py` uses `create_deep_agent`; the same middleware can be passed to `langchain.agents.create_agent`. Both should be invoked asynchronously because A2A I/O is async.

## Configure agents

Most applications can pass an Agent Card or HTTPS base URL directly:

```python
from a2a.types import AgentCard
from middleware import A2ASupervisorMiddleware

middleware = A2ASupervisorMiddleware(
    agents=[agent_card, "https://agent.example.com"],
    allowed_hosts=["agent.example.com"],
)
```

Base URLs resolve `/.well-known/agent-card.json`. Static cards appear as compact summaries in the system prompt. Keep URL-backed middleware in an `async with` scope, as in `supervisor.py`, or call `aclose()`. Large or dynamic catalogs should use search on demand:

```python
from collections.abc import Sequence
from middleware import AgentCardReference


class CompanyRegistry:
    async def discover(
        self,
        query: str | None = None,
        *,
        limit: int = 20,
    ) -> Sequence[AgentCardReference]:
        # Vendor-specific authentication, pagination, and parsing belong here.
        return [AgentCardReference(id="research-v2", url="https://research.example.com")]


middleware = A2ASupervisorMiddleware(
    registry=CompanyRegistry(),
    allowed_hosts=["research.example.com"],
)
```

An async callable with the same signature also works. A2A does not standardize registry HTTP APIs, so the adapter owns its vendor-specific transport while the middleware resolves and validates returned cards.

## Task lifecycle

`delegate_to_a2a_agent` can return a direct message or task. Its normalized result includes `taskId`, `contextId`, status, and artifacts. It defaults to `return_immediately=True`, so long-running work returns a task for `get_a2a_task` rather than holding a tool call open; pass `return_immediately=False` when blocking is intentional. Pass returned IDs to continue interrupted work. Terminal tasks are immutable, so refinements should start a new task in the same context rather than reuse a completed task ID.

## Security defaults

Remote network actions should require human approval. `supervisor.py` configures:

```python
interrupt_on = {
    "delegate_to_a2a_agent": True,
    "cancel_a2a_task": True,
}
```

When using `create_agent`, add `HumanInTheLoopMiddleware` with the same `interrupt_on` map and a checkpointer. The caller resumes an approved interrupt with `Command(resume={"decisions": [{"type": "approve"}]})`.

The example also:

- requires HTTPS, rejects URL credentials and non-public/metadata targets, revalidates DNS before each call, and supports an explicit host allowlist;
- caps Agent Card, event count, artifact count, part count, individual content, and final tool-result sizes and applies request timeouts;
- closes every A2A client and exposes `async with middleware` for resolver cleanup;
- never reads or forwards arbitrary credentials from model arguments.

For local development only, `allow_local_http=True` permits HTTP to loopback addresses. Do not enable it for model-selected or untrusted URLs. DNS checks are preflight validation, not connection pinning: use the allowlist and production egress firewall rules to prevent DNS-rebinding SSRF.

## Authentication

Agent Cards declare authentication schemes, but credential acquisition is out of band. Supply trusted official SDK `ClientCallInterceptor` instances through `interceptors=[...]`. Keep credentials in your application secret store, scope them per destination, and never expose them as tool arguments or registry/card text.

## Tests

Tests inject fake registry/client implementations and make no network requests:

```bash
uv run pytest tests
uv run ruff check .
uv run ruff format --check .
```
