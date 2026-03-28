# Custom Middleware

Demonstrates how to write custom middleware for Deep Agents, covering the three main middleware hooks: `wrap_model_call`, `awrap_model_call`, and `before_agent`.

## What this example shows

- **`ToolCallLoggerMiddleware`** — Intercepts every LLM response via `wrap_model_call` to log tool calls with timestamps. Useful for debugging and auditing.
- **`SessionContextMiddleware`** — Injects dynamic context (session ID, timestamp, metadata) into the system prompt on every LLM call.
- **`GuardrailMiddleware`** — Uses `before_agent` to modify agent state before the loop starts.

## Middleware hooks

Deep Agents middleware extends `AgentMiddleware` and can override these hooks:

| Hook | When it runs | Use case |
|------|-------------|----------|
| `wrap_model_call` / `awrap_model_call` | Before and after every LLM call | Modify system prompt, filter tools, log responses, transform messages |
| `before_agent` | Once, before the agent loop starts | Initialize state, inject context, validate configuration |

Middleware in the `middleware` parameter runs **after** the built-in stack (TodoList, Filesystem, SubAgent, Summarization) but **before** AnthropicPromptCaching and Memory.

## Quickstart

**Prerequisites**: Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Navigate to this directory and install:

```bash
cd examples/custom_middleware
uv sync
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Run:

```bash
uv run python agent.py
```

## Writing your own middleware

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # Modify the request before the LLM call
        new_system = (request.system_message or "") + "\nExtra context here."
        response = handler(request.override(system_message=new_system))
        # Inspect or modify the response after the LLM call
        return response
```

Then pass it to `create_deep_agent`:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[MyMiddleware()],
)
```
