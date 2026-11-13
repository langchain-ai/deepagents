# Annotating execute calls

`DescribeExecuteMiddleware` adds a required, human-readable `description` argument to the model-facing `execute` definition. The original tool still receives only its existing arguments, including `command` and optional `timeout`. Both synchronous and asynchronous calls are supported.

Import `middleware.py` from this directory and pass the middleware to your agent:

```python
from deepagents import create_deep_agent
from middleware import DescribeExecuteMiddleware

agent = create_deep_agent(
    model=model,
    backend=sandbox,
    middleware=[DescribeExecuteMiddleware()],
)
```

Use your configured chat model and a backend implementing `SandboxBackendProtocol`. The middleware does not provide a sandbox or enable execution on backends that lack it.

A model-generated call can now look like:

```json
{
  "name": "execute",
  "args": {
    "command": "pwd",
    "description": "Show the working directory before choosing file paths."
  },
  "id": "call_example",
  "type": "tool_call"
}
```

The description stays in the model's tool-call message, so a UI or trace can display it. The tool-call wrapper removes it from a copy of the arguments before delegating to the existing tool; it does not mutate the recorded message or change backend execution behavior.

Descriptions are untrusted model output, not a security check or approval. Review the actual command and retain your existing sandbox restrictions and human-in-the-loop policy. If configuring a separate `HumanInTheLoopMiddleware`, put it before `DescribeExecuteMiddleware` so it can inspect the annotation before it is stripped. This middleware applies to the agent receiving it; configure it separately for subagents that also need annotated execution.
