# Nest sandbox LLM calls in a Deep Agents trace

This example keeps LangChain model calls made by generated code in a remote sandbox
inside the Deep Agent's existing LangSmith trace. It uses LangSmith distributed
tracing to bridge the process boundary:

1. `TracedLangSmithSandbox` reads the active run tree when the `execute` tool calls
   the backend.
2. It passes `run_tree.to_headers()` to the sandbox as
   `LANGSMITH_TRACE_HEADERS`.
3. Generated sandbox code opens `tracing_context(parent=headers)` before calling
   its traced entrypoint.

Because the backend reads the context while `execute` is running, the active run
is the `execute` tool span. The resulting trace shape is:

```text
Deep Agent
└── execute
    └── sandbox_rlm_task
        ├── model
        ├── model
        ├── model
        └── model
```

The sandbox runs an ordinary shell command, so the agent can keep writing all code
to the same sandbox workspace. No separate code-interpreter tool is involved.

## Setup

Export the host and sandbox model configuration:

```bash
export LANGSMITH_API_KEY=<key>
export LANGSMITH_PROJECT=deepagents-sandbox-distributed-tracing
export LANGSMITH_TRACING=true
export OPENAI_API_KEY=<key>
export HOST_MODEL=openai:gpt-4.1
export SANDBOX_MODEL=openai:gpt-4.1-mini
```

Install and run from this directory:

```bash
uv run python agent.py
```

The snapshot image includes `uv`; generated code can use a PEP 723 inline script
or `uv run --with langchain --with langchain-openai --with langsmith` to install
its sandbox-side runtime.

## Why backend propagation instead of command rewriting?

Passing environment variables through the sandbox SDK avoids interpolating API
keys or trace headers into an LLM-authored shell command. The backend also handles
both synchronous and asynchronous execution, including internal wrappers used to
offload large command output.

Only an explicit environment allowlist is copied into the sandbox. Add provider
credentials to `_sandbox_env()` if the sandbox model uses a provider other than
OpenAI.

## Security

Distributed-tracing headers are trusted context, and the sandbox receives model
and LangSmith credentials. Use this only with a sandbox you trust, never accept
trace headers from an untrusted caller, and scope injected credentials as narrowly
as possible. The sandbox code must parse only `LANGSMITH_TRACE_HEADERS`; do not pass
its full environment to `tracing_context(parent=...)`.

## Generalizing to other sandbox backends

Apply the same boundary in another backend's `execute` and `aexecute` methods:
read `get_current_run_tree()`, serialize `run_tree.to_headers()`, and pass that
value through the provider's structured environment API. In sandbox code, parse
the value and use `tracing_context(parent=headers)` around the traced entrypoint.
Capturing the headers earlier, before the tool call starts, would attach sandbox
runs to the agent's parent span instead of the `execute` span.
