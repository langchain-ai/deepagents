# Shared sandbox with async subagents

Two per-run graph factories hosted by LangGraph Agent Server share one LangSmith
sandbox without sharing a checkpoint thread or running the worker in the parent's
process. No sandbox ID is exposed in the model's `start_async_task` arguments.

- `parent` derives `shared-<UUID hex>` from `configurable.thread_id` and creates or
  reconnects that named sandbox.
- `start_async_task` creates a separate worker thread and passes the parent's
  thread ID as `configurable.parent_thread_id`. Updates retain this original ID.
- `worker` derives the same name from `parent_thread_id` and attaches to the
  existing sandbox. It does not provision a second sandbox.

## Run

Use this repository's SDK checkout (the parent-ID propagation is required).
Install the SDK's declared dependencies with `uv sync --all-groups` in
`libs/deepagents`. Install the LangGraph CLI with its `inmem` extra using your
usual development tooling if it is not already available.

Create `.env` here with:

```dotenv
ANTHROPIC_API_KEY=...
LANGSMITH_API_KEY=...
SANDBOX_SNAPSHOT=your-ready-snapshot-name
AGENT_SERVER_URL=http://localhost:2024
```

Use a ready LangSmith sandbox snapshot with a writable `/workspace` directory.
Both graphs use `create_deep_agent`'s default model. From this directory, start:

```bash
langgraph dev
```

In Studio, select `parent`, create a thread, and send:

> Write /workspace/parent.txt containing "from parent", then start a background
> worker that reads it and writes /workspace/worker.txt containing "from worker".
> Return the task ID without polling.

Then ask the parent to check that task ID. After completion, ask it to read
`/workspace/worker.txt`. The parent seeing the worker's file demonstrates shared
storage; the worker still has its own thread and independently tracked run.

## Ownership and limitations

The factories close their client connections after each run, but do not delete
or stop the shared sandbox. Delete it explicitly after **all** parent and worker
runs finish, using `SandboxClient.delete_sandbox` with the name derived from the
parent thread UUID. Cancellation of one worker must not delete shared storage.

This is a single-owner example: Agent Server serializes runs on each parent
thread, and only that parent creates the sandbox. Sandbox expiry/deletion while
workers are running, concurrent writes to the same file, nested delegation, and
cross-tenant authorization need application-level policies. A predictable name
is not an authorization mechanism; keep both graph endpoints and the sandbox
credentials within the same trusted application. Direct worker invocations must
supply a valid `parent_thread_id` and an already existing sandbox.

## Network-free verification

From `libs/deepagents`:

```bash
uv run --group test pytest ../../examples/shared-async-sandbox/test_graph.py tests/unit_tests/test_async_subagents.py --color=no
```

These checks cover identity propagation (including updates), parent provisioning,
and worker attachment without creating a second sandbox. A live run additionally
requires model credentials and LangSmith sandbox access.
