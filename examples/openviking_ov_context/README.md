# OpenViking lifecycle hooks for Deep Agents

`OVMiddleware` is an `AgentMiddleware` example for applications that already
initialize an OpenViking `OVContext` in `abefore_agent`. It writes each new graph
message batch through an injected `OVMessageWriter`, then evaluates the
capacity and completed-run thresholds in `CommitPolicy` using live
`pending_tokens` from OpenViking.

The example targets the OpenViking API merged in
[OpenViking PR #4](https://gitcode.com/weixin_59153604/OpenViking-0.4.1/pull/4).
The dependency is pinned to that merge commit because this fork's `OVContext`
contract is not in a published release.

## Install and test

From this directory:

```bash
uv sync
uv run python -m unittest discover -s tests -v
```

The hook expects `state["ov_context"]` to contain a checkpoint mapping with an
active `run_id` and the matching OpenViking `session_id`. The application must
initialize the session and start the run before these hooks execute, serialize
operations for each session, and inject:

- An `OVMessageWriter` that persists the full batch and returns `WriteResult`.
- A session client with async `get_session(session_id)` returning
  `pending_tokens`.
- A `CommitPolicy` backed by a client with async
  `commit_session(session_id, keep_recent_count=...)`.

For example, when constructing the caller's middleware stack:

```python
from commit_policy import CommitConfig, CommitPolicy
from ov_middleware import OVMiddleware

policy = CommitPolicy(
    CommitConfig(
        token_budget=128_000,
        commit_token_threshold_ratio=0.7,
        completed_run_threshold=5,
        keep_recent_count=8,
    ),
    ov_client,
)
middleware = OVMiddleware(
    writer=writer,
    session_client=ov_client,
    commit_policy=policy,
)
```

Pass `middleware` through `create_deep_agent(middleware=[...])` as part of an
application that provides the `abefore_agent` initialization above. The hook
returns only checkpoint-safe `ov_context` updates. `aafter_model` defers a
commit while tool calls are pending; `aafter_agent` flushes any final tool
result, records one completed run, and ends it. Writer and client failures
propagate so the graph does not advance its cursor on an unconfirmed write.

This example owns the write and commit hooks for its session. Do not install a
second writer middleware for the same `OVContext`.
