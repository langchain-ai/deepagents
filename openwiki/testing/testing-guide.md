---
type: testing guide
title: Testing Guide
description: Focused, hermetic validation routes for dcode configuration, MCP and JavaScript cost behavior; SDK messages and subagents; ACP; Talon; QuickJS; and Harbor adapters. It distinguishes network-restricted unit checks from credentialed evaluation workflows.
tags: [testing, hermetic-tests, dcode, mcp, acp, talon, quickjs, harbor]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-21T08:04:00.255Z
sources:
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-8288b43b279d5cf7aaf1505d
    resource: repo://libs/acp/tests/test_agent.py
  - id: openwiki-source-f188d13aef04b643ced86520
    resource: repo://libs/acp/tests/unit_tests/test_execute_permissions.py
  - id: openwiki-source-d4716b8ae162796c2c7ad991
    resource: repo://libs/code/deepagents_code/_js_cost.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-2843d86bbaef4173c77ca3e1
    resource: repo://libs/code/tests/unit_tests/test_config.py
  - id: openwiki-source-84638d6cd0c222d6264dbef8
    resource: repo://libs/code/tests/unit_tests/test_js_cost_sync.py
  - id: openwiki-source-8574be7f7f29e3e1dd328837
    resource: repo://libs/code/tests/unit_tests/test_js_cost_tracking.py
  - id: openwiki-source-1ce25590f75ba42bdd04fce2
    resource: repo://libs/code/tests/unit_tests/test_mcp_tools.py
  - id: openwiki-source-0dfd43fc1de1af401946bc38
    resource: repo://libs/code/tests/unit_tests/test_model_config.py
  - id: openwiki-source-dc64f28a66d10932b86fcd61
    resource: repo://libs/deepagents/tests/unit_tests/test_messages_reducer.py
  - id: openwiki-source-ca8183c87e6002c442ee2d62
    resource: repo://libs/deepagents/tests/unit_tests/test_subagents.py
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-78f970513a111c43fc9971ce
    resource: repo://libs/evals/tests/unit_tests/test_harbor_langgraph_agent.py
  - id: openwiki-source-224af38a099019bb2a44f0a7
    resource: repo://libs/partners/quickjs/Makefile
  - id: openwiki-source-cf029b27d1ea908d55e21938
    resource: repo://libs/partners/quickjs/tests/unit_tests/test_slot_cleanup.py
  - id: openwiki-source-284870d46988a0957cf01b5c
    resource: repo://libs/partners/quickjs/tests/unit_tests/test_snapshot_persistence.py
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-4d6726e17c8a0c78539a7d33
    resource: repo://libs/talon/tests/test_runtime.py
  - id: openwiki-source-82dab853903c3a574614fd1e
    resource: repo://libs/talon/tests/unit_tests/test_background.py
generated: { by: "openwiki/0.4.2", at: "2026-11-21T08:04:00.255Z" }
---

# Testing Guide

Run the smallest package-owned route that exercises the behavior a caller observes. These unit tests use fake models, temporary files, in-memory or temporary SQLite checkpointers, in-process MCP servers, mock transports, and synchronization events; they must not require a provider credential or a live service. Warnings are errors unless a test narrowly permits a reviewed warning.

Start in the owning package, synchronize dependencies, and pass the nearest file via `TEST_FILE`:

```bash
# dcode: configuration, MCP, and JavaScript cost behavior
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_config.py
make test TEST_FILE=tests/unit_tests/test_model_config.py
make test TEST_FILE=tests/unit_tests/test_mcp_tools.py
make test TEST_FILE=tests/unit_tests/test_js_cost_sync.py
make test TEST_FILE=tests/unit_tests/test_js_cost_tracking.py

# SDK message/checkpoint and delegation semantics
cd ../deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_messages_reducer.py
make test TEST_FILE=tests/unit_tests/test_subagents.py

# ACP protocol adaptation and permission caching
cd ../acp
uv sync --all-groups
make test TEST_FILE=tests/test_agent.py
make test TEST_FILE=tests/unit_tests/test_execute_permissions.py

# Talon lifecycle and detached-work ownership
cd ../talon
uv sync --all-groups
make test TEST_FILE=tests/test_runtime.py
make test TEST_FILE=tests/unit_tests/test_background.py

# QuickJS snapshots and cleanup lifecycle
cd ../partners/quickjs
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_snapshot_persistence.py
make test TEST_FILE=tests/unit_tests/test_slot_cleanup.py

# Harbor LangGraph adapter
cd ../../evals
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_harbor_langgraph_agent.py
```

The package Makefiles are the command authority. dcode runs pytest with xdist, disabled sockets except Unix sockets, disabled benchmarks, and coverage. Talon runs its Node WhatsApp-bridge suite before pytest with the same socket restriction, a ten-second timeout, and coverage. ACP also disables sockets except Unix sockets, has a ten-second timeout, and collects coverage. QuickJS and evals disable sockets except Unix sockets; QuickJS includes coverage. See [State persistence](../concepts/state-persistence.md), [MCP](../integrations/mcp.md), [Sandbox partners](../integrations/sandbox-partners.md), [Development](../operations/development.md), and [Run evals](../workflows/run-evals.md).

## Choose the boundary

| Change | Focused route | Observable invariant |
| --- | --- | --- |
| Workspace dotenv, credential, model policy, or configuration handoff | `libs/code/tests/unit_tests/test_config.py`, `test_model_config.py` | Workspace snapshots are isolated; precedence and endpoint/credential policy cannot leak or bypass a restriction. |
| MCP discovery, subprocess working directory, transport, or failure isolation | `libs/code/tests/unit_tests/test_mcp_tools.py` | Config source determines the server working directory; a test never reads the real profile or contacts a remote server. |
| JS interpreter execution or subagent cost transfer | `libs/code/tests/unit_tests/test_js_cost_sync.py`, `test_js_cost_tracking.py` | Tool-message semantics hold synchronously and receipts remain durable without double charge. |
| Core checkpoint message IDs or synchronous/async delegation | `libs/deepagents/tests/unit_tests/test_messages_reducer.py`, `test_subagents.py` | Messages retain stable IDs and a parent receives only each child's final result with isolated child state. |
| ACP streaming, replay, cancellation, or execute approval | `libs/acp/tests/test_agent.py`, `tests/unit_tests/test_execute_permissions.py` | Clients see ordered public blocks, only the intended session is cancelled, and reusable execute permission is exact and session-scoped. |
| Talon graph construction, tool reload, environment, or recovery | `libs/talon/tests/test_runtime.py` | A replacement graph is atomic, conversation identity survives turns, and shells receive no credentials. |
| Detached subagents, cancellation, or delivery recovery | `libs/talon/tests/unit_tests/test_background.py` | Jobs and results remain conversation-owned and are consumed only after successful delivery. |
| QuickJS cross-turn state or leaked context | `libs/partners/quickjs/tests/unit_tests/test_snapshot_persistence.py`, `test_slot_cleanup.py` | Only a valid saved thread snapshot persists; stale contexts cannot become state. |
| Harbor graph configuration, skills, MCP, shell isolation, or web search | `libs/evals/tests/unit_tests/test_harbor_langgraph_agent.py` | The adapter constructs a constrained local graph without credential/tracing egress. |

## dcode configuration, MCP, and model policy

Use `test_config.py` for process environment, global/project dotenv files, workspace preview, or server-handoff changes. Keep `os.environ` outside the `use_environment()` binding: bindings copy their input, are immutable, and restore the process environment after an exception. Two workspaces should build two `.env` files under `tmp_path` and verify construction-time consumers—credentials, config options, and MCP interpolation—resolve their own snapshot without process mutation.

Precedence and interpolation are security boundaries: shell values win over project and global dotenv values, while a project value must not interpolate into trusted global dotenv input. Use `test_model_config.py` for exact allowlist parsing (including an explicit empty deny-all), canonical provider-wide wildcards, and stored credentials. A stored personal key with no base URL must clear inherited gateway URL and applicable headers so it cannot be sent to a stale endpoint.

Use `test_mcp_tools.py` rather than a live MCP endpoint. Its `FakeMCPServer` routes real FastMCP client/session/schema handling through an in-process transport while retaining the constructed transport for assertions. When changing config-source or `cwd` behavior, make a temporary project and server that reads a relative file: project and explicit configs use the session directory, user/plugin configs use the backend directory, and an explicit server `cwd` wins. Redirect profile paths and `Path.home()` to `tmp_path`, and clean a stateful `MCPSessionManager` in `finally`.

## JavaScript costs: assert durable receipts

`CostAwareCodeInterpreterMiddleware` leaves synchronous tool execution untouched. The sync matrix covers `invoke` and `stream`, `thread`, `turn`, and `call` modes, plus `js_eval` and an unrelated ordinary tool. Assert one successful `ToolMessage`, the original call ID, and observable content—not middleware call order.

The asynchronous route composes a parent agent, `SubAgentMiddleware`, `CostTrackingMiddleware`, a cost-aware interpreter, and an in-memory or SQLite saver. Give fake model responses deterministic usage/pricing and assert persisted `_session_cost_usd` and its breakdown after invocation, interruption, rebuild, and resume. For SQLite, close and reopen the saver around a reconstructed graph; that proves persistence rather than process-local retention.

```mermaid
sequenceDiagram
    participant Parent as Parent graph
    participant JS as Cost aware JS middleware
    participant Child as Subagent graph
    participant Saver as Checkpoint saver
    Parent->>JS: js_eval task call
    JS->>Child: isolated dispatch namespace
    Child->>Saver: persist cost receipt
    JS->>Saver: read owned receipt total
    JS-->>Parent: cost transfer update
    Parent->>Saver: persist session total and breakdown
```

*Cost transfers from a child receipt only after the JavaScript dispatch has settled.*

Cover a completed sibling alongside interruption, failure, cancellation, replay, and parallel calls. Those cases protect occurrence identity, avoid double charging, and ensure graph-control exceptions propagate rather than being turned into ordinary tool failures.

## SDK messages, subagents, and ACP

For checkpoint/message changes, `test_messages_reducer.py` builds a minimal graph with `InMemorySaver`. It checks that `get_state()` returns non-`None`, stable IDs for object and dict-form messages, across sync and async resumed turns; it also covers replay when the messages checkpoint blob is `None` and `REMOVE_ALL_MESSAGES` resets prior/batched writes. This is the nearest regression route for serialization and reducer behavior.

For delegation changes, `test_subagents.py` verifies that remote definitions expose `start_async_task` and `check_async_task`, while local definitions remain behind `task`. A synchronous child returns its final AI message as the parent `ToolMessage`, not its transcript. Parallel children must keep distinct call IDs/results, and forked or isolated children must neither inherit nor overwrite a parent rubric; use that route whenever adding shared or private state.

ACP tests use a fake connected client and a memory checkpointer. Public prompt streaming preserves text/reasoning block order and whitespace, suppresses local-context messages, puts content before tool calls, and restricts cancellation to the selected session. A restarted server can replay persisted public history, reasoning, and completed tool calls, but must reject a session it did not create or a request with a different `cwd`.

Execute permissions are an authorization cache, not a command-type allowlist: only an identical command and canonical arguments in the same session context can reuse approval. Any argument, `cwd`, mode, model, or session change requires a new review; only the explicit always-allow choice is reusable, and forgetting a session discards it.

## Talon runtime and background-worker boundaries

`DeepAgentRuntime` owns a graph and request-scoped configuration. Use recording graphs and patched construction to verify refresh/reload transactionality: a failed replacement leaves the prior graph and tools active. Assert a conversation uses the same `thread_id` across turns and an authorization handler is bound only during that request. The local shell backend is also a security boundary: provider, tracing, MCP, cloud, and OAuth credentials must be absent, while its path is hardened.

`BackgroundSubagents` holds in-memory jobs keyed by owner conversation. Use events to make lifecycle edges deterministic: wait for worker start, trigger cancellation or delivery, then await the worker. A different conversation cannot inspect, cancel, or consume a job; capacity is global to configured copies and cancellation frees it.

```mermaid
flowchart TD
    Turn["Chat turn delegates task"] --> Scheduled{"Scheduled turn"}
    Scheduled -->|yes| Inline["Run inline with concurrency limit"]
    Scheduled -->|no| Worker["Create owner scoped background job"]
    Worker --> Result["Store sanitized result"]
    Result --> Next["Later owner turn consumes result"]
    Next --> Delivered{"Reply delivered"}
    Delivered -->|yes| Ack["Keep result acknowledged"]
    Delivered -->|no| Requeue["Requeue acknowledged result"]
```

*Chat delegation is detached, whereas scheduled delegation runs inline because no later conversational turn exists to deliver a result.*

Results are acknowledged only after a successful consuming turn. Failed or cancelled turns leave them available; a host can requeue precisely the acknowledged IDs if delivery is undone. Test sanitized failure/timeout results and that a worker inherits relevant caller context but not the authorization handler.

## QuickJS snapshots and cleanup

QuickJS slots are per LangGraph thread. In `thread` mode, `after_agent` snapshots REPL state and the next turn restores it; `turn` mode resets it and `call` mode isolates each eval. Run both `invoke` and `ainvoke`. Snapshot tests cover ordinary values, top-level-await bindings, and transformed top-level `const`/`let` declarations; they also prove a model-only turn allocates neither a runtime nor worker and does not disturb a signed snapshot across an agent rebuild.

Use `test_slot_cleanup.py` for lifecycle paths that skip `after_agent`. After failure or cancellation, the next invocation must close a leaked live context and restore the last checkpointed snapshot. A `Command(resume=...)` instead continues the live context because it does not re-enter `before_agent`; time travel or a fork to an older snapshot must discard a mismatched warm context. Assert the next JavaScript result and context replacement.

## Harbor adapter and evaluation separation

The Harbor unit route tests the adapter without Harbor or an external model. Patch model/agent factories, search, and MCP adapters; use in-process `FastMCP`; disable LangSmith tracing before tool invocation. `make_graph` must build headless local dcode using a bounded ASCII-safe, collision-resistant assistant ID, scrub shell-visible model/tracing credentials during construction, restore the parent environment, and populate runtime model identity.

The bare and tau3 routes cover extension boundaries: the bare graph validates an absolute non-traversing `skills_dir`, uses `inherit_env=False`, and attaches task-provided HTTP MCP tools. Tau3 maps streamable HTTP but rejects dataset-provided stdio MCP servers. Web search is absent without `TAVILY_API_KEY`; with it, tests clamp result count and output length, avoid tracing egress, and require upstream failure to return tool output rather than end an evaluation.

Do **not** treat `make evals MODEL=<id>` or the `run-*` Harbor targets as hermetic tests. They are credentialed and potentially paid workflows: `make evals` requires `MODEL`, and Harbor targets stage local dependencies and inject model/tracing provider environment. Run those only when validating evaluation behavior, following [Run evals](../workflows/run-evals.md), after the focused network-restricted unit route passes.

## Before finishing

1. Run the focused route and its paired persistence/handoff route when a change crosses a state or ownership boundary.
2. Use `tmp_path`, fake models, in-process adapters, mock transports, or temporary SQLite; never a developer credential or live service.
3. Add failure, cancellation, and rebuild/resume coverage whenever state, credentials, a worker, a receipt, or a snapshot has an owner.
4. Assert replies, serialized/checkpoint state, a sanitized environment, or provider payloads rather than private implementation order.
5. Run `make lint` in each changed package after focused tests pass.
