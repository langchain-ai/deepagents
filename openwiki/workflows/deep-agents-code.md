---
type: Engineering Workflow
title: Deep Agents Code runtime, approvals, and MCP trust
description: Maintainer guide to dcode’s Textual client and LangGraph server, human approval modes, experimental Auto policy, sandboxes, and MCP configuration trust.
tags: [dcode, security, approvals, mcp, workflow]
---
# Deep Agents Code: runtime, approvals, and MCP trust

`libs/code` packages the prebuilt terminal coding agent (`dcode` / `deepagents-code`). It is the coding-specific consumer of the SDK described in [Runtime and package architecture](../architecture/overview.md), not a standalone agent runtime.

## Process and graph flow

Deep Agents Code intentionally separates UI from graph execution:

```text
CLI parsing (`main.py`)
  -> Textual client/app (`app.py`, UI widgets)
  -> `langgraph dev` server subprocess
  -> cached `server_graph.make_graph()`
  -> `create_cli_agent()` middleware/tool/subagent assembly
  -> core `create_deep_agent()` / LangGraph execution
```

- `libs/code/deepagents_code/main.py` validates CLI/configuration, prevents autonomous flags in ACP or headless modes, constructs server arguments, and starts the Textual app.
- `server_graph.py` reads `DEEPAGENTS_CODE_SERVER_*` config, resolves models off the event loop, loads MCP/plugins, optionally builds a persistent sandbox, and caches the graph for the server process lifetime behind a lock.
- `agent.py` configures the SDK with model selection, goal/resume state, ask-user, memory/skills/plugins, local context, shell/interpreter support, compaction, rubric grading, approval middleware, and main/general/async subagents.
- Local execution uses `LocalShellBackend` rooted at the working directory; remote execution delegates filesystem and shell operations to the selected sandbox.

`libs/code/ARCHITECTURE.md` and `DEVELOPMENT.md` are the first primary docs to read when changing this path. Changes to the server-side graph construction should also account for the core assembly rules in [Runtime and package architecture](../architecture/overview.md).

## Approval modes are safety policy, not containment

The README says that starting in a directory trusts its artifacts before approval. Remote sandboxes are the recommended boundary for untrusted repositories. Human approval complements that boundary but does not turn local execution into a sandbox.

| Mode | Behavior | Important constraint |
| --- | --- | --- |
| `manual` | Interrupts gated operations for user approval. | Default/fail-closed mode. |
| `auto` | Experimental deterministic policy plus classifier review may approve eligible operations. | Limited to local interactive, unsandboxed use with `DEEPAGENTS_CODE_EXPERIMENTAL`; otherwise it downgrades to manual. |
| `yolo` | Bypasses HITL. | Requires a versioned local acknowledgement stored with restrictive permissions. |

Approval state is a hashed per-thread record in LangGraph Store, read and validated by the server against the active thread. Missing, malformed, or unreadable state falls back to Manual. This server/client synchronization exists so a user can change modes during an active conversation; failure to synchronize a return to Manual must not leave an action running under a stale permissive policy.

The gated inventory includes writes/edits/deletes, execute, web search/fetch, subagent/task operations, optional compaction, and non-read-only MCP tools. Keep that inventory synchronized with the middleware’s interrupt configuration when adding a tool.

### Auto-mode authority boundary

The recent classifier-backed Auto feature is deliberately narrow:

- Fast-path writes must stay inside the trusted root and exclude sensitive paths such as CI/hooks, shell scripts, and dependency/config locations.
- Fast-path shell approval permits a small read-only Git set or narrow configured commands; shell control operators and broad/wildcard commands are rejected.
- Classifier input may be authorized only by **literal, pre-expansion user text attached by the client**. File content, tool output, and assistant prose cannot expand authority.
- The implementation redacts/sanitizes persisted reasons and validates tool-call identities/batches exactly.
- `readOnlyHint` only bypasses gating when it is literal, coherent boolean metadata with no destructive hint. Ambiguous metadata fails closed.

Auto is neither an OS boundary nor a guarantee that delegated work is classifier-reviewed. Parent Auto review must not be assumed to cover all subagent internals, and PTC/interpreter host-bridge calls have their own policy boundary. Security-sensitive changes here need both a code review focused on authority propagation and explicit top-level/delegated-path tests.

## MCP sources and project trust

MCP configuration is resolved low-to-high from user `~/.deepagents/.mcp.json`, project `.deepagents/.mcp.json`, project `.mcp.json`, then explicit configuration. Plugin configurations are also composed server-side. Supported transports are stdio, HTTP, and SSE; config validation covers server shape, headers/auth, and mutually exclusive tool filters.

Project-declared MCP configuration is a trust boundary: it can spawn a local command, cause SSRF, or exfiltrate interpolated headers. Thus project stdio **and remote** servers are gated. Whole-config `--trust-project-mcp` is possible, but scoped user-owned approvals/environment allowlists can authorize individual servers; explicit denial wins. `${VAR}` values are interpolated only at activation, and the loader isolates individual server errors while redacting resolved values when interpolation was used.

Runtime discovery uses throwaway sessions; tool wrappers use a lazy process-wide session manager with retry/invalidation for transient/dead/reauth sessions. Loading is bounded-concurrent while output ordering remains deterministic.

### Cached MCP tool failure boundary

`_build_cached_mcp_tool()` in `libs/code/deepagents_code/mcp_tools.py` turns each discovered tool into a LangChain `StructuredTool`. Its coroutine obtains a cached session, retries a transient session failure once after invalidation, and raises a `ToolException` for failures the model must see. `_handle_cached_mcp_tool_error()` is the sole `WARNING`/traceback logging boundary for those recoverable `ToolException`s and returns the tool-local error text. Do not add a second warning in the coroutine: one failed tool call must yield one failure warning, not duplicate diagnostics.

Cleanup warnings are separate: invalidating a failed retry session or closing a session can warn independently because they describe a resource-cleanup problem rather than a duplicate tool failure. Preserve that distinction when changing retries or error handling. Re-raise cancellation, keyboard interrupt, system exit, and existing `ToolException` values unchanged; the wrapper must not turn control flow or actionable MCP errors into a generic retry result.

## Tests and safe modification sequence

Run from `libs/code`:

```bash
uv sync --all-groups
make check                 # package full local suite
make test                  # unit/no-network
make integration_test      # network-enabled tests
```

The pytest defaults enforce a 30-second timeout and strict markers/configuration. Relevant anchors:

- `tests/unit_tests/test_approval_mode.py`: store failures/malformed state fail closed; YOLO acknowledgement behavior.
- `tests/unit_tests/test_auto_mode.py`: provenance, annotation coherence, path/Git policies, classifier failures, replay/escalation, denials, and headless MCP guards.
- `tests/unit_tests/test_server_graph.py`: graph cache, startup error handling, MCP discovery, off-loop construction, and no-MCP/read-only conditions.
- `tests/unit_tests/test_mcp_tools.py::TestCachedSessionProxy::test_repeated_transient_error_surfaces_tool_message`: a second transient failure becomes a model-visible error after one retry and logs exactly one tool-failure warning with traceback.
- `tests/unit_tests/test_mcp_tools.py::TestCachedSessionProxy::test_generic_oserror_is_not_retried`: a non-transient `OSError` is model-visible without session retry and has the same single-warning contract.
- `tests/integration_tests/test_auto_approve_remote.py`: actual approved/rejected remote writes, including subagent behavior.

For the cached MCP error seam, use the quiet focused check before broader package checks:

```bash
cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_mcp_tools.py -k 'repeated_transient_error_surfaces_tool_message or generic_oserror_is_not_retried'
```

Before changing dcode: identify whether the behavior is client UI, persisted approval state, graph construction, middleware, backend/sandbox, or MCP session lifecycle; make the change at that boundary; then test both failure-to-manual and success paths. For repository-wide CI/release context, see [Evaluation and release](evaluation-and-release.md) and [Operations and testing](../engineering/operations-and-testing.md).
