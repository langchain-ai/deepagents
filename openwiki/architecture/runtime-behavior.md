---
type: runtime behavior
title: dcode Runtime Behavior and Failure Handling
description: How dcode launches and owns a workspace-aware LangGraph runtime, constructs its agent resources, selects bounded workspace runtimes, and surfaces startup and request failures.
tags: [dcode, runtime, server-startup, workspace, sandbox, extensions, mcp, retry, shutdown]
sources:
  - id: openwiki-source-1728494bdd59604ce9b5f65b
    resource: repo://libs/code/deepagents_code/_server_config.py
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-074ce96a8baea27a6c43328b
    resource: repo://libs/code/deepagents_code/client/launch/server.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-c101168dc0286ff6c29ed37f
    resource: repo://libs/code/deepagents_code/model_retry.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-c8dacdfd6192dd22d24a9362
    resource: repo://libs/code/tests/integration_tests/test_pending_work_recovery.py
  - id: openwiki-source-c04c6318f6e59e0d1c9d6182
    resource: repo://libs/code/tests/unit_tests/test_model_retry.py
  - id: openwiki-source-439d3e6c6f1b62e6d282df3f
    resource: repo://libs/code/tests/unit_tests/test_remote_client.py
  - id: openwiki-source-784e764f7f5eb5169220c3d2
    resource: repo://libs/code/tests/unit_tests/test_server_graph.py
  - id: openwiki-source-f598809da8d8fbff2d7ae090
    resource: repo://libs/code/tests/unit_tests/test_server_manager.py
verified:
  - by: openwiki/0.4.2
    at: 2026-09-09T08:05:37.706Z
generated: { by: "openwiki/0.4.2", at: "2026-09-09T08:05:37.706Z" }
---

# dcode Runtime Behavior and Failure Handling

Interactive dcode runs its agent in an owned, loopback `langgraph dev` subprocess and accesses the `agent` graph through `RemoteAgent` over HTTP and SSE. The split is intentional: the client retains UI and session concerns, while the server owns the compiled graph, its backend, MCP sessions, sandbox lifetime, and server-side offload operation. ACP's in-process stdio path is outside this page. See [Deep Agents Code Architecture](/openwiki/architecture/code-agent.md), [Configuration Layering](/openwiki/concepts/config-layering.md), [State Persistence](/openwiki/concepts/state-persistence.md), [MCP](/openwiki/integrations/mcp.md), [Sandbox partners](/openwiki/integrations/sandbox-partners.md), and [Run a dcode session](/openwiki/workflows/run-dcode-session.md).

## Launch and runtime construction

`start_server_and_get_agent` captures an explicit or current project context, pre-validates an explicit MCP configuration, derives `ServerConfig` from the invocation, exports its `DEEPAGENTS_CODE_SERVER_*` representation, and scaffolds a temporary server project. It starts `langgraph dev` on `127.0.0.1` and port `0` by default, waits for the `agent` graph, then configures the returned `RemoteAgent` with the workspace's **session** policy claim and fingerprint. Project-scoped policy is not included in that claim: it is resolved and trusted by the server for the target directory.

```mermaid
sequenceDiagram
    participant Client as dcode client
    participant Manager as server manager
    participant Child as langgraph dev
    participant Factory as graph factory
    participant Remote as RemoteAgent
    Client->>Manager: start_server_and_get_agent
    Manager->>Manager: capture workspace and resolve ServerConfig
    Manager->>Manager: validate explicit MCP config and scaffold runtime
    Manager->>Child: launch loopback server
    Child->>Factory: load make_graph
    Factory->>Factory: build or retrieve launch runtime
    Manager->>Child: wait for agent readiness
    Manager->>Remote: create remote graph client
    Manager->>Remote: set workspace claim and fingerprint
    Manager-->>Client: hand off agent and process ownership
```

This sequence shows the construction handoff. Before a successful return, the manager owns the child; its `finally` invokes idempotent cleanup after every setup failure, including cancellation. An explicit malformed or missing MCP config fails before process creation, whereas project- and user-discovered MCP configuration is handled by server-side discovery.

`ServerConfig` is the process-boundary contract: the parent serializes it to environment variables and the server reconstructs it from the same prefix. It carries the model and model parameters, interaction and tool choices, sandbox selection, MCP settings, extension settings, and project context. A supplied filesystem-tool allowlist is a security control: malformed, empty, or unknown values received from the environment fail closed, and an explicit list must include `read_file`.

At server construction, `_make_graphs` snapshots dotenv-derived workspace environment and credentials before activating that immutable environment for runtime assembly. Blocking path/bootstrap, model, plugin-discovery, and synchronous graph-construction work is moved off the event loop. It creates the configured model; adds built-in `fetch_url` and thread-ID tools; conditionally adds web search when workspace credentials provide a Tavily key; and discovers MCP tools unless `no_mcp` is set. Criteria and rubric agents receive only read-only built-ins and MCP tools explicitly and coherently annotated read-only, rather than a name-based approximation.

Sandbox creation is server-process lifetime state. A configured provider is opened while building the runtime, held for cleanup through `atexit`, and passed into `create_cli_agent`; unsupported, absent, invalid, or failed sandbox setup emits a startup error and exits. With experimental extensions enabled, the server loads extensions using the workspace path, interactive/headless mode, project-trust decision, and explicit extension paths; load errors are warnings, while an active registry is bound to server extensions and shut down if graph construction later fails.

`create_cli_agent` receives the resolved model, tool/MCP set, sandbox, approval and shell policy, filesystem and interpreter choices, memory/skills/subagents, rubric settings, extension registry, environment, and credential snapshot. The resulting `ServerRuntime` keeps the compiled agent, the exact `CompositeBackend` used to construct it, its MCP metadata, and an offload operation derived from that backend. Failure to expose that operation is a construction failure, not a partially functional server: `/offload` must share the same backend and archive policy as the agent.

## Workspace selection, caching, and policy drift

The initial graph load is not enough to select a request runtime. `make_graph` receives LangGraph execution context when serving a run; it requires a nonempty thread ID and workspace context, verifies the durable thread-to-workspace binding, then selects that binding's runtime. Calls without execution context use the configured launch runtime. This prevents an arbitrary request from selecting a workspace merely by naming a directory.

```mermaid
flowchart TD
    Request["make_graph invocation"] --> HasExecution{"Execution context present"}
    HasExecution -- No --> Launch["Get launch ServerRuntime"]
    HasExecution -- Yes --> Valid{"Thread ID and workspace context valid"}
    Valid -- No --> Reject["Raise ValueError"]
    Valid -- Yes --> Binding["Verify durable thread workspace binding"]
    Binding --> Policy["Resolve current workspace policy"]
    Policy --> Drift{"Policy and fingerprint unchanged"}
    Drift -- No --> Conflict["Raise workspace conflict"]
    Drift -- Yes --> Cache{"Runtime cached by resource key"}
    Cache -- Yes --> Touch["Refresh LRU position"]
    Touch --> Agent["Return bound runtime agent"]
    Cache -- No --> Build["Claim sandbox and build workspace runtime"]
    Build --> Store["Insert into bounded LRU"]
    Store --> Agent
    Launch --> Agent
```

This flow distinguishes launch-time construction from execution-time workspace routing.

The process runtime is lock-protected and constructed once. Its cache is load-bearing: rebuilding per request would repeat MCP discovery, create/leak sandbox sessions, register duplicate `atexit` handlers, and make the graph and offload route disagree about backend state. Workspace-specific runtimes use a second lock-protected LRU keyed by the binding resource key; an access refreshes recency and insertion evicts the oldest entry once the cache exceeds 32 entries.

Before reusing or building a workspace runtime, the server resolves current policy for the binding and compares both project-policy fields and the full workspace fingerprint against the durable binding. It preserves bound extension trust while resolving current policy, but reports any remaining project policy drift or fingerprint change as `WorkspaceConflictError`; even a trust-store read failure is treated as a policy change. A process-wide sandbox may be claimed only by one workspace ID, so a second workspace cannot silently share it.

On first use of a thread, `RemoteAgent` posts `cwd`, and when configured the session claim plus fingerprint, to `/dcode/threads/{thread_id}/workspace`; it validates and caches the returned descriptor per thread. `set_workspace` requires the policy and fingerprint together and clears that cache. This client cache avoids repeated binding requests but is not the authority: the durable server binding and per-request validation govern selection.

## Streaming, state, and operation failures

`RemoteAgent.astream` requires a thread ID, gets that thread's workspace descriptor, adds it to the runtime context, and delegates SSE framing, `messages-tuple` negotiation, namespace extraction, and interrupt detection to `RemoteGraph`. It defaults to `messages` and `updates`, deserializes message dictionaries for the UI, converts `__interrupt__` update data, and deliberately leaves state snapshots serialized. A message conversion failure is counted and reported after streaming rather than aborting unrelated events.

Checkpoint data and the development server's HTTP thread row have separate lifecycles. `aget_state` returns empty state for a missing remote thread or the SDK's known no-checkpoint `TypeError`, but logs and re-raises other errors. `aensure_thread` idempotently creates the live HTTP row, allowing a persisted thread to receive state mutations after a server restart. For a state-update conflict, the client cancels pending/running runs concurrently with bounded per-run waits and retries the update once.

Lost or cancelled work is recovered destructively, not resumed. `aabandon_pending_work` cancels active runs, calculates error `ToolMessage` values only for unanswered calls in the trailing AI-message turn, writes `__end__` to discard queued work, then re-reads state and fails if queued nodes, tasks, or interrupts remain. The trailing-turn restriction preserves tool-use/result adjacency and avoids producing invalid history for older interrupted calls.

The server also exposes backend-owned offload through an authenticated custom HTTP route rather than a second addressable graph. The generated configuration enables custom-route authentication when a deployment supplies auth; local process launch explicitly uses noop auth and loopback binding. The remote offload client ensures the thread exists, forwards workspace context, validates protocol responses, and treats an absent route as a server compatibility error.

## Retry, startup, and shutdown semantics

`CodeModelRetryMiddleware` wraps the model-node handler rather than an entire agent turn and is installed inside side-effecting automatic compaction. Thus a transient provider retry does not replay completed tools, summary generation, or archive append. It remains installed even with a zero startup budget because a runtime-selected model can provide a request-time budget. `GraphBubbleUp` passes through as graph control flow; classified transient failures are retried while budget remains using usable `Retry-After` guidance or jittered exponential backoff, while terminal failures are re-raised rather than turned into artificial AI output. Correlated attempt and retry events record whether visible output may already have started, without allowing diagnostic-event failure to fail the run.

Runtime construction is a startup barrier. The process runtime factory checks managed configuration health, emits `DEEPAGENTS_STARTUP_ERROR:` and exits nonzero when construction fails; early child-exit health polling extracts that marker and adds it to the parent error. Request-scoped callers must catch `SystemExit` and map it appropriately rather than terminating an already-serving child. This is especially relevant to operation routes.

The child environment strips `PYTHONPATH` and other startup-influencing values before launching the server interpreter. It preserves the launch `PYTHONPATH` only in a dedicated carrier for downstream agent execute commands, and protects immutable profile/carrier values from later environment overrides.

`ServerProcess` owns shutdown. On POSIX the child starts in a dedicated session/process group; stop sends `SIGTERM` to that group, waits for the whole group, then escalates with `SIGKILL` if necessary, refusing to target dcode's own group. Windows sends Ctrl+Break for graceful shutdown and escalates by killing the root process; a surviving descendant can therefore be orphaned. Signaling failures are logged because cleanup cannot guarantee that the child is gone.

## Focused regression coverage and safe changes

Server-graph unit tests inject builders to verify one construction under repeated and concurrent factory access, the startup marker/exit contract, and nonblocking bootstrap. They also exercise ordered, identity-based criteria tool selection and fail-closed MCP annotations. Server-manager tests cover config serialization, filesystem allowlist rejection, project-relative MCP path normalization, session-only workspace claims, scaffold forwarding, generated built-in graph/operation registration, and option forwarding.

When changing this area:

1. Keep policy partitioning and durable binding validation aligned with the workspace cache key.
2. Do not make a process-wide sandbox reusable across workspace IDs.
3. Keep graph and offload construction on the same `ServerRuntime` backend.
4. Preserve startup-marker parsing, request-scope `SystemExit` containment, and cancellation-safe launch cleanup.
5. Keep model retries inside compaction and pending-work recovery ordered as cancel, trailing-turn repair, `__end__`, then verification.
6. Test both platform shutdown scopes when changing process ownership or escalation.
