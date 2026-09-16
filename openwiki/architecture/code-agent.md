---
type: architecture-overview
title: Deep Agents Code Architecture
description: Architecture of dcode's normal terminal-client and local LangGraph-server boundary, workspace-scoped runtimes, streaming, persistence, cleanup, and separate ACP stdio mode.
tags: [deepagents-code, dcode, architecture, client-server, langgraph, acp, streaming]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-09T08:05:37.706Z
sources:
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-1728494bdd59604ce9b5f65b
    resource: repo://libs/code/deepagents_code/_server_config.py
  - id: openwiki-source-4d4186e9d62fb4abe495cdd0
    resource: repo://libs/code/deepagents_code/acp.py
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-074ce96a8baea27a6c43328b
    resource: repo://libs/code/deepagents_code/client/launch/server.py
  - id: openwiki-source-ecf20e7a2684ba0d2ae7d701
    resource: repo://libs/code/deepagents_code/client/non_interactive.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-52d96f61bc4737f02a18cf79
    resource: repo://libs/code/deepagents_code/configuration/resolver.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-030d8bd153a9c3ea2a99cb7d
    resource: repo://libs/code/deepagents_code/workspace.py
  - id: openwiki-source-5dc287d30945406e0821cb29
    resource: repo://libs/code/tests/integration_tests/test_acp_mode.py
  - id: openwiki-source-439d3e6c6f1b62e6d282df3f
    resource: repo://libs/code/tests/unit_tests/test_remote_client.py
  - id: openwiki-source-784e764f7f5eb5169220c3d2
    resource: repo://libs/code/tests/unit_tests/test_server_graph.py
  - id: openwiki-source-877b53371bf970f1b38a1809
    resource: repo://libs/code/tests/unit_tests/test_workspace.py
generated: { by: "openwiki/0.4.2", at: "2026-09-09T08:05:37.706Z" }
---

# Deep Agents Code Architecture

`deepagents-code` (`dcode`) is a reference terminal coding-agent product built on the `deepagents` SDK. It combines the SDK harness with terminal UX, persistence, tools, skills, and optional sandboxed execution.

There are two deliberately distinct execution paths:

- **Normal interactive and headless dcode** run a terminal client and an owned local `langgraph dev` server in separate processes. The client owns presentation, input, and approvals; the server owns the model, graph, tools, memory, skills, backend, and checkpoints.
- **`dcode --acp`** runs an ACP server in the launching process over stdio. It builds local session graphs and does not start `langgraph dev` or use `RemoteAgent`.

This is an ownership boundary, not merely a transport choice: changes to the normal-server path must be evaluated against ACP separately.

## Normal client-server run

Interactive mode renders and collects input in the Textual app. Headless mode runs one task against the same `RemoteAgent`/server arrangement but streams machine-oriented output to stdout; `--quiet` suppresses tool and file-operation diagnostics so stdout contains response text only.

```mermaid
sequenceDiagram
    participant User
    participant Client as Terminal client
    participant Manager as Server manager
    participant Server as LangGraph server
    participant Graph as Agent graph

    Client->>Manager: Resolve inputs and project context
    Manager->>Manager: Validate MCP config and scaffold workspace
    Manager->>Server: Spawn langgraph dev on loopback
    Manager->>Server: Wait for agent graph readiness
    Manager-->>Client: Return configured RemoteAgent
    User->>Client: Prompt or approval
    Client->>Server: Bind thread workspace when needed
    Client->>Server: Send run and receive SSE events
    Server->>Graph: Validate context and choose runtime
    Graph-->>Server: Events and checkpoint changes
    Server-->>Client: SSE events
    Client-->>User: Render or print output
```

This sequence shows the normal local path. The workspace bind and execution context make runtime selection server-authoritative; ACP has its own boundary below.

### Startup and configuration handoff

`start_server_and_get_agent` captures project context (or uses an explicit cwd), validates an explicit MCP file before spawning, resolves `ServerConfig`, and scaffolds a temporary LangGraph workspace. That workspace contains `pyproject.toml`, `langgraph.json`, and a generated checkpointer module. The module reads the application session database path from an environment variable and yields `AsyncSqliteSaver`, rather than baking the path into generated code.

The generated graph reference is `deepagents_code.server_graph:make_graph`. For that built-in reference, `langgraph.json` also registers dcode's offload HTTP app with custom-route auth enabled; a custom graph reference does not get `/offload`. The normal local server binds `127.0.0.1` and defaults to port `0`, deliberately obtaining an ephemeral port instead of occupying `langgraph dev`'s conventional port 2024.

The client writes resolved launch settings as `DEEPAGENTS_CODE_SERVER_*` variables. `ServerConfig.to_env()` and `ServerConfig.from_env()` are the shared schema, keeping variable names, serialization, and defaults in one place. Configuration precedence uses lower numeric ranks first: managed policy, CLI arguments, retained reload values, environment, user `config.toml`, then typed defaults. See [configuration layering](/openwiki/concepts/config-layering.md).

The child environment is also a security boundary. Startup-sensitive inherited variables, including `PYTHONPATH`, are stripped before the server interpreter starts. The original `PYTHONPATH` is carried separately solely for approval-gated shell execution, and immutable child environment handling pins profile selection to the client launch profile. See [security](/openwiki/operations/security.md).

## Workspace binding and runtime selection

`RemoteAgent` wraps LangGraph's `RemoteGraph`: the underlying client performs HTTP/SSE parsing, `messages-tuple` negotiation, namespace extraction, and interrupt detection. dcode normalizes thread IDs and converts streamed message dictionaries to message objects for the Textual adapter, while state snapshots remain in the server's serialized form. It also retries one state update after an HTTP 409 by cancelling active runs, which addresses a stream cancellation that races server-side completion.

Before a thread is used, `RemoteAgent` posts the configured cwd, workspace policy, and fingerprint to `/dcode/threads/{thread_id}/workspace`, then caches the server-returned descriptor. It separately registers the HTTP thread record because SQLite checkpoint data can survive a server restart even when the dev server has no live thread row.

```mermaid
flowchart TD
    Begin["RemoteAgent needs a thread workspace"] --> Bind["POST workspace claim"]
    Bind --> Persist["Server resolves policy and persists binding"]
    Persist --> Run["Client sends thread ID and workspace context"]
    Run --> Check["make_graph validates durable binding"]
    Check --> Match{"Context and policy match"}
    Match -- No --> Reject["Reject workspace conflict"]
    Match -- Yes --> Cached{"Runtime cached by resource key"}
    Cached -- Yes --> Use["Use cached workspace runtime"]
    Cached -- No --> Build["Resolve bound config and build runtime"]
    Build --> Use
```

This flow shows how an untrusted workspace claim becomes a durable, server-validated runtime choice.

The server canonicalizes absolute cwd and project-root identity and stores a binding atomically in the session SQLite database. A thread cannot silently move to another workspace or configuration fingerprint: later binds and execution contexts must match the durable payload or raise `WorkspaceConflictError`. The durable policy excludes credentials and prompt material. Old binding schemas are migrated only after the preserved identity/session controls are validated.

With an execution runtime context, `make_graph` requires a nonempty thread ID and workspace context, verifies the durable binding, and selects the runtime keyed by its persisted resource key. Without execution context, it uses the configured launch workspace if present; otherwise it uses the lock-protected process runtime. Every workspace-runtime lookup re-resolves configuration and rejects project-policy or server-configuration drift before returning even a cached runtime. A revoked project-extension trust therefore invalidates use, while a new grant is pinned out of an already-bound thread. See [state persistence](/openwiki/concepts/state-persistence.md).

## Graph assembly and shared server resources

`create_cli_agent` is the composition entry point for the resolved model, built-in and MCP tools, optional sandbox, filesystem and approval policy, memory, skills, interpreter configuration, subagents, grading context, and workspace credentials/environment. It returns the compiled graph and composite backend; the server derives its offload operation from that backend, so normal graph execution and `/offload` share backend ownership.

Criteria creation and rubric grading receive built-in external-context tools and only MCP tools explicitly and coherently annotated read-only. Missing, malformed, contradictory, or mutating MCP annotations fail closed.

The process runtime is constructed once under a lock. That cache is correctness-critical: it prevents repeated MCP discovery, sandbox creation, and duplicate `atexit` registration, and ensures the graph and offload route share compatible resources. Workspace runtimes live in a shared-lock LRU keyed by the durable resource key and capped at 32 entries. Runtime construction uses the workspace's immutable environment snapshot, including its dotenv-derived credentials, without mutating the server process environment.

Some resources constrain what one server process may host. A configured sandbox is process-wide and the first workspace claims it; another workspace is rejected even if the first build failed. LangSmith tracing settings are also process-lifetime: a workspace with different tracing/redaction settings is rejected rather than rerouting cached or concurrent runtime traces. Run a separate normal server when either constraint prevents co-hosting.

## Failure handling and teardown

Runtime construction is a startup barrier. A build failure emits a `DEEPAGENTS_STARTUP_ERROR:` marker and exits with code 1, allowing the parent to extract a specific cause from child logs rather than report only a readiness timeout. In request-scoped offload handling, that `SystemExit` is contained and translated to a service-unavailable response rather than killing the server mid-request.

The manager stops its owned process when spawning, graph readiness, remote-client setup, or workspace setup fails. Its `finally` cleanup handles cancellation as well as ordinary exceptions. `server_session` stops the process on normal exit and drains queued notices for debug-preserved logs.

On POSIX, the server is started in a dedicated process group. Teardown signals the group, waits for descendants as well as the root, then escalates to `SIGKILL` if necessary. Windows uses a graceful console signal where possible, but hard-kill escalation reaches only the root process handle; a surviving descendant can be orphaned.

Focused tests cover message and interrupt conversion at the remote stream boundary, binding idempotence and first-bind races, substituted workspace and policy conflicts, non-secret persisted policy, schema migration, workspace runtime cache and policy-drift behavior, sandbox ownership, and ACP protocol startup. These tests are useful guardrails when changing the boundaries described here.

## ACP stdio lifecycle

`--acp` calls `_run_acp_cli_async` in the launching process. It resolves the initial model and MCP tools, keeps dcode's checkpointer open while serving, and passes a `build_agent(context)` callback to the ACP server. That callback uses the ACP session's selected model (or the resolved default) and cwd to make `ProjectContext` and call `create_cli_agent` with the shared checkpointer. These session-local graphs are not normal workspace-runtime-cache entries.

In Auto mode, dcode uses its `AgentServerACP` adapter. The adapter wraps local graph streaming to store trusted Auto approval state, attach prompt metadata, and supply CLI context. YOLO requires prior acknowledgement, and `--auto-classifier-model` is accepted in ACP only when the resolved mode is Auto. ACP failures go to stderr and return a nonzero status; they do not follow the normal subprocess startup-marker path. A `finally` block cleans up the ACP MCP session manager.

The ACP smoke test starts `deepagents --acp --no-mcp`, initializes the protocol, opens a session using the current cwd, and asserts that it has an ID. It protects the stdio lifecycle without conflating it with the loopback normal-server path. See [ACP](/openwiki/integrations/acp.md).

## Extension and operations guidance

Tools, MCP servers, skills, subagents, sandbox providers, hooks/commands, and authorized Python extensions are composition points. In normal-server mode, resource-affecting settings become part of the durable workspace policy/fingerprint; do not expect a bound thread to live-reconfigure into a different resource policy. Custom graph references also need an independent offload strategy because dcode does not register `/offload` for them.

For operational usage, see [run a dcode session](/openwiki/workflows/run-dcode-session.md), [runtime behavior](/openwiki/architecture/runtime-behavior.md), [context management](/openwiki/concepts/context-management.md), and [configuration layering](/openwiki/concepts/config-layering.md).
