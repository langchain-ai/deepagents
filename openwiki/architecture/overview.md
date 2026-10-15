---
type: system architecture
title: Repository Architecture Overview
description: Architecture and release map for the Deep Agents Python monorepo, covering the SDK, dcode, ACP, evaluations, Talon, and partner integrations. Explains current package dependencies, graph assembly and delegation boundaries, persistence, and host lifecycle ownership.
tags: [architecture, deepagents, langgraph, sdk, talon, integrations]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-15T08:05:41.400Z
sources:
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-ffc41789c892ca61e2829a4c
    resource: repo://libs/acp/deepagents_acp/server.py
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-4d4186e9d62fb4abe495cdd0
    resource: repo://libs/code/deepagents_code/acp.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-6a038e6e1a11f450bcafce54
    resource: repo://libs/talon/deepagents_talon/__main__.py
  - id: openwiki-source-517d471fea32c6a16331f5e4
    resource: repo://libs/talon/deepagents_talon/channels/__init__.py
  - id: openwiki-source-c2be68f237284dc06b9c12f7
    resource: repo://libs/talon/deepagents_talon/checkpoint_backends.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-cebe4ea270e21dce4de9b074
    resource: repo://libs/talon/deepagents_talon/interfaces.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-a69daa62c9a3eb9a49f09bf9
    resource: repo://libs/talon/tests/test_host.py
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-10-15T08:05:41.400Z" }
---

# Repository Architecture Overview

This repository is a monorepo of independently versioned Python packages, not one deployable service. `deepagents` is the reusable graph-construction harness. dcode is the terminal product, ACP translates graphs for editor clients, Talon hosts graphs for long-running channels and schedules, and `deepagents-evals` measures behavior. Partner packages add optional integrations rather than forming a required runtime layer.

- [SDK construction and execution](./sdk-construction-execution.md)
- [Code agent](./code-agent.md)
- [Runtime behavior](./runtime-behavior.md)
- [ACP integration](../integrations/acp.md)
- [Talon integration](../integrations/talon.md)
- [Development](../operations/development.md)

## Packages, releases, and dependency direction

| Package | Current version and Python baseline | Architectural role |
| --- | --- | --- |
| `deepagents` | `0.7.25`; `>=3.11,<4.0` | Core SDK: `create_deep_agent`, middleware, and pluggable backends. |
| `deepagents-acp` | `0.0.13`; `>=3.11` | Agent Client Protocol bridge for serving graphs to editor clients. |
| `deepagents-code` (`dcode`) | `0.1.85`; `>=3.12,<4.0` | Reference terminal coding-agent product, including interactive and headless clients. |
| `deepagents-talon` | `0.0.9`; `>=3.12` | Experimental local host for channels, turns, and cron scheduling. |
| `deepagents-evals` | `0.0.1`; `>=3.12,<3.14` | End-to-end behavioral evaluation and Harbor benchmark suite. |
| `langchain-quickjs` | `0.3.9`; `>=3.11,<4.0` | Optional JavaScript REPL middleware partner integration. |

Release Please manages separate release lines for the SDK, ACP, dcode, Talon, and five partner packages: Daytona, Modal, Runloop, Vercel, and QuickJS. The manifest records dcode at `0.1.85`; `deepagents-evals` is absent from both the manifest and configured package list. Release Please creates separate draft release pull requests and skips automated GitHub releases. Thus, a package under `libs/` does not itself imply participation in the automated release line.

Published package dependencies point toward the SDK: ACP depends on `deepagents`; dcode pins `deepagents==0.7.25` and depends on ACP; Talon requires both the SDK and dcode; evals requires the SDK and dcode; and QuickJS requires the 0.7.x SDK plus LangChain and LangGraph. Their Python baselines are, respectively, ACP `>=3.11`, dcode `>=3.12,<4.0`, Talon `>=3.12`, evals `>=3.12,<3.14`, and QuickJS `>=3.11,<4.0`. In monorepo development, local `uv` source mappings make the SDK dependencies editable where declared.

```mermaid
flowchart TD
  Dcode["deepagents-code"] --> SDK["deepagents SDK"]
  Dcode --> ACP["deepagents-acp"]
  ACP --> SDK
  Talon["deepagents-talon"] --> SDK
  Talon --> Dcode
  Evals["deepagents-evals"] --> SDK
  Evals --> Dcode
  Evals --> QuickJS["langchain-quickjs"]
  QuickJS --> SDK
  SDK --> LangChain["LangChain create_agent"]
  LangChain --> LangGraph["LangGraph runtime"]
```
This shows the principal package and runtime dependency direction: an arrow points from a consumer to a required package or lower layer.

The SDK base dependencies include LangChain, LangChain Core, Anthropic and Google model integrations, LangSmith, `packaging`, and `wcmatch`. AWS, QuickJS, and video are optional extras rather than base requirements.

## Runtime stack and graph assembly

Deep Agents is a three-layer stack:

- **LangGraph** is the runtime for state, checkpoints, streaming, and interrupts.
- **LangChain `create_agent()`** is the agent abstraction that builds the model-plus-tools-plus-middleware loop on that runtime.
- **Deep Agents** is the opinionated harness above `create_agent()`, supplying long-horizon defaults such as filesystem backends, subagents, context management, skills, memory, profiles, and tool policy.

`create_deep_agent()` is the SDK assembly point. It resolves the model and harness profile, resolves the backend, assembles the main-agent middleware stack, builds the default general-purpose subagent, composes the final system prompt, and delegates to LangChain's `create_agent(...)` to produce the runnable graph. Its public arguments are the primary application extension boundary: callers can supply tools, middleware, subagents, skills, memory, filesystem permissions, interrupt settings, schemas, a checkpointer, store, cache, and graph name.

### Delegation boundaries

Synchronous declarative and compiled subagents are exposed through the main agent's `task` tool. The SDK adds a default synchronous `general-purpose` subagent unless the active profile disables it; if there are no synchronous subagents, the `task` tool is absent. Declarative subagents inherit the parent custom state schema and top-level approval configuration unless they override it. A compiled runnable is already assembled, so it must be compiled with a compatible schema and configures its own approvals.

`AsyncSubAgent` is a separate remote-delegation boundary: `create_deep_agent()` routes specifications with `graph_id` to `AsyncSubAgentMiddleware`, rather than `SubAgentMiddleware`. It uses the LangGraph SDK and Agent Protocol to create a remote background run, returns a task ID immediately, and gives the parent tools to check, update, cancel, and list tasks. The remote graph owns its schema and approval behavior. Its persisted task state records remote thread and run IDs plus status, allowing the parent to monitor work without blocking its own turn. Local ASGI transport requires an asynchronous parent entry point; synchronous invocation requires a reachable URL.

Several constraints protect this assembly boundary:

- A harness profile cannot exclude protected `FilesystemMiddleware` or synchronous `SubAgentMiddleware`; invalid or unmatched exclusions fail construction.
- Filesystem permissions are ordered first-match rules enforced only for built-in filesystem tools. `deny` returns a permission error and `interrupt` installs or extends approval behavior; this is not backend-wide authorization.
- A custom state schema must extend `DeepAgentState` to preserve its `DeltaChannel` messages reducer. Parallel subagent updates to a shared custom field need a reducer or LangGraph raises `InvalidUpdateError`.

For middleware ordering and execution details, see [SDK construction and execution](./sdk-construction-execution.md).

| Boundary | Owns | Does not own |
| --- | --- | --- |
| SDK | Harness composition, middleware, backend routing, profile policy, and SDK subagents. | UI, editor transport, channel connections, or cron delivery. |
| LangGraph | Graph state, checkpoints, and stream/interrupt semantics. | Product-specific transport and host lifecycle. |
| dcode | Terminal client/server product and coding-agent extensions. | Generic harness behavior. |
| ACP | Protocol and session translation around a graph. | Terminal presentation or channel policy. |
| Talon | Process lifecycle, channel adapters, turn coordination, schedules, and host-owned stores. | Production multi-tenant containment. |
| Evals | Repeatable behavioral measurement and benchmark integration. | Live request serving. |

Persistence is intentionally split. A LangGraph checkpointer is passed through SDK graph construction for execution state; a Deep Agents backend independently governs file, memory, and execution capabilities. Talon adds a conversation archive and cron-job persistence. On Talon's model-backed CLI path, the checkpointer and archive open together and are wrapped in `ConversationSaver`, so checkpoint state alone is not channel delivery history.

## Product, protocol, and evaluation boundaries

### dcode and ACP

`deepagents-code` is a reference terminal coding-agent product on the SDK. Its terminal client and agent server run in separate processes: the client owns presentation, input, and approval interaction; the server owns graph execution and streams events back. Headless mode uses the same server runtime.

`AgentServerACP` accepts either a compiled graph or a graph factory given an `AgentSessionContext` containing working directory, mode, and optional model. With `load_sessions=True`, it advertises and implements session loading, verifying persisted checkpoint metadata and working-directory identity before replaying a session. Dcode's ACP specialization wraps each session graph to set trusted Auto-mode approval state and prompt metadata during streaming. Keep generic protocol translation in ACP and terminal policy in dcode.

### Evaluations and partners

`deepagents-evals` runs agents against real LLMs, captures tool calls, file mutations, and final responses, and scores correctness and efficiency. Harbor integration supports sandboxed benchmarks such as Terminal Bench 2.0. It is a quality-measurement boundary rather than an online serving component.

Partner packages add external execution or middleware capabilities. `langchain-quickjs`, for example, provides JavaScript REPL middleware and depends on Deep Agents 0.7.x, LangChain, and LangGraph. Configure such integrations at the consuming product or host boundary instead of making them mandatory SDK dependencies.

## Talon: long-running local host

Talon is an experimental alpha local runtime host. One process event loop coordinates channel adapters, an agent runtime, and optional cron scheduling. It is not a production containment or multi-tenant security boundary: channel access should be treated as access to configured agent capabilities, credentials, MCP tools, and host resources. Its opt-in sandbox relocates shell and filesystem tools, but MCP tools, web tools, and channel media handling stay on the host.

```mermaid
sequenceDiagram
  participant Cli as Talon CLI
  participant Checkpoints as Checkpoint backend
  participant Archive as History archive
  participant Runtime as DeepAgentRuntime
  participant Host as TalonHost
  participant Channel as Channel adapter
  participant Graph as SDK graph
  Cli->>Checkpoints: Open configured saver
  Cli->>Archive: Open history store
  Cli->>Runtime: Construct ConversationSaver runtime
  Cli->>Host: Construct host
  Host->>Runtime: Start and build graph
  Host->>Channel: Bind callbacks and start
  Channel->>Host: Inbound message
  Host->>Runtime: Invoke serialized turn
  Runtime->>Graph: Run graph
  Graph-->>Runtime: Result or interrupt
  Runtime-->>Host: Agent result
  Host->>Channel: Deliver result
```
This sequence separates CLI-owned resource opening, host-owned transport and delivery, and runtime-owned graph construction and invocation.

### Construction, turns, and lifecycle safety

If no model is configured, the CLI uses `EchoAgentRuntime`, which permits lifecycle and channel wiring without provider credentials. The model-backed path can open a sandbox, load MCP tools, and create `DeepAgentRuntime`; `PersistentCronScheduler` is attached only when at least one channel is configured.

At `DeepAgentRuntime.start()`, Talon resolves subagents, obtains a tool-approval snapshot, and creates its SDK graph. Its composition adds filesystem behavior, selected-model summarization and model selection, progress messages, task tooling, background-subagent behavior, Talon tools, and caller middleware before calling `create_deep_agent()`.

Each `invoke()` rejects calls before startup, refreshes runtime tools, and rebuilds the graph if the approval snapshot changed. The turn binds request-scoped approvals, authorization, history/archive scope, cron origin, selected model, progress messaging, the invocation graph, and pending background results. It resets every binding in `finally`, preventing state from one channel or scheduled turn leaking into another. `stop()` cancels background workers before clearing the graph and closing the checkpointer; when cancellation fails, it intentionally leaves resources open so an active worker cannot write to a closed saver.

`TalonHost` starts the agent runtime before binding and starting channels, then starts an optional scheduler. A partial startup is unwound in reverse order. Shutdown cancels active work before stopping channels, scheduler, and runtime; failures in one component stop are isolated so later components are still stopped. Focused host tests cover normal lifecycle, partial channel startup, cancellation and interruption recovery, authorization callback routing, and delivery behavior.

### Transport and storage extension points

A channel adapter carries provider-specific `ChannelMessage` values to the host and may provide reactions. The required adapter operations cover start/stop, inbound callback registration, text and media send, edit, typing, and connection status. Built-in exports cover WhatsApp, Telegram, Discord, and Slack.

Talon selects its checkpointer by URI. Built-in SQLite (`sqlite` and `file`), PostgreSQL, and MongoDB schemes are selected before a trusted `deepagents_talon.checkpoint_backends` entry-point plugin. The plugin receives the unchanged URI and owns setup and cleanup. Unexpected opening failures become a credential-safe configuration error. Use separate remote databases for separate assistants: remote checkpoint thread IDs are not automatically namespaced by assistant ID.

## Safe change and test starting points

- Put reusable graph composition changes in `deepagents`, starting at `create_deep_agent()`; do not pull UI, channel, or transport policy into the SDK.
- Keep synchronous `task` delegation distinct from `AsyncSubAgent` remote task orchestration; changes to cross-boundary schema, approvals, or cancellation must be applied on the graph that owns them.
- Put terminal client/server behavior in dcode, preserving the common server path for headless execution.
- Put ACP streaming and session replay in `deepagents-acp`, retaining persisted-metadata and working-directory validation.
- Put Talon delivery, conversation serialization, and scheduler behavior in `TalonHost`; put graph composition, resource refresh, and request-context handling in `DeepAgentRuntime`.
- Treat checkpoint backend entry points and partner integrations as trusted operator-installed extension boundaries. Preserve their cleanup ownership and credential-safe failure behavior.
- For Talon lifecycle or cancellation changes, begin with `libs/talon/tests/test_host.py`, then run runtime and package-specific tests for the changed boundary.
