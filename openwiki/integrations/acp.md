---
type: protocol integration
title: Agent Client Protocol Integration
description: How deepagents-acp projects LangGraph agents into editor-facing ACP sessions, including durable recovery, permission handling, and hidden local context. It also explains the versioned dcode ACP surface and its in-process stdio boundary.
tags: [acp, deepagents, langgraph, dcode, stdio, sessions, streaming]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-06T08:04:19.814Z
sources:
  - id: openwiki-source-532ea636a0657c1d2714bd7a
    resource: repo://libs/acp/CHANGELOG.md
  - id: openwiki-source-ffc41789c892ca61e2829a4c
    resource: repo://libs/acp/deepagents_acp/server.py
  - id: openwiki-source-1ffb4d0f447fcc4e9ca248ef
    resource: repo://libs/acp/deepagents_acp/utils.py
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-f188d13aef04b643ced86520
    resource: repo://libs/acp/tests/unit_tests/test_execute_permissions.py
  - id: openwiki-source-986e87e057d803bd0fa17738
    resource: repo://libs/acp/tests/unit_tests/test_local_context.py
  - id: openwiki-source-4d4186e9d62fb4abe495cdd0
    resource: repo://libs/code/deepagents_code/acp.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-8896ee5af2462c92e51bfb6e
    resource: repo://libs/code/deepagents_code/local_context.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-91c9283d1547adfffd627c43
    resource: repo://libs/code/deepagents_code/thread_ownership.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-e0e6f6b6ce0dbf7a671d682f
    resource: repo://libs/code/tests/unit_tests/test_main_acp_mode.py
  - id: openwiki-source-a5e918d96b1dae3f7adec3f5
    resource: repo://libs/code/tests/unit_tests/test_thread_ownership.py
generated: { by: "openwiki/0.4.2", at: "2026-11-06T08:04:19.814Z" }
---

# Agent Client Protocol Integration

[Agent Client Protocol (ACP)](https://agentclientprotocol.com/overview/introduction) lets an editor communicate with an agent process over stdio. `deepagents-acp` provides `AgentServerACP`, which implements ACP while projecting a LangGraph graph's sessions, stream events, and compatible human-in-the-loop interrupts. It is an adapter, not an agent runtime: the embedding host owns graph construction, tools, checkpoint storage, and the agent's interrupt policy.

There are two host choices:

- A custom host can wrap a compatible compiled LangGraph graph, or supply a graph factory.
- `dcode --acp` runs the prebuilt dcode agent as an ACP stdio server. It builds graphs, opens the checkpointer, and handles ACP **inside the dcode process**.

## Package contract and compatibility

`deepagents-acp` 0.0.13 is an alpha MIT package for Python 3.11 through Python 3.x below 4. It depends on `deepagents`, `agent-client-protocol>=0.10.1`, and `python-dotenv>=1.2.2`. The standalone package is therefore the reusable editor-to-graph adapter, not a distribution of dcode's coding-agent policy or tool set.

`deepagents-code` 0.2.5 requires Python 3.14 through below 3.15 and declares `deepagents-acp>=0.0.13,<1.0.0`. A dcode ACP deployment consequently follows dcode's stricter Python requirement. Version 0.0.13 fixed reusable shell authorization to apply only to exact commands and preserved local-context prompts across compaction.

For a custom graph, install `deepagents-acp`, construct the graph (including a checkpointer if it must be recoverable), pass it to `AgentServerACP`, and run it with `acp.run_agent`. The README's demo uses `MemorySaver`, which is appropriate for a process-local example but not restart recovery. A factory is the extension point when the graph must vary by ACP session cwd, selected mode, or model.

## Transport boundary: ACP versus normal dcode

ACP mode must not be confused with a normal interactive dcode session. In the normal terminal path, the Textual client uses `RemoteAgent`; it lazily creates a LangGraph `RemoteGraph` that talks HTTP and SSE to the normal local LangGraph server path. In ACP mode, dcode does **not** launch that Textual UI or normal local LangGraph server path, and no `RemoteGraph` participates. Instead, the editor speaks ACP over stdio to an in-process adapter and graph factory.

```mermaid
flowchart TD
    Editor["ACP editor"] --> Stdio["stdio ACP connection"]
    Stdio --> Acp["dcode ACP process"]
    Acp --> Factory["in-process per-session graph factory"]
    Factory --> LocalGraph["in-process LangGraph graph"]
    LocalGraph --> Saver["dcode checkpointer"]

    Terminal["dcode terminal user"] --> Tui["Textual client"]
    Tui --> Remote["RemoteAgent"]
    Remote --> RemoteGraph["RemoteGraph over HTTP and SSE"]
    RemoteGraph --> Server["normal local LangGraph server"]
```

*ACP directly hosts an in-process graph and never launches the normal local server path; the terminal UI uses a remote graph client.*

## Generic adapter lifecycle

`AgentServerACP` accepts either a compiled `CompiledStateGraph` or a factory receiving `AgentSessionContext(cwd, mode, model)`. Modes and model selectors are valid only for a factory; passing either with a compiled graph raises `ValueError`. The ACP `session_id` is also the LangGraph `thread_id`.

At `initialize`, the server advertises image input and advertises `session/load` only when `load_sessions=True`. `new_session` creates an ID, retains its cwd and supplied ACP MCP descriptors, initializes selectors, and—when loading is enabled—writes ACP session metadata to the checkpoint before returning. Mode and model option values must be strings and must be recognized; a valid change resets the graph and persists the selection when durable loading is enabled.

```mermaid
sequenceDiagram
    participant Client as ACP client
    participant Bridge as AgentServerACP
    participant Factory as Graph factory
    participant Graph as LangGraph graph
    participant Store as Checkpointer
    Client->>Bridge: initialize
    Bridge-->>Client: capabilities
    Client->>Bridge: new session with cwd
    Bridge->>Bridge: allocate session ID and options
    Bridge->>Factory: context with cwd mode model
    Factory-->>Bridge: graph
    Bridge->>Store: persist ACP metadata
    Bridge-->>Client: session ID and options
    Client->>Bridge: prompt or selector change
    Bridge->>Graph: stream or rebuild using thread ID
    Graph-->>Bridge: messages updates or interrupt
    Bridge-->>Client: session updates and result
    Client->>Bridge: load session
    Bridge->>Store: validate metadata and cwd
    Bridge-->>Client: replay history and options
```

*The session ID scopes the checkpoint thread; ACP projects graph state rather than storing a separate durable transcript.*

### Durable recovery and thread ownership

`load_sessions=True` only exposes the protocol capability. New durable sessions and loads require a graph compiled with a checkpointer, and restart recovery requires that storage to outlive the server. The prompt-time `MemorySaver` fallback for a graph without a saver is ephemeral and cannot provide recovery. Loading accepts only an ACP-marked checkpoint thread with the original cwd, restores supported persisted model and mode selections, and replays user and assistant content, thoughts, tool activity, and plans through session updates. Missing, unrelated, or relocated sessions are rejected. See [State persistence](/openwiki/concepts/state-persistence.md) for the underlying model.

Dcode adds ownership fencing to those durable threads. Its ACP wrapper obtains a local lease before producing a session graph configuration, inserts its token into that configuration, and releases it when a session is forgotten or the runner exits. A second server cannot load a leased thread. The ownership-aware SQLite saver validates the token before checkpoint and pending-write mutations; released, stale, or superseded clients therefore cannot write after ownership changes. Its deletion path also reserves the thread, preventing deletion while another client holds it.

## Prompt and stream projection

The adapter converts text, inline images, resource links, and embedded resources to LangChain content. Resource-link paths under the session root become relative and embedded blobs become data URIs; audio raises `NotImplementedError`. It streams LangGraph `messages` and `updates` with subgraphs enabled, but projects assistant text and reasoning only from the top-level graph. Plain-text reasoning becomes ACP thought chunks, `write_todos` becomes plan updates, and partial tool-call arguments are accumulated until they parse as JSON before emitting a tool start.

Cancellation is in-memory and scoped to an ACP session. `cancel(session_id)` marks that session; a new prompt clears any old mark, checks before and during streaming, and returns `stop_reason="cancelled"` when cancellation is observed. On completion it returns `end_turn`.

ACP can display fixed permission decisions, not arbitrary LangGraph UI. A free-form `interrupt()` value is rejected. For `HumanInTheLoopMiddleware`-style action requests, the adapter waits for the stream iterator to close before reading checkpoint state—avoiding a stale persistent-checkpoint snapshot—asks the client for a decision, then resumes with `Command(resume={"decisions": ...})`. A cancelled permission request is treated as rejection.

The adapter offers Approve, Reject, and where valid Always allow. These grants exist only in adapter memory for one ACP session. Non-`execute` tools are allowlisted by name. An `execute` grant is keyed by canonical JSON for the complete arguments and by the current `AgentSessionContext` (cwd, mode, model); a changed command, arguments, context, or session must be reviewed again, while different object-key ordering does not change an equivalent request. Forgetting a session discards its grants. For policy concepts, see [Permissions and human-in-the-loop](/openwiki/concepts/permissions-hitl.md).

## Local context stays model-only

Dcode's `LocalContextMiddleware` detects backend environment details such as git state, project structure, package managers, runtimes, and directory layout. The initial snapshot is retained in private state and appended to the model system prompt. After a summarization cutoff it detects again: a changed snapshot is added as an escaped, fingerprinted internal `HumanMessage` tagged `lc_source="local_context"`; an unchanged or failed refresh records the cutoff without retrying or adding a duplicate. This keeps the initial prompt stable for cacheability while retaining refreshed context in graph state after compaction.

That tag is a visibility boundary. `AgentServerACP` suppresses tagged messages when projecting live assistant content and when replaying both user and assistant history. Hosts that transform these messages must retain the tag or can expose internal environment context in the editor transcript.

## Dcode startup, tools, and policy

`--acp` is detected in raw argv before normal UI dependency checks, and ACP packages are imported lazily. If `acp` or `deepagents-acp` is unavailable, dcode prints an installation hint and exits nonzero. `--no-mcp` and `--mcp-config` are mutually exclusive; YOLO in ACP requires a previous acknowledgement in the interactive TUI; and `--auto-classifier-model` is valid only in Auto mode.

Before opening the ACP server, dcode resolves its initial model and model selector choices, web tools, MCP tools, and subagents; it opens its checkpointer and supplies a graph factory whose cwd and selected model come from the ACP session context. MCP is deliberately configuration-owned here: the generic adapter retains editor-provided MCP descriptors but neither passes them to `AgentSessionContext` nor makes them graph tools. Dcode instead resolves its configured, project-trusted, and plugin MCP configurations before graph construction. Missing configuration files or MCP tool-loading failures go to stderr and exit 1; the MCP session manager is cleaned up when ACP ends.

Dcode passes `auto_approve=yolo` and `auto_mode_enabled=auto` into `create_cli_agent`. Thus ACP fixed-decision rendering applies only if dcode's own policy leaves an interrupt for the user. In Auto mode dcode uses a specialized adapter that writes trusted Auto approval state, attaches text-prompt metadata to the final human message, and streams the graph with `CLIContextSchema` Auto settings.

### Editor configuration

Install dcode with ACP support:

```sh
uv tool install -U deepagents-code --with deepagents-acp
```

For example, configure an ACP-capable editor with:

```json
{
  "agent_servers": {
    "Deep Agents Code": {
      "type": "custom",
      "command": "dcode",
      "args": ["--acp", "--model", "anthropic:claude-sonnet-5"]
    }
  }
}
```

Provider API keys, such as `ANTHROPIC_API_KEY`, are read from the environment just as they are for the terminal client. Use `dcode --help` for the supported ACP flags, including `--mcp-config` and `--no-mcp`. For an end-to-end operating flow, see [Run a dcode session](/openwiki/workflows/run-dcode-session.md).

## Focused verification

The dcode ACP smoke test launches `deepagents --acp --no-mcp`, connects an ACP pipe client, initializes, creates a session, and checks that it receives a session ID. ACP launcher tests verify that one ownership-aware server blocks another from loading an open thread and that writes using the old configuration fail after ownership changes. Thread-ownership tests additionally cover cross-process exclusion, client crash fencing, serialized writers, cancellation-safe deletion, and fenced checkpoint writes.

The generic adapter tests exercise exact-command approvals and local-context compaction behavior. In particular, permission tests cover changed command text, arguments, session context, invalid reusable grants, cancellation, and forgetting a session; local-context tests cover stable initial system prompts, escaping, changed and unchanged refreshes, and failed-refresh behavior.
