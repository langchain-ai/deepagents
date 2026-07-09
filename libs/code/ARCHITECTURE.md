# Deep Agents Code Architecture

## What this package is

`deepagents-code` is a prebuilt terminal coding agent built on top of the `deepagents` SDK. It is a reference implementation: one design for packaging the SDK into a useful coding-agent product, based on patterns that have worked well in our experience.

The SDK provides the agent harness. This package shows how to combine that harness with a terminal experience, persistence, tools, skills, and optional sandboxed execution.

## The big picture

Deep Agents Code has two runtime halves:

```text
┌──────────────────── Terminal client ─────────────────────┐
│  Presents interactive or headless output                 │
│  Collects user input and approvals                       │
└──────────────────────────┬───────────────────────────────┘
                           │ streaming protocol
                           ▼
┌──────────────────── Agent server ────────────────────────┐
│  Runs the coding agent graph                             │
│  Connects the model, tools, memory, skills, and backend  │
└──────────────────────────────────────────────────────────┘
```

The client and server run in separate processes. The client owns presentation and input. The server owns the agent runtime. Keeping that boundary narrow makes the UI responsive while letting the agent use LangGraph's streaming, checkpointing, and resume behavior.

## Request flow

A request follows the same shape in interactive and headless mode:

1. The client receives user input.
2. The client sends that input to the agent server.
3. The server runs the agent and streams events back.
4. The client renders those events and collects any needed human response.
5. Session state is preserved so the conversation can continue later.

Headless mode uses the same agent runtime as the interactive UI, but swaps the terminal interface for machine-friendly input and output.

## Configuration and extension

Configuration is layered across user, project, session, and runtime scopes. That lets teams share project defaults while individual users keep their own credentials, preferences, skills, and local settings.

The main extension points are:

- **Skills and subagents** for reusable agent workflows
- **Tools and MCP servers** for external capabilities
- **Sandboxes** for changing where tool execution happens
- **Hooks and commands** for integrating with local workflows

These pieces are designed to compose. A project can provide shared defaults and integrations, while each user can layer personal configuration on top.

## Plugin lifecycle

The plugin subsystem separates configuration intent, installed artifacts, and
the active runtime:

1. Marketplace records identify available plugins.
2. Installation copies a validated artifact into the versioned cache and
   records its user, project, or local scope.
3. Discovery resolves scoped enablement and session-only `--plugin-dir`
   overrides.
4. A runtime snapshot validates and groups skills, commands, agents, hooks, and
   MCP configuration.
5. The server consumes one immutable snapshot. `/reload-plugins` builds a
   replacement before swapping and restarts MCP-owning server state.

Executable plugin surfaces require a versioned fingerprint trust decision.
Client-side hooks handle lifecycle and prompt events; server middleware handles
pre/post tool decisions so a plugin denial occurs on the execution path.

## Design tradeoffs

This architecture optimizes for:

- A responsive local terminal experience
- A reusable agent core that can be tested apart from the UI
- Durable sessions that can be resumed
- Controlled tool execution, locally or in a sandbox
- Practical extension points without rewriting the core app

The main cost is the client/server boundary. When debugging, first decide which side owns the failure: presentation and input usually belong to the client; model execution, tools, memory, and graph startup usually belong to the server.

## Where to go next

- For local setup and debugging, see [`DEVELOPMENT.md`](./DEVELOPMENT.md).
- For command behavior, see [`COMMANDS.md`](./COMMANDS.md).
- For security boundaries, see [`THREAT_MODEL.md`](./THREAT_MODEL.md).
- For plugin lifecycle and compatibility, see [`PLUGINS.md`](./PLUGINS.md).
- For package-specific coding conventions, see [`AGENTS.md`](./AGENTS.md).
