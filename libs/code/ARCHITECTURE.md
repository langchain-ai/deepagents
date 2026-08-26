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

Configuration files are read into a single process-wide generation, built on the first read and reused after that. Readers that resolve through the shared resolver all observe that one generation. They cannot disagree about a setting.

An edit to `config.toml` while the app runs has no effect on those readers until the generation advances. This happens in two places: an in-app write to the default config path, which refreshes the generation itself, and `/reload`. Each source keeps its last usable snapshot, so a file that fails to parse leaves that tier unchanged instead of erasing it.

The app does not watch files for edits. A partly applied configuration is a worse failure than a stale one.

Some readers sit outside the shared generation. Callers that take their own snapshot inspect one file generation instead of process state, and report it next to its health: `get_config_sources`, the `dcode config` command, and the `dcode doctor` command, which reads the managed file against an empty user tier. A few readers parse a file on each call, because the shared generation cannot serve them: `resolve_read_project_dotenv` runs before the project `.env` is layered into the environment, `resolve_startup_mode_with_source` needs the raw user table, and `update_check` reports the value next to the file health it just read. The reload preview also reads the user file fresh, because a dry run must show the edit under review.

The environment tier is always live. `EnvProvider` reads `os.environ` at resolution time, because the process changes it during dotenv bootstrap and on each cwd switch.

These exceptions are per caller, not per setting. A caller decides to snapshot a file itself. No option is intended to be live for one reader and cached for another, which would make the effective configuration unpredictable per option.

The main extension points are:

- **Skills and subagents** for reusable agent workflows
- **Tools and MCP servers** for external capabilities
- **Sandboxes** for changing where tool execution happens
- **Hooks and commands** for integrating with local workflows

These pieces are designed to compose. A project can provide shared defaults and integrations, while each user can layer personal configuration on top.

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
- For lifecycle hooks (`hooks.json`), see [`HOOKS.md`](./HOOKS.md).
- For cost estimates and local pricing overrides (`prices.json`), see
  [`PRICING.md`](./PRICING.md).
- For security boundaries, see [`THREAT_MODEL.md`](./THREAT_MODEL.md).
- For package-specific coding conventions, see [`AGENTS.md`](./AGENTS.md).
