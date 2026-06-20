# Deep Agents Code Architecture

A map of how `deepagents-code` (the `dcode` TUI) fits together, for people
making their first change. It is intentionally high-level — file-level details
live in module docstrings, and the surrounding guides are linked at the end.

> Draft / living document. If you touch a subsystem and this drifts, fix the
> paragraph you just invalidated.

## What this package is

`deepagents-code` is a terminal coding agent (think Claude Code / Cursor in the
shell) built on top of the `deepagents` SDK. The SDK supplies the agent harness
(`create_deep_agent`, backends, and middleware like memory and skills); this
package wraps it in an interactive Textual UI, a headless mode, conversation
persistence, MCP integration, slash commands, skills, and remote sandboxes.

## The big picture

The single most important thing to internalize: the UI and the agent run in
**two different processes**. The client (the TUI or headless runner) talks to a
LangGraph server that runs the agent graph, over HTTP/SSE.

```text
┌─────────────────────────── client process ───────────────────────────┐
│                                                                       │
│  main.py (entry, arg parsing, mode select)                            │
│     │                                                                 │
│     ├── interactive ──▶ app.py (DeepAgentsApp, Textual)               │
│     │                      │  widgets/ render; textual_adapter.py     │
│     │                      │  translates the agent stream → widgets   │
│     └── headless (-x) ─▶ non_interactive.py                           │
│                            │                                          │
│              server_manager.start_server_and_get_agent()             │
│                            │ spawns + configures                      │
│                            ▼                                          │
│                   remote_client.RemoteAgent ──(RemoteGraph)──┐        │
└──────────────────────────────────────────────────────────────┼───────┘
                                                                 │ HTTP/SSE
┌──────────────────── `langgraph dev` subprocess ───────────────┼───────┐
│  server.py / server_manager.py manage its lifecycle           ▼       │
│  langgraph.json ──loads──▶ server_graph.py ──builds──▶ agent.py       │
│                                                          │            │
│                            deepagents SDK create_deep_agent           │
│                              + middleware (memory, skills,            │
│                                local_context, ask_user, resume_state) │
│                              + tools.py / managed_tools.py / MCP      │
│                              + backend: LocalShell | Sandbox          │
└───────────────────────────────────────────────────────────────────────┘
```

### Why the subprocess?

The agent graph is served by `langgraph dev` so the client gets LangGraph's
streaming, checkpointing, and state management "for free" via `RemoteGraph`.
The cost is a process boundary: when the agent crashes at startup, the client
only sees a one-line banner — the real traceback is in the subprocess's
captured log. Debugging that boundary (the `DEEPAGENTS_CODE_DEBUG` env var, the
`deepagents_server_log_*.txt` files, the triage flow) is documented in
[`DEV.md`](./DEV.md). Read that section before debugging a "server failed to
start" report.

## Request lifecycle (interactive)

1. A keystroke / submit in `widgets/chat_input.py` raises a Textual message.
2. `DeepAgentsApp` (`app.py`) handles it, updates the `MessageStore`, and sends
   the prompt to the `RemoteAgent`.
3. `RemoteAgent` (`remote_client.py`) streams the run from the server graph and
   converts streamed message dicts back into LangChain message objects.
4. `textual_adapter.py` consumes that stream and drives the UI — appending
   assistant text, tool-call widgets, approvals, and `ask_user` prompts.
5. Human-in-the-loop interrupts (tool approval, `ask_user`) round-trip back to
   the server as resumed input.

`event_bus.py` is a side door: it lets external local processes push prompts /
commands / signals into a *running* session over a Unix-domain socket.

## Headless lifecycle (`-x`)

`non_interactive.py` runs a single task against the same server subprocess,
streams results to stdout, and exits with a status code. It reuses the agent
and server machinery but skips Textual entirely — this is the scripting/CI
path. Several flags in `main.py` (`--update`, `--install`, model defaults) are
also handled headlessly without ever starting a session.

## Module map

The package is a flat ~70-file directory; grouping by concern:

| Concern | Key modules |
| --- | --- |
| **Entry / lifecycle** | `main.py`, `__init__.py` (lazy `cli_main`), `server.py`, `server_manager.py`, `server_graph.py` |
| **UI (Textual)** | `app.py` (the `DeepAgentsApp` god object), `widgets/`, `textual_adapter.py`, `theme.py`, `app.tcss`, `input.py`, `tool_display.py`, `formatting.py` |
| **Client ↔ server** | `remote_client.py`, `_server_config.py`, `_env_vars.py`, `event_bus.py` |
| **Agent construction** | `agent.py`, `configurable_model.py`, `model_config.py`, `config.py`, prompt files (`system_prompt.md`, `default_agent_prompt.md`) |
| **Middleware (client-side)** | `local_context.py`, `ask_user.py`, `resume_state.py`, `memory_guard.py`, `filesystem_empty_result.py` |
| **Tools** | `tools.py`, `managed_tools.py` (auto-installs `rg`), `hooks.py` |
| **MCP** | `mcp_tools.py`, `mcp_auth.py`, `mcp_oauth_ui.py`, `mcp_trust.py`, `mcp_commands.py`, `mcp_providers/` |
| **Skills & subagents** | `skills/`, `built_in_skills/`, `subagents.py` |
| **Slash commands** | `command_registry.py` (single source of truth), `config_commands.py`; generated catalog in [`COMMANDS.md`](./COMMANDS.md) |
| **Sessions / persistence** | `sessions.py`, `resume_state.py`, `state_migration.py`, `auth_store.py` |
| **Sandboxes** | `integrations/` (`sandbox_factory.py`, `sandbox_registry.py`, `sandbox_provider.py`, `sandbox_config.py`) |
| **Headless / machine output** | `non_interactive.py`, `output.py` |
| **Onboarding / updates** | `onboarding.py`, `update_check.py`, `notifications.py` |

## On-disk layout (runtime)

User and project state lives under `~/.deepagents/`, with user instructions,
custom skills, and custom subagents scoped to the selected agent name. The
default agent name is `agent`. Project-local `.deepagents/` and shared
`.agents/` directories override or extend some user-level locations:

```text
~/.deepagents/
├── config.toml          # models, sandboxes, settings
├── hooks.json           # external tool hooks (hooks.py)
├── .mcp.json            # MCP server config (also discovered project-local)
├── <agent>/
│   ├── AGENTS.md        # user-level instructions for that agent
│   ├── agents/          # user custom subagents ({name}/AGENTS.md)
│   └── skills/          # user custom skills
├── bin/                 # managed binaries (e.g. ripgrep)
└── .state/
    ├── auth.json, chatgpt-auth.json, mcp-tokens/   # credentials
    ├── history.jsonl, recent_models.json
    └── mcp_trust.json

<project>/
├── AGENTS.md            # project instructions
├── .mcp.json            # project root MCP server config
├── .deepagents/
│   ├── AGENTS.md        # higher-priority project instructions
│   ├── agents/          # project custom subagents ({name}/AGENTS.md)
│   ├── skills/          # project custom skills
│   └── .mcp.json        # project MCP server config
└── .agents/
    └── skills/          # shared project skills alias

~/.agents/
└── skills/              # shared user skills alias
```

## Conventions & where to look

- **Textual footguns** (Content vs Rich Text, `notify(markup=True)` crashes,
  glyph/spinner sourcing, modal/worker rules) — [`AGENTS.md`](./AGENTS.md).
  Read it before touching `app.py` or `widgets/`.
- **Local dev, debugging, CSS hot-reload** — [`DEV.md`](./DEV.md).
- **Slash commands** are declared once in `command_registry.py`; regenerate
  [`COMMANDS.md`](./COMMANDS.md) with `make commands-catalog`.
- **SDK coupling**: `pyproject.toml` pins an exact `deepagents==X.Y.Z`. Bump it
  in the same PR as features that need new SDK behavior.
- **Security model** for tool execution / untrusted content —
  [`THREAT_MODEL.md`](./THREAT_MODEL.md).
- **Lazy imports**: heavy imports are deferred to call sites on purpose to
  protect startup time (guarded by CodSpeed benchmarks). The module-top
  `if TYPE_CHECKING:` block is the canonical dependency list for a file.

## "I want to change X" cheat sheet

| Goal | Start here |
| --- | --- |
| Add/modify a slash command | `command_registry.py` (+ `make commands-catalog`) |
| Change how a tool call renders | `widgets/tool_renderers.py`, `tool_display.py` |
| Add a tool | `tools.py` (and `agent.py` wiring) |
| Add an MCP provider | `mcp_providers/` |
| Add a sandbox backend | `integrations/` (`sandbox_registry.py`) |
| Change the system prompt | `system_prompt.md` / `default_agent_prompt.md` |
| Touch agent graph / middleware | `agent.py`, `server_graph.py` |
| Debug a startup crash | `DEV.md` → `DEEPAGENTS_CODE_DEBUG=1` |
| Add a model provider | `model_config.py`, `pyproject.toml` extras |
