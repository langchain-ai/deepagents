# Threat Model: deepagents-cli

> Generated: 2026-03-28 | Commit: e859077f | Scope: libs/cli only | Visibility: Open Source | Mode: deep

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference — findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. We welcome suggestions and corrections to improve this document.

## Scope

### In Scope

- `deepagents_cli/` — all Python source modules shipped as the `deepagents-cli` package
- CLI entry point (`main.py`, `__init__.py`)
- Interactive TUI (`app.py`, `textual_adapter.py`, `widgets/`)
- Non-interactive pipeline runner (`non_interactive.py`)
- Agent creation (`agent.py`)
- Built-in tools (`tools.py`: `http_request`, `web_search`, `fetch_url`)
- MCP config loader and trust store (`mcp_tools.py`, `mcp_trust.py`)
- Hook dispatcher (`hooks.py`)
- Sandbox integration factory (`integrations/sandbox_factory.py`, `integrations/sandbox_provider.py`)
- Session persistence (`sessions.py`)
- Configuration system (`config.py`, `model_config.py`)
- Unicode/URL safety helpers (`unicode_security.py`)
- LangGraph dev server subprocess management (`server.py`, `server_manager.py`, `server_graph.py`)
- Remote agent client (`remote_client.py`)
- Local context middleware (`local_context.py`)
- Custom subagent loader (`subagents.py`, `agent.py:load_async_subagents`)
- Conversation offload (`offload.py`)
- Skill management (`skills/commands.py`)

### Out of Scope

- `libs/deepagents/` (SDK library) — separate package with its own threat model
- `libs/acp/`, `libs/evals/`, `libs/partners/` — separate packages
- `tests/` — not shipped code; used during analysis only
- `scripts/`, `examples/` — developer tooling, not shipped
- Deployment infrastructure, CI/CD pipelines
- LLM provider behavior (model outputs, jailbreaks) — user-controlled
- Sandbox provider internals (Daytona, LangSmith, Modal, Runloop, AgentCore) — third-party
- LangGraph server internals — consumed as a subprocess dependency

### Assumptions

1. The CLI runs locally on the user's machine; the user is a developer who invoked `deepagents` themselves.
2. The project provides the HITL approval framework; users control model selection, API keys, and whether to disable approval gates.
3. `~/.deepagents/` is only writable by the authenticated local user — no multi-user shared home directories.
4. Sandbox backends are trusted third-party services. CLI responsibility ends at correctly constructing and dispatching requests to them.
5. LangSmith tracing, if enabled, is user-opted-in via environment variables.
6. The LangGraph dev server subprocess binds to `127.0.0.1` by default (`server.py:_DEFAULT_HOST`) and is ephemeral — started and stopped per CLI session.
7. `DA_SERVER_*` environment variables are readable only by the CLI process and its child server subprocess (OS process isolation assumption).
8. Users who set `class_path` in `config.toml` accept the same trust model as `pyproject.toml` build scripts — they control their own machine.

---

## System Overview

`deepagents-cli` is a terminal-based AI coding assistant. It wraps the `deepagents` SDK in an interactive TUI (Textual) and a headless non-interactive mode. Both modes route agent execution through a local `langgraph dev` subprocess: the CLI spawns a server, passes configuration via `DA_SERVER_*` environment variables, and communicates via a `RemoteAgent` HTTP+SSE client. The agent receives user prompts, reasons with a configurable LLM, and executes side-effecting tools (file read/write, shell commands, web search, HTTP requests) subject to a human-in-the-loop (HITL) approval gate. Sessions are persisted in a local SQLite checkpoint database. Users can extend the agent with MCP servers (stdio processes or remote HTTP/SSE endpoints), hooks (event-driven subprocesses), custom subagents (AGENTS.md files in `.deepagents/agents/`), async remote subagents (LangGraph deployments configured in `config.toml`), and pluggable sandbox backends for remote code execution.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        User (local machine)                          │
│                                                                      │
│  CLI Args / Env Vars ──► C1: CLI Entry Point (main.py)               │
│                                  │                                   │
│                 ┌────────────────┼─────────────────────┐             │
│                 ▼                ▼                     ▼             │
│        C2: TUI (app.py)  C2b: Non-interactive   C13: Config Channel  │
│                 │         (non_interactive.py)  (DA_SERVER_* env vars)│
│ - - - - - - - - - - TB8: CLI / Server IPC - - - - - │ - - - - - - - │
│                 │                                     ▼              │
│                 │               C11: Server Manager (server_manager) │
│                 │                      │ spawns                      │
│                 │                      ▼                             │
│                 │           C11b: LangGraph Dev Server               │
│                 │           (server.py:ServerProcess)                │
│                 │           LANGGRAPH_AUTH_TYPE=noop                 │
│ - - - - - - - - │ - - - - - TB10: RemoteAgent / Dev Server - - - -  │
│                 │                      ▲                             │
│                 └─►C12: RemoteAgent────┘ (HTTP+SSE on 127.0.0.1)    │
│                     (remote_client.py)                               │
│                            │                                         │
│                     C3: Agent Engine (server_graph.py)               │
│                     (create_cli_agent, deepagents SDK)               │
│                            │                                         │
│  User Prompt ──────────────┘                                         │
│                            │                                         │
│ - - - - - - - - - TB1: User→Agent Input - - - - - - - - - - - - -   │
│                            │                                         │
│                     LLM Decision                                     │
│ - - - - - - - - - TB2: LLM → Tool Execution (HITL) - - - - - - - -  │
│                            │                                         │
│   ┌──────────┬─────────┬───┴────────┬──────────┬──────────────┐     │
│   ▼          ▼         ▼            ▼          ▼              ▼     │
│  C4:Tools  C5:MCP   C6:Hooks     C8:Sessions C7:Sandbox  C14:Async  │
│  (file,    (procs/  (subprocs    (SQLite)    (Daytona/   Subagents   │
│  shell,    remote)  hooks.json)             Modal/etc.) (LangGraph   │
│  HTTP)                                                   remotes)   │
│   │          │                     │          │              │       │
│ - │ - - - - -│- - - - TB3 - - - - -│- - - - - │ - - - - - - -│ - -  │
│   │          ▼                     ▼          ▼              │       │
│   │      External               Local FS   Remote     External LG   │
│   ▼      MCP Server            (~/.deep   Sandbox    Deployment     │
│ External  (proc/net)           agents/)   API                       │
│ Web/APIs                                                             │
│ - - - TB4: Web content → Context - - - - - - - - - - - - - - - - -  │
│   Tool results re-enter agent context window                         │
│ - - - TB9: LocalContextMiddleware / Host environment - - - - - - -   │
│   C15: LocalContextMiddleware runs bash detect script                │
│                                                                      │
│  C9: Config System ──► C17: Model Config (class_path → importlib)   │
│  (config.toml, .env)                                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Components

| ID  | Component                   | Description                                                                                                         | Trust Level          | Default? | Entry Points                                                                                      |
|-----|-----------------------------|---------------------------------------------------------------------------------------------------------------------|----------------------|----------|---------------------------------------------------------------------------------------------------|
| C1  | CLI Entry Point             | Parses argv, loads config/env, bootstraps session                                                                   | framework-controlled | Yes      | `main.cli_main`, `main.parse_args`                                                                |
| C2  | TUI / Non-interactive       | Textual UI for interactive chat; `non_interactive.py` for headless pipelines (both use `RemoteAgent`)               | framework-controlled | Yes      | `app.DeepAgentsApp.run`, `non_interactive.run_non_interactive`                                    |
| C3  | Agent Engine                | LangGraph agent graph running inside `langgraph dev` server, assembled by `create_cli_agent`                        | framework-controlled | Yes      | `agent.create_cli_agent`, `agent._add_interrupt_on`, `server_graph.make_graph`                    |
| C4  | Built-in Tools              | `http_request`, `web_search` (Tavily), `fetch_url` (HTML→markdown)                                                 | framework-controlled | Partial¹ | `tools.http_request`, `tools.web_search`, `tools.fetch_url`                                       |
| C5  | MCP Loader & Trust          | Discovers/loads `.mcp.json`, validates server configs, manages trust store                                          | framework-controlled | No²      | `mcp_tools.resolve_and_load_mcp_tools`, `mcp_trust.compute_config_fingerprint`                   |
| C6  | Hook Dispatcher             | Fires subprocess commands on agent lifecycle events                                                                 | framework-controlled | No³      | `hooks.dispatch_hook`, `hooks._run_single_hook`                                                   |
| C7  | Sandbox Integration         | Creates/destroys remote sandboxes (Daytona, LangSmith, Modal, Runloop, AgentCore)                                  | framework-controlled | No⁴      | `integrations.sandbox_factory.create_sandbox`                                                     |
| C8  | Session Persistence         | SQLite checkpoint store for LangGraph thread state                                                                  | framework-controlled | Yes      | `sessions.get_db_path`, `sessions.generate_thread_id`                                             |
| C9  | Configuration System        | TOML config, env vars, `AGENTS.md` system prompts, model config                                                    | user-controlled      | N/A      | `config.settings`, `model_config.ModelConfig`, `~/.deepagents/config.toml`                        |
| C10 | Unicode/URL Safety          | Detects hidden Unicode, checks URL domain spoofing for approval UI warnings                                         | framework-controlled | Yes      | `unicode_security.detect_dangerous_unicode`, `unicode_security.check_url_safety`                  |
| C11 | LangGraph Dev Server        | Subprocess running `langgraph dev` with `LANGGRAPH_AUTH_TYPE=noop`; managed by `ServerProcess`                      | framework-controlled | Yes⁵     | `server.ServerProcess.start`, `server.generate_langgraph_json`, `server_manager.start_server_and_get_agent` |
| C12 | Remote Agent Client         | HTTP+SSE client wrapping `RemoteGraph`; connects to C11 on localhost                                                | framework-controlled | Yes⁵     | `remote_client.RemoteAgent.astream`, `remote_client.RemoteAgent.aget_state`                       |
| C13 | Server Config Channel       | Passes CLI config to server subprocess via `DA_SERVER_*` environment variables                                      | framework-controlled | Yes      | `_server_config.ServerConfig.to_env`, `_server_config.ServerConfig.from_env`                      |
| C14 | Async Subagent Config       | Loads remote LangGraph deployment specs from `[async_subagents]` in `config.toml`                                  | user-controlled      | No       | `agent.load_async_subagents`                                                                      |
| C15 | LocalContext Middleware      | Runs a bash detection script via backend; injects git/project/env context into system prompt each turn              | framework-controlled | Yes⁶     | `local_context.LocalContextMiddleware.before_agent`, `local_context.build_detect_script`          |
| C16 | Custom Subagent Loader      | Reads `{dir}/{name}/AGENTS.md` YAML frontmatter from `.deepagents/agents/` and project `.agents/` directories      | user-controlled      | No       | `subagents.list_subagents`, `subagents._parse_subagent_file`                                      |
| C17 | Model Config Loader         | Resolves model provider, supports `class_path` for arbitrary `BaseChatModel` instantiation via `importlib`          | user-controlled      | N/A      | `config.create_model`, `config._create_model_from_class`, `model_config.ModelConfig.load`         |

**Notes:**
1. `http_request` and `fetch_url` enabled by default; `web_search` requires `TAVILY_API_KEY`.
2. MCP servers only load if `.mcp.json` config files are present.
3. Hooks only execute if `~/.deepagents/hooks.json` exists.
4. Sandbox mode requires explicit `--sandbox` CLI flag.
5. Both TUI and non-interactive modes now always spawn a local LangGraph dev server and connect via `RemoteAgent`.
6. `LocalContextMiddleware` is added whenever `LocalShellBackend` or an `_AsyncExecutableBackend` is in use (`agent.py:create_cli_agent`).

---

## Data Classification

| ID  | PII Category           | Specific Fields                                   | Sensitivity | Storage Location(s)              | Encrypted at Rest | Retention          | Regulatory |
|-----|------------------------|---------------------------------------------------|-------------|----------------------------------|-------------------|--------------------|------------|
| DC1 | API Keys / Credentials | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGGRAPH_API_KEY` | Critical | Process environment only; never written to disk by CLI code | N/A (in-memory) | Process lifetime | All — breach trigger |
| DC2 | Conversation Messages  | User prompts, LLM responses, tool args/results   | High        | SQLite (`~/.deepagents/*.db`) via LangGraph checkpointer | No (local file, unencrypted) | Unbounded (session files persist) | GDPR if personal data is discussed |
| DC3 | System Prompt Content  | `DA_SERVER_SYSTEM_PROMPT` env var; custom AGENTS.md contents | Medium | Process environment (transient); `~/.deepagents/{agent}/AGENTS.md` on disk | No | Config lifetime | None direct |
| DC4 | MCP Trust Fingerprints | `mcp_trust.projects.*` in `config.toml`          | Low         | `~/.deepagents/config.toml`      | No                | Until revoked      | None       |
| DC5 | Offloaded Conversation History | Summarized + raw conversation messages written to sandbox backend | High | Sandbox filesystem at `/conversation_history/{thread_id}.md` | Depends on sandbox provider | Sandbox session lifetime | GDPR if personal data is discussed |

### Data Classification Details

#### DC1: API Keys / Credentials

- **Fields**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`, `LANGGRAPH_API_KEY`, and any provider-specific keys in the user's environment.
- **Storage**: Loaded from environment at process start (`config.py`). The CLI explicitly strips `LANGGRAPH_CLOUD_LICENSE_KEY` and related auth vars from the server subprocess environment (`server.py:_build_server_env`) but passes the remaining env vars (including provider API keys) to the server subprocess via `os.environ.copy()`.
- **Access**: Available to both the CLI process and the server subprocess (which inherits the full environment minus stripped vars).
- **Encryption**: Not encrypted — in-memory process environment only.
- **Retention**: Process lifetime; cleared on CLI exit.
- **Logging exposure**: Provider SDK error messages may include partial key information. The CLI does not log keys directly.
- **Gaps**: API keys are passed to the server subprocess via `_build_server_env` which does `os.environ.copy()` — all keys in the parent's environment become available to the child.

#### DC2: Conversation Messages

- **Fields**: Full conversation history (HumanMessage, AIMessage, ToolMessage) stored as LangGraph checkpoint state.
- **Storage**: SQLite at `~/.deepagents/{agent}/{thread_id}.db` (via `sessions.get_db_path`).
- **Access**: Local filesystem; readable by any process running as the same user.
- **Encryption**: None — plaintext SQLite.
- **Retention**: Unbounded — session files persist until manually deleted.
- **Logging exposure**: Tool call arguments and results (including fetched web content) are in the checkpoint. File contents read by the agent are stored there too.
- **Gaps**: Unencrypted on disk; no retention policy enforced by the CLI.

#### DC5: Offloaded Conversation History

- **Fields**: Timestamped, formatted conversation messages written by `offload.offload_messages_to_backend`.
- **Storage**: Sandbox backend filesystem at path `/conversation_history/{thread_id}.md` where `thread_id` is a UUID7 (via `sessions.generate_thread_id`).
- **Access**: Accessible within the sandbox session; depends on provider access controls.
- **Encryption**: Depends on sandbox provider storage backend.
- **Retention**: Sandbox session lifetime (destroyed when sandbox is deleted).
- **Logging exposure**: Contains full message history including tool results.
- **Gaps**: Thread ID is UUID7 (no path injection risk), but offloaded content is unstructured markdown containing raw conversation data.

---

## Trust Boundaries

| ID   | Boundary                              | Description                                                                | Controls (Inside)                                                               | Does NOT Control (Outside)                                    |
|------|---------------------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------|
| TB1  | User Input → Agent Engine             | Where user-typed prompts enter the agent graph                             | Prompt routing, session threading, UI rendering                                 | Prompt content — any text accepted                            |
| TB2  | LLM Decision → Tool Execution         | HITL gate on all side-effecting tool calls                                 | Interrupt map, allow-list check, auto-approve toggle                            | LLM reasoning; what user approves                             |
| TB3  | Tool Result → LLM Context             | Tool outputs re-enter the context window                                   | Unicode warnings on URLs in args; markdownify HTML conversion                   | Content of fetched web pages, MCP responses, search results   |
| TB4  | MCP Config → Process / Network        | `.mcp.json` triggers subprocess spawn or network connection                | Schema validation, SHA-256 fingerprint trust gate                               | What the MCP process does once trusted and running            |
| TB5  | Hooks Config → Subprocess             | `hooks.json` commands execute as local subprocesses                        | JSON structure validation, 5-second timeout                                     | Command content (user-authored); env vars available to child  |
| TB6  | Setup Script → Sandbox Execution      | User-supplied script runs inside sandbox at startup                        | `shlex.quote()` wrapping; `string.Template.safe_substitute`                     | Script content (user-authored); sandbox network access        |
| TB7  | External LLM API → Agent State        | LLM API responses drive tool call decisions                                | LLM client configuration, model selection, request params                       | LLM output content; provider-side safety                      |
| TB8  | CLI → Server Subprocess IPC           | Config passed from CLI to `langgraph dev` subprocess via `DA_SERVER_*` env vars | `ServerConfig.to_env()` serialization; env var scoping to parent+child process | Subprocess environment post-fork; /proc visibility to same-uid processes |
| TB9  | LocalContextMiddleware → Host Env     | Bash detect script output (git info, project files, Makefile) injected into system prompt | Script is framework-generated static code; 30s timeout; exit-code check | Content of Makefile, pyproject.toml, git branch names, directory listing |
| TB10 | RemoteAgent → LangGraph Dev Server    | CLI communicates with agent via HTTP+SSE on localhost                       | Server bound to `127.0.0.1` (`server.py:_DEFAULT_HOST`); ephemeral per session | No authentication (`LANGGRAPH_AUTH_TYPE=noop`); any localhost process can reach the API |
| TB11 | Config File → Code Execution          | `class_path` in `config.toml` triggers `importlib.import_module()` to load arbitrary Python modules | Format validation (`module:ClassName`); `issubclass(BaseChatModel)` check | Module-level side effects execute during import; user controls config file |

### Boundary Details

#### TB1: User Input → Agent Engine

- **Inside**: `app.DeepAgentsApp` routes keystrokes to message queue; `non_interactive.run_non_interactive` reads from stdin/argv. Both paths now go through `RemoteAgent.astream` to the server. Session ID assigned per thread (`sessions.generate_thread_id`).
- **Outside**: Prompt content — the agent accepts any text the user types. No content filtering at this layer (HITL handles downstream tool calls, not prompt intent).
- **Crossing mechanism**: HTTP POST to `127.0.0.1:{port}` via `remote_client.RemoteAgent.astream`.

#### TB2: LLM Decision → Tool Execution Gate (HITL)

- **Inside**: `agent._add_interrupt_on` registers interrupt configs for `execute`, `write_file`, `edit_file`, `web_search`, `fetch_url`, `task`, `compact_conversation`, `launch_async_subagent`, `update_async_subagent`, `cancel_async_subagent`. In non-interactive mode, `non_interactive._handle_action_request` enforces the shell allow-list via `config.is_shell_command_allowed`.
- **Outside**: Once the user clicks "approve" (interactive) or a command passes the allow-list check (non-interactive), the tool executes with no further framework-level gating.
- **Crossing mechanism**: LangGraph HITL interrupt routed through `RemoteAgent` SSE stream.
- **Key note**: `auto_approve` mode bypasses all HITL approval prompts while still displaying Unicode/URL warnings.

#### TB3: Tool Result → LLM Context

- **Inside**: `agent._format_execute_description` / `_format_fetch_url_description` scan tool *arguments* (not results) for hidden Unicode and suspicious URLs. `fetch_url` converts HTML to markdown via `markdownify` before returning.
- **Outside**: The *content* of tool results (fetched web pages, web search snippets, MCP tool responses, `execute` stdout) is passed verbatim into the LLM context window. No prompt-injection scanning of results.
- **Crossing mechanism**: LangGraph `ToolMessage` returned to agent graph state; streamed to CLI via SSE.

#### TB4: MCP Config → Process / Network

- **Inside**: `mcp_tools._validate_server_config` validates JSON structure and field types. `mcp_trust.compute_config_fingerprint` computes SHA-256 over config file contents. Project-level stdio servers require trust approval (fingerprint prompt) unless `--trust-project-mcp` is set.
- **Outside**: Stdio server `command`, `args`, and `env` fields are user-controlled strings passed directly to `StdioConnection`. The `env` dict from MCP config is forwarded without filtering — users can set arbitrary environment variables (including `PATH`, `LD_PRELOAD`, `PYTHONPATH`) for the MCP subprocess.
- **Crossing mechanism**: `subprocess.Popen` (via `langchain_mcp_adapters`) for stdio; HTTP/SSE for remote servers.

#### TB8: CLI → Server Subprocess IPC

- **Inside**: `ServerConfig.to_env()` serializes all config (model, assistant_id, system_prompt, sandbox settings, cwd, MCP config path, shell-enable flags) to `DA_SERVER_*` env vars. `server_manager._apply_server_config` writes these to `os.environ` before `subprocess.Popen`. `server.py:_build_server_env` strips cloud auth vars (`LANGGRAPH_CLOUD_LICENSE_KEY`, `LANGSMITH_CONTROL_PLANE_API_KEY`, `LANGSMITH_TENANT_ID`, `LANGGRAPH_AUTH`).
- **Outside**: The full parent environment including provider API keys flows to the child via `os.environ.copy()`. The system prompt string becomes `DA_SERVER_SYSTEM_PROMPT`. After process start, the env is fixed — no runtime mutation across processes.
- **Crossing mechanism**: `subprocess.Popen` env kwarg; subsequent `os.environ` reads in `_server_config.ServerConfig.from_env`.

#### TB9: LocalContextMiddleware → Host Environment

- **Inside**: `local_context.build_detect_script` generates a static bash heredoc that runs git commands, checks for lock files, reads Makefile headings, and produces a structured markdown summary. Script is static framework code — not constructed from user input. Exit code checked via `_handle_detect_result`. 30-second timeout.
- **Outside**: The script reads and includes the first 20 lines of `Makefile` (`_section_makefile`) and a directory listing. These file contents are not sanitized before injection into the system prompt. An attacker who can write to the CWD's Makefile can influence the system prompt.
- **Crossing mechanism**: `backend.execute(DETECT_CONTEXT_SCRIPT)` → result appended to `system_prompt` via `LocalContextMiddleware._get_modified_request`.

#### TB10: RemoteAgent → LangGraph Dev Server

- **Inside**: Server bound to `127.0.0.1` by default; `server.py:_DEFAULT_HOST = "127.0.0.1"`. `RemoteAgent` only connects to the URL returned by `ServerProcess.url`. Server is ephemeral — started at session start, stopped at session end. Port defaults to 2024 but auto-selects a free port if occupied.
- **Outside**: `LANGGRAPH_AUTH_TYPE=noop` disables all LangGraph server authentication. Any process on localhost that discovers the port can submit requests, read thread state, or inject messages.
- **Crossing mechanism**: HTTP POST/GET to `http://127.0.0.1:{port}` using `langgraph.pregel.remote.RemoteGraph`.

#### TB11: Config File → Code Execution

- **Inside**: `config._create_model_from_class` validates `class_path` format (`module:ClassName`), imports the module via `importlib.import_module()`, and checks `issubclass(cls, BaseChatModel)` before instantiation.
- **Outside**: Module-level code in the imported module executes unconditionally during `import_module()`. The `issubclass` check only runs after import. Any side effects (file I/O, network calls, subprocess spawning) in the module's top-level scope execute before the type check.
- **Crossing mechanism**: `importlib.import_module(module_path)` in `config._create_model_from_class`.

---

## Data Flows

| ID   | Source       | Destination  | Data Type                                      | Classification | Crosses Boundary | Protocol               |
|------|-------------|-------------|------------------------------------------------|----------------|------------------|------------------------|
| DF1  | User         | C2 TUI       | User prompt text                               | —              | TB1              | Keystrokes / stdin     |
| DF2  | C2 TUI       | C12 RemoteAgent | Human message + thread config                | —              | TB1              | Function call          |
| DF3  | C12 RemoteAgent | C11 LangGraph Dev Server | Input messages, thread ID    | DC2            | TB10             | HTTP POST (localhost)  |
| DF4  | C11 LangGraph Dev Server | C12 RemoteAgent | SSE stream (AI responses, tool calls, interrupts) | DC2 | TB10 | HTTP+SSE (localhost) |
| DF5  | C3 Agent     | External LLM | System prompt + message history               | DC1, DC2       | TB7              | HTTPS / LangChain      |
| DF6  | External LLM | C3 Agent     | AI response + tool call decisions             | —              | TB7              | HTTPS / LangChain      |
| DF7  | C3 Agent     | C4 Tools     | Tool call arguments (file/URL/command)        | —              | TB2              | Function call + HITL   |
| DF8  | External     | C4 Tools     | HTTP response bodies (web/API content)        | —              | TB3              | HTTPS                  |
| DF9  | C4 Tools     | C3 Agent     | Tool results (file content, web pages)        | —              | TB3              | ToolMessage            |
| DF10 | C9 Config    | C1 Entry     | TOML config, env vars, API keys               | DC1            | None             | File + environ         |
| DF11 | C5 MCP       | C3 Agent     | MCP tool call results                         | —              | TB3, TB4         | MCP protocol           |
| DF12 | C9 Config    | C5 MCP       | `.mcp.json` server definitions (command, args, env) | —        | TB4              | File read              |
| DF13 | C3 Agent     | C8 Sessions  | Agent state snapshots                         | DC2            | None             | SQLite async write     |
| DF14 | C8 Sessions  | C3 Agent     | Restored thread state on resume               | DC2            | None             | SQLite async read      |
| DF15 | C9 Config    | C6 Hooks     | `hooks.json` command definitions              | —              | TB5              | File read              |
| DF16 | C3 Agent     | C6 Hooks     | Event payload (JSON)                          | —              | TB5              | subprocess stdin       |
| DF17 | User         | C7 Sandbox   | Setup script path + content                   | —              | TB6              | File read + execute    |
| DF18 | C1 Entry     | C11 Server   | `DA_SERVER_*` env vars (config + credentials) | DC1, DC3       | TB8              | Process environment    |
| DF19 | Host FS      | C15 LocalContext | Makefile, project file contents (first 20 lines), directory listing | DC3 | TB9 | bash subprocess stdout |
| DF20 | C15 LocalContext | C3 Agent | Project context markdown appended to system prompt | DC3       | TB9              | string append to prompt|
| DF21 | C16 Subagent Loader | C3 Agent | AGENTS.md body (raw text) used as subagent system_prompt | DC3 | None | YAML parse + dict |
| DF22 | C14 Async Config | C3 Agent | AsyncSubAgent specs (URL, graph_id, headers) from config.toml | — | None | TOML parse + dict |
| DF23 | C9 Config    | C17 Model Config | `class_path` string from `config.toml` | —              | TB11             | TOML parse → importlib |
| DF24 | C5 MCP Config | MCP Subprocess | `env` dict from `.mcp.json` forwarded to stdio subprocess | DC1 | TB4 | subprocess environment |
| DF25 | C3 Agent     | C7 Sandbox   | Conversation messages for offload             | DC5            | TB6              | `backend.awrite()`     |

### Flow Details

#### DF8/DF9: External Web Content → Agent Context

- **Data**: Arbitrary HTML/JSON from the internet, converted to markdown by `markdownify`. Can be megabytes.
- **Validation**: URL domain checked for Unicode spoofing / script mixing (`unicode_security.check_url_safety`); displayed as warning in approval dialog. Content not scanned for prompt-injection patterns.
- **Trust assumption**: User approved the fetch. Content is data — but the LLM may interpret adversarial content as instructions.

#### DF11: MCP Tool Results → Agent Context

- **Data**: Arbitrary strings/objects from MCP tool calls.
- **Validation**: MCP config fingerprinted at load time. Tool result content not validated after server is trusted.
- **Trust assumption**: User trusted the MCP server; its outputs are as reliable as the server.

#### DF18: DA_SERVER_* Env Vars

- **Data**: Model name, system prompt text, sandbox settings, CWD, MCP config path, shell-enable flags — plus inherited provider API keys.
- **Validation**: No validation on server side beyond TOML/JSON decoding for structured fields (`_read_env_json`). The system prompt and model name are passed through verbatim.
- **Trust assumption**: The server subprocess is trusted with the same access as the CLI process.

#### DF19/DF20: LocalContextMiddleware Bash Script

- **Data**: Git branch/status, project language/structure, Makefile first 20 lines, directory listing (up to 20 files), runtime versions.
- **Validation**: Script exit code checked (`_handle_detect_result`). 30-second timeout. Script itself is static framework code — not interpolated from user input.
- **Trust assumption**: Files in the working directory are trustworthy. A malicious Makefile could inject content into the system prompt (requires write access to CWD).

#### DF23: class_path Config → importlib Code Execution

- **Data**: Fully-qualified Python class path string (e.g., `my_package.models:MyChatModel`) from `[models.providers.<name>]` section of `config.toml`.
- **Validation**: Format check (`module:ClassName` with `:` separator). After import, `issubclass(cls, BaseChatModel)` check. No validation of module contents before import.
- **Trust assumption**: User controls `~/.deepagents/config.toml` (same trust model as `pyproject.toml` build scripts).

#### DF24: MCP Stdio Env Dict → Subprocess

- **Data**: Arbitrary key-value pairs from the `"env"` field of stdio server definitions in `.mcp.json`.
- **Validation**: Type check only — `env` must be a dict (`mcp_tools._validate_server_config`). No filtering of key names or values. Forwarded directly to `StdioConnection` which passes to `subprocess.Popen`.
- **Trust assumption**: User authored or approved the MCP config. Project-level configs go through fingerprint trust gate before loading.

---

## Threats

| ID  | Data Flow | Classification | Threat                                                                                      | Boundary | Severity | Validation | Code Reference                                                         |
|-----|-----------|----------------|---------------------------------------------------------------------------------------------|----------|----------|------------|------------------------------------------------------------------------|
| T1  | DF8, DF9  | —              | Prompt injection via fetched web content causes LLM to request harmful actions              | TB3      | Medium   | Likely     | `tools.fetch_url`, `agent._add_interrupt_on`                          |
| T2  | DF7       | —              | `--shell-allow-list all` removes pattern checks; LLM-injected shell commands execute without approval in non-interactive mode | TB2 | Medium | Verified | `config.is_shell_command_allowed`, `non_interactive._handle_action_request` |
| T3  | DF7       | —              | Unicode-homoglyph URL in LLM-generated tool args deceives user during approval              | TB2      | Low      | Disproven  | `unicode_security.check_url_safety`, `agent._format_fetch_url_description` |
| T4  | DF5, DF9  | —              | Auto-approve mode bypasses all HITL gates; any LLM-initiated tool call executes             | TB2      | Low      | Verified   | `agent.create_cli_agent` (`auto_approve` param), `agent._add_interrupt_on` |
| T5  | DF13, DF14| DC2            | Local SQLite checkpoint file tampered with to inject adversarial content into future LLM context | None | Low   | Unverified | `sessions.get_db_path`                                                 |
| T6  | DF3, DF4  | DC2            | Unauthenticated LangGraph dev server on localhost can be accessed by any local process     | TB10     | Medium   | Verified   | `server._build_server_env`, `server._DEFAULT_HOST`                    |
| T7  | DF19, DF20| DC3            | Makefile or project file content injected into system prompt via LocalContextMiddleware    | TB9      | Low      | Verified   | `local_context._section_makefile`, `local_context.LocalContextMiddleware._get_modified_request` |
| T8  | DF21      | DC3            | Custom subagent AGENTS.md body used verbatim as system_prompt without content validation   | None     | Low      | Verified   | `subagents._parse_subagent_file`, `agent.create_cli_agent`            |
| T9  | DF23      | —              | `class_path` in config.toml triggers arbitrary Python code execution via `importlib.import_module()` | TB11 | Low | Verified | `config._create_model_from_class`, `model_config.ProviderConfig`      |
| T10 | DF24      | DC1            | MCP stdio subprocess env dict accepts arbitrary keys including `PATH`, `LD_PRELOAD`, `PYTHONPATH` without filtering | TB4 | Low | Verified | `mcp_tools._validate_server_config`, `mcp_tools._load_tools_from_config` |

### Threat Details

#### T1: Prompt Injection via Fetched Web Content

- **Flow**: DF8 (external web) → DF9 (tool result) → C3 Agent context
- **Description**: When the agent calls `fetch_url` or `web_search`, the response body enters the LLM's context window as a `ToolMessage`. A maliciously crafted web page or search snippet can embed natural-language instructions that the LLM may interpret as authoritative commands, leading to unexpected tool call requests in the next turn.
- **Preconditions**: (1) User or LLM-initiated call to `fetch_url`/`web_search` reaches a malicious page; (2) LLM interprets injected instructions as directives; (3) In interactive mode, user must still approve the resulting tool call.

#### T2: Shell Allow-List Bypass via `SHELL_ALLOW_ALL`

- **Flow**: DF7 (LLM tool call) → C4 Tools (execute)
- **Description**: When `--shell-allow-list all` (or `DEEPAGENTS_SHELL_ALLOW_LIST=all`) is set, `is_shell_command_allowed` returns `True` for any non-empty command without invoking `contains_dangerous_patterns`. In non-interactive mode, any shell command the LLM requests executes unconditionally. Combined with T1, an attacker-controlled page could cause arbitrary command execution.
- **Preconditions**: (1) User has configured `--shell-allow-list all`; (2) Non-interactive mode; (3) Successful prompt injection via DF8/DF9.

#### T3: Unicode Homoglyph URL in Approval Dialog

- **Flow**: DF7 (LLM-generated fetch_url args) → C2 TUI approval dialog
- **Description**: An LLM influenced by adversarial input could generate a `fetch_url` call with a URL containing mixed-script or confusable characters visually identical to ASCII.
- **Preconditions**: LLM generates a confusable URL (requires adversarial steering). `check_url_safety` detects mixed-script domain labels and `strip_dangerous_unicode` removes invisible BiDi/zero-width characters; warnings displayed in approval dialog. Classified Disproven as a project vulnerability — the UI warnings are the intended control.

#### T4: Auto-Approve Removes All Execution Safeguards

- **Flow**: DF7 (all tool calls) when `auto_approve=True`
- **Description**: When auto-approve is enabled (via `--auto-approve` flag or `Shift+Tab` in TUI), all tool calls including `execute`, `write_file`, `edit_file`, `fetch_url`, `launch_async_subagent` execute without user confirmation.
- **Preconditions**: User explicitly enables auto-approve. Default is approval-required.

#### T5: Local SQLite Checkpoint Tampering

- **Flow**: DF13/DF14 (session persist/restore)
- **Description**: LangGraph checkpoints stored in `~/.deepagents/*.db` contain the full conversation history and agent state. An attacker with local filesystem write access could inject adversarial messages that re-enter the LLM context on session resume.
- **Preconditions**: Attacker has write access to the user's home directory — equivalent to a fully compromised user account.

#### T6: Unauthenticated LangGraph Dev Server on Localhost

- **Flow**: DF3/DF4 (CLI ↔ LangGraph dev server)
- **Description**: The CLI spawns a `langgraph dev` server subprocess with `LANGGRAPH_AUTH_TYPE=noop` (`server.py:_build_server_env`). This disables all server-side authentication. The server binds to `127.0.0.1:{port}` (default port 2024, or a random free port if 2024 is occupied). Any local process that discovers the port can: send arbitrary inputs to the running agent thread, read the agent's conversation state (including tool results that may contain file contents or secrets), inject messages into the conversation history, or trigger state updates. The server is ephemeral — it lives only for the duration of the CLI session — but this is the entire attack window. Port discovery is feasible via localhost port scanning or by reading `/proc/{pid}/cmdline` which contains the `--port` argument.
- **Preconditions**: (1) Attacker has a local process running as the same user (or as root); (2) Attacker discovers the server port (port scan on localhost, or reads process arguments).

#### T7: LocalContextMiddleware Injects Host File Contents into System Prompt

- **Flow**: DF19 → DF20
- **Description**: `LocalContextMiddleware` runs a bash script (`build_detect_script`) that reads the first 20 lines of `Makefile` (`_section_makefile`) and a filtered directory listing, then injects this output verbatim into the system prompt on every turn. An attacker with write access to the project's working directory could craft `Makefile` content designed to manipulate the agent's behavior.
- **Preconditions**: Attacker has write access to the `Makefile` in the agent's working directory. The agent must be running in that directory (local mode, not sandbox mode).

#### T8: Custom Subagent Body Used as System Prompt Without Validation

- **Flow**: DF21
- **Description**: `subagents._parse_subagent_file` reads AGENTS.md files from `.deepagents/agents/{name}/AGENTS.md` and project-level `.agents/{name}/AGENTS.md`. The markdown body after the YAML frontmatter is used verbatim as the subagent's `system_prompt`. No content filtering is applied.
- **Preconditions**: Attacker has write access to `~/.deepagents/agents/` or the project's `.agents/` directory. User or LLM must invoke the malicious subagent via the `task` tool.

#### T9: Arbitrary Python Code Execution via `class_path` Config

- **Flow**: DF23 (config.toml → importlib)
- **Description**: The `class_path` field in `[models.providers.<name>]` config triggers `importlib.import_module()` in `config._create_model_from_class`. While the imported class is validated as a `BaseChatModel` subclass, module-level code executes unconditionally during import — before the type check runs. A malicious or compromised `config.toml` pointing to a hostile module causes arbitrary code execution at model initialization time. This applies to both `class_path` and the `_load_provider_profiles` path that uses `exec_module()` to load `_profiles.py` from provider packages.
- **Preconditions**: Attacker has write access to `~/.deepagents/config.toml` AND a malicious Python package installed in the user's environment (or on `sys.path`). The code comments document this as intentional: "same trust model as `pyproject.toml` build scripts — the user controls their own machine."

#### T10: MCP Stdio Env Dict Forwarded Without Filtering

- **Flow**: DF24 (MCP config → subprocess environment)
- **Description**: The `"env"` field in stdio MCP server definitions (`.mcp.json`) accepts an arbitrary key-value dict. `mcp_tools._validate_server_config` only checks that the field is a dict — it does not filter key names or values. The dict is forwarded directly to `StdioConnection(env=...)` which passes it to the subprocess. An attacker who can modify a project-level `.mcp.json` could set `PATH` to redirect command resolution, `LD_PRELOAD` to inject shared libraries, or `PYTHONPATH` to hijack Python imports in the MCP subprocess.
- **Preconditions**: (1) Attacker has write access to a project-level `.mcp.json`; (2) The project MCP config must be trusted by the user (fingerprint approval gate via `mcp_trust`). For user-level `~/.deepagents/.mcp.json`, the attacker already has home directory write access. Note: the `env` dict from MCP config is passed to `StdioConnection` — whether it replaces or merges with `os.environ` depends on the `langchain_mcp_adapters` library implementation.

---

## Input Source Coverage

| Input Source          | Data Flows            | Threats       | Validation Points                                                          | Responsibility | Gaps                                                                                         |
|-----------------------|-----------------------|---------------|----------------------------------------------------------------------------|----------------|----------------------------------------------------------------------------------------------|
| User direct input     | DF1, DF2              | None (TB1)    | None — prompts accepted verbatim                                           | User           | No content filtering — intentional; HITL gates downstream tool calls                        |
| LLM output            | DF6, DF7              | T1, T2, T3, T4| HITL gate; shell allow-list; Unicode/URL warnings on tool args             | Project        | LLM-generated tool args not scanned for injection beyond Unicode/URL                        |
| Tool/function results | DF9, DF11             | T1            | Unicode warning on URL args; `markdownify` HTML conversion                 | Shared         | Tool *results* pass to context without prompt-injection scan                                 |
| URL-fetched content   | DF8, DF9              | T1            | `check_url_safety` on URL arg; HTML→markdown conversion                    | Shared         | Markup-embedded instructions survive markdownify; no LLM-layer guardrail                    |
| Configuration         | DF10, DF12, DF15, DF23| T9, T10       | TOML schema; MCP schema + fingerprint; JSON structure check; `class_path` format check | User | `class_path` executes module code before type check; MCP env dict unfiltered               |
| Session restore       | DF14                  | T5            | OS file permissions; SQLite                                                | Project        | Unencrypted at rest                                                                           |
| Server IPC (env vars) | DF18                  | T6            | `ServerConfig` serialization; parent env passed to child                   | Project        | Provider API keys flow to server subprocess; system prompt in env                            |
| Host environment      | DF19, DF20            | T7            | Static script; exit code check; 30s timeout                                | Shared         | Makefile content injected into system prompt without sanitization                            |
| Custom subagents (FS) | DF21                  | T8            | `yaml.safe_load`; HITL on `task` tool                                      | User           | Subagent body text not content-filtered                                                      |
| Async subagent config | DF22                  | None direct   | TOML parse; type validation in `load_async_subagents`                      | User           | URL and headers for remote subagents are user-controlled; no URL validation                 |
| MCP subprocess env    | DF24                  | T10           | Dict type check only (`_validate_server_config`)                           | User           | No key/value filtering; arbitrary env vars forwarded to subprocess                           |
| Offloaded history     | DF25                  | None direct   | Thread ID is UUID7 (no path injection); backend handles storage            | Shared         | Raw conversation content written to sandbox filesystem                                       |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern                                                                 | Why Out of Scope                                                                                                                                                                     | Project Responsibility Ends At                                                                                     |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Prompt injection leading to arbitrary code execution (interactive mode) | In interactive mode, every side-effecting tool call requires explicit user approval via the HITL dialog. The user is the final gatekeeper.                                          | Providing HITL for all side-effecting tools (`agent._add_interrupt_on`) and Unicode/URL warnings in the approval dialog. |
| API key exfiltration via LLM-directed `http_request`                   | `http_request` requires HITL in interactive mode. Keys it could exfiltrate are user-supplied env vars. In non-interactive mode, user has opted into autonomous operation.             | Providing HITL gate for HTTP tools. Users control which env vars are in scope.                                      |
| Malicious MCP server injecting prompt instructions                     | Users configure MCP servers and explicitly trust project-level configs. Once trusted, MCP tool outputs are data from a system the user controls.                                     | Fingerprint-based trust prompts for project-level stdio configs (`mcp_trust.compute_config_fingerprint`).           |
| LLM jailbreak / safety bypass                                          | Model selection and safety configuration are user-controlled. The project routes prompts to the configured LLM but cannot guarantee model behavior.                                   | Correctly routing prompts to the configured LLM; applying the system prompt from `agent.get_system_prompt`.         |
| Sandbox provider security vulnerabilities                              | Daytona, LangSmith, Modal, Runloop, and AgentCore are third-party services. Their internal security is not this project's responsibility.                                            | Correctly initializing sandbox sessions via `integrations.sandbox_factory.create_sandbox`.                          |
| Hook commands doing harmful things                                     | Hooks in `~/.deepagents/hooks.json` are 100% user-authored. The payload is data-only (JSON on stdin).                                                                               | JSON structure validation (`hooks._load_hooks`); 5-second timeout.                                                 |
| Async subagent traffic interception / MitM                             | Async subagents connect to user-configured LangGraph deployment URLs. The project does not control those endpoints or their TLS certificates.                                        | Accepting URL/headers from user config and passing them to the LangGraph SDK (`agent.load_async_subagents`).        |
| LangGraph dev server port enumeration / discovery                     | Discovering the local dev server port requires local access. Port scanning localhost is a general OS security concern, not a framework vulnerability.                                 | Binding to `127.0.0.1` by default (`server._DEFAULT_HOST`); ephemeral server lifetime.                              |
| `.env` file from parent directory changes behavior                     | `config._find_dotenv_from_start_path` walks up the directory tree to find `.env` files. This is standard `python-dotenv` behavior for project discovery. The user controls their filesystem. | Finding `.env` from the project root (`config._find_dotenv_from_start_path`); `override=False` by default (existing env vars preserved). |

### Rationale

**Prompt injection in interactive mode**: The HITL interrupt means every file write, shell command, web search, URL fetch, task delegation, and async subagent action shows the user a confirmation dialog with full tool arguments. Even a successful prompt injection can only execute what the user explicitly approves. The project's responsibility is to make that dialog accurate — hence the Unicode/URL warning layer in `unicode_security.py`.

**LangGraph dev server without auth**: The `LANGGRAPH_AUTH_TYPE=noop` setting is intentional for local dev server use. Adding authentication would require users to manage tokens for a locally-spawned ephemeral process, creating more friction than security benefit in this context. The 127.0.0.1 binding limits exposure to the local machine. T6 documents this as an accepted risk for the threat model.

**Custom subagent system prompts**: Subagent definitions in `.deepagents/agents/` are user-authored files. The framework correctly treats them as user-controlled content. The HITL gate on `task` tool calls ensures the user approves subagent delegation before it occurs.

**`class_path` code execution**: This follows the same trust model as `pyproject.toml` build scripts — the user edits their own config file on their own machine. The `issubclass(BaseChatModel)` check provides a post-import guard, though module-level side effects execute before it. Documented as intentional in `model_config.py`.

---

## Investigated and Dismissed

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | Unsafe msgpack deserialization in langgraph checkpoint loading | Verified fix status — confirmed fixed and closed upstream. | Users on current `langgraph` versions are not exposed. | Upstream langgraph has patched the unsafe msgpack deserialization. No longer an active risk. |
| D2 | Unicode URL homoglyph as project vulnerability | Traced `check_url_safety` + `strip_dangerous_unicode` + `format_warning_detail` → approval dialog display | `unicode_security.check_url_safety`, `agent._format_fetch_url_description` | Warning system is the intended control — the project correctly surfaces the risk to the user in the approval dialog. Not a project vulnerability; classified as mitigated by design (UI warning). |
| D3 | SSRF via `http_request` / `fetch_url` to internal services | Traced `tools.http_request` and `tools.fetch_url` — no URL scheme or host blocklist. However, both tools require HITL approval in interactive mode. In non-interactive mode, only shell commands are auto-approved via the allow-list; HTTP tools still go through the HITL interrupt gate. | `tools.http_request`, `tools.fetch_url`, `agent._add_interrupt_on` | Not a project vulnerability in isolation — the HITL gate is the intended control for all HTTP tool calls. The user sees the full URL before approving. SSRF is only reachable if the user approves the request (interactive) or enables auto-approve (explicit opt-in). Classified as out-of-scope for the same reason as prompt injection in interactive mode. |
| D4 | Offload path injection via thread_id | Checked `sessions.generate_thread_id` — returns UUID7 string (alphanumeric + hyphens only, no path separators). | `sessions.generate_thread_id`, `offload.offload_messages_to_backend` | Thread IDs are UUID7 strings generated by the framework. No user-controlled path components reach the file path. Not exploitable. |

---

## Revision History

| Date       | Author                             | Changes                                                                                          |
|------------|------------------------------------|--------------------------------------------------------------------------------------------------|
| 2026-03-10 | langster-threat-model (automated)  | Initial threat model                                                                             |
| 2026-03-27 | langster-threat-model (automated)  | Deep expansion: added C11-C16 (server subprocess, RemoteAgent, LocalContextMiddleware, async subagent config, custom subagent loader); added TB8-TB10 (CLI/server IPC, LocalContext/host env, RemoteAgent/dev server); added DF18-DF22; added T6 (unauthenticated dev server), T7 (Makefile injection), T8 (subagent body injection); updated T5 (upstream msgpack fix confirmed); added data classification; added Investigated and Dismissed section; updated architecture diagram to reflect server-subprocess model |
| 2026-03-28 | langster-threat-model (automated)  | Deep validation pass: added C17 (Model Config Loader with class_path), TB11 (Config→Code Execution); added DC5 (offloaded conversation history); added DF23-DF25 (class_path flow, MCP env dict flow, offload flow); added T9 (class_path arbitrary code execution), T10 (MCP env dict unfiltered); added D3 (SSRF dismissed — HITL is intended control), D4 (offload path injection dismissed — UUID7); **removed Status column and Mitigations/Residual Risk fields from all threats** (open source visibility compliance — mitigation status must not appear in public threat models); updated T6 validation from Likely to Verified (port is deterministic at 2024 default, discoverable via /proc); updated external context (no published advisories found); updated architecture diagram |
