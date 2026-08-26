---
type: architecture-navigation
title: Source Map
description: Directory-to-responsibility index across the Deep Agents monorepo, mapping create_deep_agent assembly, SDK middleware/backends/profiles, and the deepagents-code coding agent to the files that own each behavior.
tags: [source-map, navigation, monorepo, deepagents, deepagents-code, architecture]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T21:35:57.774Z
sources:
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-3396dda6599f7426e19ed526
    resource: repo://libs/code/deepagents_code/__init__.py
  - id: openwiki-source-dc8749c06f6da0ecc0666f26
    resource: repo://libs/code/deepagents_code/_session_stats.py
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-a9143c1c174362216a1cfa2c
    resource: repo://libs/code/deepagents_code/approval_mode.py
  - id: openwiki-source-18abc7e59899514f067032b2
    resource: repo://libs/code/deepagents_code/auto_mode.py
  - id: openwiki-source-7f6b98925b5f1ba065df3a04
    resource: repo://libs/code/deepagents_code/config.py
  - id: openwiki-source-f2ac9d5fb6c7c6a21f241281
    resource: repo://libs/code/deepagents_code/cost_tracking.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-f6d553e7afdf54acac36e7d3
    resource: repo://libs/code/deepagents_code/mcp_tools.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-c100a7d2ff8c43af8ad1b816
    resource: repo://libs/code/deepagents_code/offload_middleware.py
  - id: openwiki-source-9b6cab59e92c8914079f0f53
    resource: repo://libs/code/deepagents_code/offload.py
  - id: openwiki-source-620b4c9d0fcbd4c7e6aa0120
    resource: repo://libs/code/deepagents_code/resume_state.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-0f8622164498a685abc913d5
    resource: repo://libs/code/deepagents_code/sessions.py
  - id: openwiki-source-3300d75e0c132882e2e3b4ce
    resource: repo://libs/code/deepagents_code/tool_catalog.py
  - id: openwiki-source-fd64c1b88759a3b897a5452c
    resource: repo://libs/deepagents/deepagents/__init__.py
  - id: openwiki-source-7661ce56409855dfd168bb2c
    resource: repo://libs/deepagents/deepagents/backends/__init__.py
  - id: openwiki-source-a1549ea98d425efea270be93
    resource: repo://libs/deepagents/deepagents/backends/composite.py
  - id: openwiki-source-d70fe6f8bf81e2aa641a4950
    resource: repo://libs/deepagents/deepagents/backends/context_hub.py
  - id: openwiki-source-e3efb5f3e4a9e8517eb6d8f5
    resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fc54598423086acf9d53d9fd
    resource: repo://libs/deepagents/deepagents/middleware/__init__.py
  - id: openwiki-source-f01b7478b818ecc507f2ed5d
    resource: repo://libs/deepagents/deepagents/middleware/permissions.py
  - id: openwiki-source-b27554b5c0e5b26fae2efb38
    resource: repo://libs/deepagents/deepagents/profiles/__init__.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
generated: {by: "openwiki/0.4.0", at: "2026-08-26T21:35:57.774Z"}
---

# Source Map

This page is a navigational index, not a reimplementation of each module. It maps
directories and key files to the behavior they own so an agent can jump to the
right place fast. For how the pieces fit together at runtime, read
[Architecture Overview](/openwiki/architecture/overview.md) and the
[Code Agent](/openwiki/architecture/code-agent.md). For deeper treatment of the
pluggable storage layer and the middleware list, see
[Backends](/openwiki/concepts/backends.md) and the
[Middleware Catalog](/openwiki/concepts/middleware-catalog.md).

The authoritative, in-repo maps are `libs/ARCHITECTURE.md` (SDK) and
`libs/code/ARCHITECTURE.md` (coding agent); prefer those and the per-package
`README.md` files over any duplicated detail here.

## How to navigate this codebase

The repository is a monorepo of independently versioned packages under `libs/`,
each with its own `pyproject.toml`, `Makefile`, and `README.md`; there is no root
`pyproject.toml`, and you work inside the package you are changing.

The single most useful navigation heuristic comes from `libs/ARCHITECTURE.md`:
**trace from a `create_deep_agent()` argument to the middleware or backend it
installs, then follow how that component participates during execution.** If a
tool is missing, look at middleware assembly and profile exclusions; if a tool is
visible but fails, look at backend capability and permission enforcement.

Deep Agents sits in three layers. Knowing which layer owns a behavior narrows the
search before you open a file.

```mermaid
flowchart TD
    DA["Deep Agents harness: defaults, middleware, backends, profiles"]
    LC["LangChain create_agent: model plus tools plus middleware to agent loop"]
    LG["LangGraph runtime: state, checkpoints, streaming, interrupts"]
    DA --> LC --> LG
    CODE["deepagents-code: terminal coding agent built on the SDK"]
    CODE --> DA
```

Layered dependency map: `deepagents-code` builds on the `deepagents` SDK, which builds on LangChain's `create_agent`, which builds on the LangGraph runtime.

## Core SDK: `libs/deepagents/deepagents/`

Most Deep Agents-specific code lives here. The package root exposes the public
surface (`create_deep_agent`, `DeepAgentState`, middleware, backends, and profile
registration helpers) through `__init__.py`.

### Assembly entrypoint — `graph.py`

`graph.py` is the assembly point. `create_deep_agent()` resolves the model and any
provider/harness profile, resolves the backend, assembles the main-agent
middleware stack, builds the default and caller-supplied subagents, composes the
system prompt, and delegates to LangChain's `create_agent(...)`. It is the file to
open first for middleware ordering, prompt assembly, and default model behavior.

### Middleware — `middleware/`

Middleware adds harness behavior to the agent loop: it can filter tools per model
call, inject system-prompt context, transform messages, and maintain typed
cross-turn state — things a plain `tools=` callable cannot do because that
callable only runs *after* the model chooses it.

| Module | Responsibility |
| --- | --- |
| `filesystem.py` | Built-in filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), removal of `execute` when the backend lacks shell support, and `FilesystemPermission` policy types. |
| `summarization.py` | Token counting, tool-argument truncation, and history summarization/compaction as the context window fills. |
| `skills.py` | `SkillsMiddleware`: injects skill instructions and loads reusable behaviors on demand. |
| `subagents.py` | `SubAgentMiddleware`, declarative `SubAgent`, and `CompiledSubAgent`; runs nested `create_agent` delegation via the `task` tool. |
| `async_subagents.py` | `AsyncSubAgentMiddleware` / `AsyncSubAgent` for asynchronous or remote delegated work. |
| `memory.py` | `MemoryMiddleware`: injects memory instructions and wires long-term recall. |
| `rubric.py` | `RubricMiddleware` and grader types for evaluating outputs against rubrics. |
| `permissions.py` | Backward-compatible re-export of `FilesystemPermission` from `filesystem.py`. |
| `_tool_exclusion.py` | `_ToolExclusionMiddleware`: hides tools a harness profile lists in `excluded_tools`. |
| `patch_tool_calls.py`, `_prompt_caching.py`, `_fs_interrupt.py`, `_message_eviction.py`, `_overflow_clip.py` | Tail/support behaviors: tool-call patching, prompt caching, filesystem human-in-the-loop interrupts, and history clipping/eviction. |

### Backends — `backends/`

Backends decide where files, memory, and shell execution live behind one uniform
interface. `protocol.py` defines `BackendProtocol` (and `SandboxBackendProtocol`
for shell-capable backends) that every implementation follows.

| Module | Responsibility |
| --- | --- |
| `protocol.py` | `BackendProtocol` / `SandboxBackendProtocol`, standardized file-operation error codes, and result types. |
| `state.py` | `StateBackend`: default thread-scoped storage in LangGraph state. |
| `store.py` | `StoreBackend` and `NamespaceFactory`: durable store-backed files across threads. |
| `filesystem.py` | `FilesystemBackend`: maps files to local disk, including ripgrep-backed grep. |
| `sandbox.py` | Sandboxed/remote execution backends implementing shell `execute`. |
| `composite.py` | `CompositeBackend`: routes operations to different backends by path prefix. |
| `langsmith.py` | `LangSmithSandbox` integration. |
| `local_shell.py` | `LocalShellBackend` and `DEFAULT_EXECUTE_TIMEOUT` for local command execution. |
| `context_hub.py` | `ContextHubBackend`: stores files in a LangSmith Hub agent repo. |

### Profiles — `profiles/`

Profiles tune the harness for a provider or model spec across two orthogonal
phases. **Provider profiles** (`provider/provider_profiles.py`) control
model construction (`init_chat_model` kwargs and pre-init side effects).
**Harness profiles** (`harness/harness_profiles.py`) control the runtime phase —
prompt assembly, tool visibility, extra middleware, and default subagent
behavior. `_builtin_profiles.py` registers built-ins and lazily loads third-party
plugins on first registry access; `_keys.py` validates the `provider` /
`provider:model` registry keys. Built-in modules live under `provider/` (e.g.
`_openai`, `_openrouter`, `_nvidia`) and `harness/` (e.g. `_anthropic_sonnet_4_6`,
`_openai_codex`).

## Coding agent: `libs/code/deepagents_code/`

`deepagents-code` is a prebuilt terminal coding agent (the `dcode` CLI) built on
the SDK. It runs as two halves in separate processes: a **terminal client** that
owns presentation and input, and an **agent server** that owns the agent runtime,
connected by a streaming protocol. When debugging, first decide which half owns
the failure. The package is large; the table below sketches the major modules by
domain rather than listing every file.

| Domain | Key modules | Responsibility |
| --- | --- | --- |
| Runtime & entry | `__init__.py`, `__main__.py`, `main.py`, `agent.py`, `server_graph.py` | `cli_main` startup and the main loop (`main.py`); coding-agent construction over `create_deep_agent` (`agent.py`); the `make_graph()` factory served to `langgraph dev` via `ServerConfig` (`server_graph.py`). |
| Application/UI | `app.py`, `ui.py`, `tui/`, `input.py`, `theme.py` | Textual terminal application, the `textual_adapter`, screens/modals/widgets, input handling, and theming. |
| Configuration | `config.py`, `model_config.py`, `config_manifest.py`, `configuration/` | Layered user/project/session/runtime configuration, model/provider definitions loaded from TOML, and the shared config resolver/generation model. |
| Tools & MCP | `tools.py`, `tool_catalog.py`, `managed_tools.py`, `mcp_tools.py`, `mcp_config.py`, `mcp_providers/` | Consumer-provided tools, the tool catalog behind `dcode tools list` / `/tools` read from real bound tools, and MCP server discovery/loading. |
| Cost & stats | `cost_tracking.py`, `_session_stats.py` | Durable per-thread cost accumulation via `CostTrackingMiddleware` riding graph checkpoints, and lightweight session/usage statistics and token formatting. |
| Sessions & resume | `sessions.py`, `resume_state.py`, `state_migration.py` | Thread management over LangGraph checkpoints, per-checkpoint `ResumeState` channels for `dcode -r`, and state migration. |
| Context offload | `offload.py`, `offload_api.py`, `offload_middleware.py` | Storage paths for offloaded conversation history and CLI-specific compaction/offload middleware. |
| Modes & approval | `auto_mode.py`, `approval_mode.py`, `ask_user.py` | Classifier-backed Auto mode, per-thread approval-mode state shared by the client and server, and human-in-the-loop prompting. |
| Goals & rubric | `goal_tools.py`, `goal_rubric.py`, `goal_state_*.py`, `reliable_rubric.py` | Goal tracking, rubric grading, and goal-state limits/notices. |
| Skills & subagents | `skills/`, `built_in_skills/`, `subagents.py` | Bundled skills and coding-agent subagent definitions. |

## Other packages under `libs/`

Beyond the core SDK and the coding agent, `libs/` hosts sibling packages that are
useful to know exist but out of scope for detailed mapping here: `acp/` (Agent
Client Protocol integration), `evals/` and `harbor/` (evaluation suite and
Harbor integration), `talon/` (local runtime host for long-running agents),
`cli/`, and `partners/` (provider/sandbox integrations such as `daytona`,
`modal`, `vercel`, `runloop`, and `quickjs`).

## Where authoritative detail lives

Treat these as the source of truth and follow them rather than duplicating their
contents:

- `libs/ARCHITECTURE.md` — SDK layers, construction/execution, middleware stack,
  and "Common starting points".
- `libs/DEVELOPMENT.md` — monorepo layout, `uv`/`make` workflows, and commands.
- `libs/code/ARCHITECTURE.md` — the coding agent's client/server split, request
  flow, and configuration model.
- Per-package `README.md` and `pyproject.toml` — package scope and supported
  Python range.
