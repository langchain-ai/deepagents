---
type: operator workflow guide
title: Run a dcode Session
description: Run dcode interactively, headlessly, or as an ACP server, and understand the startup, workspace-binding, approval, persistence, offload, and diagnostic boundaries that govern each mode.
tags: [dcode, deepagents-code, cli, sessions, headless, acp, approvals, mcp, hooks, sandboxes]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-15T08:05:27.526Z
sources:
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-a9143c1c174362216a1cfa2c
    resource: repo://libs/code/deepagents_code/approval_mode.py
  - id: openwiki-source-b9ef532d79a0667acf40e58b
    resource: repo://libs/code/deepagents_code/client/launch/server_manager.py
  - id: openwiki-source-074ce96a8baea27a6c43328b
    resource: repo://libs/code/deepagents_code/client/launch/server.py
  - id: openwiki-source-ecf20e7a2684ba0d2ae7d701
    resource: repo://libs/code/deepagents_code/client/non_interactive.py
  - id: openwiki-source-b7d66cbdbe9dae9f133a7c5e
    resource: repo://libs/code/deepagents_code/client/remote_client.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-ea1089f0d7536fbc96c64866
    resource: repo://libs/code/deepagents_code/offload_api.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-0f8622164498a685abc913d5
    resource: repo://libs/code/deepagents_code/sessions.py
  - id: openwiki-source-29a60a7d68da0bf4ec625403
    resource: repo://libs/code/deepagents_code/tui/textual_adapter.py
  - id: openwiki-source-030d8bd153a9c3ea2a99cb7d
    resource: repo://libs/code/deepagents_code/workspace.py
  - id: openwiki-source-88fb8e5a1d032ebc6b6d11b3
    resource: repo://libs/code/EXTENSIONS.md
  - id: openwiki-source-a7917911d186cc47811a1430
    resource: repo://libs/code/HOOKS.md
  - id: openwiki-source-1d73b3e2b56b5f0d27273379
    resource: repo://libs/code/README.md
  - id: openwiki-source-6e002fd7a8a5dcb5186cae05
    resource: repo://libs/code/tests/integration_tests/test_compact_resume.py
  - id: openwiki-source-c8dacdfd6192dd22d24a9362
    resource: repo://libs/code/tests/integration_tests/test_pending_work_recovery.py
generated: { by: "openwiki/0.4.2", at: "2026-09-15T08:05:27.526Z" }
---

# Run a dcode Session

`deepagents-code` (`dcode`) has three deliberately different launch shapes:

- **Interactive** `dcode` starts the Textual terminal UI.
- **Headless** `dcode -n "…"` runs one task for scripts and CI.
- **ACP** `dcode --acp` serves the Agent Client Protocol over standard input/output for an editor or other ACP client.

Interactive and headless mode are clients of a temporary local LangGraph server. ACP is an in-process stdio path, not headless mode with a different renderer. See [code-agent architecture](../architecture/code-agent.md), [runtime behavior](../architecture/runtime-behavior.md), [configuration layering](../concepts/config-layering.md), [state persistence](../concepts/state-persistence.md), [MCP](../integrations/mcp.md), and [cost and sessions](../operations/cost-and-sessions.md).

## Start safely and select a mode

```bash
curl -LsSf https://langch.in/dcode | bash

# Interactive TUI
dcode

# One bounded task for CI or a script
dcode -n "run the focused tests" --max-turns 8 --timeout 600

# ACP server for an ACP-capable host
dcode --acp
```

The installer includes OpenAI, Anthropic, and Gemini; set `DEEPAGENTS_CODE_EXTRAS` to request other provider extras. Treat the directory as a trust boundary. dcode reads project artifacts before the approval UI exists, so approval does **not** make an untrusted checkout safe. For an untrusted repository, use a remote sandbox rather than running on the host.

No `--sandbox` means local execution. A bare `--sandbox` selects `[sandboxes].default`; a value selects that provider. `--sandbox-id`, `--sandbox-snapshot-name`, and `--sandbox-setup` attach to or provision the selected remote environment.

## Understand CLI dispatch and bounded input

Managed configuration is checked before normal agent operations. If it cannot be enforced, dcode fails closed with exit code 78. The help path and diagnostic `config`, `doctor`, and `auth path` commands remain available; `threads list`/`ls` and `threads delete` dispatch directly against persisted state without launching an agent.

Piped stdin is text-only and capped at 10 MiB. Its destination is chosen in this order:

1. prepend it to an existing `-n` task;
2. prepend it to an interactive `-m` initial prompt;
3. seed an auto-detected interactive `--skill` prompt; or
4. make it a new headless task.

`--stdin` explicitly requires non-terminal stdin. It deliberately bypasses the interactive skill convenience and produces a headless task. In the interactive-prompt case dcode attempts to restore `/dev/tty`; without a controlling terminal the later TUI launch reports that it is not a terminal.

Headless output and controls such as `--quiet`, `--no-stream`, `--max-turns`, `--timeout`, and `--rubric` require a headless task, whether supplied with `-n` or derived from a pipe. Headless shell access is disabled unless `--shell-allow-list` is supplied. Turn-budget and wall-clock timeout exhaustion exit 124, so CI should distinguish it from a general error.

## Run a normal interactive or headless session

For either normal mode, `server_session` creates a resolved `ServerConfig`, preflights an explicitly supplied MCP configuration, and starts a temporary `langgraph dev` runtime. The server normally listens only on loopback `127.0.0.1` and an ephemeral port. Once its `agent` graph is ready, the client creates a `RemoteAgent`, gives it the launch workspace and a configuration fingerprint, and hands it to either the TUI or the headless client.

Startup ownership matters operationally: failed, cancelled, or incomplete setup stops the child process; `server_session` also stops it during ordinary teardown. An explicit malformed or missing `--mcp-config` therefore fails in the parent before a child is launched, while discovered project and user configuration is handled by server-side discovery.

### Bind the workspace before streaming

A normal client cannot simply choose a directory for each request. Before using a thread it binds the workspace through the server. The server resolves the canonical workspace and its policy, refuses a client claim that includes project-policy fields or whose session claim/fingerprint disagrees, persists the binding, then registers thread metadata. Changed or conflicting workspace information produces a validation or conflict response rather than a silent policy change.

Every `RemoteAgent` stream needs a thread ID and carries that thread's workspace context. At execution time the graph verifies the durable binding, re-resolves configuration to detect project-policy or fingerprint drift, and returns a cached or newly built workspace runtime. Workspace runtimes use an LRU cache capped at 32 entries. Sandbox ownership is process-wide: after one workspace claims a sandbox backend, another workspace cannot bind it in that server process.

The selected server runtime—not the terminal client—owns the frozen workspace environment and credentials, built-in and MCP tools, optional sandbox, compiled agent, backend, and derived offload operation. Keep changes to tools, environment handling, compaction, or archive storage on this server-side boundary.

### Stream, approve, and resume a turn

`RemoteAgent` translates server streams, messages, and human-in-the-loop interrupts into values for the client. The TUI requests `messages`, `updates`, and `custom` streams, renders tool and model activity, obtains an approval or `ask_user` response, then resumes the graph. The headless client uses the same normal server arrangement but creates a new thread for each process.

Interactive approval modes are **Manual**, **Auto**, and **YOLO**. An invalid persisted mode falls back to Manual; the mode switcher skips modes unavailable in the current session; and YOLO requires acknowledgement of the current policy version. Use this as a tool-call review control, not as a substitute for workspace trust.

## Resume durable state

Checkpoints are in global cached SQLite state. New thread IDs are UUID7 values and therefore naturally time-sortable. `threads list` maintains a covering index so it can list metadata without reading large checkpoint blobs; if creating the index fails, listing still returns correct results through a slower scan.

Use `dcode -r` to choose the most recent eligible thread or `dcode -r <ID>` for an explicit thread. The TUI refuses a resume beyond a configured absolute or rolling-age cutoff. It can offer to switch to a thread's stored working directory, suggest similar IDs for a miss, and fall back to a fresh thread after a missed lookup, declined resume, or database error. This resume behavior is TUI-specific: headless mode always creates a new thread.

## Offload without bypassing the server

`/offload` (also `/compact`) is a server-owned HTTP operation that uses the same workspace runtime and backend as the graph. It operates only on a registered, durable-workspace-bound thread whose HTTP status is idle or error and whose checkpoint is quiescent—no pending graph work, tasks, or interrupts. A conflicting, unregistered, active, interrupted, changed, or pending-work thread is rejected before commit.

The route may write only state channels declared by `OffloadStateUpdate`; in particular, it cannot overwrite conversation `messages`. This makes compaction safe against a concurrent normal turn and keeps archive storage on the backend that later server agents use. A fresh server can consequently resume the thread and read the persisted archive.

If an offload hook requires input, the client does not keep a suspended server coroutine. It reposts the operation with the same operation ID and all accumulated hook responses; the server re-executes the operation and replays the answered hooks.

## Configure extensions and hooks deliberately

`--mcp-config` has the highest MCP configuration precedence, while `--no-mcp` disables all MCP loading. Project MCP servers require trust. Experimental project Python extensions also require trust; their tools are not automatically placed in dcode's human-approval map.

Hooks execute user-privilege commands with JSON lifecycle payloads. Project hooks require workspace trust. Matching handlers run concurrently, with results reduced in project → user → plugin order; exit code 2 blocks only where that lifecycle event defines blocking behavior. Treat hooks and extensions as trusted code execution, particularly in CI where the explicit project-trust flags control whether repository configuration is loaded.

## Serve ACP separately

`dcode --acp` does not call `server_session`, launch a loopback `langgraph dev` server, or create a `RemoteAgent`. It resolves the model and project context in the launching process, loads MCP tools, opens the SQLite checkpointer, builds an agent for each ACP session context, and serves it with `run_acp_agent`. MCP cleanup runs in `finally`.

ACP model-configuration, dependency-import, or MCP-loading failures are reported before serving with exit code 1. An exception while serving is reported as an ACP server failure and also returns 1. The ACP smoke test launches `deepagents --acp --no-mcp`, performs protocol initialization, creates a session for the current working directory, and verifies that a session ID is returned.

## Focused regression checks

When changing this workflow, exercise the boundary that owns the behavior rather than only the UI:

| Change | Focused verification |
| --- | --- |
| CLI routing, pipe precedence, budgets, and exit codes | CLI/unit coverage around `main.py` |
| Spawn, MCP preflight, workspace handoff, and cleanup | server-manager tests plus a normal-session smoke test |
| Binding, drift detection, runtime cache, and sandbox ownership | workspace/server-graph and route conflict tests |
| Stream conversion, HITL resumption, and abandoned work | remote-client tests and `test_pending_work_recovery.py` |
| Compaction across restart | `test_compact_resume.py` and server-side offload integration coverage |
| ACP protocol lifecycle | `test_acp_mode.py` |

The compaction-resume integration test persists a thread through one temporary server, runs `/offload` in a fresh production-style app with no client-owned backend, then verifies later server agents can read the archive. The pending-work recovery test abandons a graph paused before a tool node and verifies that the tool does not execute, queued work is cleared, and an error `ToolMessage` records cancellation. Together these protect the key operational invariants: persistence and compaction are server-owned, and abandoned pending work must not later execute.
