---
type: repository source map
title: Repository Source Map
description: Navigate Deep Agents changes from a public surface to the package that owns the behavior, its focused tests, and its independent release unit. Includes SDK assembly, Code, ACP, Talon, evaluations, partner integrations, and release wiring.
tags: [source-map, architecture, monorepo, deepagents, releases, integrations]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-21T08:06:25.442Z
sources:
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-b64c485d8d3cdc25e7b4db1a
    resource: repo://libs/code/deepagents_code/_debug.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-e930bbb03b92760cf9d657ce
    resource: repo://libs/code/tests/unit_tests/test_debug.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ef66a16bd57d322614dc349d
    resource: repo://libs/talon/deepagents_talon/async_subagents.py
  - id: openwiki-source-d98b6d615a63b95a7c893810
    resource: repo://libs/talon/deepagents_talon/mcp_middleware.py
  - id: openwiki-source-82cac27adeecff8a900a40fa
    resource: repo://libs/talon/deepagents_talon/mcp.py
  - id: openwiki-source-2d1f686d24d8182f60108ae7
    resource: repo://libs/talon/deepagents_talon/subagents.py
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-8ca4576d19f02a613c296c83
    resource: repo://libs/talon/tests/test_async_subagents.py
  - id: openwiki-source-4c1a7e831a8cd578116d1f18
    resource: repo://libs/talon/tests/test_mcp_middleware.py
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-09-21T08:06:25.442Z" }
---

# Repository Source Map

Use this page to choose the owning boundary before changing behavior. The repository is a monorepo of independently versioned packages: the SDK owns reusable agent-harness composition, while Code, ACP, Talon, evaluations, and partners own their product or integration concerns. For the architectural rationale, see the [overview](./overview.md); for package-local commands and release operations, see [Development, CI, and Releases](../operations/development.md).

## Package and dependency boundaries

```mermaid
flowchart TD
  App["Application"] --> SDK["deepagents SDK"]
  CodeClient["dcode client"] --> CodeServer["Code agent server"]
  CodeServer --> SDK
  Editor["ACP editor"] --> ACP["deepagents-acp"]
  ACP --> SDK
  Channels["Talon channels and cron"] --> Talon["Talon host"]
  Talon --> SDK
  Evals["Evaluation suite"] --> SDK
  Partners["Sandbox and provider packages"] --> CodeServer
  SDK --> LangChain["LangChain create_agent"]
  LangChain --> LangGraph["LangGraph runtime"]
```
This shows dependency direction and the boundary at which a behavior should normally be changed.

`deepagents` is an opinionated harness over LangChain's `create_agent()` and LangGraph; it does not replace the LangGraph runtime. Start reusable graph-construction work in `libs/deepagents/deepagents/graph.py` at `create_deep_agent()`: it accepts the model, tools, middleware, subagents, backend, permissions, persistence, and response configuration, then assembles the agent. Built-in filesystem, shell, and delegation tools are part of that surface; the `execute` tool only succeeds when the chosen backend is sandbox-capable.

The consuming packages deliberately retain their distinct responsibilities:

| Surface or change | Owner and first seam | Focused validation |
| --- | --- | --- |
| Reusable model/profile behavior, prompts, middleware ordering, backends, SDK subagents, permissions, or graph/checkpointer options | **SDK** — `libs/deepagents/deepagents/graph.py` and the appropriate `middleware/`, `backends/`, or `profiles/` module. | `libs/deepagents/tests/`, beginning with graph and feature-specific tests. |
| Terminal commands, TUI rendering, headless interaction, Code-only approval/configuration, client/server execution, sandbox selection, extensions, or skills | **Deep Agents Code** — `libs/code/deepagents_code/`; console scripts enter through `deepagents_code:cli_main`. Follow from `main.py` into `agent.py`, `server_graph.py`, `tui/`, or the relevant product subsystem. | `libs/code/tests/unit_tests/`, including the focused CLI, server, TUI, MCP, or sandbox test. |
| Editor protocol conversion, ACP sessions/options, graph-event streaming, session loading, or replay | **ACP** — `libs/acp/deepagents_acp/server.py`, centered on `AgentServerACP`. | `libs/acp/tests/`, especially agent/session, model-switching, and command-policy coverage. |
| Long-running local channels, cron, host lifecycle, conversation persistence, approval UX, Talon MCP lifecycle, or Talon delegation | **Talon** — `libs/talon/deepagents_talon/__main__.py`, `host.py`, `runtime.py`, and the focused `mcp*.py`, `subagents.py`, `background.py`, or `cron/` module. | `libs/talon/tests/`, selecting host, runtime, MCP, authorization, background, cron, or subagent tests. |
| Benchmark trajectories, reports, catalog/model groups, or Harbor evaluation integration | **Evals** — `libs/evals/`; product behavior still needs an owning-package regression test. | `libs/evals/tests/` plus the affected package test. |
| Provider-specific sandbox or JavaScript REPL mechanics | **Partner package** — `libs/partners/<provider>/`. QuickJS, for example, ships `langchain-quickjs`, a JavaScript REPL middleware for Deep Agents. | That partner's tests and required integration workflow. |
| Headless GitHub Action inputs, cache, outputs, and orchestration | **Repository Action** — `action.yml`, consuming dcode rather than becoming SDK policy. | Action/workflow scenario plus compatible Code behavior. |

## Release-unit navigation

The release manifest records nine managed release baselines. These are release units—not merely directories—and their distribution/component names determine release PR and tag navigation.

| Path | Distribution and component | Manifest baseline |
| --- | --- | --- |
| `libs/deepagents` | `deepagents` / `deepagents` | `0.7.15` |
| `libs/acp` | `deepagents-acp` / `deepagents-acp` | `0.0.12` |
| `libs/code` | `deepagents-code` / `deepagents-code` | `0.1.72` |
| `libs/talon` | `deepagents-talon` / `deepagents-talon` | `0.0.8` |
| `libs/partners/daytona` | `langchain-daytona` / `langchain-daytona` | `0.0.8` |
| `libs/partners/modal` | `langchain-modal` / `langchain-modal` | `0.0.6` |
| `libs/partners/runloop` | `langchain-runloop` / `langchain-runloop` | `0.0.7` |
| `libs/partners/vercel` | `langchain-vercel-sandbox` / `langchain-vercel-sandbox` | `0.0.2` |
| `libs/partners/quickjs` | `langchain-quickjs` / `langchain-quickjs` | `0.3.7` |

Release Please creates separate draft Python-package release PRs. Each configured unit supplies its changelog and version-bearing files, and package test paths are excluded from release analysis; tags include the component, use `==` as the separator, and have no `v` prefix. `libs/evals` is a package in the monorepo but is not a manifest release unit.

Do not infer compatibility from the manifest alone. The source package metadata is the install contract: Code currently declares `deepagents==0.7.15`, while Talon accepts `deepagents>=0.7.0` and uses `deepagents-code>=0.1.71,<1.0.0`; QuickJS accepts `deepagents>=0.7.0,<0.8.0`. When Code needs an SDK capability, update and validate its exact SDK pin with the consuming release change.

## Entrypoints and lifecycle seams

**Code.** `deepagents-code` and `dcode` both resolve to `deepagents_code:cli_main`, so command parsing and terminal behavior belong to Code. Its package has a client/server architecture and depends on the SDK, ACP, and partner integrations; avoid moving terminal interaction or product-specific runtime policy into the generic SDK.

**ACP.** `deepagents-acp` is the editor-facing adapter. Keep ACP session identity, working-directory/options policy, event translation, and replay in ACP, using a graph factory rather than SDK-global state when session context changes graph construction.

**Talon.** `deepagents-talon` enters at `deepagents_talon.__main__:main` and is explicitly an experimental local runtime host. Keep channel, scheduler, local persistence, operator-mediated authorization, MCP configuration, and local/background delegation there. Consult the [Talon integration guide](../integrations/talon.md) before changing host lifecycle or security-sensitive behavior, and the [sandbox partners guide](../integrations/sandbox-partners.md) for vendor execution boundaries.

### dcode debug logging

dcode debug logging is opt-in and per-thread: configured loggers attach a tagged file handler only after the debug directory and file are secured, and rebinding a thread replaces stale tagged handlers rather than stacking them.

dcode refuses unsafe debug-log destinations: POSIX debug directories must be current-user-owned real directories and files are opened without following symlinks and tightened to owner-only access; on hardening failure it removes debug handlers and warns rather than continuing to log.

### Talon MCP and subagent seams

Talon's MCP provider loads each available server independently, prefixes and metadata-marks its tools, adds management capabilities, rejects tool-name conflicts, and serializes revision-gated refreshes so a configuration change is applied before a later agent turn.

```mermaid
sequenceDiagram
  participant Runtime as Talon runtime
  participant Provider as MCP tool provider
  participant Middleware as MCP middleware
  Runtime->>Provider: Load or refresh tools
  Provider-->>Runtime: Marked tools and status
  Runtime->>Middleware: Invoke marked MCP tool
  Middleware->>Middleware: Normalize arguments and bind call ID
  Middleware-->>Runtime: Result or safe protocol error
```
This is the Talon-owned call path; ordinary tools bypass the MCP middleware.

Talon's MCP middleware only wraps metadata-marked MCP tools, normalizes empty optional string arguments, scopes authorization to the exact tool-call ID, and converts MCP protocol errors to a safe ToolMessage while allowing other exceptions to propagate.

Talon local subagents run as fresh graphs with selected tools, Talon MCP middleware, applicable approval middleware, and no checkpointer; unsupported fork mode is rejected, while malformed async-subagent configuration fails closed rather than silently omitting definitions.

## Safe change sequence

1. Identify the user-visible contract in the table and enter its package rather than treating `libs/` as one Python project.
2. Trace to the component that assembles or owns the lifecycle; preserve the SDK-versus-consumer responsibility boundary.
3. Check the release table and the package's `pyproject.toml` before altering dependency pins or release-facing files.
4. Add the smallest observable test at the owning package, escalating to integration, UI, workflow, or evaluation coverage only when the contract crosses that boundary.
5. Run the package-local target documented by `make help`; use the development guide for aggregate lock or release validation.
