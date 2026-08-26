---
type: orientation-and-navigation
title: Quickstart & Wiki Map
description: Orientation to the Deep Agents monorepo layout under libs/ and a task-routing map that sends common jobs (build an agent, run dcode, benchmark, host, sandbox) to the right wiki section.
tags: [quickstart, monorepo, navigation, deepagents, dcode, routing]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T21:35:57.774Z
sources:
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: {by: "openwiki/0.4.0", at: "2026-08-26T21:35:57.774Z"}
---

# Quickstart & Wiki Map

Deep Agents is an opinionated, batteries-included agent harness: `create_deep_agent()`
assembles a default middleware stack, pluggable backends, subagents, skills, memory,
and profiles on top of LangChain's `create_agent`, which in turn runs on the LangGraph
runtime. This page orients a coding agent to the repository and routes each common task
to the wiki section that answers it. It is navigation-first; deep content lives on the
pages linked below.

## The stack in one glance

Three layers own different behavior, and most questions get easier once you know which
layer to look at:

- **LangGraph** — the runtime: state, checkpoints, streaming, interrupts.
- **LangChain `create_agent`** — the agent abstraction: model + tools + middleware → the agent loop.
- **Deep Agents** — the opinionated harness on top of `create_agent`: default middleware, backends, and profiles.

See [Architecture Overview](/openwiki/architecture/overview.md) for how the three layers relate.

## Monorepo layout

The repository is a monorepo of **independently versioned packages under `libs/`**.
There is **no root `pyproject.toml`**; each package carries its own
`pyproject.toml`, `Makefile`, and `README.md`, and you work inside the package you are
changing. Local package dependencies are editable, so changes in one package are visible
to sibling packages during development.

| Package | Path | Responsibility |
| --- | --- | --- |
| `deepagents` | `libs/deepagents/` | Core SDK: `create_deep_agent`, middleware, and pluggable backends for building your own agents. |
| `code` (`deepagents-code`, run via `dcode`) | `libs/code/` | Prebuilt terminal coding agent — interactive Textual TUI, remote sandboxes, memory, skills, and headless mode. |
| `acp` | `libs/acp/` | Agent Client Protocol integration for running a Deep Agent (or `dcode`) inside editors like Zed. |
| `evals` | `libs/evals/` | Evaluation suite and Harbor integration for benchmarking agent behavior. |
| `talon` | `libs/talon/` | Experimental local runtime host for long-running agents (channel adapters, cron schedulers). |
| `partners` | `libs/partners/` | Provider/sandbox integrations: `daytona/`, `modal/`, `runloop/`, `vercel/`, `quickjs/`. |
| `harbor` | `libs/harbor/` | Benchmark harness used with the evals suite. |

For a directory-to-responsibility map, see the [Source Map](/openwiki/architecture/source-map.md).

### Package dependency direction

The packages depend downward toward the SDK; the SDK does not depend on the packages
built on top of it:

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
    subgraph consumers["Harness consumers"]
        code["code / dcode"]
        acp["acp"]
        talon["talon"]
        evals["evals"]
    end
    partners["partners<br/>(daytona, modal, runloop, vercel, quickjs)"]
    harbor["harbor"]
    sdk["deepagents (SDK)"]

    code --> sdk
    acp --> sdk
    talon --> sdk
    evals --> sdk
    code --> partners
    partners --> sdk
    evals --> harbor
```

## Task routing

Start here, then follow the link into the relevant hierarchy folder.

| I want to… | Go to |
| --- | --- |
| **Build my own agent** with the SDK | [Build a Deep Agent](/openwiki/workflows/build-a-deep-agent.md) → [SDK: Construction & Execution](/openwiki/architecture/sdk-construction-execution.md), [Middleware Stack](/openwiki/architecture/middleware-stack.md) |
| **Use or change the terminal coding agent** (`dcode`) | [Run & Extend a dcode Session](/openwiki/workflows/run-dcode-session.md) → [Deep Agents Code Architecture](/openwiki/architecture/code-agent.md) |
| **Run `dcode` inside an editor** | [ACP Integration](/openwiki/integrations/acp.md) |
| **Benchmark / evaluate agents** | [Evaluate & Benchmark Agents](/openwiki/workflows/run-evals.md) |
| **Host a long-running agent** | [Talon: Local Runtime Host](/openwiki/integrations/talon.md) |
| **Add a sandbox / provider integration** | [Sandbox & Partner Integrations](/openwiki/integrations/sandbox-partners.md), [MCP Integration](/openwiki/integrations/mcp.md) |
| **Set up, build, test, or release** | [Development & Build Operations](/openwiki/operations/development.md), [Testing Guide](/openwiki/testing/testing-guide.md) |

## Wiki hierarchy

Browse by domain rather than by individual page:

- **[/openwiki/architecture/](/openwiki/architecture/overview.md)** — the three-layer stack, SDK construction/execution, the middleware stack, the `dcode` client/server split, and the source map.
- **[/openwiki/concepts/](/openwiki/concepts/backends.md)** — backends, state & persistence, context management, permissions & HITL, profiles & models, subagents & skills, tools, and the middleware catalog.
- **[/openwiki/workflows/](/openwiki/workflows/build-a-deep-agent.md)** — end-to-end guides for building an agent, running a `dcode` session, and running evals.
- **[/openwiki/operations/](/openwiki/operations/development.md)** — development/build ops, security & threat model, and `dcode` cost/session tracking.
- **[/openwiki/integrations/](/openwiki/integrations/acp.md)** — ACP, MCP, sandbox/partner providers, and Talon.
- **[/openwiki/testing/](/openwiki/testing/testing-guide.md)** — unit/integration/benchmark layout and how to run and add tests.

## Getting started fast

- **Try the coding agent:** `curl -LsSf https://langch.in/dcode | bash` then `dcode`.
- **Build your own:** `uv add deepagents`, then call `create_deep_agent(model=..., tools=..., system_prompt=...)`.
- **Develop in the repo:** work inside a package (e.g. `cd libs/deepagents`), run `uv sync --all-groups`, and use that package's `make` targets. `uv` provisions the interpreter; there is no global Python version to pin.

## Runtime behavior

The static docs and this wiki describe how the system is *built*. Production trace
findings — actual run shape, hotspots, failures, and any code-vs-production
divergence — are consolidated on
[Runtime Behavior & Findings (LangSmith)](/openwiki/runtime-behavior.md). Consult it
whenever you need to complement the static picture with observed behavior.
