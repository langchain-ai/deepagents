---
type: Codebase Guide
title: Deep Agents monorepo quickstart
description: "Entry point for engineers working on the Deep Agents Python monorepo: package roles, runtime boundaries, validation, and change-sensitive areas."
tags: [deepagents, python, monorepo, engineering]
---
# Deep Agents monorepo

Deep Agents is an opinionated, extensible agent harness built on LangChain and LangGraph. It packages the long-horizon features that a basic tool-calling agent does not provide by default: filesystem and shell backends, planning, context management, skills, persistent memory, human approval, and subagents. The root README is the product-level starting point; this wiki is the maintainer map.

## Start here

- Read [Architecture overview](architecture/overview.md) to trace `create_deep_agent()` from SDK construction into LangChain/LangGraph execution and to understand package boundaries.
- Read [Deep Agents Code](workflows/deep-agents-code.md) before changing the terminal agent, approval routing, auto mode, sandboxes, or MCP loading.
- Read [Evaluation and release](workflows/evaluation-and-release.md) before changing eval harnesses, Harbor workflows, score aggregation, or package-release automation.
- Use [Operations and testing](engineering/operations-and-testing.md) for the package-local edit/test/lint loop, CI controls, integrations, and source map.

## Repository shape

`libs/` is a set of independently versioned Python packages; there is deliberately no root `pyproject.toml`. Work inside the package being changed, where its `pyproject.toml`, `uv.lock`, `Makefile`, and tests define the local contract.

| Area | Role | First source anchor |
| --- | --- | --- |
| `libs/deepagents/` | Core SDK: `create_deep_agent`, middleware, profiles, backends, and subagent machinery. | `libs/deepagents/deepagents/graph.py` |
| `libs/code/` | `dcode` / Deep Agents Code terminal coding agent, with a Textual client and LangGraph server process. | `libs/code/deepagents_code/main.py` |
| `libs/acp/` | Agent Client Protocol adapter for compiled Deep Agent graphs and ACP-capable editors. | `libs/acp/deepagents_acp/server.py` |
| `libs/cli/` | Managed Deep Agents deployment CLI; not the interactive terminal agent. | `libs/cli/deepagents_cli/main.py` |
| `libs/evals/` | Unit/live evaluation tooling, Harbor integrations, datasets, and scorecard documentation. | `libs/evals/README.md` |
| `libs/talon/` | Local runtime host for long-running agents. | `libs/talon/README.md` |
| `libs/partners/` | Sandbox/provider integrations: Daytona, Modal, QuickJS, Runloop, and Vercel. | `libs/partners/` |
| `.github/` | Reusable CI, Harbor evaluations, release, and repository policy automation. | `.github/workflows/ci.yml` |
| `examples/` | Focused patterns and deployable-reference agents rather than a shared product runtime. | `examples/README.md` |

The core SDK in [Architecture overview](architecture/overview.md) supplies the harness that [Deep Agents Code](workflows/deep-agents-code.md) configures for interactive coding. That agent is exercised and compared through [Evaluation and release](workflows/evaluation-and-release.md); package checks and publishing rules live in [Operations and testing](engineering/operations-and-testing.md).

## Fast local loop

Use `uv`; repository guidance explicitly disallows using `pip`, Poetry, or Conda for environment/dependency operations. Install dependencies within the affected package and use its Makefile as the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

The common package targets are `make test` (socket-restricted unit tests), `make integration_test` (network permitted), `make lint`, `make format`, and `make type`. From `libs/`, `make lint` and `make lock-check` fan out across packages. See [Operations and testing](engineering/operations-and-testing.md) for checks by subsystem and CI behavior.

## Product and security boundaries

- The SDK is a harness, not a new graph runtime: LangChain owns the agent loop and LangGraph owns state, checkpointing, streaming, and interrupts.
- Tool authority follows the configured backend and middleware. The root README’s security model is **trust the LLM**: enforce containment at tool/sandbox boundaries rather than treating model intent as a security control.
- Deep Agents Code adds approval UX and policy, but approval is not containment. For untrusted repositories, use a remote sandbox; read [Deep Agents Code](workflows/deep-agents-code.md) before changing approval/MCP behavior.
- Real model/Harbor evaluations have separate credentials, costs, and semantics from unit tests; they are documented in [Evaluation and release](workflows/evaluation-and-release.md).

## Current repository context

The supplied working-tree snapshot had a modified `AGENTS.md` plus untracked OpenWiki workflow/wiki files; source documentation work should not overwrite that unrelated state. Recent history indicates active work in two high-risk areas:

- `feat(code): classifier-backed Auto approval mode` introduced a large Auto-mode and approval-routing surface in `libs/code`; fail-closed behavior and prompt-authority provenance are critical.
- HEAD, `feat(evals): compare branch variants with a neutral harness`, expanded unified Harbor evaluation to compare branch variants without changing the evaluator/harness baseline.

Treat both as change-sensitive seams and follow the linked workflow pages for their exact checks and limits.

## Backlog

- **Talon runtime host** — `libs/talon/README.md`; deferred from this first pass because the core SDK, dcode, and evaluation/release pathways dominate current repository changes.
- **Partner implementations** — `libs/partners/{daytona,modal,quickjs,runloop,vercel}`; catalogued above but not individually documented because each is an integration package with its own boundary and should be expanded when modified.
- **Examples** — `examples/README.md`; examples are intentionally navigated from their own READMEs and were not duplicated into the maintainer wiki.
