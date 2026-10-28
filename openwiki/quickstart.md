---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package, distinguish published dependency contracts from local checkout wiring, and run the narrowest relevant validation. Use the linked architecture, integration, operations, and testing guides for implementation detail.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-28T08:04:17.781Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-10-28T08:04:17.781Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, rather than the product where a symptom is visible. This is a monorepo of independently versioned packages: the `deepagents` SDK, `deepagents-code` terminal product (`dcode`), ACP integration, evals, experimental Talon host, and provider integrations. Work in the owning package: its `pyproject.toml`, `Makefile`, source, and tests define its supported environment and commands.

For the package graph, use [Repository Architecture Overview](./architecture/overview.md). For an entrypoint, lifecycle owner, or representative test, use [System Source Map](./architecture/source-map.md).

## Route the change

| Change class | Own it in / first validation | Read next |
| --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/`; `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [Source map](./architecture/source-map.md), [Testing guide](./testing/testing-guide.md) |
| dcode CLI/TUI, sessions, workspaces, configuration, persistence, plugins, MCP, or headless behavior | `libs/code/`; `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [dcode architecture](./architecture/code-agent.md), [MCP integration](./integrations/mcp.md), [Testing guide](./testing/testing-guide.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/`; `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py`; also validate dcode when its ACP specialization changes | [Source map](./architecture/source-map.md), [MCP integration](./integrations/mcp.md) |
| Talon channels, schedules, approvals, history, host runtime, or managed MCP configuration | `libs/talon/`; `cd libs/talon && make test TEST_FILE=tests/<nearest test>.py` | [Talon runtime integration](./integrations/talon.md), [MCP integration](./integrations/mcp.md), [Testing guide](./testing/testing-guide.md) |
| Daytona, Modal, Runloop, Vercel, QuickJS, or another provider/sandbox boundary | Matching `libs/partners/<provider>/`, then the consuming product seam | [Source map](./architecture/source-map.md), [MCP integration](./integrations/mcp.md) |
| Evaluation harness, Harbor, behavioral evaluation, or results | `libs/evals/`; run `make test` before an evaluation; use `make evals MODEL=<id>` only for model-backed evaluation | [Testing guide](./testing/testing-guide.md), [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |
| Dependency constraints, lockfiles, CI, labels, or releases | Affected package manifest and lockfile; begin with package-local checks | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md), [Architecture overview](./architecture/overview.md) |

Run `make help` inside the package instead of assuming uniform targets. After the focused test, run its `make lint`. For dcode, `make check` is the local CI-parity gate: lint, imports, unit tests, extras/version/lockfile checks, plus an advisory stale-SDK-pin report.

## Current release and dependency boundary

Release Please currently manages separate components at these manifest baselines: `deepagents` `0.7.32`, `deepagents-acp` `0.0.13`, `deepagents-code` `0.2.2`, `deepagents-talon` `0.1.0`, and partner packages Daytona `0.0.9`, Modal `0.0.7`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.11`. Evals is independently versioned (`0.0.1`) but is not Release Please-managed. A managed component receives its own draft release PR; merging a checked release PR publishes PyPI and creates the GitHub release. Follow the operations guide rather than manually racing release automation.

Treat these two manifest layers differently:

- **`[project.dependencies]` is published contract.** It constrains what a released wheel installs and is the compatibility boundary to review first.
- **`[tool.uv.sources]` is checkout wiring.** It substitutes a sibling path, often editable, only while developing locally; it does not publish an unreleased cross-package promise.

The main consumer direction is ACP → SDK; dcode → SDK, ACP, and partner integrations; evals → SDK, dcode, and QuickJS; and Talon → SDK and dcode. dcode ships an exact `deepagents==0.7.32` requirement: raise that pin in the same pull request that first needs newer SDK functionality. Its local resolution maps the SDK, ACP, and partner packages to editable siblings. Evals maps the SDK and QuickJS editably but uses dcode as a sibling directory source; Talon maps both SDK and dcode editably.

When changing a manifest or public cross-package behavior, update and validate the contract owner first, then inspect every consumer source mapping and lockfile. A dcode manifest change can require its own, evals', and Talon's resolution refresh. From `libs/`, use `make lock-check` to verify the affected lock closure (and `make lock` or `make lock-bump DEP=<package>` when updating it). See [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) for the fan-out and release workflow.

## Local loop and test boundary

Use `uv` for interpreters, environments, and dependencies; use each package's Makefile as the command authority. `uv` provisions the applicable interpreter, so there is no global Python version to pin. Install dependencies explicitly in the package being changed:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The SDK and dcode unit targets disable network sockets while allowing Unix sockets. ACP and Talon do the same; Talon first runs the WhatsApp bridge Node tests. Escalate focused test selection and integration/evaluation decisions through the [Testing Guide](./testing/testing-guide.md).

## Continue by question

- **What owns this runtime behavior or package boundary?** [Repository Architecture Overview](./architecture/overview.md)
- **Where is the entrypoint, state/lifecycle owner, or representative test?** [System Source Map](./architecture/source-map.md)
- **How does dcode assemble and operate its agent?** [dcode architecture](./architecture/code-agent.md)
- **How should dcode or Talon integrate MCP?** [MCP integration](./integrations/mcp.md) and [Talon runtime integration](./integrations/talon.md)
- **How do I update dependencies, lockfiles, CI, or releases?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Which focused validation is appropriate?** [Testing guide](./testing/testing-guide.md)
