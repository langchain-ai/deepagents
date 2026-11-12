---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to the independently versioned package that owns its contract or state, then select focused validation and lockfile work. Links lead to the architecture, operations, workflow, integration, and testing guidance for each domain.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-11T08:04:08.950Z
sources:
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-11-11T08:04:08.950Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime state, rather than the package where a symptom happens to appear. `libs/` is an independently versioned monorepo: every package owns its `pyproject.toml`, `Makefile`, README, source, tests, and `uv.lock`. The release manifest tracks separate release baselines for the SDK, ACP, dcode, Talon, and partner distributions.

Use `uv` for interpreters, environments, and dependency synchronization, and use `make help` plus the package Makefile as the command contract. `uv` provisions the package's required interpreter, so maintainers do not need a global Python version. First synchronize dependencies explicitly, then make, test, lint, and lock changes within the owning package.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

For package relationships and published versus checkout dependency boundaries, read the [architecture overview](./architecture/overview.md). Use the [source map](./architecture/source-map.md) when a task spans runtime components, and [development operations](./operations/development.md) for release and lockfile procedure.

## Route the task

| If the change concerns… | Start in | Focused validation | Then read |
| --- | --- | --- | --- |
| SDK graph construction, middleware, backends, profiles, filesystem tools, or subagents | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| dcode CLI or TUI behavior, local server lifecycle, workspace policy, model/configuration resolution, headless behavior, costs, or traces | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Run a dcode session](./workflows/run-dcode-session.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| ACP protocol translation or dcode ACP launch mode | `libs/acp/`, then the dcode seam if it selects or consumes the protocol | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md), [Run a dcode session](./workflows/run-dcode-session.md) |
| Talon channels, host lifecycle, MCP, history, sandbox selection, scheduling, or background delegation | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon runtime integration](./integrations/talon.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| Evaluation definitions, Harbor execution, datasets, or reports | `libs/evals/` | `make help`, then the closest package target | [Development operations](./operations/development.md), [testing guide](./testing/testing-guide.md) |
| Daytona, Modal, QuickJS, Runloop, or Vercel sandbox behavior | `libs/partners/<provider>/` | `make help`, then the owning package target | [Sandbox and partner integrations](./integrations/sandbox-partners.md), [testing guide](./testing/testing-guide.md) |
| A dependency bound, Python support, extra, or local source mapping | The changed package's `pyproject.toml`, then local consumers | Refresh the lock closure; test a direct consumer | [Architecture overview](./architecture/overview.md), [development operations](./operations/development.md) |

The usual test layout separates deterministic, network-free tests in `tests/unit_tests/` from networked contracts in `tests/integration_tests/`; unexpected pytest warnings fail. Follow the nearest existing test and the owning Makefile when a package has a different local layout. Run the narrowest owning suite first, then its `make lint`.

## Package and compatibility boundaries

The core `deepagents` SDK supports Python `>=3.11,<4.0`. dcode is the `deepagents-code` distribution (also exposed by the `dcode` command), requires Python `>=3.14,<3.15`, and pins `deepagents==0.7.35`. Treat an SDK behavior needed by dcode as a cross-package compatibility change: update the published pin and lock resolution as appropriate, then validate dcode rather than testing only the SDK.

`deepagents-talon` is an experimental Python `>=3.14,<3.15` local host for long-running channels and schedules. Its published base requirements include `deepagents>=0.7.35`, `deepagents-code>=0.2.6,<1.0.0`, channel and MCP libraries, LangChain/LangGraph, and SQLite checkpointing. Talon's `deepagents` and `deepagents-code` source mappings are editable checkout wiring; they do not replace its published requirements. For its host/runtime ownership and operational authority boundaries, use the [Talon runtime integration](./integrations/talon.md).

Talon capabilities beyond the base host are extras: history embeddings, MongoDB or PostgreSQL persistence, OpenShell, and media. Choose one of the mutually exclusive `media` and `media-cpu` extras; on Linux, `media-cpu` selects Talon's explicit PyTorch CPU index. Extras install implementation dependencies only—they do not configure credentials, remote resources, or authorization.

ACP and Talon `make test` disable network sockets. Talon runs the WhatsApp bridge Node tests before Python pytest, permits Unix sockets, applies a 10-second timeout, and collects coverage; its lint target runs Ruff checking, format verification, and `ty`.

## Dependency and lockfile workflow

Keep these layers distinct:

1. **Published metadata:** `[project]` dependencies, optional extras, and `requires-python` are the downstream installation contract.
2. **Checkout wiring:** `[tool.uv.sources]` can substitute editable sibling source for monorepo development; it does not strengthen the published dependency constraint.
3. **Generated resolution:** each package's `uv.lock` is the concrete development and CI resolution. Regenerate it with `uv`; do not edit it by hand.

A manifest edit can invalidate its owner plus transitive reverse consumers that declare local uv path sources. From the repository root, refresh from the changed manifest and commit every lockfile printed by the refresher:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/talon/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/talon/pyproject.toml
```

The refresher computes that reverse-consumer closure and locks every selected package with the checker's package policy interpreter. The targeted checker instead selects package directories touched by the supplied paths; without paths it checks every discovered package and prints an executable non-check `uv lock` repair command if a lock is stale. The CI lockfile workflow uses this no-argument route.

Do not use `make -C libs lock-check` as a CI-equivalent check for dcode, evals, or Talon: its fan-out mapping uses Python 3.14 only for ACP and 3.12 for every other package, whereas the checker uses Python 3.14 for ACP, dcode, evals, and Talon.

## Finish a focused change

1. Select the package that owns the state or published contract; run `uv sync` there and inspect `make help` when its targets are unfamiliar.
2. Change the owning layer. For dependency work, review both the published constraint and any local source substitution.
3. Run the nearest behavioral test and the package lint target. Test dcode after an SDK pin or compatibility change; test Talon after an SDK or dcode local-source change because it consumes both.
4. For a manifest edit, refresh the reverse-consumer lock closure, commit every printed `uv.lock`, and run the targeted checker for affected paths. Use its no-argument form for a repository-wide final check.
5. Continue with the linked [architecture](./architecture/overview.md), [operations](./operations/development.md), [workflow](./workflows/run-dcode-session.md), [integrations](./integrations/talon.md), and [testing](./testing/testing-guide.md) pages for behavior-specific design and validation.
