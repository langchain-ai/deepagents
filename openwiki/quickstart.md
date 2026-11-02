---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Choose the Deep Agents package that owns a change, run its narrowest meaningful validation, and continue to the runtime, evaluation, CI, dependency, or release guide that owns the detailed procedure.
tags: [deepagents, maintenance, routing, testing, ci]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-02T08:04:35.090Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
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
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-224af38a099019bb2a44f0a7
    resource: repo://libs/partners/quickjs/Makefile
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-11-02T08:04:35.090Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the behavior or published contract—not necessarily the product in which the symptom appeared. `libs/` contains independently versioned packages. Each package owns its manifest, Makefile, source, tests, and README; run `make help` in that directory rather than assuming a target is shared across packages.

## Route the change

| If the change concerns… | Owning package | First focused validation | Continue with |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| The `dcode` CLI/TUI, headless sessions, workspace, configuration, persistence, plugins, MCP, or terminal behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` | [Development operations](./operations/development.md), [testing guide](./testing/testing-guide.md) |
| An editor-facing Agent Client Protocol server or protocol behavior | `libs/acp/` | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) and then Code if its ACP specialization changes |
| Long-running host behavior—channels, schedules, approvals, history, managed MCP, or `/effort` | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon integration](./integrations/talon.md) |
| REPL lifecycle, prompt/tool bindings, snapshots, persistence, or subagent guidance | `libs/partners/quickjs/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Sandbox and partner integrations](./integrations/sandbox-partners.md), then the Code or evals consumer seam |
| A Daytona, Modal, Runloop, Vercel, or other provider/sandbox integration | `libs/partners/<provider>/` | Its `make help`, then its narrowest test target | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Deterministic eval-harness behavior, real-model behavioral evaluation, Harbor, or result reporting | `libs/evals/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Run Evals](./workflows/run-evals.md) |
| Root composite GitHub Action behavior | Root `action.yml` and `.github/` workflow coverage | The action/workflow contract test named by the integration guide | [GitHub Action integration](./integrations/github-action.md) |
| CI selection, workflow security, labels, release-note automation, or a CI gate | `.github/workflows/` and its helper/test domain | Workflow contract tests | [CI, labels, and workflow security](./operations/ci-labeling.md) |
| Published requirements, sibling sources, a package lock, or a release | Affected package `pyproject.toml` and `uv.lock` | Package validation; use the documented lock closure/checker | [Development, dependencies, lockfiles, and releases](./operations/development.md) |

QuickJS is a provider/REPL package even when the visible behavior is in Code or evals. Talon is a host that consumes the SDK and Code. Test the owner first, then validate the consumer seam when a changed dependency crosses it.

## Work package-first

Use `uv` for interpreters, environments, and dependencies, and the owning package's Makefile as the command source of truth. `uv` provisions the applicable interpreter automatically; do not introduce a repository-wide Python pin. Synchronize and validate from the package directory:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The standard SDK, Code, ACP, Talon, and QuickJS test targets disable network sockets while permitting Unix sockets. Talon's target also runs WhatsApp bridge Node tests before its Python tests, so even a narrowed Python target exercises that bridge prerequisite.

The focused interface differs by package. SDK and Code accept `TEST_FILE`; Code's `make check` adds its local CI-parity gates, while QuickJS provides `make update_snapshots` for intentional prompt-smoke snapshot updates. Read the package Makefile before choosing a target.

## Dependencies and lockfiles

Treat the three manifest layers as different contracts:

- **`[project.dependencies]` and `requires-python`** are the publishable downstream contract.
- **`[tool.uv.sources]`** maps dependencies to sibling checkouts during monorepo development; it does not alter published metadata.
- **`uv.lock`** is the owning project's generated resolution, not a repository-wide environment.

The main local dependency direction is SDK → ACP, Code, evals, Talon, and QuickJS; Code also consumes ACP and partner integrations, and evals consumes Code and QuickJS. Local editable sources can mask an inadequate published requirement, so validate the installed compatibility boundary when changing a package interface. In particular, raise Code's exact SDK pin in the same PR that first uses newer SDK behavior.

For a manifest change, regenerate the owner lock and its reverse local-source consumers—not merely the file that was edited. The dependency guide documents the closure helper and targeted Python policy. `make -C libs lock-check` is useful additional fan-out validation; it discovers library packages and example projects, but package locks remain project-specific.

## What CI will validate

On pull requests, the main CI workflow selects changed package jobs; an SDK change also selects dependent package tests. A push to `main` runs the package matrix. CI and action changes are included in the relevant package filters, so workflow changes receive package coverage rather than being treated as documentation-only edits. Use the CI guide for the security and exception semantics of the final gate—especially the narrow Talon failure waiver and curated-release-notes inheritance—rather than encoding those exceptions into package changes.

For a QuickJS prompt-snapshot compatibility seam, an SDK-only pull request additionally runs the QuickJS system-prompt smoke test when the QuickJS package itself is unchanged. This protects snapshots against SDK wording drift without running the full QuickJS suite redundantly.

## Choose the operational guide

- **How the runtime is assembled, executes, and persists state:** [Architecture overview](./architecture/overview.md)
- **Where a runtime concern is implemented:** [Architecture source map](./architecture/source-map.md)
- **Focused, integration, regression, and workflow validation:** [Testing guide](./testing/testing-guide.md)
- **Real-model evals, Harbor, artifacts, and unified evaluation:** [Run Evals](./workflows/run-evals.md)
- **CI job selection, labels, trusted workflow boundaries, and gates:** [CI, labels, and workflow security](./operations/ci-labeling.md)
- **Setup, dependency contracts, lock closure, and releases:** [Development, dependencies, lockfiles, and releases](./operations/development.md)
- **Release Please:** each managed component receives a separate draft release PR; after its reviewed PR is merged, the release workflow publishes the GitHub and PyPI releases. Start with the [development operations guide](./operations/development.md) for the complete release path.

Adding a partner is a repository-wide extension, not just a new `libs/partners/` directory: package CI selection, labels, release configuration, secret inventory, integration coverage, and related workflow wiring must be coordinated. Follow the partner onboarding guidance before opening that work.
