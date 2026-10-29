---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package, distinguish published dependency contracts from local checkout wiring, and run the narrowest relevant validation. Use the linked architecture, integration, operations, and testing guides for implementation detail.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-29T08:04:29.358Z
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
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
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
generated: { by: "openwiki/0.4.2", at: "2026-10-29T08:04:29.358Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not merely the product where a symptom appears. `libs/` contains independently versioned packages: the `deepagents` SDK, the `deepagents-code` terminal product (`dcode`), ACP integration, evals, experimental Talon host, and provider integrations. Each package owns its `pyproject.toml`, Makefile, README, source, and tests; work there, since these are the authority for its supported environment and validation.

For package relationships and public boundaries, see [Repository Architecture Overview](./architecture/overview.md). For runtime entrypoints and state owners, see [System Source Map](./architecture/source-map.md).

## Route the change

| Change | Start here | Focused first check | Read next |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [Source map](./architecture/source-map.md), [Testing guide](./testing/testing-guide.md) |
| dcode CLI/TUI, sessions, workspaces, configuration, persistence, plugins, MCP, or headless behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [dcode architecture](./architecture/code-agent.md), [Testing guide](./testing/testing-guide.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/` | `make test TEST_FILE=tests/<nearest test>.py` | [Source map](./architecture/source-map.md); also validate dcode when its ACP specialization changes |
| Talon channels, schedules, approvals, history, host runtime, or managed MCP configuration | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [Testing guide](./testing/testing-guide.md) |
| Daytona, Modal, Runloop, Vercel, QuickJS, or another provider/sandbox boundary | Matching `libs/partners/<provider>/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Sandbox and partner integrations](./integrations/sandbox-partners.md), then the consuming product seam |
| Evaluation harness, Harbor, behavioral evaluation, or results | `libs/evals/` | `make test` | [Testing guide](./testing/testing-guide.md), [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |
| Dependency constraints, lockfiles, CI, or releases | Affected package manifest and lockfile | Package-local checks, then `cd libs && make lock-check` when lockfiles are affected | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |

Run `make help` in the package before assuming targets are uniform. After a focused test, run `make lint`. dcode additionally offers `make check`, its local CI-parity gate: lint, import checks, unit tests, extras/version/lockfile checks, and an advisory stale-SDK-pin report.

## Published contract versus local checkout

Read two manifest layers separately:

- **`[project.dependencies]` is the published contract.** It is the version constraint a released distribution exposes to an installer and is the compatibility boundary to review first.
- **`[tool.uv.sources]` is local checkout wiring.** It replaces a resolved distribution with a sibling path—commonly editable—during local development; it does not publish an unreleased cross-package compatibility promise.

The primary dependency direction is ACP → SDK; dcode → SDK, ACP, and partner integrations; evals → SDK, dcode, and QuickJS; and Talon → SDK and dcode. ACP locally resolves the SDK editably. dcode locally resolves the SDK, ACP, and its partner packages from editable siblings. Evals resolves the SDK and QuickJS editably but maps dcode to its sibling directory without the editable flag; Talon maps both SDK and dcode editably. QuickJS itself resolves the SDK editably.

`deepagents-code` publishes an exact `deepagents==0.7.32` requirement. Raise that pin in the same pull request that first requires newer SDK behavior; do not defer the compatibility change to release time. When changing a manifest or public cross-package behavior, update and validate the contract owner first, then inspect downstream source mappings and lockfiles. From `libs/`, `make lock-check` fans out lock validation; use `make lock` to refresh all locks or `make lock-bump DEP=<package>` to bump one dependency across them.

## Local loop and test boundary

Use `uv` for interpreters, environments, and dependencies and use the package Makefile for ordinary commands. `uv` provisions the applicable Python interpreter, so there is no repository-wide Python version to install or pin. Sync explicitly in the package being changed:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Network-free, deterministic coverage belongs in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. Unaccepted pytest warnings fail the suite. Standard SDK and dcode unit tests disable network sockets while allowing Unix sockets. ACP and Talon tests do likewise; Talon runs its WhatsApp bridge Node tests before its Python tests. Use the [Testing Guide](./testing/testing-guide.md) to choose broader coverage, integration tests, or model-backed evaluation. In evals, `make evals MODEL=<id>` requires a model identifier and is distinct from the network-free unit-test target.

## Releases and package ownership

Release Please manages separate components for the SDK, ACP, dcode, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partners. Each managed package gets a separate draft release PR; merging its checked release PR triggers the workflow that creates the PyPI and GitHub releases. Evals is independently versioned but is not Release Please-managed.

The current Release Please manifest baselines are `0.7.32` for the SDK, `0.0.13` for ACP, `0.2.2` for dcode, `0.1.0` for Talon, and `0.0.9`, `0.0.7`, `0.0.7`, `0.0.2`, and `0.3.11` for Daytona, Modal, Runloop, Vercel, and QuickJS respectively; evals is `0.0.1`. Adding a partner is not only a package change: its guidance requires CI, labeling, release, and related repository wiring as well.

## Continue by question

- **What owns this runtime behavior or package boundary?** [Repository Architecture Overview](./architecture/overview.md)
- **Where is the entrypoint, state/lifecycle owner, or representative test?** [System Source Map](./architecture/source-map.md)
- **How does dcode assemble and operate its agent?** [dcode architecture](./architecture/code-agent.md)
- **How do sandbox and partner integrations fit dcode?** [Sandbox and partner integrations](./integrations/sandbox-partners.md)
- **How do I update dependencies, lockfiles, CI, or releases?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Which focused validation is appropriate?** [Testing guide](./testing/testing-guide.md)
