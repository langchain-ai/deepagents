---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package and runtime boundary, distinguish published dependency contracts from local checkout mappings, and run the narrowest package-supported validation.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-30T08:04:27.172Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
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
generated: { by: "openwiki/0.4.2", at: "2026-10-30T08:04:27.172Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the behavior, state, or published contract—not the product where a symptom is visible. `libs/` contains independently versioned packages: the `deepagents` SDK, `deepagents-code` terminal product (`dcode`), ACP integration, evals, experimental Talon host, and provider integrations. Each package owns its manifest, Makefile, README, source, and tests; its `pyproject.toml` and Makefile define its compatibility and supported commands.

For the package graph and public boundaries, see [Repository Architecture Overview](./architecture/overview.md). For graph, middleware, Talon, and persistence state owners, see [System Source Map](./architecture/source-map.md).

## Route the change

| Change affects | Start in | Focused first validation | Continue with |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [Build a Deep Agent](./workflows/build-a-deep-agent.md), [Testing Guide](./testing/testing-guide.md) |
| dcode CLI/TUI, sessions, workspaces, configuration, persistence, plugins, MCP, or headless behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [Testing Guide](./testing/testing-guide.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/` | `make test TEST_FILE=tests/<nearest test>.py` | [Source map](./architecture/source-map.md); validate dcode too when its ACP specialization changes |
| Talon channels, schedules, approvals, history, host runtime, hooks, or managed MCP configuration | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon Runtime Integration](./integrations/talon.md), [Source map](./architecture/source-map.md), [Testing Guide](./testing/testing-guide.md) |
| Daytona, Modal, Runloop, Vercel, QuickJS, or another provider/sandbox boundary | Matching `libs/partners/<provider>/` | The owning package's `make help`, then its narrowest test target | [Sandbox and partner integrations](./integrations/sandbox-partners.md), then the consuming product seam |
| Evaluation harness, Harbor, behavioral evaluation, or results | `libs/evals/` | `make test` | [Testing Guide](./testing/testing-guide.md), [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |
| Dependency constraints, lockfiles, CI, or releases | Affected package manifest and lockfile | Package-local checks; for changed locks, `cd libs && make lock-check` | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |

Run `make help` in the owning package before assuming target names or test paths are uniform. After a focused test, run that package's `make lint`. dcode also provides `make check`: its local CI-parity gate runs lint, import checks, unit tests, extras/version/lockfile checks, and reports a stale workspace SDK pin without failing solely for that report.

## Published contract versus local checkout

Read the two manifest layers separately:

- **`[project.dependencies]` is the published contract.** It is the version constraint a released distribution gives installers and the compatibility boundary to review first.
- **`[tool.uv.sources]` is local checkout wiring.** It replaces a resolved distribution with a sibling path during local development—often as an editable source. It does not itself publish an unreleased cross-package compatibility promise.

The primary dependency direction is ACP → SDK; dcode → SDK, ACP, and partner integrations; evals → SDK, dcode, and QuickJS; and Talon → SDK and dcode. ACP resolves the SDK from an editable sibling. dcode resolves the SDK, ACP, and partner packages from editable siblings. Evals maps the SDK and QuickJS as editable sources, but maps dcode to its sibling directory without `editable = true`; Talon maps SDK and dcode editably. QuickJS maps the SDK editably.

`deepagents-code` publishes an exact `deepagents==0.7.32` requirement. Raise that pin in the pull request that first requires newer SDK behavior; do not postpone the compatibility change to release time. When changing a manifest or public cross-package behavior, update and validate the contract owner first, then inspect direct consumers' source mappings and lockfiles.

## Local development and tests

Use `uv` for interpreters, environments, and dependencies, and use the owning package's Makefile for normal commands. `uv` provisions the applicable interpreter; do not impose a repository-wide Python pin—honor the package's `requires-python`. Synchronize explicitly in the package being changed:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Network-free, deterministic coverage belongs in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. Unaccepted pytest warnings fail the suite. The standard SDK and dcode unit-test targets disable network sockets while allowing Unix sockets. ACP and Talon standard test targets do the same; Talon first runs WhatsApp bridge Node tests. In evals, `make evals MODEL=<id>` requires a model identifier and is distinct from `make test`.

For all package locks, run `cd libs && make lock-check`; `make lock` refreshes the fan-out set, and `make lock-bump DEP=<package>` requests a dependency bump across it. These commands select the lock interpreter per package, so use them rather than inventing a repository-wide Python version.

## Release and extension boundaries

Release Please manages separate release components for the SDK, ACP, dcode, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partners. It opens a separate draft release PR for each managed package; merging the checked release PR triggers the PyPI and GitHub releases. Evals is independently versioned but is not Release Please-managed. For current release state and the merge procedure, use the [development and release guide](./operations/development.md).

Adding a partner is a repository integration, not only a new package: it requires coordinated CI, labeling, release, and related repository wiring. Follow the partner guidance before treating a provider package as complete.

## Continue by question

- **What owns this package boundary or runtime behavior?** [Repository Architecture Overview](./architecture/overview.md)
- **Where are the entrypoint, lifecycle/state owner, and representative focused tests?** [System Source Map](./architecture/source-map.md)
- **How do I construct and customize a Deep Agents graph?** [Build a Deep Agent](./workflows/build-a-deep-agent.md)
- **How does the experimental long-running host integrate channels, schedules, hooks, and persistence?** [Talon Runtime Integration](./integrations/talon.md)
- **How do sandbox and provider integrations fit the products?** [Sandbox and partner integrations](./integrations/sandbox-partners.md)
- **How do I update dependencies, locks, CI, or releases?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Which focused validation is appropriate?** [Testing Guide](./testing/testing-guide.md)
