---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning package and focused operational or persistence guide. Use the package Makefile and narrow tests before expanding validation across dependency boundaries.
tags: [deepagents, maintenance, routing, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-04T08:04:03.075Z
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
generated: { by: "openwiki/0.4.2", at: "2026-11-04T08:04:03.075Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or state—not necessarily the product where a symptom is visible. The repository contains independently versioned packages under `libs/`; each package owns its `pyproject.toml`, `Makefile`, source, tests, and README. Run `make help` in that package rather than assuming commands, test layout, or Python versions are shared.

## Route the task

| Change concerns… | Start in | First focused validation | Continue in |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, or graph state | `libs/deepagents/` (SDK) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md) |
| Terminal CLI/TUI, server, workspace, configuration, MCP, dcode persistence, or self-management | `libs/code/` (`deepagents-code` / `dcode`) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` | [Run a dcode session](./workflows/run-dcode-session.md) and [Testing guide](./testing/testing-guide.md) |
| dcode install, uninstall, update, prerelease, Python migration, or editable-install behavior | `libs/code/` lifecycle tests | `make test TEST_FILE=tests/unit_tests/<nearest update/install test>.py` | **[dcode Installation and Update Lifecycle](./operations/dcode-install-and-update.md)** |
| Editor-facing Agent Client Protocol hosting or translation | `libs/acp/` | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Talon channel turns, approvals, schedules, MCP, sandboxing, progress, or observability | `libs/talon/` host/runtime | `make test TEST_FILE=tests/<nearest host/runtime test>.py` | **[Talon Runtime Integration](./integrations/talon.md)** and [Talon scheduling](./concepts/talon-scheduling.md) |
| Talon checkpoints, archive transcripts, vector search, reindexing, deletion, or a storage backend | `libs/talon/` storage seam | `make test TEST_FILE=tests/unit_tests/<nearest archive/history test>.py` | **[Talon Conversation History and Storage](./concepts/talon-history-storage.md)** |
| Evaluation harnesses, real-model trajectories, Harbor, or reports | `libs/evals/` | Its `make help`, then the nearest test/evaluation seam | [Architecture overview](./architecture/overview.md) |
| QuickJS prompt snapshots or a sandbox/provider implementation | `libs/partners/quickjs/` or `libs/partners/<provider>/` | QuickJS: `make test TEST_FILE=tests/unit_tests/<nearest test>.py`; otherwise package `make help` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Dependency metadata, local source wiring, lockfiles, or release configuration | Changed package manifest, then reverse local consumers | Package validation, lock refresh/check | [Architecture overview](./architecture/overview.md) and [Development](./operations/development.md) |

The dependency direction is a useful second check: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; evals consumes the SDK, dcode, and QuickJS; and Talon consumes the SDK and dcode through sibling sources. Test the owner first, then the consumer seam when an interface crosses one of those boundaries.

For Talon specifically, do not conflate persistence roles: checkpoint state, durable conversation archive metadata, and optional vector search are separate resources. A storage change belongs in the storage guide above; a channel-to-graph, delivery, sandbox, or host-lifecycle change belongs in the runtime guide. Talon’s base distribution depends on `deepagents>=0.7.32` and `deepagents-code>=0.2.4,<1.0.0`; history embedding adapters and remote storage support are optional extras, so an adapter or backend change can alter installation requirements as well as runtime behavior.

## Work package-first

Use `uv` for interpreters, environments, and dependencies. It provisions the applicable interpreter automatically, so do not add a repository-wide Python pin; each package declares its supported range. The owning Makefile is the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The standard SDK and dcode targets disable network sockets while permitting Unix sockets. ACP and Talon do likewise; Talon runs its WhatsApp bridge Node tests before Python tests.

Use the narrowest test first. SDK and dcode accept `TEST_FILE`; dcode’s `make check` adds its local CI-parity gates. QuickJS defaults `make test` to socket-disabled unit tests, accepts `TEST_FILE`, and has `make update_snapshots` for reviewed prompt-smoke changes. See the [Testing guide](./testing/testing-guide.md) for the behavior seams and exact dcode/Talon test paths; do not replace Talon’s Makefile target with a raw Python run when the bridge changes.

## Dependencies and locks

Keep three layers separate:

- **Published metadata** — `[project.dependencies]`, extras, and `requires-python` are the installer-visible contract.
- **Checkout wiring** — `[tool.uv.sources]` maps sibling sources for monorepo development; it does not change wheel metadata.
- **Resolution** — each package’s `uv.lock` is generated for that package, not a repository-wide environment.

An editable source can hide an inadequate published requirement. `deepagents-code` requires `deepagents==0.7.32`; raise that exact pin in the same pull request that first depends on newer SDK behavior. After a manifest change, refresh the owner and reverse local-source consumers, then use `make -C libs lock-check` as fan-out validation. The [architecture overview](./architecture/overview.md) explains Talon’s direct dcode range, optional history/storage extras, and local source mappings.

## CI and release orientation

On pull requests, CI selects lint and test jobs from package path filters. SDK changes select dependent package jobs, and relevant workflow or composite-action changes are included in those filters. An SDK-only pull request that does not change QuickJS also runs QuickJS system-prompt smoke tests against the editable SDK; pushes and QuickJS-changing pull requests run the full QuickJS suite.

Release Please creates a separate draft release PR for each managed package; merging it triggers the release workflow that creates the PyPI and GitHub releases. Adding a partner therefore requires coordinated CI, labeling, release, and related repository wiring—not just a new package.

## Detailed guides

- **Package graph, dependency contracts, and lock closure:** [Repository Architecture Overview](./architecture/overview.md)
- **Find the responsible implementation state owner:** [System Source Map](./architecture/source-map.md)
- **Talon checkpoint/archive/vector lifecycle and storage safety:** [Talon Conversation History and Storage](./concepts/talon-history-storage.md)
- **Talon host, channel, sandbox, MCP, progress, and observability behavior:** [Talon Runtime Integration](./integrations/talon.md)
- **dcode update/install recovery and editable-install safeguards:** [dcode Installation and Update Lifecycle](./operations/dcode-install-and-update.md)
- **Focused regression, Node bridge, and integration coverage:** [Testing Guide](./testing/testing-guide.md)
- **Setup, lockfiles, and managed publication:** [Development](./operations/development.md)
