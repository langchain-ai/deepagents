---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning package, compatibility contract, and focused validation command. Summarizes current package versions, Python ranges, dependency boundaries, and links to detailed guides.
tags: [deepagents, maintenance, routing, testing, dependencies]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-05T08:04:42.777Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
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
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
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
generated: { by: "openwiki/0.4.2", at: "2026-11-05T08:04:42.777Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime state—not necessarily the product where a symptom appears. This is a monorepo of independently versioned packages under `libs/`; each package owns its `pyproject.toml`, `Makefile`, source, tests, and README. Run `make help` in that directory before assuming a command, test layout, or interpreter applies everywhere.

## Route the task

| Change concerns… | Start in | First focused validation | Then read |
| --- | --- | --- | --- |
| Agent assembly, middleware, profiles, backends, filesystem/tools, permissions, subagents, or graph state | `libs/deepagents/` (SDK) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), and [Build a Deep Agent](./workflows/build-a-deep-agent.md) |
| Terminal CLI/TUI, agent server, workspace/configuration, MCP, persistence, skills, or self-management | `libs/code/` (`deepagents-code` / `dcode`) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` | [Testing guide](./testing/testing-guide.md) |
| Editor-facing Agent Client Protocol hosting or graph/session translation | `libs/acp/` | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Channels, approvals, schedules, sandbox selection, progress, MCP, or host observability | `libs/talon/` | `make test TEST_FILE=tests/<nearest host/runtime test>.py` | [Talon Runtime Integration](./integrations/talon.md) |
| Talon checkpoints, archives, vector search, reindexing, deletion, or a storage backend | `libs/talon/` storage seam | `make test TEST_FILE=tests/unit_tests/<nearest archive/history test>.py` | [Talon Conversation History and Storage](./concepts/talon-history-storage.md) |
| Real-model trajectories, Harbor, datasets, or evaluation reports | `libs/evals/` | `make help`, then the nearest evaluation seam | [Architecture overview](./architecture/overview.md) |
| QuickJS prompt snapshots or a sandbox/provider implementation | `libs/partners/quickjs/` or `libs/partners/<provider>/` | QuickJS: `make test TEST_FILE=tests/unit_tests/<nearest test>.py`; otherwise package `make help` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Dependency metadata, sibling sources, lockfiles, CI selection, or releases | Changed package manifest, then reverse local consumers | Package checks and lock refresh/check | [Development](./operations/development.md) and [Architecture overview](./architecture/overview.md) |

The dependency direction is a useful second check: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; evals consumes the SDK, dcode, and QuickJS; and Talon consumes the SDK and dcode through sibling sources. Test the owner first, then its direct consumer seam when a changed interface crosses one of those boundaries.

## Current package contracts

| Distribution (source) | Version | Python contract | Maintainer boundary |
| --- | --- | --- | --- |
| `deepagents` (`libs/deepagents/`) | `0.7.32` | `>=3.11,<4.0` | Reusable agent-harness SDK. |
| `deepagents-acp` (`libs/acp/`) | `0.0.13` | `>=3.11,<4.0` | Agent Client Protocol bridge. |
| `deepagents-code` / `dcode` (`libs/code/`) | `0.2.4` | `>=3.14,<3.15` | Terminal coding-agent product and server. |
| `deepagents-talon` (`libs/talon/`) | `0.1.0` | `>=3.14,<3.15` | Experimental local host for long-running channels and schedules. |
| `deepagents-evals` (`libs/evals/`) | `0.0.1` | `>=3.14,<3.15` | Evaluation suite and Harbor integration. |

The SDK is the reusable harness; dcode owns terminal product behavior, ACP owns protocol translation, and Talon owns host lifecycle, channels, and schedules. Provider and sandbox packages remain integration boundaries rather than SDK base dependencies. For the exact published relationship graph and operational implications, see the [architecture overview](./architecture/overview.md).

## Work package-first

Use `uv` for interpreters, environments, and dependencies. It provisions the applicable interpreter automatically, so do not add a repository-wide Python pin; each package declares its own `requires-python`. The owning Makefile is the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. Standard SDK and dcode unit targets disable network sockets while allowing Unix sockets. ACP and Talon do likewise; Talon first runs its WhatsApp bridge Node tests. Use the narrowest relevant test before expanding validation.

SDK and dcode accept `TEST_FILE`. Dcode's `make check` adds local CI-parity checks, including metadata, lock, and SDK-pin checks. QuickJS defaults `make test` to socket-disabled unit tests, accepts `TEST_FILE`, and provides `make update_snapshots` for reviewed prompt-smoke updates. Use Talon's Makefile target rather than a raw Python run when a bridge change is involved.

## Dependencies, local sources, and locks

Keep these layers separate:

- **Published metadata** — `[project.dependencies]`, extras, and `requires-python` define what an installer receives.
- **Checkout wiring** — `[tool.uv.sources]` maps sibling source directories for monorepo development; it is not wheel metadata.
- **Resolution** — each package's `uv.lock` is generated for that package, not for one repository-wide environment.

An editable sibling can make a local test pass while the published requirement is too weak. `deepagents-code` publishes the exact `deepagents==0.7.32` pin; raise it in the same pull request that first uses newer SDK behavior. Its normal install also requires ACP `>=0.0.13,<1.0.0` and QuickJS `>=0.3.11,<0.4.0`. Talon publishes `deepagents>=0.7.32` and `deepagents-code>=0.2.4,<1.0.0`; history embedding adapters plus MongoDB/PostgreSQL support are optional extras, not base-host requirements.

After a manifest change, refresh the owner and its reverse local-source consumers, then validate broadly:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
make -C libs lock-check
```

Do not hand-edit lockfiles. The [Development guide](./operations/development.md) explains lockfile closure, the automated lower-bound updater, and the separate ownership of dcode's exact SDK pin.

## CI and release orientation

On pull requests, CI selects lint and test jobs using package path filters. SDK changes select dependent packages, and relevant workflow or composite-action changes are included in those filters. For an SDK-only pull request that does not change QuickJS, CI runs QuickJS's system-prompt smoke test against the editable SDK; pushes and QuickJS-changing pull requests run the full QuickJS suite.

Release Please creates a separate draft release PR for each managed package. Merging its checked release PR triggers the release workflow that publishes PyPI and creates the GitHub release. Adding a partner package therefore needs coordinated package, CI, labeling, release, and repository wiring—not only a new distribution.

## Detailed guides

- **Package graph, Python/dependency contracts, and lock closure:** [Repository Architecture Overview](./architecture/overview.md)
- **Find the implementation and state owner:** [System Source Map](./architecture/source-map.md)
- **SDK graph customization:** [Build a Deep Agent](./workflows/build-a-deep-agent.md)
- **Talon host, channel, sandbox, MCP, and observability behavior:** [Talon Runtime Integration](./integrations/talon.md)
- **Talon checkpoint/archive/vector lifecycle and storage safety:** [Talon Conversation History and Storage](./concepts/talon-history-storage.md)
- **Focused regression and integration coverage:** [Testing Guide](./testing/testing-guide.md)
- **Setup, dependency floors, lockfiles, and managed publication:** [Development](./operations/development.md)
