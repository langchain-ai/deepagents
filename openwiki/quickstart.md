---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning package, compatibility contract, and focused validation command. Summarizes package versions, dependency and lockfile boundaries, CI selection, and the detailed architecture, workflow, integration, operations, and testing guides.
tags: [deepagents, maintenance, routing, testing, dependencies]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-06T08:04:19.814Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
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
generated: { by: "openwiki/0.4.2", at: "2026-11-06T08:04:19.814Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime state, not necessarily the product where a symptom appears. `libs/` is an independently versioned-package monorepo: each package owns its `pyproject.toml`, `Makefile`, source, tests, and README. Run `make help` in that package before assuming a command or test layout applies everywhere.

## Route the task

| Change concerns… | Start in | First focused validation | Continue with |
| --- | --- | --- | --- |
| Agent construction, middleware, profiles, backends, filesystem tools, permissions, subagents, or graph state | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [Build a Deep Agent](./workflows/build-a-deep-agent.md) |
| Terminal CLI/TUI, agent server, workspace/configuration, MCP, persistence, skills, or dcode product behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` | [Testing guide](./testing/testing-guide.md), [GitHub Action integration](./integrations/github-action.md) |
| Editor-facing Agent Client Protocol session translation or dcode's ACP mode | `libs/acp/` or the dcode ACP seam | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Long-running channels, approvals, schedules, sandbox selection, progress, MCP, or host observability | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon integration](./integrations/talon.md), [source map](./architecture/source-map.md) |
| Real-model trajectories, Harbor, datasets, or evaluation reports | `libs/evals/` | `make help`, then the nearest evaluation target | [Architecture overview](./architecture/overview.md), [Testing guide](./testing/testing-guide.md) |
| QuickJS prompt snapshots or a sandbox/provider implementation | `libs/partners/quickjs/` or `libs/partners/<provider>/` | QuickJS: `make test TEST_FILE=tests/unit_tests/<nearest test>.py`; otherwise package `make help` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Dependency metadata, editable sibling wiring, locks, CI selection, or release automation | Changed package manifest, then its local-source consumers | Refresh/check locks and run the owning package target | [Development](./operations/development.md), [CI selection and labeling](./operations/ci-labeling.md) |

The dependency direction is a useful second check: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; evals consumes the SDK, dcode, and QuickJS; Talon consumes the SDK and dcode. Test the owner first, then a direct consumer when an interface crosses one of those boundaries.

## Current core package contracts

| Distribution (source) | Version | Python contract | Maintainer boundary |
| --- | --- | --- | --- |
| `deepagents` (`libs/deepagents/`) | `0.7.34` | `>=3.11,<4.0` | Reusable agent-harness SDK: graph construction, middleware, and pluggable backends. |
| `deepagents-acp` (`libs/acp/`) | `0.0.13` | `>=3.11,<4.0` | Editor-facing Agent Client Protocol adapter. |
| `deepagents-code` / `dcode` (`libs/code/`) | `0.2.5` | `>=3.14,<3.15` | Terminal coding-agent product and server. |
| `deepagents-talon` (`libs/talon/`) | `0.1.0` | `>=3.14,<3.15` | Experimental local host for long-running channels and schedules. |
| `deepagents-evals` (`libs/evals/`) | `0.0.1` | `>=3.14,<3.15` | Evaluation suite and Harbor integration. |

The SDK is the reusable harness; dcode owns terminal product policy, ACP owns protocol translation, and Talon owns host lifecycle, channels, and schedules. Provider and sandbox distributions are integration boundaries rather than SDK base dependencies. Release Please's manifest currently records releases for the SDK, ACP, dcode, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partner packages; consult the manifests rather than treating this table as a release-authority list.

## Work package-first

Use `uv` for interpreters, environments, and dependencies. It provisions the applicable interpreter automatically, so do not add a repository-wide Python pin; each package declares `requires-python`. The owning Makefile is the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. SDK and dcode unit targets disable network sockets while allowing Unix sockets. ACP does the same; Talon's target first runs WhatsApp bridge Node tests and then runs Python tests with sockets disabled. Use the narrowest relevant test before widening validation.

SDK and dcode accept `TEST_FILE`. Dcode's `make check` adds local CI-parity gates, including generated-command, metadata, lock, and SDK-pin checks. QuickJS defaults `make test` to socket-disabled unit tests, accepts `TEST_FILE`, and provides `make update_snapshots` for reviewed prompt-smoke updates. Use Talon's Makefile target rather than a raw Python invocation for bridge changes.

## Dependencies, local sources, and locks

Keep three layers separate:

- **Published metadata** — `[project.dependencies]`, extras, and `requires-python` define what an installer receives.
- **Checkout wiring** — `[tool.uv.sources]` maps sibling directories for monorepo development; it is not wheel metadata.
- **Resolution** — every package has a generated, package-local `uv.lock`; there is no repository-wide environment lock.

An editable sibling can make local tests pass while a published requirement is too weak. Dcode publishes an exact `deepagents==0.7.34` pin; raise it in the same pull request that first uses newer SDK behavior. Its normal dependencies also include ACP `>=0.0.13,<1.0.0` and QuickJS `>=0.3.11,<0.4.0`. Talon publishes `deepagents>=0.7.32` and `deepagents-code>=0.2.4,<1.0.0`; history embedding adapters plus MongoDB/PostgreSQL support are optional extras rather than base-host dependencies.

After a manifest change, refresh the owner and every affected transitive local consumer, then validate the complete lock set:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
make -C libs lock-check
```

Do not hand-edit lockfiles. The helper determines stale local consumers from the named manifest, while the fan-out target runs `uv lock --check` for every package directory using its configured Python version. See [Development](./operations/development.md) for lockfile closure and release guidance.

## CI and release orientation

On pull requests, CI selects package lint and test jobs from path filters. SDK changes select dependent package jobs, and edits to the shared CI workflow, reusable lint/test workflows, or composite actions select the corresponding package filters. Dcode, Talon, evals, and ACP therefore see SDK changes in CI; Talon also runs for dcode changes. CI tests the SDK across Python 3.11–3.14, ACP across the same range, and dcode, Talon, and evals on Python 3.14.

For an SDK-only pull request that does not change QuickJS, CI runs QuickJS's system-prompt smoke test against the editable local SDK to catch snapshot drift. Pushes and QuickJS-changing pull requests use the full QuickJS suite instead.

Release Please uses separate draft release PRs per managed package. Once a release PR is refreshed, reviewed, and merged with required checks, the release workflow publishes its PyPI and GitHub releases. Adding a partner package therefore requires coordinated package, CI, labeling, release, and repository wiring—not only a new distribution. Follow [Development](./operations/development.md) and the repository release documentation for the operational procedure.

## Detailed guides

- **Package graph, versions, Python/dependency contracts, and ownership:** [Repository Architecture Overview](./architecture/overview.md)
- **Implementation and state-owner routing:** [System Source Map](./architecture/source-map.md)
- **Agent construction and customization:** [Build or Modify a Deep Agent](./workflows/build-a-deep-agent.md)
- **ACP session and dcode ACP boundary:** [Agent Client Protocol Integration](./integrations/acp.md)
- **Sandbox and provider package boundaries:** [Sandbox and Partner Integrations](./integrations/sandbox-partners.md)
- **Dependency setup, local sources, locks, and releases:** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **CI filters, labels, and required gates:** [CI Selection and Labeling](./operations/ci-labeling.md)
- **Focused regression and integration coverage:** [Testing Guide](./testing/testing-guide.md)
