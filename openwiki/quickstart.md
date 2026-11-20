---
type: maintainer routing guide
title: Repository Quickstart
description: Route a Deep Agents monorepo change to the package that owns its public contract, runtime boundary, tests, and release metadata. Use package-local uv and Make workflows, and treat manifest changes as published-contract and consumer-lockfile work.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
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
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
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
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-11-20T08:04:05.944Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-11-20T08:04:05.944Z
---

# Repository Quickstart

This is a monorepo of independently versioned distributions under `libs/`. Start at the package that owns the public contract, persistent state, or process lifecycle—not merely where a symptom appears. Each package owns its `pyproject.toml`, `Makefile`, README, source, tests, and generated lockfile; its local instructions and nearest behavioral regression test are the authority for a focused change.

Use `uv` for interpreters, environments, and dependencies, and use the package-local `Makefile` for normal commands. `uv` provisions the required Python interpreter, so there is no repository-wide environment or global Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` before assuming a target exists. Install dependencies explicitly with `uv sync`; reserve `uv run ...` for direct one-off commands.

## Route the change

| If the change concerns… | Start in | Read next | Focused validation |
| --- | --- | --- | --- |
| SDK graph construction, middleware, backends, profiles, filesystem tools, permissions, or subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md) | Nearest SDK unit test, then `make lint` |
| dcode CLI/TUI, server, approvals, workspace policy, model configuration, MCP, sessions, or JavaScript costs | `libs/code/` | [Profiles and Models](./concepts/profiles-models.md), [GitHub Action integration](./integrations/github-action.md), and [Cost and sessions](./operations/cost-and-sessions.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| ACP graph-to-protocol translation or dcode ACP mode | `libs/acp/`; also inspect dcode if it launches or configures ACP | [Agent Client Protocol integration](./integrations/acp.md) | Nearest ACP test via `make test` |
| Talon channels, graph/runtime lifecycle, approvals, schedules, detached work, archives, or history | `libs/talon/` | [Talon runtime integration](./integrations/talon.md), [Talon scheduling](./concepts/talon-scheduling.md), and [Subagents and skills](./concepts/subagents-skills.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| QuickJS or Daytona, Modal, Runloop, and Vercel sandbox behavior | `libs/partners/<provider>/`; inspect dcode when it loads the integration | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | `make help`, then the owning package's focused target |
| Evaluation definitions, behavioral evals, Harbor graphs/tasks, or reports | `libs/evals/` | [Run Evals](./workflows/run-evals.md) and the [Testing Guide](./testing/testing-guide.md) | `make help`, then the closest owning target |
| Dependency floors, Python ranges, extras, editable sibling mappings, locks, or release automation | Changed package manifest, then its local-source consumers | [Development and Release Operations](./operations/development.md) | Refresh the lock closure, check locks, and test a direct consumer |

Repository convention places deterministic, network-free tests in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; unaccepted pytest warnings fail the suite. Follow the owning package's actual test layout when it differs. The [Testing Guide](./testing/testing-guide.md) provides the current hermetic routes for dcode, Talon, QuickJS, and Harbor.

## Package and compatibility map

`deepagents` is the reusable SDK; ACP adapts a Deep Agents graph to the Agent Client Protocol; `deepagents-code` is the terminal coding-agent product launched by `dcode`; Talon is the experimental local host for long-running channels and schedules; `deepagents-evals` owns evaluation and Harbor integration; and partner packages provide provider or sandbox integrations. The linked domain pages define their runtime semantics.

| Distribution / owner | Current package version | Published compatibility that matters for routing |
| --- | --- | --- |
| SDK: `libs/deepagents/` | `deepagents` 0.7.41 | Python `>=3.11,<4.0` |
| dcode: `libs/code/` | `deepagents-code` 0.2.12 | Python `>=3.14,<3.15`; exact `deepagents==0.7.41` |
| ACP: `libs/acp/` | `deepagents-acp` 0.0.13 | Python `>=3.11,<4.0`; consumes `deepagents` |
| Talon: `libs/talon/` | `deepagents-talon` 0.1.1 | Python `>=3.14,<3.15`; `deepagents>=0.7.41`, `deepagents-code>=0.2.11,<1.0.0` |
| Evals: `libs/evals/` | `deepagents-evals` 0.0.1 | Python `>=3.14,<3.15`; SDK, dcode, QuickJS, and Harbor evaluation boundary |
| QuickJS: `libs/partners/quickjs/` | `langchain-quickjs` 0.4.0 | Python `>=3.11,<4.0`; `deepagents>=0.7.41,<0.8.0` |

Dcode's exact SDK requirement is a release dependency: raise it in the feature PR when dcode starts using new SDK functionality, rather than at release time. Talon's checkout maps SDK and dcode siblings as editable sources, and QuickJS maps the SDK likewise. These mappings make sibling changes immediately visible locally, but they do not alter wheel metadata; a green editable environment does not prove a released installation will resolve.

Talon's history embedding adapters and MongoDB/PostgreSQL support are optional extras rather than base-host dependencies. Its `media` and `media-cpu` extras conflict; on Linux, `media-cpu` resolves PyTorch from the explicit CPU index. See [Talon runtime integration](./integrations/talon.md) for runtime composition and [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) for QuickJS lifecycle and persistence behavior.

## Manifest and lockfile changes

Keep three layers distinct:

1. **Published metadata** — `[project]` requirements, extras, and `requires-python` define downstream installation compatibility.
2. **Checkout wiring** — `[tool.uv.sources]` substitutes sibling paths during repository development; it does not change wheel metadata.
3. **Generated resolution** — a package-local `uv.lock` records concrete development/CI resolution. Regenerate it with `uv`; never hand-edit it.

After changing a manifest, refresh the changed package and its transitive reverse consumers from the repository root:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

The refresher follows local `uv` path sources, locks each affected package with its selected interpreter, and prints every `uv.lock` to commit. Check either the directly touched package paths or all locks afterward:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

With supplied paths, the checker deliberately validates only packages touched by those paths; it does not calculate the reverse-consumer closure. With no arguments it discovers and validates every package lockfile, which is also the reusable CI route. An out-of-date lock error prints an executable non-check `uv lock` repair command. The targeted scripts use Python 3.14 for ACP, dcode, evals, and Talon and Python 3.12 for other packages. The `libs/` fan-out `lock-check` uses Python 3.14 only for dcode, evals, and Talon, so prefer the checker when CI-equivalent ACP lock validation matters. For the full lock and wheel-release process, use [Development and Release Operations](./operations/development.md).

For local focused validation, ACP and Talon `make test` disable network sockets while allowing Unix sockets; Talon first runs WhatsApp-bridge Node tests. Dcode's unit-test target is also socket-disabled, while `make integration_test` runs network-capable integration tests. Do not substitute an invented repository-wide test command for a package-owned target.

## Release handoff

The manifest records separate release baselines: `deepagents` 0.7.41, `deepagents-acp` 0.0.13, `deepagents-code` 0.2.12, `deepagents-talon` 0.1.1, Daytona 0.0.9, Modal 0.0.8, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.4.0. These are last-released baselines; each package's `pyproject.toml` remains the installer-visible contract.

Release Please creates or updates a separate draft release PR per managed package from conventional commits on `main`. Merging that release PR triggers the publish and GitHub-release workflow. Use [Development and Release Operations](./operations/development.md) and `.github/RELEASING.md` for the release gate, source-free wheel validation, fan-out rules, and curated notes; do not hand-edit release-automation artifacts.

Before submitting a focused change:

1. Inspect the owner package's instructions, manifest, Makefile, source, and nearest regression test.
2. Run `uv sync --all-groups`, make the narrowest ownership-boundary change, and run the relevant package target plus `make lint`.
3. For a manifest change, review wheel requirements separately from `uv` source wiring; refresh the complete reverse-consumer lock closure; then test an immediate compatibility consumer.
4. For runtime, evaluation, test, or release work, follow the linked domain page instead of broadening the patch by guesswork.
