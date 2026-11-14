---
type: maintainer routing guide
title: Repository Quickstart
description: Route a Deep Agents monorepo change to its owning package, use the relevant architecture or workflow guide, and begin package-local development with safe validation and lockfile handling.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
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
verified:
  - by: openwiki/0.4.2
    at: 2026-11-14T08:03:44.053Z
generated: { by: "openwiki/0.4.2", at: "2026-11-14T08:03:44.053Z" }
---

# Repository Quickstart

Start with the package that owns the public contract, persistent state, or process lifecycle—not the package where a symptom appears. This is an independently versioned Python monorepo: packages under `libs/` own their own `pyproject.toml`, `Makefile`, README, source, tests, and lockfile. The release manifest tracks separate baselines for the SDK, ACP, dcode, Talon, and partner distributions.

Use `uv` for the interpreter, environment, and dependencies, and use the local `Makefile` as the command interface. `uv` selects the interpreter required by the package, so do not create a repository-wide Python pin or a shared virtual environment.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` when a target is unfamiliar. Install dependencies explicitly with `uv sync`; use `uv run ...` only for a one-off command. The package-local Makefile, nearest existing test, and package-specific contributor guidance take precedence over assumptions from another package.

## Route the change

| If the change concerns… | Owning package / first boundary | Read next | Start validation with |
| --- | --- | --- | --- |
| SDK graph assembly, middleware, backends, profiles, filesystem tools, permissions, or SDK subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md), [SDK construction and execution](./architecture/sdk-construction-execution.md), [source map](./architecture/source-map.md) | A nearest SDK unit test, then `make lint` |
| Context compaction, state/checkpoint behavior, declarative/forked/remote subagents, profiles, or filesystem semantics | `libs/deepagents/` | [Context management](./concepts/context-management.md), [State persistence](./concepts/state-persistence.md), [Subagents and skills](./concepts/subagents-skills.md), [Profiles and models](./concepts/profiles-models.md), or [Tools and filesystem](./concepts/tools-filesystem.md) | The focused SDK behavior test named by the source map/testing guide |
| dcode CLI/TUI, local server, headless run, workspace policy, models, approvals, or output | `libs/code/` | [Run a dcode session](./workflows/run-dcode-session.md), [Profiles and models](./concepts/profiles-models.md), [source map](./architecture/source-map.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| ACP graph-to-protocol adaptation or dcode ACP mode | `libs/acp/`; inspect dcode too when it launches or configures ACP | [Agent Client Protocol integration](./integrations/acp.md) and [Run a dcode session](./workflows/run-dcode-session.md) | The nearest ACP test via `make test` |
| Talon channel host, runtime graph/MCP reload, schedules, detached work, or archives | `libs/talon/` | [source map](./architecture/source-map.md), [Talon conversation history and storage](./concepts/talon-history-storage.md), and [Testing guide](./testing/testing-guide.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| Evaluation definitions, datasets, Harbor execution, or reports | `libs/evals/` | [Development and release operations](./operations/development.md) and [Testing guide](./testing/testing-guide.md) | `make help`, then the closest supported target |
| Daytona, Modal, QuickJS, Runloop, or Vercel sandbox behavior | `libs/partners/<provider>/` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) and [Testing guide](./testing/testing-guide.md) | `make help`, then the owning package target |
| A dependency floor/pin, Python range, optional extra, local source mapping, lockfile, or release baseline | The changed package manifest; then every local consumer affected by it | [Architecture overview](./architecture/overview.md) and [Development and release operations](./operations/development.md) | Refresh the lock closure, run the targeted checker, and test a direct consumer |

The [Testing guide](./testing/testing-guide.md) supplies behavior-specific suites. As a general rule, put deterministic, network-free coverage in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; unaccepted pytest warnings fail the suite. Follow local layout where a package differs.

## Package relationships that change the safe plan

`deepagents` is the reusable SDK; ACP adapts a graph to the Agent Client Protocol; dcode is the terminal coding-agent product; Talon is the experimental local host for long-running channels and schedules; evals is the evaluation/Harbor boundary; and partner packages provide optional sandbox or middleware integrations. The architecture overview explains their installer-visible contracts and development-only workspace edges.

The SDK supports Python `>=3.11,<4.0`. dcode supports Python `>=3.14,<3.15` and publishes an exact `deepagents==0.7.38` dependency. Talon also supports Python `>=3.14,<3.15`, publishes `deepagents>=0.7.38` and `deepagents-code>=0.2.7,<1.0.0`, and maps the SDK and dcode siblings as editable checkout sources. Therefore, an SDK capability change that dcode needs is a coordinated published-pin, lockfile, and dcode-validation change; a successful editable checkout alone does not prove a released installation works.

Talon's history embedding adapters and MongoDB/PostgreSQL persistence are optional extras, not base-host dependencies. Its `make test` runs WhatsApp bridge Node tests before the Python suite; the ACP and Talon unit targets disable network sockets, and Talon's Python invocation allows Unix sockets, applies a 10-second timeout, and collects coverage.

## Safe dependency and lockfile path

Keep three layers separate:

1. **Published metadata** — `[project]` dependencies, extras, and `requires-python` are the downstream installation contract.
2. **Checkout wiring** — `[tool.uv.sources]` can select sibling checkouts for repository development; it does not change the published constraint.
3. **Generated resolution** — each package's `uv.lock` is the concrete development/CI resolution. Regenerate it with `uv`; do not edit it by hand.

After changing a manifest, refresh the owner and its transitive reverse consumers that use local `uv` sources, then check the affected paths:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/code/pyproject.toml
```

The refresh helper finds reverse consumers and chooses the policy interpreter for each lock. The checker with supplied paths intentionally checks only the named package directories; without paths it checks every discovered package and prints a non-check `uv lock` repair command when it finds stale resolution. CI uses that no-argument route. Do not regard `make -C libs lock-check` as CI-equivalent for dcode, evals, or Talon: its interpreter mapping differs from the checker's.

## Finish a focused change

1. Identify the owner using the routing table and inspect its local instructions, manifest, Makefile, source, and nearest regression test.
2. Synchronize that package with `uv sync --all-groups`; make the change at the state, lifecycle, or published-contract owner.
3. Run the narrowest behavioral test and `make lint`. Add deterministic coverage only when it protects observable behavior.
4. If a package manifest changed, review published metadata separately from `uv` source wiring, refresh every affected lockfile, and test an immediate consumer where the compatibility boundary is crossed.
5. For release or dependency automation, use [Development and release operations](./operations/development.md). For runtime-specific behavior, continue from the linked architecture, concept, integration, workflow, or testing guide rather than broadening the change by guesswork.
