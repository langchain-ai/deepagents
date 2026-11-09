---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning independently versioned package, focused validation path, and generated lockfile closure. Covers the current SDK, dcode, and Talon dependency and Python constraints.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-09T08:04:29.143Z
sources:
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
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
  - id: openwiki-source-966f1489233b6fc6ee5f2e4c
    resource: repo://libs/talon/uv.lock
generated: { by: "openwiki/0.4.2", at: "2026-11-09T08:04:29.143Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime state—not necessarily the package where a symptom appears. This is an independently versioned `libs/` monorepo: each package owns its manifest, Makefile, README, source, tests, and package-local `uv.lock`. Work from that package directory, synchronize its dependencies explicitly with `uv sync`, and use `make help` and the package Makefile as the normal command contract. `uv` provisions the applicable interpreter; there is no repository-wide Python version to install or pin.

## Route the task

| Change concerns… | Start in | Focused validation | Read next |
| --- | --- | --- | --- |
| SDK graph construction, middleware, backends, profiles, filesystem tools, or SDK subagents | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [profiles and models](./concepts/profiles-models.md) |
| dcode CLI or terminal UI, local agent-server lifecycle, model metadata, workspace policy, or product behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Code Agent architecture](./architecture/code-agent.md), [source map](./architecture/source-map.md), [testing guide](./testing/testing-guide.md) |
| ACP session translation or dcode ACP mode | `libs/acp/`, then the dcode seam if applicable | `make help`, then the nearest package target | [Architecture overview](./architecture/overview.md), [ACP integration](./integrations/acp.md) |
| Talon channels, admission, host lifecycle, cron or background work, speech, MCP, history, sandbox selection, or Talon metadata | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon runtime integration](./integrations/talon.md), [channel admission](./concepts/talon-channel-admission.md), [scheduling](./concepts/talon-scheduling.md) |
| Evals, Harbor, datasets, or reports | `libs/evals/` | `make help`, then the nearest evaluation target | [Testing guide](./testing/testing-guide.md) |
| Sandbox provider or QuickJS implementation | `libs/partners/<provider>/` | `make help` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Published dependency, local source, or resolved dependency change | Changed package `pyproject.toml`, then its reverse local consumers | Refresh the lock closure, check every affected package, then test a direct consumer | [Development](./operations/development.md) |

Keep deterministic tests network-free under `tests/unit_tests/`; reserve `tests/integration_tests/` for networked integration contracts. The repository treats unaccepted pytest warnings as errors. Talon has some package-level tests directly under `tests/`, so use its nearest test path rather than assuming the SDK/dcode directory layout.

## Package constraints that affect routing

The SDK distribution is `deepagents` and supports Python `>=3.11,<4.0`. dcode is the `deepagents-code` distribution and `dcode` command; it requires Python `>=3.14,<3.15` and pins its SDK dependency exactly to `deepagents==0.7.34`. Treat that pin as a compatibility boundary: a dcode behavior that starts using new SDK behavior requires a coordinated SDK-pin change and validation of dcode, not merely an SDK test.

`deepagents-talon` is an experimental Python `>=3.14,<3.15` local host for long-running channels and schedules. Its base dependencies intentionally include `deepagents>=0.7.34`, `deepagents-code>=0.2.5,<1.0.0`, the dcode-facing channel/MCP stack, LangChain/LangGraph, and SQLite checkpoint support. In the monorepo, Talon's `deepagents` and `deepagents-code` source entries are editable sibling-checkout wiring; they are not published dependency requirements.

### Talon capability extras

Install only the extra needed for the behavior under change or deployment:

| Capability | Extra | Boundary |
| --- | --- | --- |
| Local, Voyage, or OpenAI-compatible history embeddings | `history-local`, `history-voyage`, `history-openai` | Embedding adapters are optional rather than base-host dependencies. |
| MongoDB or PostgreSQL checkpoint/history support | `mongodb`, `postgres` | Persistence backends are optional; SQLite remains in the base installation. |
| OpenShell sandbox provider | `openshell` | Pulls the matching dcode provider capability. |
| Audio/media and transcription dependencies | `media` or `media-cpu` | Choose exactly one: they conflict. On Linux, `media-cpu` takes `torch` and `torchaudio` from Talon's explicit PyTorch CPU index. |

For example:

```bash
cd libs/talon
uv sync --group test --extra postgres
make test TEST_FILE=tests/<nearest test>.py
make lint
```

The ACP and Talon standard `make test` targets disable network sockets. Talon's target runs WhatsApp bridge Node tests before Python tests; its Python invocation uses the test dependency group, allows Unix sockets, applies a 10-second timeout, and records coverage. Talon's `make lint` checks Ruff formatting and lint rules and runs `ty`. Use that target instead of raw pytest when package-equivalent coverage is needed.

Talon is an experimental host, not a security boundary supplied by extras. In particular, selecting a persistence, embedding, sandbox, or media extra adds code dependencies; it neither supplies credentials nor creates remote resources. Consult [Talon runtime integration](./integrations/talon.md) for MCP behavior and [Talon channel admission](./concepts/talon-channel-admission.md) before changing an externally reachable channel boundary.

## Manifest, sources, and locks are separate layers

A dependency edit has three different consequences:

1. **Published metadata:** `[project.dependencies]`, `[project.optional-dependencies]`, and `requires-python` are the downstream wheel contract.
2. **Checkout wiring:** `[tool.uv.sources]` can substitute editable sibling source while working in this checkout. It can conceal a published constraint that is too weak, so review the installer-visible requirement separately.
3. **Generated resolution:** each package's `uv.lock` records the concrete resolution selected for that package's development and CI environment. Do not hand-edit it.

For example, Talon's generated lock resolves Python `==3.14.*` and records resolution branches and a conflict for mutually exclusive `media` and `media-cpu`. That describes the generated resolution; the manifest remains the dependency declaration.

A local path-source change can affect more than the edited package. The targeted refresher starts from changed manifests, finds transitive reverse consumers of local uv paths, and locks that complete closure with its selected interpreter. Its interpreter mapping selects Python 3.14 for `acp`, `code`, `evals`, and `talon`; other package directories use Python 3.12.

From the repository root, begin with the changed manifest and commit every lockfile the refresh script prints:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/talon/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/talon/pyproject.toml
```

When changing dcode or SDK metadata, refresh from that owner. Talon locally consumes both, so it is a direct consumer to validate after the refresh:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/code/pyproject.toml libs/talon/pyproject.toml
make -C libs/talon test TEST_FILE=tests/<nearest test>.py
```

The checker examines packages named by supplied paths, or every discovered package when no paths are supplied. A stale lock error includes the non-check `uv lock --directory … --python …` repair command. `make -C libs lock-check` remains useful as a broad fan-out, but it maps only ACP to Python 3.14 and all other packages to Python 3.12. It therefore is not CI-equivalent lock validation for Talon, dcode, or evals; use the targeted scripts for those packages.

## Finish a focused change

1. Identify the state or published-contract owner with the routing table and synchronize that package environment. Select Talon extras only when the changed capability requires them.
2. Change code or metadata in the layer that owns it. Preserve the distinction between published constraints and monorepo source mappings.
3. Run the nearest owning test through its Makefile, then its lint target. A boundary change should also exercise the direct consumer: dcode for an SDK-pin or SDK-compatibility change, and Talon for an SDK/dcode path-source contract change.
4. For every manifest edit, refresh from the changed manifest, commit the printed lockfile closure, and run the targeted lock checker for all affected package paths.
5. Use the hierarchy for behavior-specific design and tests: [architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md); [profiles and models](./concepts/profiles-models.md), [channel admission](./concepts/talon-channel-admission.md), and [scheduling](./concepts/talon-scheduling.md); [Talon integration](./integrations/talon.md); [development operations](./operations/development.md); and the [testing guide](./testing/testing-guide.md).
