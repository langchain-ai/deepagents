---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning independently versioned package, focused validation path, and generated lockfile closure. Covers current SDK and dcode pins, Talon capabilities, and CI lockfile ownership.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-10T08:04:01.717Z
sources:
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
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
generated: { by: "openwiki/0.4.2", at: "2026-11-10T08:04:01.717Z" }
---

# Deep Agents Monorepo Quickstart

Start in the package that owns the published contract or runtime state—not necessarily the package where a symptom appears. This is an independently versioned `libs/` monorepo: each package owns its manifest, Makefile, README, source, tests, and package-local `uv.lock`. Use `uv` for interpreters, environments, and explicit dependency synchronization, and use `make help` and the package Makefile as the command contract. `uv` provisions the package's required interpreter, so there is no global Python version to install or pin.

For system ownership rather than command routing, see the [architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md). For detailed dependency, CI, and test policy, use [development operations](./operations/development.md) and the [testing guide](./testing/testing-guide.md).

## Route the task

| Change concerns… | Start in | Focused validation | Continue in |
| --- | --- | --- | --- |
| SDK graph construction, middleware, backends, profiles, filesystem tools, or SDK subagents | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [SDK construction](./architecture/sdk-construction-execution.md), [middleware assembly](./architecture/middleware-stack.md), [tools and filesystem access](./concepts/tools-filesystem.md) |
| dcode CLI/TUI, local server lifecycle, workspace policy, offload, or model configuration | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [dcode architecture](./architecture/code-agent.md), [run a dcode session](./workflows/run-dcode-session.md) |
| ACP session translation or dcode ACP mode | `libs/acp/`, then the dcode seam when applicable | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Talon channels, admission, host lifecycle, cron/background work, speech, MCP, history, sandbox, or metadata | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon integration](./integrations/talon.md), [channel admission](./concepts/talon-channel-admission.md), [scheduling](./concepts/talon-scheduling.md) |
| Evals, Harbor, datasets, or reports | `libs/evals/` | `make help`, then the nearest evaluation target | [testing guide](./testing/testing-guide.md) |
| Sandbox provider or QuickJS implementation | `libs/partners/<provider>/` | `make help`, then the owning package target | [sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Published dependency, local source mapping, or resolution | Changed package `pyproject.toml`, then reverse local consumers | Refresh the lock closure, check affected paths, then test a direct consumer | [development operations](./operations/development.md) |

Use the nearest package test through its Makefile before expanding scope. Repository convention puts deterministic, network-free tests under `tests/unit_tests/` and networked contracts under `tests/integration_tests/`; unaccepted pytest warnings fail. Talon also has package-level tests directly under `tests/`, so its nearest test path is authoritative.

## Compatibility boundaries that change the route

The reusable SDK distribution is `deepagents` `0.7.35`, with Python `>=3.11,<4.0`. The terminal product is `deepagents-code` `0.2.6`, exposed by both `deepagents-code` and `dcode`; it requires Python `>=3.14,<3.15` and publishes the exact dependency `deepagents==0.7.35`. Treat this as a release compatibility boundary: if dcode begins to require newer SDK behavior, update the SDK pin and its lockfile and validate dcode—not merely the SDK.

`deepagents-talon` `0.1.0` is an experimental Python `>=3.14,<3.15` local host for long-running channels and schedules. Its base dependency contract includes `deepagents>=0.7.34`, `deepagents-code>=0.2.5,<1.0.0`, channel/MCP dependencies, LangChain/LangGraph, and SQLite checkpoint support. Talon's `[tool.uv.sources]` mappings make its SDK and dcode siblings editable in this checkout; those mappings are development wiring, not published requirements.

### Talon installation choices

| Capability | Extra | Operational boundary |
| --- | --- | --- |
| Local, Voyage, or OpenAI-compatible history embeddings | `history-local`, `history-voyage`, `history-openai` | Embedding adapters are optional, not base-host dependencies. |
| MongoDB or PostgreSQL persistence | `mongodb`, `postgres` | Remote persistence support is optional; SQLite checkpoint support is base. |
| OpenShell sandbox provider | `openshell` | Pulls the matching dcode provider capability. |
| Audio/media and transcription | `media` or `media-cpu` | Select exactly one. On Linux, `media-cpu` selects `torch` and `torchaudio` from Talon's explicit PyTorch CPU index. |

Extras install code; they do not provide credentials, create remote resources, or establish authorization policy. For channel-facing behavior and unattended authority, follow [Talon integration](./integrations/talon.md), [channel admission](./concepts/talon-channel-admission.md), and [MCP integration](./integrations/mcp.md).

For a Talon change requiring an extra:

```bash
cd libs/talon
uv sync --group test --extra postgres
make test TEST_FILE=tests/<nearest test>.py
make lint
```

The ACP and Talon standard `make test` targets disable network sockets. Talon first runs WhatsApp bridge Node tests, then runs pytest from its test group with Unix sockets allowed, a 10-second timeout, and coverage; `make lint` runs Ruff checks, format verification, and `ty`.

## Dependency metadata, checkout wiring, and locks

Keep three layers distinct:

1. **Published metadata**—`[project.dependencies]`, `[project.optional-dependencies]`, and `requires-python` are the downstream installer contract.
2. **Checkout wiring**—`[tool.uv.sources]` can substitute editable sibling source in this repository. It does not strengthen a published version constraint.
3. **Generated resolution**—each package's `uv.lock` is the concrete development/CI resolution. Regenerate it with `uv`; do not hand-edit it.

Talon's lock, for example, resolves Python `==3.14.*` and represents the mutually exclusive `media` and `media-cpu` branches; it is not the dependency manifest.

A manifest edit can invalidate the owner and transitive reverse consumers that use local uv path sources. From the repository root, refresh from the changed manifest and commit every printed lockfile:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/talon/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/talon/pyproject.toml
```

The refresher computes that reverse-consumer closure and locks each package with its policy interpreter. The targeted checker has different selection semantics: supplied paths check only directly touched package directories, while no paths checks every discovered package. On a stale lock, it reports the exact non-check `uv lock` repair command.

The checker uses Python 3.14 for ACP, dcode, evals, and Talon, and Python 3.12 for the other discovered packages. Do not treat `make -C libs lock-check` as CI-equivalent for dcode, evals, or Talon: its fan-out maps only ACP to 3.14 and everything else to 3.12. CI calls the reusable `Check Lockfiles` workflow, which runs the targeted checker with no paths and therefore verifies every discovered lockfile.

## Finish a focused change

1. Select the package that owns the state or published contract, run `uv sync` there, and use `make help` if its targets are unfamiliar.
2. Change the owning layer. For dependencies, verify both the published constraint and any local source substitution.
3. Run the nearest owning test and `make lint`. For an SDK compatibility or pin change, test dcode; for an SDK/dcode local-source change, test Talon as a direct consumer.
4. For a manifest edit, refresh the reverse-consumer lock closure, commit all printed `uv.lock` files, and run the targeted checker for the affected package paths. Use its no-argument form for a repository-wide final check.
5. Continue through the linked [architecture](./architecture/overview.md), [concepts](./concepts/profiles-models.md), [workflows](./workflows/build-a-deep-agent.md), [integrations](./integrations/talon.md), [operations](./operations/development.md), and [testing](./testing/testing-guide.md) pages for behavior-specific design and validation.
