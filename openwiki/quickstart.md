---
type: maintainer routing guide
title: Repository Quickstart
description: Route a Deep Agents monorepo change to the package that owns its public contract, validation, dependency metadata, lockfile closure, and release handoff. Use package-local uv and Make workflows, and keep published requirements distinct from checkout wiring and generated locks.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-28T08:03:41.605Z
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
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
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
  - id: openwiki-source-da577cbe81ec29338f1388b2
    resource: repo://libs/partners/daytona/pyproject.toml
  - id: openwiki-source-936554ac5f0a201f8696be25
    resource: repo://libs/partners/modal/pyproject.toml
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-8d2c8381956c1c023bcdb565
    resource: repo://libs/partners/runloop/pyproject.toml
  - id: openwiki-source-03a39f44d8ccfde2fd47e57a
    resource: repo://libs/partners/vercel/pyproject.toml
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "openwiki/0.4.2", at: "2026-11-28T08:03:41.605Z" }
---

# Repository Quickstart

This is a monorepo of independently versioned distributions under `libs/`. Start in the package that owns the public contract or lifecycle—not merely where a symptom appears. Its `pyproject.toml`, `Makefile`, README, source, and nearest behavioral regression test are the authority for a focused change.

Use `uv` for interpreters, environments, and dependencies, and the package-local `Makefile` for normal commands. `uv` provisions the required interpreter, so there is no repository-wide environment or global Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` before assuming a target exists. Install dependencies explicitly with `uv sync`; reserve `uv run ...` for direct, one-off commands. Install the repository hooks once with `uv tool install pre-commit` and `pre-commit install --install-hooks`.

## Route the change

| If the change concerns… | Owning package / boundary | Read next | First focused validation |
| --- | --- | --- | --- |
| SDK graph construction, middleware, backends, profiles, filesystem tools, permissions, or subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| dcode CLI/TUI, server, approvals, workspace policy, model configuration, MCP, sessions, or cost tracking | `libs/code/` | [Development and Release Operations](./operations/development.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| ACP graph-to-protocol translation or dcode ACP mode | `libs/acp/`; inspect dcode too when it launches or configures ACP | [Agent Client Protocol integration](./integrations/acp.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| Talon channels, host/runtime lifecycle, approvals, schedules, detached work, archives, history, or media | `libs/talon/` | [Talon Runtime Integration](./integrations/talon.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| QuickJS or Daytona, Modal, Runloop, and Vercel sandbox behavior | `libs/partners/<provider>/`; inspect dcode when it loads the integration | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | `make help`, then the owning package's focused target |
| Evaluation definitions, behavioral evals, Harbor graphs/tasks, or reports | `libs/evals/` | [Testing Guide](./testing/testing-guide.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`; use the eval target for model-backed work |
| Dependency floors, Python ranges, extras, checkout source mappings, locks, or release automation | Changed package manifest, then local-source consumers | [Development and Release Operations](./operations/development.md) | Refresh the lock closure, check locks, and test a direct consumer |

Repository convention puts deterministic, network-free tests in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; unaccepted pytest warnings fail the suite. Follow the owning package's actual test layout when it differs. See the [Testing Guide](./testing/testing-guide.md) for focused test routes.

## Package ownership and published compatibility

`deepagents` is the reusable SDK; ACP adapts a Deep Agents graph to the Agent Client Protocol; `deepagents-code` is the terminal coding-agent product launched by `dcode`; Talon is the experimental local host for long-running channels and schedules; `deepagents-evals` owns evaluation and Harbor integration; partner packages provide provider or sandbox integrations.

The table is **published metadata** from package manifests: these are the constraints an installer sees, not the development checkout resolution.

| Distribution / owner | Current manifest version and entrypoint | Published compatibility important to routing |
| --- | --- | --- |
| SDK: `libs/deepagents/` | `deepagents` 0.7.42; application code constructs agents with `create_deep_agent` | Python `>=3.11,<4.0` |
| dcode: `libs/code/` | `deepagents-code` 0.2.15; `dcode` and `deepagents-code` invoke `deepagents_code:cli_main` | Python `>=3.14,<3.15`; exact `deepagents==0.7.42` |
| ACP: `libs/acp/` | `deepagents-acp` 0.1.0 | Python `>=3.11,<4.0`; depends on `deepagents` and Agent Client Protocol `>=1.0.0,<2.0.0` |
| Talon: `libs/talon/` | `deepagents-talon` 0.1.2; `deepagents-talon` invokes `deepagents_talon.__main__:main` | Python `>=3.14,<3.15`; `deepagents>=0.7.42`, `deepagents-code>=0.2.15,<1.0.0` |
| Evals: `libs/evals/` | `deepagents-evals` 0.0.1; `deepagents-evals` invokes `deepagents_evals.cli:main` | Python `>=3.14,<3.15`; `deepagents>=0.7.32`, `deepagents-code>=0.2.1` |
| Partners: `libs/partners/{daytona,modal,runloop,vercel}/` | `langchain-daytona` 0.0.9, `langchain-modal` 0.0.8, `langchain-runloop` 0.0.7, `langchain-vercel-sandbox` 0.0.2 | Python `>=3.11,<4.0`; each publishes `deepagents>=0.7.42,<0.8.0` |
| QuickJS: `libs/partners/quickjs/` | `langchain-quickjs` 0.4.0 | Python `>=3.11,<4.0`; `deepagents>=0.7.42,<0.8.0` |

Dcode's exact SDK requirement is a release dependency: raise it in the feature PR when dcode starts using new SDK functionality, rather than at release time. Talon maps SDK and dcode siblings as editable sources; the partner manifests map the SDK likewise. Evals maps local dcode, SDK, and QuickJS checkouts. These mappings make sibling changes available during repository development, but do not alter wheel metadata: a green checkout resolution does not prove a released installation will resolve.

Talon's history embedding adapters and MongoDB/PostgreSQL support are optional extras rather than base-host dependencies. Its `media` and `media-cpu` extras conflict; on Linux, `media-cpu` resolves PyTorch from the explicit CPU index. See [Talon Runtime Integration](./integrations/talon.md) for host composition.

## Manifest and lockfile changes

Keep three layers separate:

1. **Published metadata** — `[project]` requirements, extras, and `requires-python` define what downstream installations may resolve.
2. **Development-only checkout wiring** — `[tool.uv.sources]` maps a requirement to a sibling path for repository development. It is not wheel metadata.
3. **Generated resolution** — a package-local `uv.lock` records the concrete development/CI resolution. Regenerate it with `uv`; never hand-edit it.

After changing a manifest, refresh the changed package and every transitive reverse consumer that uses it as a local `uv` path source, from the repository root:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

The refresher derives that local-source closure, locks every affected package with the checker's selected interpreter, and prints each `uv.lock` to commit. Then check either affected paths or all locks:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

With supplied paths, the checker validates only packages those paths touch; it does not calculate reverse consumers. With no arguments it discovers and validates every package lockfile, which is also the reusable CI route. An out-of-date lock error prints an executable non-check `uv lock` repair command. The targeted scripts use Python 3.14 for ACP, dcode, evals, and Talon and Python 3.12 for other packages. The `libs/` fan-out `lock-check` maps only dcode, evals, and Talon to Python 3.14, so prefer the checker for CI-equivalent ACP lock validation.

For focused validation, ACP and Talon `make test` disable network sockets while allowing Unix sockets; Talon first runs WhatsApp-bridge Node tests. Dcode's unit-test target is also socket-disabled, while `make integration_test` runs the integration-test directory without that socket restriction. Do not substitute an invented repository-wide test command for a package-owned target.

## Release handoff

The release manifest records separate last-released baselines for the SDK, ACP, dcode, Talon, and managed partner packages: `deepagents` 0.7.42, `deepagents-acp` 0.1.0, `deepagents-code` 0.2.15, `deepagents-talon` 0.1.2, Daytona 0.0.9, Modal 0.0.8, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.4.0. The evals package is independently versioned but is not listed in that release manifest. These baselines are not installer compatibility constraints; the package `pyproject.toml` is.

Release Please analyzes conventional commits on `main` and creates or updates a separate draft release PR for each managed package. Wait for its refresh and lockfile update, review the generated changelog and curated notes, and merge only after required checks pass; the merge triggers pre-release checks, PyPI publication, and the GitHub release. Use [Development and Release Operations](./operations/development.md) and `.github/RELEASING.md` for release gates, fan-out, recovery, and notes.

Before submitting a focused change:

1. Inspect the owner package's instructions, manifest, Makefile, source, and nearest regression test.
2. Run `uv sync --all-groups`, make the narrowest ownership-boundary change, and run the relevant package target plus `make lint`.
3. For a manifest change, review wheel requirements separately from `uv` source wiring; refresh the complete reverse-consumer lock closure; then test an immediate compatibility consumer.
4. For architecture, runtime, evaluation, testing, integration, or release work, follow the linked domain page instead of broadening the patch by guesswork.
