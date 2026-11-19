---
type: maintainer routing guide
title: Repository Quickstart
description: Route a Deep Agents monorepo change to the package that owns its public contract, runtime boundary, tests, and release metadata. Use package-local uv and Make workflows, and handle manifest and lockfile changes as a consumer-closure operation.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-19T08:04:44.778Z
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
generated: { by: "openwiki/0.4.2", at: "2026-11-19T08:04:44.778Z" }
---

# Repository Quickstart

This repository is a monorepo of independently versioned distributions under `libs/`. Start where the public contract, persistent state, or process lifecycle is owned—not merely where a symptom appears. Each package owns its `pyproject.toml`, `Makefile`, README, source, tests, and generated lockfile; its local guidance and nearest regression test are the authority for a focused change.

Use `uv` for the interpreter, environment, and dependencies, and use the package-local `Makefile` for normal commands. `uv` provisions the package's required Python interpreter, so do not create a repository-wide environment or pin a global Python version.

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
| SDK graph assembly, middleware, backends, profiles, filesystem tools, permissions, or subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md) and [source map](./architecture/source-map.md) | Nearest SDK unit test, then `make lint` |
| dcode CLI/TUI, local server, headless execution, approvals, workspace policy, model configuration, MCP, or session status | `libs/code/` | [Profiles and Models](./concepts/profiles-models.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| ACP graph-to-protocol translation or dcode ACP mode | `libs/acp/`; also inspect dcode if it launches/configures ACP | [Agent Client Protocol integration](./integrations/acp.md) | Nearest ACP test via `make test` |
| Talon channels, runtime lifecycle, schedules, detached work, archives, or history storage | `libs/talon/` | [Talon runtime integration](./integrations/talon.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| QuickJS or Daytona, Modal, Runloop, and Vercel sandbox behavior | `libs/partners/<provider>/`; inspect dcode when it loads the integration | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | `make help`, then the owner's focused target |
| Evaluation definitions, datasets, Harbor execution, or reports | `libs/evals/` | [Testing guide](./testing/testing-guide.md) | `make help`, then the closest owning target |
| Action-facing dcode distribution or GitHub Action metadata | Root action files and `libs/code/` | [GitHub Action Integration](./integrations/github-action.md) | The action's focused checks |
| Dependency floors, Python ranges, extras, local source mappings, locks, release labels, or release automation | Changed package manifest, then local-source consumers | [Development and Release Operations](./operations/development.md) | Refresh the lock closure, check each refreshed lock, and test a direct consumer |

Repository convention places deterministic, network-free tests in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; pytest warnings not explicitly accepted fail the suite. Follow the owning package's actual layout when it differs.

## Package and compatibility map

`deepagents` is the reusable SDK; ACP adapts a Deep Agents graph to the Agent Client Protocol; `deepagents-code` is the terminal coding-agent product launched by `dcode`; Talon is the experimental local host for long-running channels and schedules; evals is the evaluation/Harbor boundary; and partner packages provide provider or sandbox integrations. The linked domain pages, rather than this routing page, define their runtime semantics.

The release manifest records separate baselines for the SDK, ACP, dcode, Talon, and each partner distribution: `deepagents` 0.7.40, `deepagents-acp` 0.0.13, `deepagents-code` 0.2.11, `deepagents-talon` 0.1.0, Daytona 0.0.9, Modal 0.0.8, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.3.12. Those are release baselines; a package's `pyproject.toml` remains the installer-visible compatibility contract.

The SDK and ACP support Python `>=3.11,<4.0`; dcode and Talon require Python `>=3.14,<3.15`. Dcode publishes the exact requirement `deepagents==0.7.40`; increase that pin in the same feature PR when dcode begins using newer SDK functionality. An editable checkout can otherwise mask an insufficient published requirement.

Talon publishes `deepagents>=0.7.40` and `deepagents-code>=0.2.11,<1.0.0`, while its checkout maps both siblings as editable sources. Its history embedding adapters and MongoDB/PostgreSQL support are extras, not base-host dependencies. The `media` and `media-cpu` extras conflict; on Linux the latter uses the explicit CPU PyTorch index. QuickJS independently publishes `deepagents>=0.7.40,<0.8.0` while mapping the local SDK for checkout development. A green editable environment therefore does not prove released-package resolution.

## Manifest and lockfile changes

Keep three layers distinct:

1. **Published metadata** — `[project]` requirements, extras, and `requires-python` define downstream installation compatibility.
2. **Checkout wiring** — `[tool.uv.sources]` substitutes sibling paths during repository development; it does not change wheel metadata.
3. **Generated resolution** — a package-local `uv.lock` records the concrete development/CI resolution. Regenerate it with `uv`; never hand-edit it.

After changing a manifest, refresh from the repository root with the changed manifest path:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

The refresher identifies the manifest owner and the transitive reverse-consumer closure of local `uv` path sources, then locks every affected package with its selected interpreter. Commit every printed `uv.lock` path. Next check either those affected package paths or all locks:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

When paths are supplied, the checker deliberately checks only packages touched by those paths; it does not calculate the reverse-consumer closure. With no arguments it discovers and validates every package lockfile, which is also the reusable CI workflow route. An out-of-date lock error prints an executable non-check `uv lock` repair command.

The targeted scripts use Python 3.14 for ACP, dcode, evals, and Talon and 3.12 for other packages. The `libs/` fan-out `lock-check` instead uses 3.14 for dcode, evals, and Talon and 3.12 for everything else, so it is not CI-equivalent for ACP locks. Prefer the checker for CI parity.

For local focused validation, ACP and Talon `make test` disable network sockets while allowing Unix sockets. Talon's test target first executes the WhatsApp bridge Node tests. Dcode's unit-test target is likewise socket-disabled, while `make integration_test` runs the integration-test directory without that socket restriction.

## Release handoff and completion

Release Please creates or updates a separate draft release PR for each managed package from conventional commits on `main`. Merging that PR triggers the publish and GitHub-release workflow. Consult `.github/RELEASING.md` and [Development and Release Operations](./operations/development.md) for the package-specific release procedure; do not defer a dcode SDK-pin update until release time.

Before submitting a focused change:

1. Inspect the owner package's instructions, manifest, Makefile, source, and nearest regression test.
2. Run `uv sync --all-groups`, change the public-contract, state, or lifecycle owner, and run the narrowest behavioral test plus `make lint`.
3. For a manifest change, review wheel requirements separately from `uv` source wiring; refresh and check the complete reverse-consumer lock closure; then test an immediate compatibility consumer.
4. For runtime behavior or release operations, use the linked architecture, integration, testing, or operations page rather than broadening the patch by guesswork.
