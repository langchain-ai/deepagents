---
type: maintainer routing guide
title: Repository Quickstart
description: Route Deep Agents monorepo changes to the independently released package and its owning runtime, integration, testing, or release guide. Use package-local uv and Make workflows while keeping published requirements, checkout source mappings, and lockfiles distinct.
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
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
verified:
  - by: openwiki/0.4.2
    at: 2026-11-18T08:03:59.609Z
generated: { by: "openwiki/0.4.2", at: "2026-11-18T08:03:59.609Z" }
---

# Repository Quickstart

Start in the package that owns the public contract, persistent state, or process lifecycle—not merely where a symptom appears. This is a monorepo of independently versioned packages under `libs/`; each owns its `pyproject.toml`, `Makefile`, README, source, tests, and lockfile. The release manifest tracks separate baselines for the SDK, ACP, dcode, Talon, and the partner distributions.

Use `uv` for the interpreter, environment, and dependencies, and the package-local `Makefile` for normal commands. `uv` provisions the interpreter required by that package; do not add a repository-wide Python pin or shared virtual environment.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` before assuming a target exists. Install dependencies explicitly with `uv sync`; reserve `uv run ...` for direct, one-off commands. The package's local guidance, Makefile, and nearest regression test override conventions inferred from another package.

## Route the change

| If the change concerns… | Own it here first | Continue in the owning guide | Begin validation with |
| --- | --- | --- | --- |
| SDK graph assembly, middleware, backends, profiles, filesystem tools, permissions, or SDK subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md), [backends](./concepts/backends.md), or [source map](./architecture/source-map.md) | Nearest SDK unit test, then `make lint` |
| dcode CLI/TUI, local server, headless execution, session output/status, workspace policy, approvals, or model configuration | `libs/code/` | [Deep Agents Code Architecture](./architecture/code-agent.md), [Run a dcode Session](./workflows/run-dcode-session.md), and [Profiles and Models](./concepts/profiles-models.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| dcode hook configuration, event selection, command effects, interrupts/resumption, or transcript behavior | `libs/code/` | [dcode Hook Lifecycle](./concepts/hooks-lifecycle.md) | The closest hook lifecycle regression test and dcode lint |
| ACP graph-to-protocol adaptation or dcode ACP mode | `libs/acp/`; inspect dcode when it launches or configures ACP | [Agent Client Protocol integration](./integrations/acp.md) | Nearest ACP test through `make test` |
| Talon channels, runtime graph/MCP reload, schedules, detached work, archives, or history storage | `libs/talon/` | [Talon runtime integration](./integrations/talon.md), [Talon scheduling](./concepts/talon-scheduling.md), or [Talon history storage](./concepts/talon-history-storage.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| dcode MCP discovery, server configuration, trust, OAuth, reconnection, or tool loading | `libs/code/` | [MCP Integration](./integrations/mcp.md) | The nearest MCP regression test |
| QuickJS, Daytona, Modal, Runloop, or Vercel sandbox behavior | `libs/partners/<provider>/` (and dcode when it loads the integration) | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | `make help`, then the owning target |
| Evaluation definitions, datasets, Harbor execution, or reports | `libs/evals/` | [Run Evals](./workflows/run-evals.md) and [Testing guide](./testing/testing-guide.md) | `make help`, then the closest owning target |
| Action-facing dcode distribution or GitHub Action metadata | Root action files and `libs/code/` | [GitHub Action Integration](./integrations/github-action.md) | The action's focused checks |
| Dependency floors/pins, Python ranges, extras, local source mappings, locks, release labels, or release automation | Changed manifest/package, then its local-source consumers | [Development and Release Operations](./operations/development.md) and [CI Gates and Labels](./operations/ci-labeling.md) | Refresh the lock closure, verify every refreshed package, and test a direct consumer |

The [Testing guide](./testing/testing-guide.md) provides behavior-specific suites. As repository convention, deterministic network-free tests belong in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; warnings not explicitly accepted fail pytest. Follow the package's actual layout where it differs.

## Package and dependency boundaries

`deepagents` is the reusable SDK; ACP adapts an agent graph to the Agent Client Protocol; `deepagents-code` is the terminal coding agent launched by `dcode`; Talon is an experimental host for long-running channels and schedules; evals is the evaluation and Harbor boundary; and partner packages carry provider or sandbox integrations. Use the linked pages for runtime semantics rather than treating this routing map as a runtime reference.

The release manifest baselines are `deepagents` 0.7.39, `deepagents-acp` 0.0.13, `deepagents-code` 0.2.10, `deepagents-talon` 0.1.0, and, respectively, Daytona 0.0.9, Modal 0.0.8, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.3.12. A package's `pyproject.toml`, not this baseline list, is the installer-visible compatibility contract.

The SDK and ACP support Python `>=3.11,<4.0`; dcode, Talon, and evals require Python `>=3.14,<3.15`. dcode publishes an exact `deepagents==0.7.39` requirement and uses editable local mappings for sibling packages during checkout development. Talon publishes `deepagents>=0.7.39` and `deepagents-code>=0.2.9,<1.0.0`, while mapping its SDK and dcode dependencies to sibling checkouts. Therefore, a green editable checkout does not by itself prove that a released installation resolves the intended published versions.

Talon's history embedding adapters, MongoDB/PostgreSQL storage, media, and OpenShell support are extras, rather than base-host dependencies. Its `media` and `media-cpu` extras conflict; on Linux the latter resolves PyTorch packages from the CPU index. QuickJS also publishes an SDK range independently of its editable checkout mapping. Treat a compatible SDK or dcode change as a manifest, lockfile, and consumer-validation change when a consumer depends on it.

## Safely change a manifest or lockfile

Keep these layers separate:

1. **Published metadata** — `[project]` dependencies, optional extras, and `requires-python` define downstream installation compatibility.
2. **Checkout wiring** — `[tool.uv.sources]` substitutes sibling paths for repository development; it does not alter what is published.
3. **Generated resolution** — package-local `uv.lock` records the concrete development/CI resolution. Regenerate it through `uv`; do not edit it manually.

When a manifest changes, give `refresh_lockfiles.py` the changed manifest path. It determines the owning package plus transitive reverse consumers that refer to local paths, then locks each with its selected interpreter:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

The helper prints each refreshed `uv.lock` path. Pass all of those package directories to the targeted checker, or run the no-argument CI-equivalent check:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

With supplied paths, the checker deliberately validates only packages those paths touch; it does not recalculate the reverse-consumer closure. With no paths, it discovers every package and checks every lockfile. An out-of-date lock failure includes an executable non-check `uv lock` repair command. The reusable CI workflow takes the no-path route.

The scripts select Python 3.14 for ACP, dcode, evals, and Talon, and 3.12 for other discovered packages. Do not use `make -C libs lock-check` as CI parity for dcode, evals, or Talon: the `libs` fan-out Makefile maps only ACP to 3.14 and all other packages to 3.12.

For focused local validation, ACP and Talon `make test` disable network sockets while allowing Unix sockets. Talon first runs the WhatsApp bridge Node tests, then its Python suite. dcode's unit target is also socket-disabled, whereas its `integration_test` target is the network-permitted route.

## Release and CI handoff

Release-please manages a separate draft release PR for each managed package; it analyzes conventional commits on `main`, and merging that release PR triggers the publish and GitHub-release workflow. The managed package table and release troubleshooting live in `.github/RELEASING.md`; use [Development and Release Operations](./operations/development.md) for the maintainer procedure and [CI Gates and Labels](./operations/ci-labeling.md) for scope, lockfile, and label gates. If dcode needs a newly released SDK capability, update its exact SDK pin in the feature PR rather than deferring it to release time.

## Finish a focused change

1. Use the routing table, then inspect the owner package's instructions, manifest, Makefile, source, and nearest regression test.
2. Run `uv sync --all-groups` inside that package and make the change at the public-contract, state, or lifecycle owner.
3. Run the narrowest behavioral test and `make lint`; add deterministic coverage only when it protects observable behavior.
4. If a manifest changed, review the published requirement separately from local `uv` source wiring, refresh the reverse-consumer lock closure, verify every refreshed lock, and test the immediate compatibility consumer.
5. For a release or CI change, follow the linked operations guide; for runtime behavior, use the linked architecture, lifecycle, integration, or testing page rather than broadening the change by guesswork.
