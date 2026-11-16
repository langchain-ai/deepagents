---
type: maintainer routing guide
title: Repository Quickstart
description: Route Deep Agents monorepo work to the package that owns the contract, then use its local uv and Make workflow. Follow the published-metadata, local-source, and lockfile boundaries when changing dependencies.
tags: [deepagents, dcode, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-16T08:11:13.198Z
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
generated: { by: "openwiki/0.4.2", at: "2026-11-16T08:11:13.198Z" }
---

# Repository Quickstart

Start in the package that owns the public contract, persistent state, or process lifecycle—not merely where a symptom appears. This is a monorepo of independently versioned packages under `libs/`; each owns its `pyproject.toml`, `Makefile`, README, source, tests, and lockfile. The release manifest keeps separate baselines for the SDK, ACP, dcode, Talon, and partner distributions.

Use `uv` for the interpreter, environment, and dependencies, and the package-local `Makefile` for normal commands. `uv` provisions the interpreter required by that package; do not add a repository-wide Python pin or shared virtual environment.

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Run `make help` before assuming a target exists. Install dependencies explicitly with `uv sync`; reserve `uv run ...` for direct, one-off commands. The package's local guidance, Makefile, and nearest regression test override conventions inferred from another package.

## Route the change

| If the change concerns… | Own it here first | Read next | Begin validation with |
| --- | --- | --- | --- |
| SDK graph assembly, middleware, backends, profiles, filesystem tools, permissions, or SDK subagents | `libs/deepagents/` | [Architecture overview](./architecture/overview.md) | Nearest SDK unit test, then `make lint` |
| Context compaction, checkpoints, subagents, profiles, or filesystem semantics | `libs/deepagents/` | [Profiles and models](./concepts/profiles-models.md) and [Testing guide](./testing/testing-guide.md) | Focused behavior test named by the local source/test layout |
| dcode CLI/TUI, local server, headless run, workspace policy, model catalog, approvals, or output | `libs/code/` | [Deep Agents Code Architecture](./architecture/code-agent.md) and [Profiles and models](./concepts/profiles-models.md) | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` |
| ACP graph-to-protocol adaptation or dcode ACP mode | `libs/acp/`; inspect dcode when it launches or configures ACP | [Agent Client Protocol integration](./integrations/acp.md) | Nearest ACP test through `make test` |
| Talon channel host, runtime graph/MCP reload, schedules, detached work, archives, or history storage | `libs/talon/` | [Testing guide](./testing/testing-guide.md) | `make test TEST_FILE=tests/<nearest test>.py`, then `make lint` |
| Evaluation definitions, datasets, Harbor execution, or reports | `libs/evals/` | [Development and release operations](./operations/development.md) | `make help`, then the closest owning target |
| Daytona, Modal, QuickJS, Runloop, or Vercel sandbox behavior | `libs/partners/<provider>/` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) | `make help`, then the owning target |
| Dependency floors/pins, Python ranges, extras, local source mappings, locks, or release baselines | Changed package manifest, then its local-source consumers | [Architecture overview](./architecture/overview.md) and [Development and release operations](./operations/development.md) | Refresh the lock closure, verify every refreshed package, and test a direct consumer |

The [Testing guide](./testing/testing-guide.md) provides behavior-specific suites. As the repository convention, deterministic network-free tests belong in `tests/unit_tests/` and networked contracts in `tests/integration_tests/`; warnings not explicitly accepted fail pytest. Follow the package's actual layout where it differs.

## Package boundaries that change the plan

`deepagents` is the reusable SDK; ACP adapts an agent graph to the Agent Client Protocol; `deepagents-code` is the terminal coding agent launched by `dcode`; Talon is an experimental host for long-running channels and schedules; evals is the evaluation and Harbor boundary; and partner packages carry provider or sandbox integrations.

Dependency metadata is an installer-visible compatibility contract. The SDK supports Python `>=3.11,<4.0`; dcode and Talon support Python `>=3.14,<3.15`. dcode publishes an exact `deepagents==0.7.38` dependency and uses editable local mappings for the SDK, ACP, and partner packages during checkout development. QuickJS supports Python `>=3.11,<4.0`, publishes `deepagents>=0.7.38,<0.8.0`, and maps the SDK locally. A green editable checkout therefore does not prove a released installation resolves the intended published versions.

Talon publishes `deepagents>=0.7.38` and `deepagents-code>=0.2.8,<1.0.0`, while its local mappings point to the SDK and dcode checkouts. History embedding adapters, MongoDB/PostgreSQL storage, media, and the OpenShell backend are opt-in extras; media and `media-cpu` are declared mutually exclusive. Treat an SDK or dcode compatibility change that Talon consumes as a manifest, lockfile, and consumer-validation change—not just a source edit.

For focused local validation, ACP and Talon `make test` disable network sockets while allowing Unix sockets. Talon first runs the WhatsApp bridge Node tests, then runs Python tests with a 10-second timeout and coverage. dcode's unit target is also socket-disabled, whereas its `integration_test` target is the network-permitted route.

## Safe manifest and lockfile path

Keep these layers separate:

1. **Published metadata** — `[project]` dependencies, optional extras, and `requires-python` define downstream installation compatibility.
2. **Checkout wiring** — `[tool.uv.sources]` substitutes sibling paths for repository development; it does not alter what is published.
3. **Generated resolution** — package-local `uv.lock` records the concrete development/CI resolution. Regenerate it through `uv`; do not edit it manually.

When a manifest changes, give `refresh_lockfiles.py` the changed manifest path. It determines the owning package plus the transitive reverse consumers that refer to local paths, and locks each with its selected interpreter:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
```

The helper prints each refreshed `uv.lock` path. Pass **all** of those package directories to the targeted checker, or run the no-argument CI-equivalent check:

```bash
python .github/scripts/checks/check_lockfiles_pre_commit.py
```

With supplied paths, the checker deliberately validates only packages whose directories those paths touch—it does not recalculate the reverse-consumer closure. With no paths, it discovers every package and checks every lockfile. A stale lock failure includes an executable non-check `uv lock --directory <package> --python <version>` repair command. The reusable CI workflow takes the no-path route.

The lock scripts select Python 3.14 for ACP, dcode, evals, and Talon, and 3.12 for other discovered packages. Do not use `make -C libs lock-check` as CI parity for dcode, evals, or Talon: the `libs` fan-out Makefile maps only ACP to 3.14 and maps every other package to 3.12.

## Finish a focused change

1. Use the routing table, then inspect the owner package's instructions, manifest, Makefile, source, and nearest regression test.
2. Run `uv sync --all-groups` inside that package and make the change at the public-contract, state, or lifecycle owner.
3. Run the narrowest behavioral test and `make lint`; add deterministic coverage only when it protects observable behavior.
4. If a manifest changed, review the published requirement separately from local `uv` source wiring, refresh the reverse-consumer lock closure, verify every refreshed lock, and test the immediate compatibility consumer.
5. For releases and automation continue to [Development and release operations](./operations/development.md); for runtime behavior use the linked architecture, concept, integration, or testing guide rather than broadening the change by guesswork.
