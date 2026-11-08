---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning package and validation target. Includes the Talon manifest contract, optional extras, generated lockfile closure, and direct-consumer checks.
tags: [deepagents, talon, maintenance, dependencies, lockfiles, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-08T08:03:51.027Z
sources:
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
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
generated: { by: "openwiki/0.4.2", at: "2026-11-08T08:03:51.027Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime behavior—not necessarily the package where a symptom appears. Packages under `libs/` are independently versioned and own their `pyproject.toml`, `Makefile`, README, source, tests, and package-local `uv.lock`. Use `uv` for the package environment and its `Makefile` for normal validation; `uv` provisions the applicable interpreter, so do not impose a repository-wide Python pin.

## Route the task

| Change concerns… | Start in | First validation | Continue with |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, profiles, filesystem tools, or subagents | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [testing guide](./testing/testing-guide.md) |
| Terminal UI/CLI, agent server, workspace policy, or dcode product behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Development](./operations/development.md), [testing guide](./testing/testing-guide.md) |
| ACP session translation or dcode ACP mode | `libs/acp/` or the dcode seam | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Channels, schedules, host lifecycle, MCP, sandbox selection, history, or Talon manifest | `libs/talon/` | `make test TEST_FILE=tests/<nearest test>.py` | [Talon integration](./integrations/talon.md), [Talon history storage](./concepts/talon-history-storage.md) |
| Evaluations, Harbor, datasets, or reports | `libs/evals/` | `make help`, then the nearest evaluation target | [Testing guide](./testing/testing-guide.md) |
| Provider or sandbox implementation, including QuickJS | `libs/partners/<provider>/` | `make help` | [Architecture overview](./architecture/overview.md) |
| Package metadata, local sources, or generated locks | The changed package manifest, then its reverse local consumers | Refresh generated locks, check them, and test a direct consumer | [Development](./operations/development.md) |

Run `make help` in the selected package before assuming a target exists. Keep ordinary deterministic coverage network-free; put networked contracts in `tests/integration_tests/`. Unexpected pytest warnings fail the suite.

## Talon: choose the right environment

`deepagents-talon` is an experimental local host for long-running channels and schedules. It requires Python `>=3.14,<3.15`. Its base install is deliberately the host baseline: it includes the SDK (`deepagents>=0.7.34`), dcode (`deepagents-code>=0.2.5,<1.0.0`), SQLite checkpoint support, channel clients, MCP, and the LangChain/LangGraph libraries it imports. Base installation does **not** install every persistence, embedding, sandbox, or media capability.

For ordinary Talon implementation work, create the Talon environment and use its target:

```bash
cd libs/talon
uv sync --group test
make test TEST_FILE=tests/<nearest test>.py
make lint
```

The Talon test target first runs the WhatsApp bridge Node tests, then runs pytest with sockets disabled except Unix sockets, a 10-second timeout, and coverage. Do not bypass it with a raw pytest command when a bridge change or package-equivalent test behavior matters.

### Extras are capability choices, not base requirements

Choose only the extra that exercises the feature being changed or deployed:

| Capability | Extra | What it adds |
| --- | --- | --- |
| Local semantic-history embeddings | `history-local` | `sentence-transformers` and `transformers` |
| Voyage or OpenAI-compatible history embeddings | `history-voyage` or `history-openai` | The corresponding provider integration/client |
| Remote checkpoint/history persistence | `mongodb` or `postgres` | The matching LangGraph backend; PostgreSQL also installs `psycopg[binary]` |
| OpenShell sandbox | `openshell` | `deepagents-code[openshell]` |
| Media/transcription | `media` or `media-cpu` | dcode media support plus audio/ML dependencies |

For example, use `uv sync --group test --extra postgres` for PostgreSQL work, or `uv sync --group test --extra history-voyage` for the Voyage adapter. `media` and `media-cpu` are mutually exclusive. On Linux, `media-cpu` selects `torch` and `torchaudio` from Talon's explicit PyTorch CPU index; use it for GPU-less hosts and use `media` when CUDA-oriented PyPI wheels are intended. Extras add implementation dependencies; they do not provide credentials, create remote resources, or change Talon's experimental security boundary.

## Manifest change checklist

Treat three similarly named things as separate contracts:

1. **Published metadata** — `[project.dependencies]`, `[project.optional-dependencies]`, and `requires-python` are what downstream installers receive.
2. **Checkout wiring** — Talon's `[tool.uv.sources]` maps `deepagents` and `deepagents-code` to editable sibling directories for this monorepo. Those paths are not published wheel requirements.
3. **Generated resolution** — `libs/talon/uv.lock` is the concrete Python 3.14 development/CI resolution. It includes marker branches for the media extras and their conflict, but it is not a second manifest.

A local editable checkout can hide an insufficient published constraint. When changing a Talon dependency or extra, review the installer-visible requirement independently from local-source behavior. When changing the SDK or dcode contract, Talon is a direct local consumer: refresh its selected lock through the generated workflow and run Talon's focused test route in addition to the owner package's tests.

Use the extra-specific test area when one exists; otherwise add a focused, deterministic test at the behavior boundary. For history and persistence work, see [Talon Conversation History and Storage](./concepts/talon-history-storage.md); for MCP adapter behavior, see [Talon Runtime Integration](./integrations/talon.md).

## Refresh generated locks—never edit them by hand

A manifest edit can stale more than its own lock. The repository refresh helper begins from one or more changed manifests, discovers every reverse transitive consumer using `[tool.uv.sources]` paths, and runs `uv lock` for that closure with its checker-selected Python version. For Talon, that version is Python 3.14.

From the repository root, refresh from the actual changed manifest and commit every lockfile it reports:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/talon/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/talon/pyproject.toml
```

If an SDK or dcode manifest changed, start with that manifest instead; the refresh closure includes Talon because Talon maps both siblings as local sources:

```bash
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml
python .github/scripts/checks/check_lockfiles_pre_commit.py libs/code/pyproject.toml libs/talon/pyproject.toml
make -C libs/talon test TEST_FILE=tests/<nearest test>.py
```

The checker checks package directories named by supplied paths; with no paths it checks every discovered package. It prints the non-check `uv lock --directory … --python …` repair command when a lock is stale. `make -C libs lock-check` is a useful broad fan-out, but its Makefile maps only ACP to Python 3.14 and therefore is not the CI-equivalent lock check for Talon, dcode, or evals. Prefer the targeted refresher and checker for a Talon manifest change.

## Compact completion route

1. Select the owning package and synchronize that package's environment. For Talon, select extras only when the changed capability needs them.
2. Change published metadata, source wiring, or code only in the layer that owns it. Do not convert an optional Talon capability into a base dependency without an intentional product decision.
3. Run the narrowest owning test and `make lint`; Talon's target includes its Node bridge prerequisite.
4. For any manifest change, run `refresh_lockfiles.py` from the changed manifest, commit the complete reported lock closure, and run `check_lockfiles_pre_commit.py` with the affected package paths.
5. Test a direct consumer when a published or local-source dependency boundary changed—Talon for SDK/dcode changes; the consuming product or integration for a Talon boundary change.

For broader package ownership and dependency direction, see the [Repository Architecture Overview](./architecture/overview.md). For CI, release, and general lockfile operations, see [Development, Dependencies, Lockfiles, and Releases](./operations/development.md). For focused regression selection, see the [Testing Guide](./testing/testing-guide.md).
