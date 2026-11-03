---
type: maintainer routing guide
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents monorepo change to its owning independently released package, run the relevant focused validation, and continue to the detailed runtime, integration, testing, or release guide.
tags: [deepagents, maintenance, routing, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-03T08:04:09.572Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
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
generated: { by: "openwiki/0.4.2", at: "2026-11-03T08:04:09.572Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the published contract or runtime behavior—not necessarily the product where a symptom appears. This is a monorepo of independently versioned packages under `libs/`; each package owns its `pyproject.toml`, `Makefile`, source, tests, and README. Use `make help` in that package rather than assuming that commands or Python versions are shared.

## Route the change

| Change concerns… | Owner and boundary | Start validation | Read next |
| --- | --- | --- | --- |
| Agent assembly, middleware, backends, tools, permissions, subagents, skills, or graph state | `libs/deepagents/`: reusable SDK | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Architecture overview](./architecture/overview.md), [source map](./architecture/source-map.md), [profiles and models](./concepts/profiles-models.md) |
| The terminal product: CLI/TUI, local server, headless session, workspace, configuration, persistence, plugins, MCP, or dcode ACP specialization | `libs/code/`: `deepagents-code`, with `dcode` as a console entry point | `make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make lint` | [Run a dcode session](./workflows/run-dcode-session.md), [MCP](./integrations/mcp.md), [testing guide](./testing/testing-guide.md) |
| Generic editor-facing Agent Client Protocol graph hosting or protocol translation | `libs/acp/`: ACP adapter | `make test TEST_FILE=tests/<nearest test>.py` | [ACP integration](./integrations/acp.md) |
| Long-running local hosting: channels, schedules, approvals, history, or managed MCP | `libs/talon/`: experimental host consuming SDK and dcode | `make test TEST_FILE=tests/<nearest test>.py` | [Talon integration](./integrations/talon.md), [testing guide](./testing/testing-guide.md) |
| Evaluation harnesses, real-model trajectory evaluation, Harbor, or result reporting | `libs/evals/`: evaluation boundary, not a serving runtime | Its `make help`, then the nearest test/evaluation seam | [Architecture overview](./architecture/overview.md), [testing guide](./testing/testing-guide.md) |
| QuickJS REPL/prompt snapshots or a sandbox/provider implementation | `libs/partners/quickjs/` or `libs/partners/<provider>/` | QuickJS: `make test TEST_FILE=tests/unit_tests/<nearest test>.py`; otherwise use that package's `make help` | [Sandbox and partner integrations](./integrations/sandbox-partners.md) |
| Dependency metadata, sibling checkout mappings, a lockfile, or a managed release | Affected package manifest and lockfile, plus reverse local consumers | Package validation, then the documented lock closure/check | [Development, dependencies, lockfiles, and releases](./operations/development.md) |

The core relationships explain the usual follow-up work: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; evals consumes the SDK, dcode, and QuickJS; and Talon consumes the SDK and dcode. Test the owner first, then test the consumer seam when an interface crosses one of those boundaries. QuickJS remains a partner package even when a visible effect appears in dcode or evals.

## Work package-first

Use `uv` for interpreters, environments, and dependencies. `uv` provisions the applicable interpreter, so do not add a repository-wide Python pin; each package declares its own supported range. Use the owning Makefile as the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/` and reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The normal SDK, dcode, ACP, Talon, and QuickJS test targets disable network sockets while allowing Unix sockets. Talon's `make test` runs WhatsApp bridge Node tests before its Python tests.

SDK and dcode accept `TEST_FILE` for focused test selection. dcode's `make check` adds its local CI-parity gates. QuickJS defaults `make test` to unit tests, accepts `TEST_FILE`, and provides `make update_snapshots` for intentional prompt-smoke snapshot changes. Consult [Testing Guide](./testing/testing-guide.md) for the behavioral seams to cover rather than expanding a narrow regression into a broad suite by default.

## Dependencies, locks, and compatibility

Keep these layers separate:

- **Published metadata**—`[project.dependencies]`, extras, and `requires-python`—is the installer-visible contract.
- **Checkout wiring**—`[tool.uv.sources]`—maps sibling sources for monorepo development and does not change wheel metadata.
- **Resolution**—the package-local `uv.lock`—is generated for that project; it is not a repository-wide environment.

Local editable sources can hide an inadequate published requirement. In particular, `deepagents-code` requires `deepagents==0.7.32`; raise that exact pin in the same pull request that first relies on newer SDK behavior. After a manifest change, regenerate the owner lock and its reverse local-source consumers, then use `make -C libs lock-check` as additional fan-out validation. The [development guide](./operations/development.md) documents the closure-aware command and release safeguards.

## CI and release orientation

On pull requests, CI selects lint and test jobs from package path filters. SDK changes select dependent package work, and relevant CI workflow or composite-action changes are included in package filters. For an SDK-only pull request that does not change QuickJS, CI additionally runs QuickJS system-prompt smoke tests against the editable SDK; pushes and QuickJS-changing pull requests run the full QuickJS suite.

Release Please manages separate draft release pull requests for the SDK, ACP, dcode, Talon, and listed partner packages. Merging a reviewed managed release PR triggers the release workflow that creates the PyPI and GitHub releases. Adding a partner is therefore repository wiring as well as package work: coordinate CI selection, labels, release configuration, secrets, and integration coverage.

## Continue in the detailed guide

- **Runtime layering, package boundaries, and ownership:** [Repository Architecture Overview](./architecture/overview.md)
- **Locate the responsible implementation area:** [Architecture Source Map](./architecture/source-map.md)
- **dcode interactive, headless, and ACP execution:** [Run a dcode Session](./workflows/run-dcode-session.md)
- **Protocol and provider boundaries:** [ACP](./integrations/acp.md), [MCP](./integrations/mcp.md), and [Sandbox and Partner Integrations](./integrations/sandbox-partners.md)
- **Focused regression and integration coverage:** [Testing Guide](./testing/testing-guide.md)
- **Setup, lockfile closure, and managed publication:** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
