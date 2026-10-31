---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route Deep Agents changes to the package that owns the behavior, dependency contract, and focused validation. Includes the QuickJS and Talon boundaries and package-local lockfile workflow.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-31T08:04:00.426Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-f05e6255636ce7459a743628
    resource: repo://examples/deep_research/uv.lock
  - id: openwiki-source-3ddcc3abc53dc97c931d911a
    resource: repo://examples/nvidia_deep_agent/uv.lock
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
  - id: openwiki-source-1315dc43c5577f5948632387
    resource: repo://libs/code/uv.lock
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-6985df297a9bf809e10bc5d8
    resource: repo://libs/evals/uv.lock
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-224af38a099019bb2a44f0a7
    resource: repo://libs/partners/quickjs/Makefile
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-10-31T08:04:00.426Z" }
---

# Deep Agents Monorepo Quickstart

Start in the package that owns the behavior or published contract, rather than the product where a symptom appears. `libs/` is a set of independently versioned packages: the `deepagents` SDK, `deepagents-code` (`dcode`), ACP, evals, the experimental Talon host, and provider integrations. Each package owns its `pyproject.toml`, `Makefile`, source, and tests; use its README and `make help` for package-specific details instead of assuming commands are uniform.

## Route the change

| Change affects | Start in | Focused validation | Read next |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Development guide](./operations/development.md), [Testing Guide](./testing/testing-guide.md) |
| dcode CLI/TUI, sessions, workspaces, configuration, persistence, plugins, MCP, or headless behavior | `libs/code/` | `make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Profiles and Models](./concepts/profiles-models.md), [Testing Guide](./testing/testing-guide.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/` | Use the package's narrowest supported test target | Validate dcode too when its ACP specialization changes |
| Talon channels, schedules, approvals, history, host runtime, managed MCP, or `/effort` | `libs/talon/` | `make test TEST_FILE=tests/unit_tests/test_effort.py` for effort behavior; otherwise the nearest test | [Talon Runtime Integration](./integrations/talon.md), [Testing Guide](./testing/testing-guide.md) |
| QuickJS REPL lifecycle, prompt/tool bindings, persistence, snapshots, or subagent guidance | `libs/partners/quickjs/` | `make test TEST_FILE=tests/unit_tests/test_repl_middleware.py`; use `make update_snapshots` only for accepted prompt-snapshot changes | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md), then the dcode/evals consumer seam |
| Another sandbox/provider integration | Matching `libs/partners/<provider>/` | Its `make help`, then the narrowest test target | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Evaluation harness, Harbor, behavioral evaluation, or results | `libs/evals/` | `make test` | [Testing Guide](./testing/testing-guide.md), [Development guide](./operations/development.md) |
| Dependency constraints, locks, CI, or releases | The affected package manifest and `uv.lock` | Package-local lock/test checks, then `cd libs && make lock-check` when fan-out is needed | [Development guide](./operations/development.md) |

QuickJS is a provider/REPL package, even when the visible behavior is in dcode or evals. Talon is the long-running host: changes to its operator command, model-effort translation, or persistence belong there, although it consumes the SDK and dcode. Follow the consumer seam after the owning package test when a change crosses that boundary.

## Work package-first

Use `uv` for interpreters, environments, and dependencies, and use the owning package's Makefile as the command source of truth. `uv` provisions the applicable Python interpreter, so do not introduce a repository-wide Python pin; honor the package's `requires-python`. A normal loop is:

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free tests in `tests/unit_tests/` and reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The SDK, dcode, QuickJS, ACP, and Talon standard test targets disable network sockets while allowing Unix sockets. Talon's `make test` also runs its WhatsApp bridge Node tests before pytest, including when `TEST_FILE` narrows the Python portion.

The focused boundary is package-specific: SDK and dcode accept `TEST_FILE`; dcode's `make check` adds local CI-parity work—lint, imports, unit tests, extras/version/lockfile checks—and reports a stale workspace SDK pin as advisory rather than failing solely for that report. QuickJS also accepts `TEST_FILE` and has a dedicated snapshot-update target. Run the package's `make lint` after the focused test.

## Dependencies and lockfiles

Read both manifest layers when a change crosses packages:

- **`[project.dependencies]` is the published contract.** Review its version constraints when deciding what a released package supports.
- **`[tool.uv.sources]` is local checkout wiring.** It replaces a distribution with a sibling path for local development and does not by itself publish an unreleased compatibility promise.

The main direction is ACP → SDK; dcode → SDK, ACP, and partner integrations (including QuickJS); evals → SDK, dcode, and QuickJS; and Talon → SDK and dcode. The local sources are mostly editable siblings; evals deliberately maps dcode by sibling path without `editable = true`. QuickJS maps the SDK editably, while Talon maps both SDK and dcode editably.

`deepagents-code` publishes an exact `deepagents==0.7.32` requirement. Raise that pin in the same pull request that first requires newer SDK behavior. For a manifest change, update the owning package's `uv.lock` in that package directory and validate it there. The `libs/` fan-out `make lock`, `make lock-bump DEP=<package>`, and `make lock-check` cover library packages and example projects with manifests; use the fan-out check when a dependency change can affect more than one lockfile. Do not treat a lockfile from dcode, evals, or an example as a repository-wide environment: their lock metadata records their own Python range and resolution markers.

## Release and extension boundaries

Release Please manages separate release components for the SDK, ACP, dcode, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partners. It opens a separate draft release PR for each managed package; merging the reviewed, passing release PR triggers the PyPI and GitHub releases. Evals is independently versioned but is not Release Please-managed.

A new partner is more than a package: it requires coordinated CI, labels, release configuration, and related repository wiring. Follow the partner guidance rather than copying an existing package in isolation.

## Continue by question

- **dcode capability profiles or model controls:** [Profiles and Models](./concepts/profiles-models.md)
- **QuickJS lifecycle, prompts, persistence, or sandbox boundary:** [Sandbox and Partner Integrations](./integrations/sandbox-partners.md)
- **Talon `/effort`, channels, runtime, or persistence:** [Talon Runtime Integration](./integrations/talon.md)
- **Dependencies, locks, CI, or releases:** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Representative focused tests and validation scope:** [Testing Guide](./testing/testing-guide.md)
