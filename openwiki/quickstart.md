---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning independently packaged component, apply the published-versus-local dependency boundary, and run focused package validation. Links point to the deeper architecture, integration, evaluation, development, and testing guidance.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-27T08:05:40.296Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
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
  - id: openwiki-source-6985df297a9bf809e10bc5d8
    resource: repo://libs/evals/uv.lock
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-966f1489233b6fc6ee5f2e4c
    resource: repo://libs/talon/uv.lock
  - id: openwiki-source-482fa4ca84f42b04ba025fc1
    resource: repo://release-please-config.json
generated: { by: "openwiki/0.4.2", at: "2026-10-27T08:05:40.296Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not necessarily the product where the symptom appears. The independently versioned packages under `libs/` include the `deepagents` SDK, `deepagents-code` terminal product (run as `dcode`), ACP integration, evaluation suite, experimental Talon host, and partner integrations. Each package owns its `pyproject.toml`, `Makefile`, and README: treat its manifest, local source/tests, and Makefile as authoritative rather than this routing page.

Use [Repository Architecture Overview](./architecture/overview.md) for the runtime and package graph, [System Source Map](./architecture/source-map.md) to locate lifecycle owners and representative tests, [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) for release procedure, and the [Testing Guide](./testing/testing-guide.md) for focused test seams.

## Route the change

| Change or question | Owner and narrowest first validation | Consult next |
| --- | --- | --- |
| Reusable agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/`; run `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/<nearest test>.py`. | [SDK architecture](./architecture/sdk-construction-execution.md); [Testing Guide](./testing/testing-guide.md) |
| dcode CLI/TUI, local server/session behavior, workspaces, MCP, plugins, persistence, configuration, or headless mode | `libs/code/`; run `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py`. | [Deep Agents Code Architecture](./architecture/code-agent.md); [Plugins](./concepts/plugins.md); [MCP](./integrations/mcp.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/`; run `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py`; validate dcode too when its ACP specialization changes. | [Agent Client Protocol Integration](./integrations/acp.md) |
| Talon channels, pairing, schedules, host runtime, approvals, or history | `libs/talon/`; run `cd libs/talon && make test TEST_FILE=tests/<nearest test>.py`. | [Talon integration](./integrations/talon.md); [Testing Guide](./testing/testing-guide.md) |
| Daytona, Modal, Runloop, Vercel, QuickJS, or another provider/sandbox boundary | The matching `libs/partners/<provider>/`, followed by the consuming product seam. | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Deterministic evaluation plumbing, behavioral evaluations, Harbor, or results | `libs/evals/`; run its package-local unit test before a model-driven evaluation. | [Run Evals](./workflows/run-evals.md); [Testing Guide](./testing/testing-guide.md) |
| Dependency constraints, locks, CI, or releases | The affected package manifest and lockfile. Start with its local check; use `cd libs && make lock-check` when the lock closure spans packages. | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |

After a focused test, run the owning package's `make lint`; run `make help` rather than assuming every package has identical targets. For dcode, `make check` is the package-local CI-parity gate: it runs lint, import checks, unit tests, extras/version/lockfile checks, and reports a stale workspace SDK pin without making that local check fail.

## Published contract versus checkout wiring

A `[project.dependencies]` entry is the constraint shipped in the released wheel. A `[tool.uv.sources]` entry only changes how a checkout resolves that named dependency locally; it is not a released compatibility promise. Review both before changing a cross-package interface.

- **ACP → SDK:** ACP publishes a `deepagents` dependency and maps it to the editable `../deepagents` checkout.
- **dcode → SDK, ACP, and partners:** `deepagents-code` publishes the exact `deepagents==0.7.32` requirement, a compatible `deepagents-acp` range, and its runtime dependencies. Its local uv sources make the SDK, ACP, and listed partner packages, including QuickJS, editable siblings. Raise the exact SDK pin in the same pull request that first requires newer SDK behavior; a local checkout must not conceal an unreleased compatibility change.
- **Evals → SDK, dcode, and QuickJS:** `deepagents-evals` publishes lower-bounded SDK and dcode requirements. Locally it maps the SDK and QuickJS as editable siblings, while its dcode source is a sibling directory source rather than an editable mapping. Its lockfile therefore records the dcode checkout as part of the evaluation resolution.
- **Talon → SDK and dcode:** Talon publishes lower-bounded requirements for both and maps both to editable siblings. Its dcode extras also resolve through that local source, so dcode distribution metadata can require a Talon lock refresh.

When a manifest changes, change and test the contract owner first, then inspect every consumer whose source mapping or lock resolves that package. For a dcode dependency or extra change, that includes the dcode lock and the evals and Talon locks; run the closest consumer test when behavior crosses the boundary. Finish with `cd libs && make lock-check` if the complete affected lock closure is uncertain.

## Package-local loop and test boundary

Use `uv` for interpreters, environments, and dependencies, and package Makefiles as the command authority. `uv` provisions the applicable interpreter automatically, so there is no repository-wide Python version to install or pin; packages declare their supported ranges.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. SDK, dcode, ACP, and Talon standard test targets disable network sockets while allowing Unix sockets; Talon's target first runs the WhatsApp bridge Node tests. Use the [Testing Guide](./testing/testing-guide.md) to select a focused test and escalation boundary; use [Run Evals](./workflows/run-evals.md) rather than treating model-backed measurements as ordinary unit tests.

## Release handoff

Release Please manages separate release components for the SDK, ACP, dcode, Talon, and the Daytona, Modal, Runloop, Vercel, and QuickJS partners. The evals package is independently versioned but is not a Release Please-managed component. Release Please opens separate draft release PRs; merging a managed release PR triggers checks, PyPI publication, and GitHub release creation. Keep a behavior change in one releasable component where possible, and move dependent package/lockfile fan-out to the planned dependency handoff described in the release guide.

## Continue by question

- **Which layer owns this behavior or package boundary?** [Repository Architecture Overview](./architecture/overview.md)
- **Where are the entry point, lifecycle owner, and representative test?** [System Source Map](./architecture/source-map.md)
- **How do I change dcode extensions?** [Plugins](./concepts/plugins.md) and [MCP Integration](./integrations/mcp.md)
- **Which sandbox or provider package is involved?** [Sandbox and Partner Integrations](./integrations/sandbox-partners.md)
- **How should I update dependencies, locks, or releases?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Which focused validation or evaluation fits this change?** [Testing Guide](./testing/testing-guide.md) and [Run Evals](./workflows/run-evals.md)
