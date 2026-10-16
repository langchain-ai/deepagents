---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: A compact routing map for maintaining the independently released Deep Agents packages. Use it to identify the behavior owner, dependency direction, and first focused validation for a change.
tags: [deepagents, dcode, acp, talon, maintenance, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-16T08:05:35.701Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-3cf9b023d0ac578794148a01
    resource: repo://.github/scripts/checks/refresh_lockfiles.py
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-18f01ea5159b63661c1c8b1c
    resource: repo://libs/acp/Makefile
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-11d6c59d85493653aee76558
    resource: repo://libs/code/tests/unit_tests/test_app.py
  - id: openwiki-source-a22e72eeda1efb40d5250020
    resource: repo://libs/code/tests/unit_tests/test_dep_floor_check.py
  - id: openwiki-source-4f80e7af3e8da661224fe1a4
    resource: repo://libs/code/tests/unit_tests/tui/widgets/test_auto_mode_notice.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-10-16T08:05:35.701Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the behavior, not the product where it appears. `libs/` contains independently versioned SDK, dcode, ACP, Talon, evals, and partner packages. The current release baselines are `deepagents` **0.7.26**, `deepagents-acp` **0.0.13**, `deepagents-code` **0.1.85**, and `deepagents-talon` **0.0.9**. Release-managed partners are Daytona 0.0.8, Modal 0.0.6, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.3.9; evals is a behavioral-test package rather than a Release Please component.

For the package graph and runtime boundaries, read [Repository Architecture Overview](./architecture/overview.md). For implementation ownership, use [Source Map and Ownership Boundaries](./architecture/source-map.md).

## Route the change

| Change | Behavior owner and dependency direction | Read next | First focused validation |
| --- | --- | --- | --- |
| Reusable graph assembly, middleware order, filesystem policy, skills, state, or subagents | `libs/deepagents/` is the SDK owner. ACP and dcode consume it; Talon consumes both the SDK and dcode. | [SDK Construction and Execution](./architecture/sdk-construction-execution.md) | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_graph.py`, then the narrow middleware/backend test named in the source map. |
| Terminal client/server behavior, Textual UI, sessions, commands, models, or Auto/YOLO approvals | `libs/code/` owns dcode and exactly pins the SDK. Talon is a downstream dcode consumer. | [Code Agent Architecture](./architecture/code-agent.md) and [Profiles and Models](./concepts/profiles-models.md) | Run the nearest `tests/unit_tests/` file through `make test TEST_FILE=...`; follow with `make lint`. For first-enable Auto UI, begin with `test_app.py` and `tui/widgets/test_auto_mode_notice.py`. |
| Editor protocol, custom graph hosting, session replay, or `dcode --acp` | `libs/acp/` owns the ACP integration and consumes the SDK. | [Agent Client Protocol](./integrations/acp.md) | `cd libs/acp && make test TEST_FILE=tests/<affected_test>.py`; also validate dcode when its ACP mode changes. |
| Channel adapters, schedules, runtime tools, MCP, sandboxing, or host lifecycle | `libs/talon/` is the experimental runtime host downstream of both dcode and the SDK. | [Talon Runtime Host](./integrations/talon.md) | `cd libs/talon && make test TEST_FILE=tests/<affected_test>.py`; this runs WhatsApp bridge Node tests first. |
| Provider sandbox, QuickJS, or a provider package | `libs/partners/<provider>/` owns the integration; dcode uses partner packages through editable local sources. | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | Run that partner's tests, then affected consumer and applicable Harbor/integration coverage. |
| Real-model quality, trajectories, scoring, or a sandbox benchmark | `libs/evals/` owns behavioral evaluation, not a release artifact. | [Testing Guide](./testing/testing-guide.md) | Add deterministic regression coverage first; then use `make evals MODEL=<id>` or the relevant Harbor job. |
| Dependency bounds, editable consumers, lockfiles, CI, or release automation | The changed manifest owns its own lock; `.github/` automation determines the consumer closure and delivery rules. | [Development, Dependencies, and Releases](./operations/development.md) | Re-lock the changed package and its local consumers, then use `cd libs && make lock-check`. For dcode dependency floors, start with `tests/unit_tests/test_dep_floor_check.py`. |

## Keep cross-package changes moving in the right direction

`create_deep_agent` is the public SDK graph-assembly seam. A caller middleware with the same name replaces the base entry in place, but harness profiles cannot remove protected filesystem or synchronous-subagent scaffolding. Put reusable agent behavior in the SDK rather than reproducing it in dcode, ACP, or Talon.

The hosts are deliberately separate: dcode is the terminal product; ACP can serve either a custom Deep Agent through `AgentServerACP` or the prebuilt dcode product over stdio; Talon is an experimental local runtime host. Do not transfer UI, lifecycle, approval, or security assumptions between hosts.

`deepagents-code` pins `deepagents==0.7.26`. Any dcode change that requires newer SDK behavior must update this exact pin in the same pull request. In local development, sibling dependencies are editable, so an interface edit reaches consumers immediately; CI schedules dcode for SDK changes and Talon for SDK or dcode changes. Each partner independently owns its environment, manifest, Makefile, tests, and version; adding one also needs CI, label, release, and applicable Harbor/integration wiring.

## Work and validate

Use `uv` for interpreters, environments, and dependencies, and use the package `Makefile` as the command authority. `uv` provisions the appropriate interpreter, so do not impose a repository-wide Python pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_graph.py
make lint
```

Keep ordinary tests deterministic and network-free in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. ACP and Talon's normal `make test` targets disable network sockets; Talon's target first runs its WhatsApp bridge Node tests. Evals is intentionally different: it runs agents against real LLMs, captures trajectories, scores correctness and efficiency, and integrates Harbor for sandboxed benchmarks. `make evals MODEL=<id>` runs `tests/evals`.

For dcode's Auto-mode disclosure, retain focused coverage that confirms the confirmation modal precedes activation, identifies the separate classifier model, and persists the notice only after confirmation. The notice-body unit test also protects Markdown escaping of the configured model label. Its editable-install dependency-floor tests must cover both skipping released installs and warning, without exit, for an editable environment below a declared floor.

### Dependency and lockfile changes

A manifest change can invalidate its package lockfile and lockfiles of transitive editable local consumers. The refresh helper finds consumers from `[tool.uv.sources]`, resolves every affected package using its required Python version, and refreshes its lock. The check script runs `uv lock --check` only for packages touched by the supplied paths (or every package if none are supplied).

```bash
# From the repository root: refresh a changed manifest and its editable consumers.
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml

# From libs/: verify every package lockfile.
make lock-check
```

## Release handoff

Release Please creates separate release PRs for the SDK, ACP, dcode, Talon, and partner packages. Review the generated changelog. For curated notes, keep the PR draft while changes accumulate, mark it ready, review and edit the bot draft, then comment `@release-bot apply`; use `ci:skip-curated-notes` only when intentionally publishing the generated changelog without that review.

When a change crosses a boundary, continue in the detailed guide rather than expanding this map:

- **Find an SDK owner or focused regression test:** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **Change SDK composition or public behavior:** [SDK Construction and Execution](./architecture/sdk-construction-execution.md)
- **Change models, profiles, permissions, state, or delegation:** [Concepts](./concepts/index.md)
- **Change ACP, Talon, partners, sandboxes, or MCP:** [Integrations](./integrations/index.md)
- **Change setup, dependencies, locks, or delivery:** [Development, Dependencies, and Releases](./operations/development.md)
- **Choose wider or real-provider coverage:** [Testing Guide](./testing/testing-guide.md)
