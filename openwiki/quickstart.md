---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: A compact routing map for maintaining the independently released Deep Agents packages. Use it to find the behavior owner and run the first focused test for graph, UI, dependency, or lockfile changes.
tags: [deepagents, dcode, acp, talon, maintenance, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-15T08:05:41.400Z
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
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
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
generated: { by: "openwiki/0.4.2", at: "2026-10-15T08:05:41.400Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the behavior, not the product where it appears. `libs/` contains independently versioned SDK, dcode, ACP, Talon, evals, and partner packages. Current release baselines are `deepagents` **0.7.25**, `deepagents-acp` **0.0.13**, `deepagents-code` **0.1.85**, and `deepagents-talon` **0.0.9**. Release-managed partners are Daytona 0.0.8, Modal 0.0.6, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.3.9. Evals is a behavioral-test package rather than a Release Please component.

For the package graph and lifecycle boundaries, read [Repository Architecture Overview](./architecture/overview.md); for SDK implementation seams, read [Source Map and Ownership Boundaries](./architecture/source-map.md).

## Route the change

| Change | Owner | Read next | First focused validation |
| --- | --- | --- | --- |
| Reusable graph assembly, middleware order, filesystem policy, skills, state, synchronous or remote subagents | `libs/deepagents/` | [SDK Construction and Execution](./architecture/sdk-construction-execution.md) and [Source Map](./architecture/source-map.md) | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_graph.py` for assembly; use the source map's subagent or summarization test for those narrower seams. |
| dcode terminal client/server behavior, TUI, sessions, commands, model configuration, Auto/YOLO approvals | `libs/code/` | [Code Agent Architecture](./architecture/code-agent.md) and [Profiles and Models](./concepts/profiles-models.md) | For first-enable approval UI, start with `tests/unit_tests/test_app.py` and `tests/unit_tests/tui/widgets/test_auto_mode_notice.py`; otherwise select the nearest `tests/unit_tests/` test, then `make lint`. |
| Editor protocol, custom graph hosting, session replay, or `dcode --acp` | `libs/acp/` | [Agent Client Protocol](./integrations/acp.md) | Run the affected ACP test. Also run dcode coverage when changing its ACP mode. |
| Channel adapters, turns, cron work, runtime tools, sandboxing, or host lifecycle | `libs/talon/` | [Talon Runtime Host](./integrations/talon.md) | Run the nearest Talon test via `make test`; it runs WhatsApp bridge Node tests before Python tests. |
| Sandbox provider, QuickJS, or a new provider package | `libs/partners/<provider>/` | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | Run that partner's tests, then its affected consumers and applicable Harbor/integration coverage. |
| Real-model quality, trajectories, scoring, or sandbox benchmark | `libs/evals/` | [Testing Guide](./testing/testing-guide.md) | Add deterministic regression coverage first; then use `make evals MODEL=<id>` or the relevant Harbor job. |
| Dependency bounds, editable consumers, lockfiles, CI, or release automation | Changed `pyproject.toml`, `uv.lock`, and `.github/` automation | [Development, Dependencies, and Releases](./operations/development.md) | For a dcode dependency floor, start with `tests/unit_tests/test_dep_floor_check.py`; re-lock the changed package and its local consumers, then verify with `cd libs && make lock-check`. |

## Preserve the ownership boundaries

`create_deep_agent` is the public SDK graph-assembly seam. A caller middleware with the same name replaces the base entry in place, but harness profiles cannot remove protected filesystem or synchronous-subagent scaffolding. Put reusable agent behavior there; do not reproduce it in dcode, ACP, or Talon.

The hosts have separate responsibilities: dcode is the terminal product; ACP can serve either a custom Deep Agent through `AgentServerACP` or the prebuilt dcode product over stdio; and Talon is an experimental local runtime host. Do not transfer one host's UI, lifecycle, approval, or security assumptions to another. See the linked destination page before crossing one of these boundaries.

Each partner owns its environment, manifest, Makefile, tests, and version. Adding one also requires repository wiring for CI, labels, releases, and, where applicable, Harbor and integration tests.

## Work and validate across packages

Use `uv` for interpreters, environments, and dependencies; each package `Makefile` is the command authority. `uv` provisions the appropriate interpreter, so do not impose a repository-wide Python pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_graph.py
make lint
```

Local sibling dependencies are editable, so a shared-interface edit is visible to consuming packages immediately. Validate all affected consumers: CI deliberately schedules dcode and Talon coverage for SDK changes. dcode pins the SDK exactly at `deepagents==0.7.25`; a dcode change that needs newer SDK behavior must update that pin in the same pull request.

Keep ordinary tests deterministic and network-free in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unexpected pytest warnings fail the suite. ACP and Talon disable network sockets in their standard test targets. Evals intentionally differs: it runs agents against real LLMs, captures trajectories, scores correctness and efficiency, and uses Harbor for sandboxed benchmarks; `make evals` requires `MODEL` and runs `tests/evals`.

### Dependency and lockfile changes

A changed manifest can invalidate the package's lockfile and the lockfiles of transitive local consumers. `refresh_lockfiles.py` computes those consumers from editable `tool.uv.sources` mappings and locks each package with its required Python version. The required lockfile workflow checks affected packages with `uv lock --check`.

Use the smallest appropriate command first:

```bash
# From the repository root: refresh a changed manifest and its editable consumers.
python .github/scripts/checks/refresh_lockfiles.py libs/code/pyproject.toml

# From libs/: verify every package lockfile.
make lock-check
```

## Release handoff

Release Please manages separate release PRs for the SDK, ACP, dcode, Talon, and partner packages. Review the generated changelog; for curated notes, leave the PR draft until ready, review and edit the bot's draft, then use `@release-bot apply`. Use `ci:skip-curated-notes` only when intentionally publishing the generated changelog without that review.

When a change crosses a boundary, hand off to the detailed guide rather than expanding this map:

- **Find an SDK owner or focused regression test:** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **Change SDK composition or public behavior:** [SDK Construction and Execution](./architecture/sdk-construction-execution.md)
- **Change models, profiles, permissions, state, or delegation:** [Concepts](./concepts/index.md)
- **Change ACP, Talon, partners, sandboxes, or MCP:** [Integrations](./integrations/index.md)
- **Change setup, dependencies, locks, or delivery:** [Development, Dependencies, and Releases](./operations/development.md)
- **Choose wider or real-provider coverage:** [Testing Guide](./testing/testing-guide.md)
