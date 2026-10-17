---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: A task-routing entry point for maintainers of the Deep Agents monorepo. Identify the package that owns a behavior, then follow the linked workflow, operations, architecture, or test guide.
tags: [deepagents, maintenance, dcode, talon, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-17T08:04:56.296Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
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
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-10-17T08:04:56.296Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract—not necessarily the product where a behavior is visible. `libs/` contains independently versioned packages for the `deepagents` SDK, dcode, ACP, evals, Talon, and provider integrations. **Package-local source, tests, `pyproject.toml`, and `Makefile` are authoritative**; this page is a routing map, not a substitute for them.

For the package graph, version lines, and runtime boundaries, start with [Repository Architecture Overview](./architecture/overview.md). Use [Source Map](./architecture/source-map.md) when the relevant entry point or focused regression test is unclear.

## Route the task

| If the change concerns… | Owning package and boundary | Continue with | First focused validation |
| --- | --- | --- | --- |
| Reusable graph construction, middleware, backends, filesystem policy, skills, state, or subagents | `libs/deepagents/` owns reusable harness behavior and `create_deep_agent`. | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md) and [Source Map](./architecture/source-map.md) | Run the nearest SDK unit test, such as `make test TEST_FILE=tests/unit_tests/test_graph.py`, then `make lint`. |
| Terminal CLI, Textual UI, session behavior, model configuration, approvals, commands, or sandbox selection | `libs/code/` owns the dcode product. It exactly pins the SDK version it requires. | [Run a dcode Session](./workflows/run-dcode-session.md), then [Source Map](./architecture/source-map.md) | Run the nearest `tests/unit_tests/` file with `make test TEST_FILE=...`, then `make lint`. |
| An editor-facing Agent Client Protocol server, custom graph hosting, or `dcode --acp` | `libs/acp/` owns the ACP bridge; dcode is also an ACP-serving product surface. | [Repository Architecture Overview](./architecture/overview.md) and [Run a dcode Session](./workflows/run-dcode-session.md) | From `libs/acp`, run `make test TEST_FILE=tests/<affected_test>.py`; also validate dcode when its ACP mode changes. |
| Long-running channels, schedules, host lifecycle, runtime tools, or Talon persistence | `libs/talon/` owns the experimental local runtime host, downstream of the SDK and dcode. | [Talon Runtime Integration](./integrations/talon.md) and [Source Map](./architecture/source-map.md) | Run `make test TEST_FILE=tests/<affected_test>.py` and `make lint` in `libs/talon`. |
| A sandbox provider or a provider-specific integration | The relevant `libs/partners/<provider>/` package owns it; dcode consumes partner integrations locally during development. | [Source Map](./architecture/source-map.md) and [Development, Dependencies, and Releases](./operations/development.md) | Run the partner tests, then the affected consumer and any applicable integration coverage. |
| Real-model quality, trajectories, scoring, or a sandbox benchmark | `libs/evals/` owns behavioral evaluation and Harbor integration; it is not a Release Please component. | [Testing Guide](./testing/testing-guide.md) | Add or run deterministic regression coverage first; then use `make evals MODEL=<id>` or the applicable Harbor job. |
| Dependency bounds, editable sibling consumers, lockfiles, CI, or delivery | The changed package manifest owns its compatibility and lock; repository automation determines the affected local-consumer closure and release process. | [Development, Dependencies, and Releases](./operations/development.md) | Refresh the changed package and affected consumer locks, then run the package checks or `cd libs && make lock-check`. |

## Respect package and dependency boundaries

The SDK is the reusable layer. Put general graph, middleware, backend, and agent-construction behavior in `deepagents` rather than reproducing it in dcode, ACP, or Talon. dcode is the terminal product; ACP is the protocol bridge; Talon is an experimental long-running host. Their UI, lifecycle, approval, and security behavior are host-specific.

Published dependencies and local development mappings answer different questions. ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode. Local `[tool.uv.sources]` mappings make these sibling dependencies editable, so a shared-interface change can affect consumers immediately. Validate each affected downstream package rather than assuming a producer-only test is sufficient.

`deepagents-code` currently pins `deepagents==0.7.28` exactly. If dcode starts relying on newer SDK behavior, update that exact pin in the same pull request and follow the dependency/release guide.

## Work in the owning package

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the command authority. `uv` provisions the appropriate interpreter; do not impose a repository-wide Python pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_graph.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. ACP and Talon's normal `make test` targets disable network sockets; Talon's target runs its WhatsApp bridge Node tests before Python tests. Use [Testing Guide](./testing/testing-guide.md) to select the observable seam and a narrow test before widening coverage.

## Release and integration handoff

The Release Please manifest currently tracks `deepagents` 0.7.28, `deepagents-acp` 0.0.13, `deepagents-code` 0.1.86, and `deepagents-talon` 0.0.9. It also tracks Daytona 0.0.8, Modal 0.0.6, Runloop 0.0.7, Vercel 0.0.2, and QuickJS 0.3.9. These are independently managed release lines, not a reason to combine unrelated package changes.

Release Please creates a package-specific draft release PR after releasable commits reach `main`. Review the generated changelog and use the curated-notes workflow: mark the PR ready, review and edit the bot's draft, then comment `@release-bot apply`. Use `ci:skip-curated-notes` only when intentionally publishing the generated changelog without curation. The full rules for lockfile fan-out, pins, release PR refreshes, and CI verification are in [Development, Dependencies, and Releases](./operations/development.md).

When a task crosses a boundary, follow the detailed page rather than expanding this quickstart:

- **Find an owner, entry point, or focused test:** [Source Map](./architecture/source-map.md)
- **Change graph construction or reusable SDK behavior:** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **Operate or change a terminal session:** [Run a dcode Session](./workflows/run-dcode-session.md)
- **Change a long-running channel host:** [Talon Runtime Integration](./integrations/talon.md)
- **Choose deterministic, integration, or real-model coverage:** [Testing Guide](./testing/testing-guide.md)
- **Change dependencies, locks, CI, or releases:** [Development, Dependencies, and Releases](./operations/development.md)
