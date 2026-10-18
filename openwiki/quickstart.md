---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Task-routing entry point for maintainers of the Deep Agents monorepo. Identify the contract owner, use the package-local workflow and focused tests, then follow the linked guide for cross-package operations.
tags: [deepagents, maintenance, dcode, talon, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-18T08:05:11.339Z
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
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
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
generated: { by: "openwiki/0.4.2", at: "2026-10-18T08:05:11.339Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract—not necessarily the product where a behavior is visible. `libs/` contains independently versioned packages for the `deepagents` SDK, dcode, ACP, evals, Talon, and provider integrations. **Package-local source, tests, `pyproject.toml`, and `Makefile` are authoritative**; this page routes work rather than replacing those guides.

For package relationships and runtime boundaries, read [Repository Architecture Overview](./architecture/overview.md). If the owning entry point or narrow regression test is unclear, use the [Source Map](./architecture/source-map.md).

## Route the task

| If the change concerns… | Owns the contract | Continue with | Start validation with |
| --- | --- | --- | --- |
| Reusable graph assembly, middleware, backends, filesystem policy, skills, state, or local/remote delegation | `libs/deepagents/` owns the SDK harness. It composes its agent graph through LangChain `create_agent`; product transports do not belong here. | [Build a Deep Agent](./workflows/build-a-deep-agent.md), [SDK construction and execution](./architecture/sdk-construction-execution.md), or [Subagents and Skills](./concepts/subagents-skills.md) | Nearest SDK unit test, for example `make test TEST_FILE=tests/unit_tests/test_graph.py`, then `make lint`. |
| Terminal CLI, Textual UI, client/server session, workspace/interpreter policy, commands/autocomplete, approvals, retries, or sandbox choice | `libs/code/` owns the dcode product. It is the terminal client and agent-server surface, not a second generic SDK. | [Run a dcode Session](./workflows/run-dcode-session.md) and [Code Agent Architecture](./architecture/code-agent.md) | Nearest `tests/unit_tests/` target, then `make lint` (or `make check` for the package CI-parity gate). |
| Editor-facing Agent Client Protocol behavior or `dcode --acp` | `libs/acp/` owns generic protocol/session translation; dcode owns its terminal-specific ACP specialization. | [Repository Architecture Overview](./architecture/overview.md) and [Run a dcode Session](./workflows/run-dcode-session.md) | From `libs/acp`, run `make test TEST_FILE=tests/<affected_test>.py`; also validate dcode when its ACP mode changes. |
| A long-running channel, Discord admission or delivery, host lifecycle, scheduling, or Talon persistence | `libs/talon/` owns the experimental local host and channel adapters, downstream of SDK and dcode. | [Talon Runtime Integration](./integrations/talon.md) and [Talon Channel Admission](./concepts/talon-channel-admission.md) | Run the closest Talon test, such as `tests/channels/test_discord.py` or `tests/test_host.py`, then `make lint`. |
| A sandbox provider or provider-specific integration | The relevant `libs/partners/<provider>/` package owns the provider; dcode and Talon are consumers at their respective product boundary. | [Sandbox Partners](./integrations/sandbox-partners.md) and [Development, Dependencies, and Releases](./operations/development.md) | Run provider tests, then affected dcode, Talon, or SDK consumer coverage. |
| Real-model quality, trajectories, scoring, or a sandbox benchmark | `libs/evals/` owns behavioral evaluation and Harbor integration; it is not Release Please-managed. | [Testing Guide](./testing/testing-guide.md) | Add/run deterministic regression coverage first; then use `make evals MODEL=<id>` or the applicable Harbor job. |
| Dependency bounds, editable siblings, locks, CI, or a publication | The changed manifest owns its compatibility contract; CI and release automation determine consumer and publication work. | [Development, Dependencies, and Releases](./operations/development.md) | Refresh affected locks, run package checks, and use `cd libs && make lock-check` when lock scope is uncertain. |

## Respect the boundaries

The SDK is the reusable layer: graph construction, middleware, backends, and delegation belong in `deepagents`. dcode owns terminal presentation and session policy; ACP owns editor protocol translation; Talon owns a local host's channel lifecycle, delivery, and schedules. Keep host-specific UI, approval, and transport decisions out of reusable SDK behavior.

Published requirements and local development mappings answer different questions. ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode. Editable sibling mappings expose a producer change to consumers immediately, so shared-interface work needs downstream validation. CI also schedules dcode and Talon coverage for SDK changes.

`deepagents-code` is `0.1.86` and exactly pins `deepagents==0.7.28`. If dcode starts requiring new SDK behavior, update that pin in the same pull request; do not defer the contract change to release time.

## Work and test in the owning package

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the command authority. `uv` provisions the appropriate interpreter, so there is no repository-wide Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_graph.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. ACP and Talon's normal `make test` targets disable network sockets; Talon's target runs WhatsApp bridge Node tests before Python tests. The [Testing Guide](./testing/testing-guide.md) maps changed SDK delegation and persistence, dcode session/sandbox/UI, and Talon Discord/host behavior to focused test seams.

## Release handoff

The Release Please manifest currently records `deepagents` `0.7.28`, `deepagents-acp` `0.0.13`, `deepagents-code` `0.1.86`, and `deepagents-talon` `0.0.9`; it also manages Daytona `0.0.8`, Modal `0.0.6`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.9`. These are independent release lines.

After releasable commits reach `main`, Release Please creates or updates a separate draft release PR for each managed package. Let its lockfile refresh finish; review the generated changelog, mark the PR ready, review/edit the release-bot draft, and comment `@release-bot apply`. Merging the reviewed PR triggers the publication workflow for its GitHub and PyPI releases. Use `ci:skip-curated-notes` only when intentionally shipping the generated changelog without curation.

## Follow the detailed guide

- **Find an owner, entry point, or focused test:** [Source Map](./architecture/source-map.md)
- **Change reusable graph construction, backend behavior, or delegation:** [SDK construction and execution](./architecture/sdk-construction-execution.md) and [Build a Deep Agent](./workflows/build-a-deep-agent.md)
- **Operate or change a terminal session:** [Run a dcode Session](./workflows/run-dcode-session.md)
- **Change a channel host or Discord admission path:** [Talon Runtime Integration](./integrations/talon.md) and [Talon Channel Admission](./concepts/talon-channel-admission.md)
- **Choose deterministic, integration, or real-model coverage:** [Testing Guide](./testing/testing-guide.md)
- **Change dependencies, locks, CI, or releases:** [Development, Dependencies, and Releases](./operations/development.md)
