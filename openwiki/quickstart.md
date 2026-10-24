---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package, its detailed guide, and the narrowest useful validation command. Use this page to identify editable dependency handoffs, the current dcode SDK pin, and the release-automation handoff.
tags: [deepagents, maintenance, testing, dependencies, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-24T08:05:08.866Z
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
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-0153e073a6645f3118ca08c4
    resource: repo://libs/evals/AGENTS.md
  - id: openwiki-source-be7f6aa28551fac7310db803
    resource: repo://libs/evals/Makefile
  - id: openwiki-source-49fbcc45434b619b68220bf9
    resource: repo://libs/Makefile
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-10-24T08:05:08.866Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not necessarily the product where a symptom appears. `libs/` is an independently versioned monorepo containing the reusable `deepagents` SDK, the `deepagents-code`/`dcode` product, ACP, evals, the experimental Talon host, and provider integrations. A package's `pyproject.toml`, `Makefile`, source, and tests are the authority for changing it.

Use [Repository Architecture Overview](./architecture/overview.md) to understand the package and runtime layers, and [System Source Map](./architecture/source-map.md) to locate an owner, construction seam, or representative test. The links below deliberately route to the detailed architecture, concepts, integrations, workflows, operations, and testing guidance rather than duplicating it here.

## Route the change

| Change or question | Start here | Narrowest first validation | Detailed guide |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, filesystem tools or policy, profiles, skills, state, or synchronous subagents | `libs/deepagents/` owns the reusable harness. | `cd libs/deepagents && make test TEST_FILE=<nearest test>` | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md); [SDK Construction and Execution](./architecture/sdk-construction-execution.md) |
| Middleware ordering, replacement, exclusions, or an extension | `libs/deepagents/` assembly and feature middleware. | `cd libs/deepagents && make test TEST_FILE=<nearest test>` | [SDK Middleware Assembly](./architecture/middleware-stack.md); [Middleware Catalog and Extension Rules](./concepts/middleware-catalog.md) |
| Async remote subagents, task updates or cancellation, and child interrupts | The SDK async-subagent middleware and its remote-child boundary. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_async_subagents.py` | [Subagents, Skills, and Delegation](./concepts/subagents-skills.md); [Runtime Behavior and Failure Boundaries](./architecture/runtime-behavior.md) |
| Summarization, compaction, offload, or retained history | SDK context-management middleware and its state/event contract. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/middleware/test_compact_tool.py` | [Context Management and Offload](./concepts/context-management.md) |
| `dcode` CLI/TUI, server, workspace or session behavior, commands, MCP, sandbox, configuration, or headless mode | `libs/code/` owns `deepagents-code`; `dcode` is its console entry point. | `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Deep Agents Code Architecture](./architecture/code-agent.md); [Run a dcode Session](./workflows/run-dcode-session.md) |
| Editor-facing Agent Client Protocol translation | `libs/acp/`; also validate dcode when its ACP specialization changes. | `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py` | [Agent Client Protocol Integration](./integrations/acp.md) |
| Talon channels, pairing, schedules, host runtime, delivery, approvals, or history | `libs/talon/` owns the experimental local host. | `cd libs/talon && make test TEST_FILE=<nearest test>` | [Talon Runtime Integration](./integrations/talon.md); [Talon Channel Admission and Pairing](./concepts/talon-channel-admission.md); [Talon Scheduling and Background Work](./concepts/talon-scheduling.md) |
| A Daytona, Modal, Runloop, Vercel, or QuickJS adapter | The matching `libs/partners/<provider>/` package, followed by its consuming product seam. | `cd libs/partners/<provider> && make test TEST_FILE=<nearest test>` | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Deterministic eval plumbing, behavioral eval selection, trial aggregation, Harbor, or benchmark results | `libs/evals/`; model-driven evaluation is separate from package unit testing. | `cd libs/evals && make test TEST_FILE=<nearest test>` | [Run Evaluations and Harbor Benchmarks](./workflows/run-evals.md); [Testing Guide](./testing/testing-guide.md) |
| Dependency constraints, locks, CI selection, labels, or publishing | The affected manifest and lockfile, then `.github/` automation. | `cd libs && make lock-check` | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md); [CI, Change Selection, and Labels](./operations/ci-labeling.md) |

After the focused test passes, run the package's `make lint`. For dcode, `make check` adds the local CI-parity gates. In an unfamiliar package, run `make help` rather than assuming that every package offers the same target.

## Dependency handoff

ACP consumes the SDK. dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode. These sibling sources are editable during local development, so a public or shared-interface change needs a focused consumer seam in addition to owner coverage. CI reflects this dependency direction: SDK changes select dcode, ACP, Talon, evals, and partner jobs; dcode changes select Talon.

**Handoff path**

1. Change and test the owning contract first.
2. Identify editable consumers whose imports, configuration, manifest constraints, or runtime behavior use that contract.
3. Run the closest consumer test, then the consumer package check when the change is broad.
4. If a manifest changes, update its lockfile and run `cd libs && make lock-check` when the affected lock scope is uncertain.

`deepagents-code` requires `deepagents==0.7.32` exactly. A dcode pull request that needs newer SDK behavior must update that pin in the same pull request; do not defer it to release work. The package's `make check` reports a stale workspace SDK pin as advisory locally.

## Local development and validation boundary

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the command authority. `uv` provisions the applicable interpreter, so there is no repository-wide Python version to install or pin; each manifest declares its own supported range.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. SDK, ACP, and Talon `make test` targets disable network sockets; Talon first runs its WhatsApp bridge Node tests. Evals also socket-disable unit tests, whereas `make evals MODEL=<id>` is a model-driven run and requires LangSmith tracing, a LangSmith API key, and provider credentials for the chosen model.

## CI and release handoff

Pull-request CI selects changed packages, while pushes to `main` run full CI. The path filters include editable SDK consumers, but CI is a safety net rather than a substitute for the local consumer seam.

Release Please manages separate draft release PRs for the managed SDK, ACP, dcode, Talon, and partner packages after releasable commits reach `main`; the current release baselines are tracked in `.release-please-manifest.json`. Let Release Please and its lockfile update finish before editing, rebasing, or preparing its release branch. Then review the generated changelog, mark the draft PR ready, review or edit the curated-note draft, and comment `@release-bot apply`. Use `ci:skip-curated-notes` only for an intentional generated-changelog release. Merging the reviewed release PR starts the GitHub and PyPI publishing workflow.

A trusted bot commit that only applies curated changelog notes may skip package jobs only if the parent `CI Success` result is conclusive. The gate reuses that parent result; inspect the linked parent check rather than treating skipped package jobs as independent success.

## Continue by question

- **Which layer owns this behavior?** [Repository Architecture Overview](./architecture/overview.md)
- **Where is the state owner, entry point, or focused test?** [System Source Map](./architecture/source-map.md)
- **How do I change the SDK safely?** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **How do I operate a dcode session?** [Run a dcode Session](./workflows/run-dcode-session.md)
- **Which integration boundary applies?** [Integrations](./integrations/index.md)
- **Which test category and command fit this change?** [Testing Guide](./testing/testing-guide.md)
- **How do locks, pins, CI, and releases work?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
