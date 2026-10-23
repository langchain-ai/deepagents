---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package, the right detailed guide, and the narrowest useful validation command. Use this page to identify dependency handoffs before expanding validation across consumers.
tags: [deepagents, maintenance, testing, dependencies, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-23T08:06:06.503Z
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
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
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
generated: { by: "openwiki/0.4.2", at: "2026-10-23T08:06:06.503Z" }
---

# Deep Agents Monorepo Quickstart

Start in the package that owns the contract, not necessarily the product where a symptom appears. `libs/` contains independently versioned packages for the reusable `deepagents` SDK, the `dcode` terminal product, ACP, evals, the experimental Talon host, and provider integrations. Each package's `pyproject.toml`, `Makefile`, source, and tests are the local authority.

Use [Repository Architecture Overview](./architecture/overview.md) for the package and runtime layers, and [System Source Map](./architecture/source-map.md) to find the owning symbol and a representative test.

## Route the change

| Requested change | Owning package and boundary | First focused validation | Continue here |
| --- | --- | --- | --- |
| Agent construction, middleware, backends, tools, filesystem policy, profiles, skills, state, or synchronous subagents | `libs/deepagents/` owns the reusable harness. It assembles its defaults around LangChain's `create_agent`; product UX and transport behavior do not belong here. | `cd libs/deepagents && make test TEST_FILE=<nearest test>` | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md); [SDK Construction and Execution](./architecture/sdk-construction-execution.md) |
| Middleware ordering, replacement, exclusions, or a custom middleware extension | `libs/deepagents/`, with the assembly path and feature middleware as the inspection boundary. | `cd libs/deepagents && make test TEST_FILE=<nearest test>` | [SDK Middleware Assembly](./architecture/middleware-stack.md); [Middleware Catalog and Extension Rules](./concepts/middleware-catalog.md) |
| Async remote Agent Protocol tasks, task updates/cancellation, or child interrupts | SDK async-subagent middleware; remote children own their own interrupt thread. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_async_subagents.py` | [Subagents, Skills, and Delegation](./concepts/subagents-skills.md); [Runtime Behavior and Failure Boundaries](./architecture/runtime-behavior.md) |
| Summarization, manual compaction, offload, or retained history | SDK summarization middleware and its state/event contract. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/middleware/test_compact_tool.py` | [Context Management and Offload](./concepts/context-management.md) |
| dcode CLI/TUI, local server, workspace/session behavior, commands, MCP, sandbox, configuration, or headless mode | `libs/code/` owns `deepagents-code`, launched as `dcode`. Its TUI and `langgraph dev` server are separate processes. | `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py` | [Deep Agents Code Architecture](./architecture/code-agent.md); [Run a dcode Session](./workflows/run-dcode-session.md) |
| Generic editor-facing ACP protocol translation | `libs/acp/`; if `dcode --acp` behavior changes, validate dcode's specialization too. | `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py` | [Agent Client Protocol Integration](./integrations/acp.md) |
| Talon channels, pairing, schedules, host runtime, graph invocation, delivery, approvals, or history | `libs/talon/` owns the experimental local host. | `cd libs/talon && make test TEST_FILE=<nearest test>` | [Talon Runtime Integration](./integrations/talon.md); [Talon Channel Admission and Pairing](./concepts/talon-channel-admission.md); [Talon Scheduling and Background Work](./concepts/talon-scheduling.md) |
| A Daytona, Modal, Runloop, Vercel, or QuickJS adapter | The matching `libs/partners/<provider>/` package; then validate the product integration that consumes it. | `cd libs/partners/<provider> && make test TEST_FILE=<nearest test>` | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Deterministic eval plumbing, behavioral eval selection, trial aggregation, Harbor, or benchmark results | `libs/evals/`; behavioral runs are distinct from package unit tests and require their configured model/tracing environment. | `cd libs/evals && make test TEST_FILE=<nearest test>` | [Run Evaluations and Harbor Benchmarks](./workflows/run-evals.md); [Testing Guide](./testing/testing-guide.md) |
| Lockfiles, CI selection, labels, version pins, or publishing | The affected package manifest plus `.github/` automation. | `cd libs && make lock-check` | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md); [CI, Change Selection, and Labels](./operations/ci-labeling.md) |

After a focused passing test, run the package's `make lint`; dcode's `make check` is its full local CI-parity suite. Run `make help` in an unfamiliar package rather than assuming that every package exposes the same target.

## Dependency handoffs

```text
Deep Agents SDK ──> ACP
       │             │
       ├──────────> dcode ──> provider integrations
       │             │
       └──────────> Talon <──┘
       └──────────> evals
```

ACP uses the SDK. dcode uses the SDK, ACP, and partner integrations; Talon uses the SDK and dcode. These are editable sibling dependencies in local development, so a public or shared-interface change needs consumer coverage in addition to the owner test. CI mirrors this direction: an SDK change selects coverage for dcode, ACP, Talon, evals, and each partner; a dcode change also selects Talon.

**Handoff checklist**

1. Change and test the owning contract first.
2. Identify every editable consumer whose import, configuration, manifest constraint, or runtime behavior relies on it.
3. Run the nearest consumer seam, then its package check if the change is broad.
4. If the change updates a dependency declaration, refresh the affected lockfile; use the `libs/` fan-out lock check when unsure.

`deepagents-code` declares an exact `deepagents==0.7.30` requirement. If dcode starts depending on new SDK behavior, update that pin in the same pull request rather than deferring it to release work.

## Local development and test boundaries

Use `uv` for interpreters, environments, and dependencies; use the package `Makefile` as the command authority. `uv` provisions the applicable interpreter, so there is no repository-wide Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free tests in `tests/unit_tests/`; use `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The SDK, ACP, and Talon `make test` targets disable network sockets; Talon also runs WhatsApp bridge Node tests before pytest. Evals' unit-test target is likewise socket-disabled, but its `make evals MODEL=<id>` path is a credentialed behavioral run.

## CI and release handoff

On pull requests CI selects changed package jobs; on pushes to `main` it runs full CI. Path filters deliberately include the SDK for its editable consumers, so CI is a safety net—not a replacement for running the affected consumer locally.

Release Please manages separate draft release PRs per managed package after releasable commits reach `main`. Let it refresh the release PR and lockfiles before editing its branch. When ready, review the generated changelog, mark the PR ready, review or edit the curated release-note draft, and use `@release-bot apply`; `ci:skip-curated-notes` is the intentional opt-out. Merging a reviewed release PR triggers the publishing workflow for its GitHub and PyPI releases.

A trusted bot commit that only applies curated changelog notes may skip package jobs only when its parent CI result is conclusive. Its `CI Success` gate reuses that parent result, so verify the current head and parent-linked CI status rather than treating skipped jobs as independent success.

## Next guide by question

- **Where is the symbol, state owner, or focused test?** [System Source Map](./architecture/source-map.md)
- **Which layer owns this behavior?** [Repository Architecture Overview](./architecture/overview.md)
- **How should I change the SDK graph safely?** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **Which test category and command fit this change?** [Testing Guide](./testing/testing-guide.md)
- **How do locks, pins, CI, and releases work?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
