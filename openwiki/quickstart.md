---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Task-routing entry point for Deep Agents maintainers. Find the owning package, run its narrowest deterministic regression seam, and follow the linked guide for cross-package behavior, CI, and releases.
tags: [deepagents, maintenance, delegation, context-management, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-22T08:05:54.651Z
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
  - id: openwiki-source-c972622237a22631e36f3625
    resource: repo://libs/deepagents/deepagents/backends/utils.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
  - id: openwiki-source-fb022ddbcc554eaabedfa8cd
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_state_backend.py
  - id: openwiki-source-e11958a6a68a079be57552fc
    resource: repo://libs/deepagents/tests/unit_tests/backends/test_store_backend.py
  - id: openwiki-source-69c260914d2dd18b60ad5900
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py
  - id: openwiki-source-6ce85b02eabe462f99e0c912
    resource: repo://libs/deepagents/tests/unit_tests/test_async_subagents.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-5b9a69640ff3d94216c614ce
    resource: repo://libs/talon/deepagents_talon/messaging.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-4d6726e17c8a0c78539a7d33
    resource: repo://libs/talon/tests/test_runtime.py
  - id: openwiki-source-1698129adea358c8813da5a5
    resource: repo://libs/talon/tests/unit_tests/test_messaging.py
  - id: openwiki-source-242a21b2da46507f58415265
    resource: repo://libs/talon/tests/unit_tests/test_tool_approval_runtime.py
  - id: openwiki-source-1e7998c9f5a94d22f3819721
    resource: repo://libs/talon/tests/unit_tests/test_verbosity.py
generated: { by: "openwiki/0.4.2", at: "2026-10-22T08:05:54.651Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not necessarily the product where behavior appears. `libs/` contains independently versioned packages for the reusable `deepagents` SDK, dcode, ACP, evals, the experimental Talon host, and provider integrations. Each package's source, tests, `pyproject.toml`, and `Makefile` are authoritative. This is a routing map, not a replacement for those local instructions.

For package relationships, see [Repository Architecture Overview](./architecture/overview.md); for symbols and narrower tests, see the [System Source Map](./architecture/source-map.md).

## Route a change request

| Change request | Owner and first inspection point | Narrow validation first | Detailed guide |
| --- | --- | --- | --- |
| Build or customize an agent; change graph assembly, middleware, profiles, permissions, skills, subagents, filesystem behavior, state, or a backend | `libs/deepagents/` owns the reusable harness. It delegates graph assembly to LangChain's `create_agent`; CLI, protocol, and host behavior belong in products. | `cd libs/deepagents && make test TEST_FILE=<nearest test>` then `make lint` | [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md), [SDK Construction and Execution](./architecture/sdk-construction-execution.md) |
| Change remote background delegation, task updates, completion notification, or child approvals | `libs/deepagents/deepagents/middleware/async_subagents.py` owns the remote Agent Protocol task contract. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_async_subagents.py` | [Subagents, Skills, and Delegation](./concepts/subagents-skills.md), [Runtime Behavior and Failure Boundaries](./architecture/runtime-behavior.md), [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md) |
| Change automatic or manual conversation compaction, retained history, offload, or overflow recovery | `libs/deepagents/deepagents/middleware/summarization.py` owns the summarization state and `compact_conversation` tool. | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/middleware/test_compact_tool.py` | [Context Management](./concepts/context-management.md), [Middleware Catalog](./concepts/middleware-catalog.md) |
| Change dcode startup, TUI, sessions, commands, MCP, sandboxes, approvals, or headless execution | `libs/code/` owns `deepagents-code`, launched as `dcode`. | `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py`, then `make check` | [Deep Agents Code Architecture](./architecture/code-agent.md) |
| Change the generic editor-facing Agent Client Protocol integration | `libs/acp/` owns ACP translation; `dcode --acp` also crosses dcode's specialization. | `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py`; validate dcode if its ACP mode changed. | [Agent Client Protocol Integration](./integrations/acp.md) |
| Change channels, schedules, runtime graph invocation, delivery, or approvals in the local host | `libs/talon/` owns the experimental Talon host, downstream of the SDK and dcode. | `cd libs/talon && make test TEST_FILE=<nearest test>` then `make lint` | [Talon Runtime Integration](./integrations/talon.md) |
| Change a provider or sandbox adapter | The matching `libs/partners/<provider>/` package owns the adapter; dcode and Talon consume it at their product boundary. | Run provider coverage, then the affected consumer seam. | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Change scoring, trajectories, Harbor integration, or benchmarks | `libs/evals/` owns evaluations and Harbor integration; it is not Release Please-managed. | Add/run deterministic regression coverage, then the applicable eval workflow. | [Testing Guide](./testing/testing-guide.md) |
| Change locks, CI scope, versions, or publication | The edited package manifest owns compatibility; CI and release automation own fan-out and publication. | Refresh affected locks and run package checks; use `cd libs && make lock-check` when scope is uncertain. | [Development, Dependencies, and Releases](./operations/development.md), [CI and Pull-Request Labeling](./operations/ci-labeling.md) |

## Two SDK regression seams

### Async remote delegation

`AsyncSubAgentMiddleware` launches remote Agent Protocol runs and immediately returns a task ID; task records live in `async_tasks`, so they survive conversation compaction. It exposes `start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`, and `list_async_tasks`. A remote child has its own tools, credentials, and human interrupts: a parent may report a `waiting` task but cannot answer or approve it.

Default updates interrupt the active run. With `update_strategy="enqueue"`, the follow-up is `pending`, earlier run IDs are recorded as superseded, and only one queued follow-up per task may be sent in a model turn. Cancellation attempts the latest run and any active superseded runs; status-observation failures retain cached status rather than destroying task state. Configure `notify_parent=False` when an application-owned webhook or delivery path should handle completion instead of waking an idle parent.

Run the full focused seam after changing lifecycle, notification, update, or cancellation behavior:

```bash
cd libs/deepagents
make test TEST_FILE=tests/unit_tests/test_async_subagents.py
```

### Manual compaction

`SummarizationToolMiddleware` adds `compact_conversation` but does not compact by itself; it shares `_summarization_event` state with `SummarizationMiddleware`, which performs automatic compaction. The tool first constructs effective history from the prior event and only proceeds when reported usage reaches roughly half of the configured automatic trigger. On success it creates a summary, attempts to offload older history through the configured backend, and atomically returns the new event plus a confirmation. A failed summary or offload returns an error `ToolMessage` without changing the event; an offload operation that reports no path can still produce a valid summary event.

```bash
cd libs/deepagents
make test TEST_FILE=tests/unit_tests/middleware/test_compact_tool.py
```

### Backend file-data compatibility

`StateBackend` and `StoreBackend` accept legacy `content: list[str]` data and normalize it for reads, listings, searches, sizes, and downloads. A read must not rewrite that legacy representation; a successful write or edit persists current string content. Treat both adapters as one observable compatibility contract:

```bash
cd libs/deepagents
make test TEST_FILE=tests/unit_tests/backends/test_state_backend.py
make test TEST_FILE=tests/unit_tests/backends/test_store_backend.py
```

## Work and test in the owning package

Use `uv` for interpreters, environments, and dependencies, and use the package `Makefile` as the command authority. `uv` provisions the applicable interpreter, so there is no repository-wide Python version to pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/middleware/test_compact_tool.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. SDK, dcode, ACP, and Talon normal test targets disable network sockets; Talon's target first runs its WhatsApp bridge Node tests.

Local sibling dependencies are editable. Validate shared-interface changes in affected consumers: ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode. CI path filters schedule dependent coverage for SDK changes, including dcode, ACP, Talon, evals, and partners where applicable.

## Release and CI handoff

The current Release Please baselines are `deepagents` `0.7.30`, `deepagents-acp` `0.0.13`, `deepagents-code` `0.1.88`, and `deepagents-talon` `0.0.9`. Managed partner baselines are Daytona `0.0.8`, Modal `0.0.6`, Runloop `0.0.7`, Vercel `0.0.2`, and QuickJS `0.3.9`. `deepagents-code` pins `deepagents==0.7.30` exactly: update the pin in the same PR whenever dcode requires new SDK behavior.

After releasable commits reach `main`, Release Please creates or updates a separate draft release PR per managed package. Wait for both its refresh and lockfile update; do not manually race its branch. Review the generated changelog, mark the PR ready, review or edit the curated-notes draft, and use `@release-bot apply` unless intentionally using the skip label. Merging the reviewed release PR triggers the GitHub and PyPI release workflow.

A trusted changelog-only `@release-bot apply` commit deliberately skips package jobs only when its parent has a conclusive CI result; the CI Success gate reuses that parent's result. Verify the current head SHA and its workflow/check status before merging rather than treating skipped package jobs as independent success.

## Talon operational seams

For every Talon request, `DeepAgentRuntime` refreshes runtime tools, reads the approval snapshot, rebuilds the graph when that snapshot changes, pins graph and request contexts for the invocation, and resets them in `finally`. Construct replacement tools/graph before swapping active state. For delivery changes, verify `DEEPAGENTS_TALON_VERBOSITY`: `1`, `2`, or `3`; only levels above `1` add `send_message`, level `2` has a shared two-message turn budget, and level `3` is unbounded. Progress is request- and originating-handler-scoped, rejects blank or unavailable delivery, hides transport errors from the model, and is cleaned up per invocation.

## Pick the next detailed guide

- **Find an owner, entry point, or focused test:** [System Source Map](./architecture/source-map.md)
- **Understand package boundaries and dependency direction:** [Repository Architecture Overview](./architecture/overview.md)
- **Build or extend an SDK agent:** [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md)
- **Choose deterministic, integration, or behavioral coverage:** [Testing Guide](./testing/testing-guide.md)
- **Change dependencies, locks, CI, or publication:** [Development, Dependencies, and Releases](./operations/development.md)
