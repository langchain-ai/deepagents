---
type: maintainer quickstart
title: Deep Agents Monorepo Quickstart
description: Route a Deep Agents change to its owning package, its detailed guide, and the narrowest useful validation command. Use this page to identify local editable handoffs and dcode's SDK compatibility pin.
tags: [deepagents, maintenance, dependencies, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-25T08:05:05.473Z
sources:
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
generated: { by: "openwiki/0.4.2", at: "2026-10-25T08:05:05.473Z" }
---

# Deep Agents Monorepo Quickstart

Start with the package that owns the contract, not necessarily the product where a symptom appears. This is a monorepo of independently versioned packages: the reusable `deepagents` SDK, `deepagents-code`/`dcode` terminal product, ACP bridge, eval suite, experimental Talon host, and partner integrations. A package's `pyproject.toml`, `Makefile`, source, and tests are the authority for changing it.

Use [Repository Architecture Overview](./architecture/overview.md) for the dependency and runtime map, and [System Source Map](./architecture/source-map.md) to find a construction seam, lifecycle owner, or representative test. The routes below intentionally point to the detailed guides instead of reproducing subsystem behavior here.

## Route the change

| Change or question | Owner and first validation | Consult next |
| --- | --- | --- |
| Reusable agent construction, middleware, backends, tools, permissions, subagents, profiles, or graph state | `libs/deepagents/`; run `cd libs/deepagents && make test TEST_FILE=<nearest test>`. | [SDK construction and execution](./architecture/sdk-construction-execution.md); [Build and Customize a Deep Agent](./workflows/build-a-deep-agent.md) |
| dcode CLI/TUI, local server startup, workspaces, sessions, MCP, persistence, configuration, or headless behavior | `libs/code/`; run `cd libs/code && make test TEST_FILE=tests/unit_tests/<nearest test>.py`. | [Deep Agents Code Architecture](./architecture/code-agent.md); [Run a dcode Session](./workflows/run-dcode-session.md) |
| Editor-facing Agent Client Protocol behavior | `libs/acp/`; then validate dcode too when its ACP specialization changes. Run `cd libs/acp && make test TEST_FILE=tests/<nearest test>.py`. | [Agent Client Protocol Integration](./integrations/acp.md) |
| Talon channels, pairing, schedules, host runtime, approvals, or history | `libs/talon/`; run `cd libs/talon && make test TEST_FILE=<nearest test>`. | [Talon Runtime Integration](./integrations/talon.md); [Talon Channel Admission and Pairing](./concepts/talon-channel-admission.md); [Talon Scheduling and Background Work](./concepts/talon-scheduling.md) |
| Daytona, Modal, Runloop, Vercel, or QuickJS integration | The matching `libs/partners/<provider>/`, then its consuming product seam. | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) |
| Deterministic eval plumbing, behavioral evaluations, Harbor, or results | `libs/evals/`; model-driven evaluation is not a substitute for package unit tests. | [Run Evaluations and Harbor Benchmarks](./workflows/run-evals.md); [Testing Guide](./testing/testing-guide.md) |
| Dependency constraints, locks, CI, or releases | The affected package manifest and lockfile, then repository automation. Start with `cd libs && make lock-check`. | [Development, Dependencies, Lockfiles, and Releases](./operations/development.md) |

After a focused test passes, run that package's `make lint`; use `make help` rather than assuming targets are uniform. For dcode, `make check` is the package-local CI-parity command: it includes lint, import checks, tests, extras/version/lock checks, and an advisory workspace-SDK-pin check. The [Testing Guide](./testing/testing-guide.md) maps dcode lifecycle, workspace, runtime-patch, and Textual changes to focused tests.

## Cross-package handoff

Published requirements and checkout wiring are different contracts. `[project.dependencies]` specifies the released wheel; `[tool.uv.sources]` may replace it with an editable sibling checkout during monorepo development. ACP consumes the SDK; dcode consumes the SDK, ACP, and partner integrations; Talon consumes the SDK and dcode. Thus, a shared interface or behavior change needs owner coverage **and** a focused test at each affected consumer seam.

`deepagents-code` publishes an exact `deepagents==0.7.32` requirement. Raise that pin in the same pull request that first needs newer SDK behavior; do not leave a dcode change dependent on an unpublished local checkout state. The dcode checkout resolves the SDK, ACP, and its partner packages from editable siblings; Talon resolves both the SDK and dcode that way. QuickJS likewise resolves the SDK from the sibling checkout while publishing its own SDK compatibility range.

A safe handoff is:

1. Change and test the contract owner.
2. Identify consumers through their manifest dependency and local source mapping.
3. Run the closest consumer test; run the consumer package check when the impact is broad.
4. If dependency metadata changes, update the package lockfile and use `cd libs && make lock-check` when the lock scope is uncertain.

## Package-local development and tests

Use `uv` for interpreters, environments, and dependencies, and package Makefiles as the command authority. `uv` provisions the applicable interpreter, so there is no repository-wide Python version to install or pin; each package declares its own supported range.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/<nearest test>.py
make lint
```

Keep deterministic, network-free coverage in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. SDK and dcode unit-test targets disable network sockets; ACP and Talon do too, and Talon's test target first runs the WhatsApp bridge Node tests. Use the [Testing Guide](./testing/testing-guide.md) for test design and the [development guide](./operations/development.md) for lockfiles, the dcode process-working-directory review gate, and release operations.

## Continue by question

- **Which layer or package owns this behavior?** [Repository Architecture Overview](./architecture/overview.md)
- **Where are the entry point, lifecycle boundary, and focused test?** [System Source Map](./architecture/source-map.md)
- **How does dcode start, bind a workspace, run, and shut down?** [Run a dcode Session](./workflows/run-dcode-session.md)
- **How does dcode project the Agent Client Protocol?** [Agent Client Protocol Integration](./integrations/acp.md)
- **How do dependencies, locks, and releases work?** [Development, Dependencies, Lockfiles, and Releases](./operations/development.md)
- **Which focused validation fits this change?** [Testing Guide](./testing/testing-guide.md)
