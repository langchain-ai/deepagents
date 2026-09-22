---
type: task routing guide
title: Deep Agents Repository Quickstart
description: A concise map from common Deep Agents repository changes to the owning package, related wiki guidance, focused validation, and release checks.
tags: [deepagents, monorepo, development, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-22T08:05:41.799Z
sources:
  - id: openwiki-source-248c9119a9fc632bf11e2c4a
    resource: repo://.github/workflows/check_partner_bounds.yml
  - id: openwiki-source-477b456c1269748d01a9f090
    resource: repo://.github/workflows/check_release_deps.yml
  - id: openwiki-source-d70f26033a54319a6c391236
    resource: repo://.github/workflows/check_sdk_pin.yml
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
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
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-da577cbe81ec29338f1388b2
    resource: repo://libs/partners/daytona/pyproject.toml
  - id: openwiki-source-936554ac5f0a201f8696be25
    resource: repo://libs/partners/modal/pyproject.toml
  - id: openwiki-source-b38d20ec21c25c8c726dc1b6
    resource: repo://libs/partners/quickjs/pyproject.toml
  - id: openwiki-source-8d2c8381956c1c023bcdb565
    resource: repo://libs/partners/runloop/pyproject.toml
  - id: openwiki-source-03a39f44d8ccfde2fd47e57a
    resource: repo://libs/partners/vercel/pyproject.toml
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
generated: { by: "openwiki/0.4.2", at: "2026-09-22T08:05:41.799Z" }
---

# Deep Agents Repository Quickstart

Start in the package that owns the behavior, not at the repository root. Deep Agents is the opinionated harness layer over LangChain's `create_agent()` and the LangGraph runtime. This page routes work to the appropriate domain page; it intentionally does not repeat their runtime contracts.

## Route the task

| Change | Owner and next wiki page | Focused validation |
| --- | --- | --- |
| SDK graph assembly, middleware, backends, skills, memory, filesystem, or permissions | `libs/deepagents/`; [Architecture overview](./architecture/overview.md) | Run the closest unit test, then `make test TEST_FILE=tests/unit_tests/<file>.py` and `make lint`. |
| dcode runtime assembly, workspace configuration, persistence/resume, offload, sandbox, costs, CLI, or TUI | `libs/code/`; [Run and debug a dcode session](./workflows/run-dcode-session.md), [Testing guide](./testing/testing-guide.md) | Use `make test TEST_FILE=tests/unit_tests/test_<area>.py`; run `make integration_test TEST_FILE=...` only for an integration boundary. |
| ACP editor or stdio behavior | `libs/acp/` and, where dcode is launched, `libs/code/`; [Source map](./architecture/source-map.md) | In `libs/acp`, use `make test TEST_FILE=tests/test_<area>.py`; test dcode ACP mode separately when its launcher changed. |
| Sandbox/provider adapter | Matching `libs/partners/<provider>/`; [Sandbox and partner backends](./integrations/sandbox-partners.md) | Run the adapter's package-local tests and the changed SDK or dcode contract test. |
| Evaluation scenario or Harbor execution | `libs/evals/`; [Testing guide](./testing/testing-guide.md) | Start with the owning unit test; use a real-model evaluation only when the scenario requires it. |
| Talon channels, scheduling, or host behavior | `libs/talon/`; [Architecture overview](./architecture/overview.md) | `make test TEST_FILE=tests/<focused-path>.py`, followed by `make lint`; the test target also runs the WhatsApp bridge Node tests. |
| Package version, dependency, lockfile, or release automation | The changed package, then `libs/`; [Development, CI, and releases](./operations/development.md) | Run package checks and `make -C libs lock-check` when a lockfile is affected. |

## Package map and compatibility

`libs/` is a monorepo of independently versioned packages. Each package owns its `pyproject.toml`, `Makefile`, and README; there is no root `pyproject.toml`. First-party dependencies are editable locally, so make an SDK change in `libs/deepagents` visible to its sibling consumers without a separate publish.

| Package or group | Current baseline | Role | Python requirement |
| --- | ---: | --- | --- |
| `deepagents` | `0.7.17` | SDK: `create_deep_agent`, middleware, and backends | `>=3.11,<4.0` |
| `deepagents-code` | `0.1.73` | Prebuilt terminal coding agent, invoked as `dcode` | `>=3.12,<4.0` |
| `deepagents-acp` | `0.0.12` | Agent Client Protocol editor integration | `>=3.11` |
| `deepagents-evals` | source version `0.0.1` | Evaluation suite and Harbor integration | `>=3.12,<3.14` |
| `deepagents-talon` | `0.0.8` | Experimental local long-running host | `>=3.12` |
| Partners | Daytona `0.0.8`; Modal `0.0.6`; Runloop `0.0.7`; Vercel `0.0.2`; QuickJS `0.3.7` | Provider and sandbox integrations | `>=3.11,<4.0` |

Choose the interpreter from the changed package's `requires-python`; `uv` provisions it, so there is no repository-wide Python version to pin. The partner packages are package-local SDK integration boundaries, not dependencies required for every repository task.

```mermaid
flowchart TD
    Code["deepagents-code and dcode"] --> SDK["deepagents SDK"]
    ACP["deepagents-acp"] --> SDK
    Evals["deepagents-evals"] --> SDK
    Evals --> Harbor["Harbor"]
    Evals --> Code
    Talon["deepagents-talon"] --> SDK
    Talon --> Code
    Partners["Partner packages"] --> SDK
```
*Published package dependencies flow from consumers and adapters to the SDK, dcode, or Harbor capability they use.*

The manifest relationship is directional: Code pins `deepagents==0.7.17`; ACP depends on `deepagents`; evals depends on Deep Agents, dcode, and Harbor; and Talon depends on Deep Agents and dcode. Treat an SDK version or dependency-bound change as a consumer and release-check change as well.

## Focused edit–test loop

Use `uv` for interpreters, environments, and dependencies, and use each package's `Makefile` as the authority for supported commands.

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_server_graph.py
make lint
```

Both the SDK and Code `test` targets accept `TEST_FILE`; their standard unit runs disable network sockets except Unix sockets, run in parallel, and report coverage. Code's `make lint` additionally checks its generated command catalog and process-current-working-directory rule. ACP uses `tests/` as its default `TEST_FILE` and applies a 10-second pytest timeout.

For a package-metadata change, update its lockfile deliberately and validate at the appropriate scope:

```bash
cd libs/code
uv lock
make check
make -C ../ lock-check
```

`libs/code`'s `make check` runs lint, import checks, unit tests, extras/version consistency checks, a lockfile check, and an advisory SDK-pin check. On a Code release PR, the SDK-pin workflow warns for a stale SDK pin, while publication enforces that the Code pin is at least the workspace SDK version. Release dependency validation removes editable local sources and resolves changed release manifests against PyPI, so a locally working sibling dependency does not by itself establish a publishable dependency graph. The partner-bounds workflow is advisory: it flags an SDK release that is excluded by a partner's declared `deepagents` upper bound.

## Where to go next

- [Repository architecture overview](./architecture/overview.md) — ownership boundaries and SDK stack.
- [Repository source map](./architecture/source-map.md) — source, entrypoint, and test navigation.
- [Development, CI, and releases](./operations/development.md) — setup, lockfiles, CI, and publishing details.
- [Testing guide](./testing/testing-guide.md) — dcode-focused test selection and test layers.
- [Run and debug a dcode session](./workflows/run-dcode-session.md) — operational dcode lifecycle and recovery.
