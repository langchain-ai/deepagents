---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: Routing map for changes across the independently released Deep Agents SDK, dcode, ACP, Talon, evals, and partner packages. Start at the behavior owner, preserve its boundary, and run focused package validation.
tags: [deepagents, dcode, talon, maintenance, development, testing, integrations]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-12T08:20:17.385Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
  - id: openwiki-source-6f5b1b7a043ee1d414708793
    resource: repo://libs/code/ARCHITECTURE.md
  - id: openwiki-source-fdf5afeb1dd1d11652374e88
    resource: repo://libs/code/deepagents_code/app.py
  - id: openwiki-source-fcc71dc507b62bee0432e12e
    resource: repo://libs/code/deepagents_code/command_registry.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-4a7b6def251b42596a410ebc
    resource: repo://libs/code/deepagents_code/model_config.py
  - id: openwiki-source-a9eb680bb6bdae179f52a3ac
    resource: repo://libs/code/deepagents_code/server_graph.py
  - id: openwiki-source-006b62af9993da1b48c11de8
    resource: repo://libs/code/Makefile
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-d1add1f969d9ef0a3687cc02
    resource: repo://libs/code/tests/unit_tests/test_textual_patches.py
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
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-12T08:20:17.385Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the observable behavior, rather than the product that displays it. The monorepo independently versions the core `deepagents` SDK, terminal `deepagents-code` product (`dcode`), ACP integration, evals, experimental Talon host, and provider integrations under `libs/`. See [Repository Architecture Overview](./architecture/overview.md) for package direction and [Source Map and Ownership Boundaries](./architecture/source-map.md) when the implementation owner is unclear.

## Route the change

| Change or question | Behavior owner and boundary | Focused validation |
| --- | --- | --- |
| Graph assembly, middleware, filesystem/tools, backend, permissions, skills, or subagents | `libs/deepagents/`; `create_deep_agent` is the public assembly seam | Matching SDK unit test; start with `tests/unit_tests/test_graph.py` for assembly/stack changes. See [SDK Construction and Execution](./architecture/sdk-construction-execution.md). |
| Terminal startup, Textual UI, commands, model configuration, sessions, hooks, graph execution, or remote runtime | `libs/code/`; client presentation/input is separate from loopback server execution/runtime state | Narrow `libs/code/tests/unit_tests/` coverage; use mounted/pilot behavior for UI changes. See [Deep Agents Code Architecture](./architecture/code-agent.md) and [Run a dcode Session](./workflows/run-dcode-session.md). |
| Editor protocol or ACP session semantics | `libs/acp/` for `AgentServerACP`, or dcode ACP mode for the prebuilt product | ACP tests, plus dcode coverage when the prebuilt `dcode --acp` boundary changes. |
| Channels, pairing/admission, schedules, tool approval, checkpoints/history, sandboxing, or host lifecycle | `libs/talon/`; adapters, scheduler, and agent runtime share the host lifecycle | Closest Talon adapter/runtime test. Treat interruption and persistence as host behavior, not a generic SDK graph regression. |
| Sandbox provider or other optional integration | Owning `libs/partners/<provider>/` package, then consumer configuration | Owner-package tests first; add Harbor/integration coverage only if that surface changes. See [Sandbox and Partner Integrations](./integrations/sandbox-partners.md). |
| Real-model quality, trajectories, scoring, or Harbor benchmark execution | `libs/evals/`; an end-to-end behavioral layer | Add a deterministic contract first, then run the relevant eval or Harbor job. See [Testing Guide](./testing/testing-guide.md). |
| Dependency floor, lock, package release, CI selection, or cross-package compatibility | Package manifest/lock and `.github/` automation | Run package checks plus the specific CI/release guard. See [Development, Dependencies, and Releases](./operations/development.md). |

## Preserve the important boundaries

### SDK assembly is the shared behavior owner

`create_deep_agent` owns public SDK graph assembly. Caller middleware with the same name replaces a base middleware entry in place; new middleware is placed before the profile/caching/memory tail. A harness profile cannot exclude the filesystem or synchronous-subagent scaffolding: filesystem middleware supplies the built-in file tools and permission enforcement, while subagent middleware supplies the `task` handler. Change this contract in the SDK, not by duplicating it in dcode or Talon.

Local sibling dependencies are editable, so an SDK interface change is immediately visible to consumers. Run affected consumer validation as well: CI includes `libs/deepagents/**` in dcode and Talon filters specifically because those products consume the SDK from editable paths.

### dcode has client and server ownership

Both `dcode` and `deepagents-code` invoke `deepagents_code:cli_main`; start in `main.py` for command-line startup and mode routing. The client owns terminal interaction and presentation, while the loopback agent server owns graph execution, model/tool work, checkpoints, workspace runtime selection, and server hooks. Keep state on its owning side of that boundary rather than creating a UI-side duplicate.

Two frequent dcode seams deserve explicit checks:

- `app.py` imports `_textual_patches` before any Textual `App` is constructed. Put compatibility changes at that import-time boundary and cover observable patch behavior in `test_textual_patches.py`.
- `COMMANDS` is the slash-command registry. Alias-aware bypass tiers and autocomplete are derived from it; after catalog-relevant changes, run `make commands-catalog`, then `make lint`, which checks catalog drift.

For model configuration, `DEEPAGENTS_CODE_<NAME>` takes precedence over an unprefixed variable. Presence of the prefixed variable—even with an empty value—suppresses the canonical variable, so diagnose resolution before changing authentication or model selection.

### ACP, Talon, evals, and partners are separate delivery boundaries

ACP supports both a custom Deep Agent hosted through `AgentServerACP` and the separately prebuilt dcode product hosted on stdio by `dcode --acp`; validate the applicable protocol side rather than assuming they are interchangeable.

Talon is an experimental local host for channel adapters, cron scheduling, and agent runtime in one event loop. It is not a production security boundary: sandboxing is opt-in and does not cover MCP tools. For a conversation, an incoming message cancels the active turn, records an interruption marker after the latest committed checkpoint, then starts the new message on the same thread. If cancellation does not finish within 30 seconds, Talon leaves the old run isolated and does not start the new turn.

The evals package is a real-LLM behavioral suite: it captures trajectories and scores correctness and efficiency, with Harbor integration for sandboxed benchmarks. It is evidence of behavioral quality, not a substitute for deterministic regression coverage. Partner packages each own their versioning, environment, manifest, Makefile, and tests; a new partner also needs the associated CI, labels, release, and applicable Harbor/integration wiring.

## Work, test, and release safely

Use `uv` for interpreters, environments, and dependencies; each package's `Makefile` is the command authority. `uv` provisions the suitable Python interpreter automatically, so do not impose a global Python pin. Work from the owning package, synchronize explicitly, inspect `make help`, run the narrowest behavior-observing test, then run package lint.

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_sessions.py
make lint
```

Network-free deterministic tests belong in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. In Code, the normal unit-test target disables network sockets, while `integration_test` is the deliberate network boundary.

For evals, `make evals` requires `MODEL` and runs `tests/evals`:

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`deepagents-code` pins an exact SDK version. When dcode requires new SDK behavior, update that pin in the same pull request. Release-please creates package-specific release PRs; curated release notes require a draft, maintainer review/edit, and `@release-bot apply` unless `ci:skip-curated-notes` intentionally skips that flow.

## Continue by question

- **Which package and runtime boundary own this behavior?** [Repository Architecture Overview](./architecture/overview.md)
- **Where are the owner and closest regression tests?** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **How does dcode start, execute, and present a session?** [Deep Agents Code Architecture](./architecture/code-agent.md) and [Run a dcode Session](./workflows/run-dcode-session.md)
- **How do I change dependencies, locks, CI, or releases?** [Development, Dependencies, and Releases](./operations/development.md)
- **Which focused test layer should I run?** [Testing Guide](./testing/testing-guide.md)
