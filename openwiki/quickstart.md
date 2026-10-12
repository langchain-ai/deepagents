---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: A routing map for maintainers of the independently versioned Deep Agents packages. Start with the behavior owner and runtime boundary, then run the narrowest package-local validation.
tags: [deepagents, dcode, talon, maintenance, development, testing, integrations]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-09T08:07:51.383Z
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
generated: { by: "openwiki/0.4.2", at: "2026-10-09T08:07:51.383Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the observable behavior, not the product that happens to display it. `libs/` contains independently versioned packages: the reusable `deepagents` SDK, the terminal `deepagents-code` product (`dcode`), ACP, evals, the experimental Talon host, and provider integrations. For package and dependency direction, see [Repository Architecture Overview](./architecture/overview.md); use [Source Map and Ownership Boundaries](./architecture/source-map.md) when the owner is not clear.

## Route the change

| Concern | Behavior owner and boundary | Read next | Focused validation |
| --- | --- | --- | --- |
| Graph construction, middleware, tools, backends, permissions, skills, or subagents | `libs/deepagents/`; public graph assembly | [Architecture overview](./architecture/overview.md), [Subagents and Skills](./concepts/subagents-skills.md), or [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md) | The matching test under `libs/deepagents/tests/unit_tests/`, beginning with `test_graph.py` for assembly changes |
| Terminal startup, Textual UI, configuration, threads, commands, graph execution, or remote runtime | `libs/code/`; distinguish terminal client/presentation from loopback server/runtime state | [Deep Agents Code Architecture](./architecture/code-agent.md), [State, Checkpoints, and Sessions](./concepts/state-persistence.md), or [Cost Tracking and Session Operations](./operations/cost-and-sessions.md) | The narrow `libs/code/tests/unit_tests/` test; use mounted/pilot UI behavior for UI changes |
| An editor protocol or ACP session semantics | `libs/acp/`, or dcode's ACP mode when exposing the prebuilt product | [Agent Client Protocol](./integrations/acp.md) | ACP package tests, plus dcode coverage if the change crosses its ACP boundary |
| Channels, admission/pairing, tool approvals, schedules, checkpoint/history storage, sandboxing, or host lifecycle | `libs/talon/`; channel adapter → Talon runtime → Deep Agents graph | [Talon Runtime Host](./integrations/talon.md) and [Talon Channels and Admission](./concepts/talon-channel-admission.md) | The closest `libs/talon/tests/` adapter/runtime test, including bridge tests when changing WhatsApp |
| A sandbox provider, MCP, or GitHub Action | The owning partner package or `.github/` action/workflow | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | Owner-package test first; add workflow/Harbor coverage only when that integration surface changes |
| Real-model quality, trajectories, scoring, or Harbor jobs | `libs/evals/`; a behavioral benchmark, not the default unit-test layer | [Run and Extend Evaluations](./workflows/run-evals.md) | First add a deterministic contract, then run `make evals MODEL=<id>` or the applicable Harbor job |
| Dependency floors, locks, CI selection, release PRs, or package fan-out | Package manifest/lock plus `.github/` automation | [Development, Dependencies, and Releases](./operations/development.md) | Package checks and the specific release/CI guard affected |

## Important seams

### SDK: change the assembly owner, not a consumer workaround

`create_deep_agent` is the SDK’s public graph-assembly seam. It composes the harness around LangChain/LangGraph; product packages should configure it rather than reproduce its middleware, tool, and backend setup. Same-named caller middleware replaces the corresponding base entry in place, but filesystem and synchronous-subagent scaffolding are protected because they provide core capability and delegation behavior. Start with the SDK architecture and prove stack behavior in `libs/deepagents/tests/unit_tests/test_graph.py` before following into the owning middleware or backend test.

### dcode: keep presentation and execution on their owning side

Both `dcode` and `deepagents-code` enter through `deepagents_code:cli_main`. From there, dcode separates terminal interaction and presentation in the client from graph execution, model/tool work, checkpoints, workspace runtime selection, and server-side hooks in its loopback agent server. Begin at `deepagents_code/main.py` for startup/mode routing; use `app.py` and `tui/` for presentation, and trace session or execution behavior into the server side rather than adding a UI-side state duplicate.

Two small seams commonly cause incomplete changes:

- **Textual compatibility:** `app.py` applies `_textual_patches` during import, before any Textual `App` exists. Keep compatibility behavior at that boundary and cover its observable behavior in `test_textual_patches.py`.
- **Slash commands:** `COMMANDS` in `command_registry.py` is the source registry. Alias-aware bypass sets and autocomplete are derived from it. After changing catalog-relevant metadata, run `make commands-catalog`; `make lint` rejects catalog drift.

For model configuration, a `DEEPAGENTS_CODE_<NAME>` environment variable takes precedence over the unprefixed name even when the prefixed value is empty. Investigate configuration resolution before changing an authentication or model-selection screen.

### Talon: treat a channel as an operator-facing host boundary

Talon owns the process lifecycle for channel adapters, cron scheduling, and agent runtime in one event loop. It is experimental and is not a production security boundary: channel access can reach the operator’s model credentials, MCP tools, and host resources; sandboxing is opt-in and does not cover MCP tools. Route channel admission and Slack-specific behavior to the channel/admission guide, runtime persistence and sandbox lifecycle to the Talon runtime guide, and tool-policy changes to Talon’s approval boundary rather than SDK permission code.

Talon’s host-level interruption behavior is also distinct from dcode sessions: a new message cancels the active turn on that conversation, records an interruption after the latest committed checkpoint, and starts on the same thread. If cancellation fails to finish within 30 seconds, it leaves the existing run isolated and does not start the new turn. Exercise that behavior in Talon runtime tests rather than a generic SDK graph test.

### ACP, evals, and partners have their own delivery boundaries

ACP can host a custom Deep Agent in an editor, while `dcode --acp` exposes the prebuilt dcode product over stdio. Do not assume a change to one is a change to the other; validate the relevant side of that protocol boundary.

The evals package is an end-to-end, real-LLM behavioral suite: it captures agent trajectories and scores correctness and efficiency, with Harbor for sandboxed benchmarks. It is evidence for model behavior, not a substitute for a deterministic regression test. Each partner package similarly owns its own manifest, environment, Makefile, tests, and versioning; adding one also requires repository wiring for CI, labels, release, and applicable Harbor/integration surfaces.

## Work and validate locally

Use `uv` for interpreters, environments, and dependencies; each package’s `Makefile` is the command authority. `uv` provisions the appropriate interpreter, so do not impose a global Python pin. Work in the owning package, synchronize explicitly, inspect `make help`, and run the narrowest behavior-observing test before the package lint target.

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_sessions.py
make lint
```

Network-free deterministic tests belong in `tests/unit_tests/`; networked contracts belong in `tests/integration_tests/`; unaccepted pytest warnings fail the suite. Code’s normal unit-test target disables network sockets, while `integration_test` is its intentional network boundary. For Textual work, test a mounted interaction—rendering, focus, bindings, screen stack, or persisted outcome—rather than incidental handler order. Use the [Testing Guide](./testing/testing-guide.md) to select the exact SDK, dcode, Talon, generated-artifact, async/remote, or integration layer.

Local sibling dependencies are editable. Therefore validate affected consumers after changing a shared interface: CI includes `libs/deepagents/**` in the Code and Talon change filters because both consume the SDK locally.

## Dependencies, releases, and evaluation escalation

For ordinary package changes, use the package-local manifest and lockfile path described in [Development, Dependencies, and Releases](./operations/development.md). `deepagents-code` has an exact SDK pin: update it in the same change when dcode needs new SDK behavior. Release-please manages release PRs per package; before merging a release PR, the curated-notes flow requires a draft, maintainer review/edit, and `@release-bot apply` unless the explicit skip label is used.

Run evals only after deterministic coverage establishes the contract:

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`make evals` requires `MODEL` and runs `tests/evals`. Follow the evaluation workflow for model groups, trajectory/scoring changes, or Harbor staging rather than interpreting a unit test as benchmark evidence.

## Continue by question

- **What owns this behavior and which package consumes which?** [Repository Architecture Overview](./architecture/overview.md)
- **Where is the implementation owner and closest regression test?** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **How do I work on dependencies, locks, CI, or releases?** [Development, Dependencies, and Releases](./operations/development.md)
- **How do I change or operate dcode?** [Deep Agents Code Architecture](./architecture/code-agent.md)
- **How do I operate Talon or change its runtime boundary?** [Talon Runtime Host](./integrations/talon.md)
- **Which test command/layer protects this change?** [Testing Guide](./testing/testing-guide.md)
