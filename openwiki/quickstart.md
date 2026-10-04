---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Route a behavioral change to its independently released package, public entrypoint, neighboring design guide, and narrowest regression seam. Use this page to choose package-local development and avoid accidental release or dependency fan-out.
tags: [deepagents, navigation, development, testing, releases]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-03T08:05:07.881Z
sources:
  - id: openwiki-source-9a1c436646ef8c4f6dde787a
    resource: repo://.github/RELEASING.md
  - id: openwiki-source-5e59f90a38f5bdf9ed76984b
    resource: repo://.release-please-manifest.json
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-bb78950c8b36b7b9f6746e96
    resource: repo://libs/acp/pyproject.toml
  - id: openwiki-source-6c2e9cfaa20096e021221d47
    resource: repo://libs/code/CHANGELOG.md
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-478a579b56d29c6928ec2320
    resource: repo://libs/deepagents/pyproject.toml
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
  - id: openwiki-source-e2a176528c4d510dcc417820
    resource: repo://libs/talon/CHANGELOG.md
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-03T08:05:07.881Z" }
---

# Repository Quickstart and Change Routing

This is a monorepo of independently versioned packages under `libs/`, not one root Python project. Start with the package that owns the observable behavior; inspect its README, `pyproject.toml`, `Makefile`, implementation, and nearest test. Shared harness behavior belongs in the SDK; product UI and session behavior in dcode; editor protocol translation in ACP; and long-running local-host policy in Talon.

## Route the observable change

| If the change affects… | Own it in / start at | Read next | Validate first |
| --- | --- | --- | --- |
| Agent graph construction, default tools, middleware, backends, filesystem, memory, skills, subagents, or approvals | **SDK — `libs/deepagents`**: `deepagents.create_deep_agent` / `deepagents/graph.py` | [Architecture overview](./architecture/overview.md) · [Build or modify a Deep Agent](./workflows/build-a-deep-agent.md) | `libs/deepagents/tests/unit_tests/test_graph.py`, then the feature-specific middleware or backend test |
| Terminal startup, command parsing, TUI, prompt, cost, session, approval, MCP, or sandbox UX | **dcode — `libs/code`**: `dcode` and `deepagents-code` both target `deepagents_code:cli_main`; execution composition is in `agent.py` and `server_graph.py` | [Run and change a dcode session](./workflows/run-dcode-session.md) | Command/server test, prompt smoke snapshot, or the affected `tests/unit_tests/tui/` widget test |
| Editor events, ACP session stream, model/mode selection, permissions, or session load | **ACP — `libs/acp`**: `AgentServerACP` in `deepagents_acp/server.py` | [System source map](./architecture/source-map.md) | `libs/acp/tests/test_agent.py` before an editor smoke test |
| Local bootstrap, channels, pairing, conversation cancellation, checkpoints/history, cron, MCP, or sandbox lifecycle | **Talon — `libs/talon`**: `deepagents-talon` targets `deepagents_talon.__main__:main`; host lifecycle is in `host.py` | [Talon runtime integration](./integrations/talon.md) | `test_host.py`, `test_runtime.py`, a channel test, or the focused persistence/cron resource test |
| Real-model trajectory, tool-use quality, or benchmark score | **Evals — `libs/evals`**: `deepagents-evals` / `deepagents_evals.cli:main` | [Testing guide](./testing/testing-guide.md) | A focused real-LLM eval; retain deterministic package coverage for the underlying contract |
| A provider sandbox integration | **Partner — `libs/partners/<provider>`** | [Development, packaging, and releases](./operations/development.md) | The partner package test plus consuming factory/configuration wiring |
| CI, labeling, release checks, or workflow helpers | **Automation — `.github/`** | [Development, packaging, and releases](./operations/development.md) | The closest helper-script or workflow-contract test, then the relevant package/lock check |

The [System Source Map](./architecture/source-map.md) expands these entrypoints and test neighborhoods. It is the right next stop when the package is clear but the state owner is not.

## Develop package-locally

Use `uv` for interpreters, environments, and dependencies. Each package's `Makefile` is the command authority; `uv` provisions the required interpreter, so do not globally pin Python or substitute `pip`, Poetry, or Conda. Deep Agents and ACP require Python `>=3.11`; Code and Talon require `>=3.12`.

```bash
cd libs/deepagents
uv sync --all-groups
make help
make test TEST_FILE=tests/unit_tests/test_specific.py
make lint
```

Run equivalent commands from the changed package. Use `uv run ...` only for a one-off direct command. From `libs/`, `make lint`, `make lock`, and `make lock-check` fan out across packages; use them for cross-package work, not as a substitute for the focused owner-boundary test. Local sibling dependencies are editable, so validate affected consumers when a shared interface changes.

Network-free deterministic coverage belongs in `tests/unit_tests/`; networked contracts belong in `tests/integration_tests/`. Warnings not explicitly accepted fail pytest. The [Testing Guide](./testing/testing-guide.md) maps focused seams across SDK, dcode, and Talon.

## Package and release routing

| Release component | Current version | Distribution / entrypoint |
| --- | ---: | --- |
| Deep Agents | `0.7.21` | `deepagents` / `create_deep_agent` |
| ACP | `0.0.12` | `deepagents-acp` / `AgentServerACP` |
| Code | `0.1.80` | `deepagents-code` / `dcode` |
| Talon | `0.0.9` | `deepagents-talon` / `deepagents-talon` |
| Daytona | `0.0.8` | `langchain-daytona` |
| Modal | `0.0.6` | `langchain-modal` |
| Runloop | `0.0.7` | `langchain-runloop` |
| Vercel | `0.0.2` | `langchain-vercel-sandbox` |
| QuickJS | `0.3.8` | `langchain-quickjs` |

The release manifest tracks these components independently. A package manifest owns its distribution name, Python range, dependencies, extras, and console entrypoints; regenerate its lock through the package workflow rather than editing it by hand. In particular, `deepagents-code` pins an exact `deepagents==` version: update that pin in the feature PR when Code begins to require new SDK behavior.

Keep a bump-worthy user-visible PR to one release component. Release-please assigns a releasable commit by changed **paths**, not its Conventional Commit scope: touching real files in several managed packages opens a release PR for each. Put cross-package dependency bounds and lockfile regeneration in a separate `chore(deps):` change; `chore` is hidden from release creation. Never use an empty commit to repair release notes: it has no package paths and can fan out to every managed component. For a deliberate new SDK line, lift in-tree consumer bounds first, then release in dependency order—partners with published caps, exact-pinned consumers such as Code, then packages that depend on those consumers. [Development, packaging, and releases](./operations/development.md) covers the release checks and recovery labels.

A new partner is both a package and automation change: beyond its environment, manifest, Makefile, and tests, add issue forms, dependency automation, labels, CI/change detection, release configuration, and applicable Harbor, integration-test, and secret wiring.

## Operational boundaries

- **dcode:** By default it trusts the directory in which it runs. Approval prompts gate model-requested tool calls, but project artifacts are read before approval; use a remote sandbox for an untrusted repository. See [Run and change a dcode session](./workflows/run-dcode-session.md).
- **Talon:** It is experimental, not a production or multi-tenant security boundary. Channel access is effectively access to the operator's agent, credentials, MCP tools, and host resources. Shell and filesystem tools use the host unless `DEEPAGENTS_TALON_SANDBOX` selects a sandbox; startup failure exits rather than silently falling back, while MCP tools and channel media remain host-resident. See [Talon runtime integration](./integrations/talon.md).

## Continue by question

- **Where are the entrypoint, state owner, and nearest test?** [System Source Map](./architecture/source-map.md)
- **How do the SDK, dcode, ACP, and Talon relate?** [Architecture Overview](./architecture/overview.md)
- **How do I change agent assembly safely?** [Build or Modify a Deep Agent](./workflows/build-a-deep-agent.md)
- **How do I run or change a terminal session?** [Run and Change a dcode Session](./workflows/run-dcode-session.md)
- **How do I configure the local host?** [Talon Runtime Integration](./integrations/talon.md)
- **How do I choose a test seam?** [Testing Guide](./testing/testing-guide.md)
- **How do I package, lock, or release?** [Development, Packaging, and Releases](./operations/development.md)
