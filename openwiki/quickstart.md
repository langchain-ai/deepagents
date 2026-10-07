---
type: task routing guide
title: Repository Quickstart and Change Routing
description: Find the owning package, lifecycle boundary, focused regression neighborhood, and companion guide for SDK, dcode, protocol hosts, partners, evals, and repository automation changes.
tags: [deepagents, dcode, development, testing, evaluation, automation]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-07T08:06:51.789Z
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
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
  - id: openwiki-source-f2bb883b9cbec377de535c00
    resource: repo://libs/evals/pyproject.toml
  - id: openwiki-source-8565b7f246ed6e34051d8dfe
    resource: repo://libs/evals/README.md
  - id: openwiki-source-667fd72e0b93552f91d3888d
    resource: repo://libs/partners/AGENTS.md
  - id: openwiki-source-7da6afe7fe64c6589cf1fed0
    resource: repo://libs/README.md
generated: { by: "openwiki/0.4.2", at: "2026-10-07T08:06:51.789Z" }
---

# Repository Quickstart and Change Routing

Start with the package that owns the observable behavior, then follow the lifecycle or state boundary rather than repairing a downstream symptom. This is a monorepo of independently versioned packages under `libs/`: `deepagents` is the reusable agent harness and `deepagents-code` (`dcode`) is the terminal product. ACP, Talon, evals, and provider integrations are separate packages with their own delivery and test surfaces.

## Choose the owning domain

| Change affects… | Start here | Focused validation | Read next |
| --- | --- | --- | --- |
| Agent construction, built-in tools, profiles, permissions, subagents, or middleware ordering | `libs/deepagents/deepagents/graph.py` | `libs/deepagents/tests/unit_tests/test_graph.py`; use the closest middleware, permissions, or subagent test too | [SDK Construction and Execution](./architecture/sdk-construction-execution.md) · [Middleware Stack and Ordering](./architecture/middleware-stack.md) |
| dcode launch arguments, startup policy, headless mode, or ACP launch selection | `libs/code/deepagents_code/main.py` | `libs/code/tests/unit_tests/test_main.py` or `test_main_acp_mode.py` | [Run and Change a dcode Session](./workflows/run-dcode-session.md) |
| Terminal rendering, input, screens, queue presentation, or Textual behavior | `libs/code/deepagents_code/app.py` and `tui/` | matching mounted app/widget test; use `test_textual_patches.py` for compatibility-patch behavior | [dcode Client and Agent Server](./architecture/code-agent.md) · [Testing Guide](./testing/testing-guide.md) |
| Graph execution, tools, checkpoints, workspace runtime, or server-side hooks | `libs/code/deepagents_code/agent.py` and `server_graph.py` | `libs/code/tests/unit_tests/test_server_graph.py`, then the feature test | [dcode Client and Agent Server](./architecture/code-agent.md) |
| Model catalog, provider configuration, credentials, retry policy, or model selection | `libs/code/deepagents_code/model_*.py` and configuration modules | `test_model_config.py`, `test_model_metadata.py`, `test_model_retry.py`, and the owning configuration test | [Models and Harness Profiles](./concepts/profiles-models.md) |
| Thread resume or selection, SQLite records, durable names, checkpoints, ownership, or local inspection | `libs/code/deepagents_code/sessions.py`, `thread_ownership.py`, and thread UI/skill modules | `test_sessions.py`; add ownership, resume, name, selector, or inspector coverage only when that boundary changes | [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md) · [Run and Change a dcode Session](./workflows/run-dcode-session.md) |
| Slash-command names, aliases, descriptions, visibility, autocomplete metadata, or queue tier | `libs/code/deepagents_code/command_registry.py` | `test_command_registry.py`, relevant `test_app.py`, then `make commands-catalog` and `make lint` | [Run and Change a dcode Session](./workflows/run-dcode-session.md) |
| ACP editor protocol translation or editor-facing agent sessions | `libs/acp/` | the affected test in `libs/acp/tests/` | [System Source Map](./architecture/source-map.md) |
| Talon channels, scheduling, host lifecycle, or long-running execution | `libs/talon/` | the affected test in `libs/talon/tests/` | [Architecture Overview](./architecture/overview.md) |
| A provider or sandbox integration | `libs/partners/<provider>/` | that package’s tests and affected factory/configuration tests | [Sandbox Provider Integrations](./integrations/sandbox-partners.md) |
| Real-model trajectories, benchmark scores, or Harbor jobs | `libs/evals/` | deterministic eval-harness test first, then a targeted eval or Harbor run | [Testing Guide](./testing/testing-guide.md) |
| CI path routing, release packaging, or automation | `.github/workflows/`, `.github/actions/`, or `.github/scripts/` | nearest workflow/helper contract test | [System Source Map](./architecture/source-map.md) |

Use the [System Source Map](./architecture/source-map.md) when the package is clear but the state owner or narrowest regression neighborhood is not.

## dcode: identify the lifecycle owner first

Both `dcode` and `deepagents-code` invoke `deepagents_code:cli_main`. The terminal client and loopback agent server are separate processes: client code owns terminal input and presentation, while the server owns graph execution, models, tools, memory, checkpoints, and backend/runtime work. Keep a fix on the side that owns the state.

- **Threads and sessions:** treat a thread as durable state, not a transcript widget. Resume, rename, deletion, and inspection can cross SQLite metadata, LangGraph checkpoints, workspace binding, and writer ownership. Begin in `sessions.py`, then follow the state-persistence and session-workflow guides before changing a selector or modal.
- **Model policy:** keep provider resolution, credentials, catalog metadata, retry behavior, and policy in the model/configuration lifecycle owner—not a client screen. `DEEPAGENTS_CODE_<NAME>` overrides the unprefixed environment variable even when its value is empty, which deliberately suppresses the canonical value. Start with the model/config tests named above.
- **Commands:** `COMMANDS` is the declaration point for static slash commands. Queue-bypass sets include aliases and autocomplete entries derive from that registry; do not maintain competing app or widget metadata. Regenerate `COMMANDS.md` rather than editing the generated catalog.
- **Textual:** `app.py` imports `_textual_patches` as an import-time side effect before an `App` is created. Put framework compatibility behavior there and protect it with behavioral Textual tests; use mounted screen/app tests for presentation, focus, bindings, and asynchronous UI lifecycle.

For the detailed boundary and request lifecycle, see [dcode Client and Agent Server](./architecture/code-agent.md). For session recovery, safe local thread inspection, and command behavior, see [Run and Change a dcode Session](./workflows/run-dcode-session.md).

## SDK middleware changes

`create_deep_agent` in `graph.py` is the public graph-construction seam. It owns the default stack, profile resolution, skills and subagent assembly, permissions/approval wiring, and final graph compilation. Do not reproduce that assembly in dcode or a product host.

Middleware is an ordering contract: a supplied middleware with the same name replaces the existing entry in place; a new one is placed after the core stack and before the profile and tail layers. Protected filesystem and synchronous-subagent scaffolding cannot be excluded, because they underpin file tools/permissions and the `task` delegation path. Route ordering, replacement, profile exclusion, skill, subagent, or approval changes through [Middleware Stack and Ordering](./architecture/middleware-stack.md), and prove the compiled-stack behavior in `test_graph.py` before expanding to the specific subsystem test.

## Validate at the narrow boundary

Repository unit tests belong in `tests/unit_tests/` and are network-free and deterministic; networked contracts belong in `tests/integration_tests/`. Unaccepted pytest warnings fail the suite. Run commands from the package that owns the change:

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_sessions.py
make lint
```

The Code package’s unit-test target disables network sockets. Its `lint` target also checks generated slash-command catalog drift, and `make check` is its broader local CI target. For Textual work, assert visible behavior with the real mounted component—rendering, focus, screen stack, or persisted UI state—rather than private handler order. The [Testing Guide](./testing/testing-guide.md) maps the current session, inspector, middleware, command, and UI regression seams.

## Work package-locally and validate consumers

Use `uv` for interpreters, environments, and dependencies; every package’s Makefile is the authority for supported commands. `uv` provisions the appropriate interpreter, so no global Python version is pinned. Install dependencies explicitly in the changed package and use `make help` there to discover its targets.

Sibling dependencies are editable. A shared SDK interface change therefore needs validation in affected consumers; Code maps the SDK, ACP, and relevant partners to local source paths, and CI includes dependent coverage when the SDK changes. CI detects changed packages on pull requests; its Code and Talon filters explicitly include `libs/deepagents/**` for this reason.

## Evals, partners, and automation are separate change types

`libs/evals` is a real-LLM end-to-end behavioral suite: it captures agent trajectories and scores correctness and efficiency, with Harbor support for sandboxed benchmarks. During development it resolves `deepagents`, `deepagents-code`, and QuickJS from local source paths; Harbor staging copies the checked-out packages into its sandbox project. Establish deterministic behavior first, then answer model-behavior questions with a targeted eval.

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

`make evals` requires `MODEL` and runs `tests/evals`. Select a Harbor target for the intended sandbox environment rather than treating a local unit test as a benchmark result.

Partner packages are independently versioned and own their own environment, manifest, Makefile, and tests. Adding one also requires repository wiring for labels, change detection, CI, releases, and applicable Harbor/integration setup; begin with `libs/partners/AGENTS.md`. For workflow and release work, follow the existing `.github/` domain and its nearest validation rather than attaching automation behavior to a product package.

## Continue by question

- **What is the package and dependency direction?** [Architecture Overview](./architecture/overview.md)
- **Where is the state owner and closest focused test?** [System Source Map](./architecture/source-map.md)
- **How does the SDK assemble an agent?** [SDK Construction and Execution](./architecture/sdk-construction-execution.md)
- **What middleware ordering must hold?** [Middleware Stack and Ordering](./architecture/middleware-stack.md)
- **How do dcode client and server divide work?** [dcode Client and Agent Server](./architecture/code-agent.md)
- **How are models and profiles selected safely?** [Models and Harness Profiles](./concepts/profiles-models.md)
- **How are threads, names, checkpoints, and leases made durable?** [State, Checkpoints, and Persistent Records](./concepts/state-persistence.md)
- **How do I run or change a dcode session?** [Run and Change a dcode Session](./workflows/run-dcode-session.md)
- **Which focused regression protects this change?** [Testing Guide](./testing/testing-guide.md)
