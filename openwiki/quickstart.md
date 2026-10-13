---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: Routing map for changes across independently released Deep Agents packages, their current dependency relationships, and focused validation paths. Start at the behavior owner, validate affected editable consumers, and release each package independently.
tags: [deepagents, dcode, talon, acp, maintenance, development, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-13T08:11:05.854Z
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
  - id: openwiki-source-8134f31fb22085cb0e6b4054
    resource: repo://libs/acp/README.md
  - id: openwiki-source-f188d13aef04b643ced86520
    resource: repo://libs/acp/tests/unit_tests/test_execute_permissions.py
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
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-fdd0c2c3830b8e9a88502a57
    resource: repo://libs/talon/README.md
  - id: openwiki-source-e0db085f967450f48196d999
    resource: repo://libs/talon/tests/test_media.py
  - id: openwiki-source-57a0613315e23277d358df76
    resource: repo://libs/talon/tests/unit_tests/test_sandbox.py
generated: { by: "openwiki/0.4.2", at: "2026-10-13T08:11:05.854Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the observable behavior, not the product that happens to expose it. `libs/` contains independently versioned SDK, dcode, ACP, evals, Talon, and provider-integration packages. The current release manifest is `deepagents` **0.7.24**, `deepagents-acp` **0.0.13**, `deepagents-code` **0.1.84**, and `deepagents-talon` **0.0.9**. See [Repository Architecture Overview](./architecture/overview.md) for dependency direction and [Source Map and Ownership Boundaries](./architecture/source-map.md) when the implementation owner is unclear.

## Route the change

| Change or question | Owner and boundary | Focused validation |
| --- | --- | --- |
| SDK graph construction, middleware, filesystem permissions, backends, skills, or subagents | `libs/deepagents/`; `create_deep_agent` is the public assembly seam | Run the nearest SDK unit test. Start with `tests/unit_tests/test_graph.py` for stack construction and `tests/unit_tests/test_permissions.py` for permission/HITL behavior. See [SDK Construction and Execution](./architecture/sdk-construction-execution.md). |
| Terminal startup, Textual UI, slash commands, model configuration, sessions, hooks, or server execution | `libs/code/`; presentation/input is client-owned, while execution/runtime state is server-owned | Run the narrowest `libs/code/tests/unit_tests/` test, then `make lint`; use mounted/pilot behavior for UI changes. See [Models, Profiles, and Provider Configuration](./concepts/profiles-models.md). |
| ACP adapter, editor session, approval, or local-context behavior | `libs/acp/`; distinguish a custom `AgentServerACP` from dcode ACP mode | Run `make test TEST_FILE=tests/unit_tests/test_execute_permissions.py` for exact-command approval changes and `make test TEST_FILE=tests/unit_tests/test_local_context.py` for compaction/context changes. Also run dcode coverage if `dcode --acp` changes. See [Agent Client Protocol](./integrations/acp.md). |
| Channel adapters, conversation interruption, sandbox routing, media, schedules, approvals, checkpoints, or host lifecycle | `libs/talon/`; the host owns these behaviors | Run the closest Talon test. Start with `tests/unit_tests/test_sandbox.py` for sandbox routing/lifecycle and `tests/test_media.py` for media bounds. See [Talon Runtime Host](./integrations/talon.md). |
| Optional sandbox/provider integration | Owning `libs/partners/<provider>/`, then the consumer configuration; each partner owns its versioning, environment, manifest, Makefile, and tests | Test the partner package first; add Harbor/integration coverage only when that surface changes. Adding a partner also needs applicable CI, labels, release, and Harbor/integration wiring. See [Sandbox and Partner Integrations](./integrations/sandbox-partners.md). |
| Real-model quality, trajectories, scoring, or Harbor benchmarks | `libs/evals/`; an end-to-end behavioral layer | Add deterministic coverage first, then run the relevant eval or Harbor job. See [Testing Guide](./testing/testing-guide.md). |
| Dependency floor, lock, package release, CI selection, or cross-package compatibility | Package manifest/lock plus `.github/` automation | Run package checks and the specific release/CI guard. See [Development, Dependencies, and Releases](./operations/development.md). |

## Respect dependency and delivery boundaries

### Package relationships and release readiness

The SDK is the shared behavior owner. `deepagents-code==0.1.84` declares the exact `deepagents==0.7.24` pin, so a dcode feature that relies on new SDK behavior must update that pin in the same pull request. ACP 0.0.13 consumes the editable sibling SDK without an upper/lower SDK specifier; dcode accepts `deepagents-acp>=0.0.10,<1.0.0`. Talon 0.0.9 consumes editable SDK and dcode siblings and publishes compatibility floors of `deepagents>=0.7.24` and `deepagents-code>=0.1.71,<1.0.0`.

Local sibling dependencies are editable. Therefore an SDK interface change is immediately visible to consumers and must be validated in each affected consumer; CI includes SDK changes in the dcode, ACP, Talon, evals, and partner filters. Release-please opens a package-specific release PR for each managed package. Before merging one, review the generated changelog and follow the curated-notes flow—draft, review/edit, then `@release-bot apply`—unless the intentional `ci:skip-curated-notes` exception is used.

### SDK changes belong in the public builder

`create_deep_agent` owns public graph assembly. Middleware supplied with the same name replaces a base entry in place; newly supplied middleware goes before the profile/caching/memory tail, while protected filesystem and synchronous-subagent scaffolding cannot be removed by a harness profile. Make permission changes in the SDK rather than reimplementing them in a host: permission tests cover conservative recursive-delete behavior, including denied descendants, as well as filesystem tool behavior. For the full stack and extension rules, see [Middleware Assembly Stack](./architecture/middleware-stack.md), [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md), and [Filesystem Tools and Policies](./concepts/tools-filesystem.md).

### dcode is a client/server product

Both `dcode` and `deepagents-code` invoke `deepagents_code:cli_main`, making `main.py` the startup and policy-routing entrypoint. The client owns terminal interaction and presentation; the loopback agent server owns graph execution, model/tool work, checkpoints, workspace runtime selection, and server hooks. Keep state on its owning side of that boundary.

Two high-frequency maintenance seams are deliberate registries:

- `_textual_patches` is imported before a Textual `App` is constructed. Compatibility changes belong at that import-time boundary and need behavioral patch tests.
- `COMMANDS` is the source registry for slash commands: aliases feed bypass sets and autocomplete is derived from it. After a catalog-relevant change, run `make commands-catalog` and then `make lint`, which rejects catalog drift.

For credentials and model selection, `DEEPAGENTS_CODE_<NAME>` wins over the unprefixed variable. A present prefixed variable, even if empty, prevents fallback to the canonical variable; reproduce this precedence with the focused model-configuration test before altering authentication logic.

### ACP and Talon are distinct hosts

ACP supports a custom Deep Agent hosted by `AgentServerACP` and the separately prebuilt dcode product hosted on stdio by `dcode --acp`; validate the applicable protocol boundary rather than treating them as interchangeable. ACP reusable execute approval is deliberately session-scoped and exact: changing command text, arguments, cwd, mode, model, or session requires a new approval. Its local-context tests protect prompt preservation and snapshot deduplication across summarization/compaction.

Talon is an experimental local host for channel adapters, cron scheduling, and agent runtime in one event loop, not a production security boundary. Sandboxing is opt-in and does not contain MCP tools. With a Talon sandbox, shell/filesystem work is routed to the remote sandbox, while assistant `skills/`, `memory/`, media blobs, MCP, web tools, and channel-media handling remain host-routed; a sandbox startup failure exits rather than falling back to host execution. Talon creates and normally cleans up sandboxes, except an explicitly attached sandbox is never deleted. Its media validation rejects absolute, traversal, home-relative, URI, and symlink-escape outbound paths.

For a conversation, a new message cancels the active turn, records an interruption marker after the latest committed checkpoint, and starts on the same thread. If cancellation exceeds 30 seconds, Talon isolates the old run and does not start the new turn. See [Runtime Behavior, State, and Delegation](./architecture/runtime-behavior.md) and [Security and Operational Boundaries](./operations/security.md) before changing these paths.

## Work and validate safely

Use `uv` for interpreters, environments, and dependencies; each package `Makefile` is the command authority. `uv` provisions the suitable interpreter, so do not impose a global Python pin. Work in the owning package, run `uv sync --all-groups` (or its documented package-specific group), inspect `make help`, run the narrowest behavior-observing test, then lint.

```bash
cd libs/code
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_model_config.py
make lint
```

Network-free, deterministic tests belong in `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unaccepted pytest warnings fail the suite. The Code, SDK, ACP, and Talon normal test targets disable network sockets; Talon also runs its WhatsApp bridge Node tests as part of `make test`.

For the real-model eval layer, `make evals` requires `MODEL` and runs `tests/evals`; it captures trajectories and scores correctness and efficiency, while Harbor supplies sandboxed benchmarks. It complements rather than replaces deterministic regression tests.

```bash
cd libs/evals
uv sync --all-groups
make test
make evals MODEL=<id>
```

## Continue by question

- **Where is the owner and closest regression test?** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **How is the SDK stack assembled and extended?** [SDK Construction and Execution](./architecture/sdk-construction-execution.md)
- **How do filesystem approvals work?** [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md)
- **How do ACP and dcode ACP differ?** [Agent Client Protocol](./integrations/acp.md)
- **How does Talon route state, sandbox work, and media?** [Talon Runtime Host](./integrations/talon.md)
- **How do I change dependencies, locks, CI, or releases?** [Development, Dependencies, and Releases](./operations/development.md)
- **Which focused test layer should I run?** [Testing Guide](./testing/testing-guide.md)
