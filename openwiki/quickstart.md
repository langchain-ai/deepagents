---
type: maintainer quickstart
title: Deep Agents Maintainer Quickstart
description: A concise routing map for maintaining independently released Deep Agents packages. Use it to identify the behavior owner, the next architecture or integration guide, and the focused validation path.
tags: [deepagents, dcode, talon, acp, maintenance, development, testing]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-14T08:05:41.101Z
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
  - id: openwiki-source-15628256fe9cd22197db74ed
    resource: repo://libs/code/AGENTS.md
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
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
generated: { by: "openwiki/0.4.2", at: "2026-10-14T08:05:41.101Z" }
---

# Deep Agents Maintainer Quickstart

Start with the package that owns the behavior rather than the product where it appears. `libs/` is a monorepo of independently versioned packages: the `deepagents` SDK, dcode (`deepagents-code`), ACP, evals, the experimental Talon host, and provider integrations. The release manifest currently records `deepagents` 0.7.25, `deepagents-acp` 0.0.13, `deepagents-code` 0.1.84, and `deepagents-talon` 0.0.9. For package direction and ownership boundaries, begin with [Repository Architecture Overview](./architecture/overview.md) and [Source Map and Ownership Boundaries](./architecture/source-map.md).

## Route by behavior

| If you need to change… | Start in | Then read | Validate first |
| --- | --- | --- | --- |
| SDK graph construction, middleware, filesystems, skills, subagents, state, or approval behavior | `libs/deepagents/` | [SDK Construction and Execution](./architecture/sdk-construction-execution.md); [Middleware Catalog](./concepts/middleware-catalog.md); [Permissions and Human-in-the-Loop](./concepts/permissions-hitl.md); [Subagents and Skills](./concepts/subagents-skills.md) | The nearest SDK unit test; use the [Testing Guide](./testing/testing-guide.md) to select payload, sandbox, or fake/provider coverage. |
| dcode terminal startup, UI, sessions, slash commands, model configuration, cost display, or loopback execution | `libs/code/` | [Code Agent Architecture](./architecture/code-agent.md); [Models, Profiles, and Provider Configuration](./concepts/profiles-models.md); [Cost Tracking and Session Operations](./operations/cost-and-sessions.md) | A focused `libs/code/tests/unit_tests/` test, then `make lint`. |
| ACP editor protocol, custom agent hosting, dcode ACP mode, or execution approval | `libs/acp/` | [Agent Client Protocol](./integrations/acp.md) | The affected ACP test; also validate dcode when changing `dcode --acp`. |
| Talon channels, admission, turns, cron work, runtime tools, sandboxing, media, or lifecycle | `libs/talon/` | [Talon Runtime Host](./integrations/talon.md); [Talon Channel Admission](./concepts/talon-channel-admission.md); [Talon Scheduling and Background Work](./concepts/talon-scheduling.md) | The closest Talon test, including bridge tests through `make test`. |
| A sandbox provider, QuickJS, or a new partner package | The owning `libs/partners/<provider>/` package | [Sandbox and Partner Integrations](./integrations/sandbox-partners.md) | The partner's own tests, then applicable consumer, Harbor, and integration coverage. |
| Real-model quality, trajectory capture, scoring, or Harbor benchmarks | `libs/evals/` | [Testing Guide](./testing/testing-guide.md) | Deterministic regression coverage first; then `make evals MODEL=<id>` or the relevant Harbor job. |
| Dependency bounds, locks, CI selection, or package delivery | The changed package manifest and `.github/` workflow/release automation | [Development, Dependencies, and Releases](./operations/development.md) | The package checks and the relevant lock, CI, or release guard. |

## Keep the boundaries intact

`create_deep_agent` is the SDK's public graph-assembly seam: same-named supplied middleware replaces a base entry in place, but protected filesystem and synchronous-subagent scaffolding remains present. Change shared agent behavior there rather than duplicating it in dcode, ACP, or Talon.

dcode, ACP, and Talon are separate integration boundaries. dcode is the terminal product; ACP can host either a custom Deep Agent through `AgentServerACP` or the prebuilt dcode product over stdio; Talon is an experimental local host for channel adapters, cron scheduling, and agent runtime. Follow the linked integration page for the applicable boundary instead of assuming one host's lifecycle, approvals, or security model applies to another.

Partner packages own their own environment, manifest, Makefile, tests, and version. A new partner also needs the repository wiring that makes it selectable, tested, labeled, and released; sandbox-backed partners may additionally require Harbor and integration-test wiring.

## Work across packages deliberately

Use `uv` for interpreters, environments, and dependencies, and treat each package's `Makefile` as the command authority. `uv` provisions the suitable interpreter, so do not impose a repository-wide Python pin.

```bash
cd libs/deepagents
uv sync --all-groups
make test TEST_FILE=tests/unit_tests/test_graph.py
make lint
```

Local sibling dependencies are editable. A shared-interface change is immediately visible to consumers, so validate every affected consumer. CI selects changed packages on pull requests and includes dependent coverage for SDK changes. In particular, dcode pins an exact SDK version; a dcode change that needs new SDK behavior must update that pin in the same pull request.

Use deterministic, network-free tests under `tests/unit_tests/`; reserve `tests/integration_tests/` for networked contracts. Unexpected pytest warnings fail the suite. The ACP and Talon `make test` targets disable network sockets, and Talon's target runs the WhatsApp bridge Node tests before its Python tests. Evals are different: they run against real LLMs, capture trajectories, score correctness and efficiency, and use Harbor for sandboxed benchmarks; `make evals` requires `MODEL` and runs `tests/evals`.

## Release and handoff

Each managed package has its own release-please release PR. Review the generated changelog, use the curated-notes flow—draft, review/edit, then `@release-bot apply`—and merge only after its required checks pass. Use `ci:skip-curated-notes` only when intentionally publishing the generated changelog without that review.

When your change crosses a boundary, use the destination guide before implementation:

- **Find an owner, runtime seam, or nearby regression test:** [Source Map and Ownership Boundaries](./architecture/source-map.md)
- **Change SDK composition or public behavior:** [SDK Construction and Execution](./architecture/sdk-construction-execution.md)
- **Change skills, models, profiles, permissions, or state:** [Concepts](./concepts/index.md)
- **Change ACP, Talon, sandboxes, partners, or MCP:** [Integrations](./integrations/index.md)
- **Change setup, dependencies, releases, sessions, or security:** [Operations](./operations/index.md)
- **Choose focused tests or determine whether real-provider coverage is needed:** [Testing Guide](./testing/testing-guide.md)
