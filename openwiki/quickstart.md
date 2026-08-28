---
type: Codebase Guide
title: Deep Agents monorepo quickstart
description: "Entry point for engineers working on the Deep Agents Python monorepo: package roles, runtime boundaries, validation, and change-sensitive areas."
tags: [deepagents, python, monorepo, engineering]
openwiki:
  roles: [repository, workflow]
  change_kinds: [task-routing, operations]
  source_paths: [libs/deepagents/deepagents/graph.py, .github/workflows/openwiki-update.yml]
  test_paths: [.github/scripts/tests/workflows/test_workflow_secret_scoping.py]
  validation_commands: ["uv run --directory libs/deepagents --group test pytest -q .github/scripts/tests/workflows/test_workflow_secret_scoping.py"]
---
# Deep Agents monorepo

Deep Agents is an opinionated, extensible agent harness built on LangChain and LangGraph. It packages the long-horizon features that a basic tool-calling agent does not provide by default: filesystem and shell backends, planning, context management, skills, persistent memory, human approval, and subagents. The root README is the product-level starting point; this wiki is the maintainer map.

## Start here

- Read [Architecture overview](architecture/overview.md) to trace `create_deep_agent()` from SDK construction into LangChain/LangGraph execution and to understand package boundaries.
- Read [Deep Agents Code](workflows/deep-agents-code.md) before changing the terminal agent, approval routing, auto mode, sandboxes, or MCP loading.
- Read [Evaluation and release](workflows/evaluation-and-release.md) before changing eval harnesses, Harbor workflows, score aggregation, or package-release automation.
- Use [Operations and testing](engineering/operations-and-testing.md) for the package-local edit/test/lint loop, CI controls, integrations, and source map.

## Repository shape

`libs/` is a set of independently versioned Python packages; there is deliberately no root `pyproject.toml`. Work inside the package being changed, where its `pyproject.toml`, `uv.lock`, `Makefile`, and tests define the local contract.

| Area | Role | First source anchor |
| --- | --- | --- |
| `libs/deepagents/` | Core SDK: `create_deep_agent`, middleware, profiles, backends, and subagent machinery. | `libs/deepagents/deepagents/graph.py` |
| `libs/code/` | `dcode` / Deep Agents Code terminal coding agent, with a Textual client and LangGraph server process. | `libs/code/deepagents_code/main.py` |
| `libs/acp/` | Agent Client Protocol adapter for compiled Deep Agent graphs and ACP-capable editors. | `libs/acp/deepagents_acp/server.py` |
| `libs/cli/` | Managed Deep Agents deployment CLI; not the interactive terminal agent. | `libs/cli/deepagents_cli/main.py` |
| `libs/evals/` | Unit/live evaluation tooling, Harbor integrations, datasets, and scorecard documentation. | `libs/evals/README.md` |
| `libs/talon/` | Local runtime host for long-running agents. | `libs/talon/README.md` |
| `libs/partners/` | Sandbox/provider integrations: Daytona, Modal, QuickJS, Runloop, and Vercel. | `libs/partners/` |
| `.github/` | Reusable CI, Harbor evaluations, release, and repository policy automation. | `.github/workflows/ci.yml` |
| `examples/` | Focused patterns and deployable-reference agents rather than a shared product runtime. | `examples/README.md` |

The core SDK in [Architecture overview](architecture/overview.md) supplies the harness that [Deep Agents Code](workflows/deep-agents-code.md) configures for interactive coding. That agent is exercised and compared through [Evaluation and release](workflows/evaluation-and-release.md); package checks and publishing rules live in [Operations and testing](engineering/operations-and-testing.md).

## Task routing

Use this table to reach the owning behavior and the smallest evidence-backed check without a repository-wide search. Package tests belong in the affected package; the workflow guard is runnable from the SDK `uv` environment.

| Change area or user intent | Relevant wiki page | Exact source entry points | Important symbols or types | Focused tests | Minimal validation command |
| --- | --- | --- | --- | --- | --- |
| Construct or extend a Deep Agent graph, middleware, backend, or profile | [Architecture overview](architecture/overview.md) | `libs/deepagents/deepagents/graph.py`, `middleware/`, `backends/`, `profiles/` | `create_deep_agent`, `DeepAgentState` | `libs/deepagents/tests/unit_tests/` nearest behavior suite | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/<focused_test>.py` |
| Change the dcode UI/server, approvals, Auto policy, sandbox, or MCP behavior | [Deep Agents Code](workflows/deep-agents-code.md) | `libs/code/deepagents_code/{main,server_graph,agent,approval_mode,auto_mode,mcp_tools}.py` | `create_cli_agent`, `make_graph` | `test_approval_mode.py`, `test_auto_mode.py`, or `test_server_graph.py` | `cd libs/code && make test TEST_FILE=tests/unit_tests/<focused_test>.py` |
| Change cached MCP tool retries, error messages, or failure diagnostics | [Deep Agents Code](workflows/deep-agents-code.md#cached-mcp-tool-failure-boundary) | `libs/code/deepagents_code/mcp_tools.py` | `_build_cached_mcp_tool`, `_handle_cached_mcp_tool_error`, `MCPSessionManager` | `test_mcp_tools.py::TestCachedSessionProxy::{test_repeated_transient_error_surfaces_tool_message,test_generic_oserror_is_not_retried}` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_mcp_tools.py -k 'repeated_transient_error_surfaces_tool_message or generic_oserror_is_not_retried'` |
| Change eval results, unified Harbor orchestration, or package release behavior | [Evaluation and release](workflows/evaluation-and-release.md) | `libs/evals/deepagents_evals/cli.py`, `.github/scripts/evals/{unified_prep,aggregate_unified}.py`, `.github/workflows/{unified_evals,release}.yml` | `deepagents-evals`, `counts.failed` | `libs/evals/tests/unit_tests/` or `.github/scripts/tests/evals/` nearest suite | `cd libs/evals && make test TEST_FILE=tests/unit_tests/<focused_test>.py` |
| Change scheduled OpenWiki updates, their secret boundary, or generated-document commit behavior | [Operations and testing](engineering/operations-and-testing.md#generated-openwiki-maintenance) | `.github/workflows/openwiki-update.yml`, `AGENTS.md` | `update` job; `test_openwiki_uses_dedicated_environment` | `.github/scripts/tests/workflows/test_workflow_secret_scoping.py` | `uv run --directory libs/deepagents --group test pytest -q .github/scripts/tests/workflows/test_workflow_secret_scoping.py` |

## Fast local loop

Use `uv`; repository guidance explicitly disallows using `pip`, Poetry, or Conda for environment/dependency operations. Install dependencies within the affected package and use its Makefile as the command source of truth:

```bash
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

The common package targets are `make test` (socket-restricted unit tests), `make integration_test` (network permitted), `make lint`, `make format`, and `make type`. From `libs/`, `make lint` and `make lock-check` fan out across packages. See [Operations and testing](engineering/operations-and-testing.md) for checks by subsystem and CI behavior.

## Product and security boundaries

- The SDK is a harness, not a new graph runtime: LangChain owns the agent loop and LangGraph owns state, checkpointing, streaming, and interrupts.
- Tool authority follows the configured backend and middleware. The root README’s security model is **trust the LLM**: enforce containment at tool/sandbox boundaries rather than treating model intent as a security control.
- Deep Agents Code adds approval UX and policy, but approval is not containment. For untrusted repositories, use a remote sandbox; read [Deep Agents Code](workflows/deep-agents-code.md) before changing approval/MCP behavior.
- Real model/Harbor evaluations have separate credentials, costs, and semantics from unit tests; they are documented in [Evaluation and release](workflows/evaluation-and-release.md).

## Current repository context

Current HEAD is `test(code): pin single-warning logging for failed MCP tool calls (#5609)`. The available checkout has one root commit, so the recorded wiki `gitHead` is not locally resolvable for a range diff; this update is grounded in the current commit message, current sources, and tests.

The latest code change tightens the dcode MCP failure-diagnostics contract: a failed cached MCP tool call is warning-logged once with a traceback by its tool-error handler, while independent session-cleanup warnings remain observable. It extends the existing transient-retry and generic-`OSError` regression cases; see [Deep Agents Code](workflows/deep-agents-code.md#cached-mcp-tool-failure-boundary) for the implementation boundary and focused check.

The OpenWiki workflow installs `openwiki@0.3.3` under Node.js 26, invokes `openwiki code --update --print` with LangSmith tracing disabled, and runs in the dedicated `openwiki` GitHub environment. Before creating an update PR, it restores its own workflow file and stages only `openwiki` and `AGENTS.md`. Those boundaries prevent a generated documentation run from committing a changed CI workflow; the focused workflow guard test currently asserts the dedicated environment. See [Operations and testing](engineering/operations-and-testing.md#generated-openwiki-maintenance) before changing that automation.

## Backlog

- **Talon runtime host** — `libs/talon/README.md`; deferred from this first pass because the core SDK, dcode, and evaluation/release pathways dominate current repository changes.
- **Partner implementations** — `libs/partners/{daytona,modal,quickjs,runloop,vercel}`; catalogued above but not individually documented because each is an integration package with its own boundary and should be expanded when modified.
- **Examples** — `examples/README.md`; examples are intentionally navigated from their own READMEs and were not duplicated into the maintainer wiki.
