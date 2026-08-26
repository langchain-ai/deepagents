---
type: Codebase Guide
title: Deep Agents monorepo quickstart
description: "Entry point for engineers working on the Deep Agents Python monorepo: package roles, runtime boundaries, validation, and change-sensitive areas."
tags: [deepagents, python, monorepo, engineering]
openwiki:
  roles: [repository, workflow]
  change_kinds: [task-routing, operations]
  source_paths: [libs/deepagents/deepagents/graph.py, libs/code/deepagents_code/config.py, libs/code/deepagents_code/configuration/providers.py, libs/code/deepagents_code/configuration/service.py, libs/code/deepagents_code/tui/widgets/messages.py, .github/workflows/openwiki-update.yml]
  test_paths: [libs/code/tests/unit_tests/test_coding_agent_metadata.py, libs/code/tests/unit_tests/test_configuration.py, libs/code/tests/unit_tests/test_configuration_resolver.py, libs/code/tests/unit_tests/test_server_graph.py, libs/code/tests/unit_tests/tui/test_textual_adapter.py, libs/code/tests/unit_tests/tui/widgets/test_messages.py, .github/scripts/tests/workflows/test_workflow_secret_scoping.py]
  validation_commands: ["cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_coding_agent_metadata.py tests/unit_tests/tui/test_textual_adapter.py -k 'ContractCompliance or versions_contains_cli_version or versions_marks_editable_cli_version'", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_configuration.py tests/unit_tests/test_configuration_resolver.py tests/unit_tests/test_server_graph.py -k 'remote_managed or failed_remote_refresh_keeps_policy_resolving_in_the_resolver or managed_health_gate_runs_off_event_loop'", "cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k TestAssistantMessageStreamCoalescing", "uv run --directory libs/deepagents --group test pytest -q .github/scripts/tests/workflows/test_workflow_secret_scoping.py"]
---
# Deep Agents monorepo

Deep Agents is an opinionated, extensible agent harness built on LangChain and LangGraph. It packages the long-horizon features that a basic tool-calling agent does not provide by default: filesystem and shell backends, planning, context management, skills, persistent memory, human approval, and subagents. The root README is the product-level starting point; this wiki is the maintainer map.

## Start here

- Read [Architecture overview](architecture/overview.md) to trace `create_deep_agent()` from SDK construction into LangChain/LangGraph execution and to understand package boundaries.
- Read [Deep Agents Code](workflows/deep-agents-code.md) before changing the terminal agent, approval routing, auto mode, sandboxes, or MCP loading.
- Read [Talon runtime](workflows/talon-runtime.md) before changing the experimental long-running local host, channels, cron work, assistant state, or approval routing.
- Read [Evaluation and release](workflows/evaluation-and-release.md) before changing eval harnesses, Harbor workflows, score aggregation, or package-release automation.
- Use [Operations and testing](engineering/operations-and-testing.md) for the package-local edit/test/lint loop, CI controls, integrations, and source map.

## Repository shape

`libs/` is a set of independently versioned Python packages; there is deliberately no root `pyproject.toml`. Work inside the package being changed, where its `pyproject.toml`, `uv.lock`, `Makefile`, and tests define the local contract.

| Area | Role | First source anchor |
| --- | --- | --- |
| `libs/deepagents/` | Core SDK: `create_deep_agent`, middleware, profiles, backends, and subagent machinery. | `libs/deepagents/deepagents/graph.py` |
| `libs/code/` | `dcode` / Deep Agents Code terminal coding agent, with a Textual client and LangGraph server process. | `libs/code/deepagents_code/main.py` |
| `libs/acp/` | Agent Client Protocol adapter for compiled Deep Agent graphs and ACP-capable editors. | `libs/acp/deepagents_acp/server.py` |
| `libs/evals/` | Unit/live evaluation tooling, Harbor integrations, datasets, and scorecard documentation. | `libs/evals/README.md` |
| `libs/talon/` | Experimental local runtime host for long-running agents, channels, and cron jobs. | `libs/talon/deepagents_talon/__main__.py` |
| `libs/partners/` | Sandbox/provider integrations: Daytona, Modal, QuickJS, Runloop, and Vercel. | `libs/partners/` |
| `.github/` | Reusable CI, Harbor evaluations, release, and repository policy automation. | `.github/workflows/ci.yml` |
| `examples/` | Focused patterns and deployable-reference agents rather than a shared product runtime. | `examples/README.md` |

The core SDK in [Architecture overview](architecture/overview.md) supplies the harness that [Deep Agents Code](workflows/deep-agents-code.md) configures for interactive coding and that [Talon runtime](workflows/talon-runtime.md) hosts for long-running channel and scheduled work. That agent is exercised and compared through [Evaluation and release](workflows/evaluation-and-release.md); package checks and publishing rules live in [Operations and testing](engineering/operations-and-testing.md).

## Task routing

Use this table to reach the owning behavior and the smallest evidence-backed check without a repository-wide search. Package tests belong in the affected package; the workflow guard is runnable from the SDK `uv` environment.

| Change area or user intent | Relevant wiki page | Exact source entry points | Important symbols or types | Focused tests | Minimal validation command |
| --- | --- | --- | --- | --- | --- |
| Construct or extend a Deep Agent graph, middleware, backend, or profile | [Architecture overview](architecture/overview.md) | `libs/deepagents/deepagents/graph.py`, `middleware/`, `backends/`, `profiles/` | `create_deep_agent`, `DeepAgentState` | `libs/deepagents/tests/unit_tests/` nearest behavior suite | `cd libs/deepagents && make test TEST_FILE=tests/unit_tests/<focused_test>.py` |
| Change the dcode UI/server, approvals, Auto policy, sandbox, or MCP behavior | [Deep Agents Code](workflows/deep-agents-code.md) | `libs/code/deepagents_code/{main,server_graph,agent,approval_mode,auto_mode,mcp_tools}.py` | `create_cli_agent`, `make_graph` | `test_approval_mode.py`, `test_auto_mode.py`, or `test_server_graph.py` | `cd libs/code && make test TEST_FILE=tests/unit_tests/<focused_test>.py` |
| Change Talon host startup, channel turn routing, approval flow, cron jobs, or assistant-local state | [Talon runtime](workflows/talon-runtime.md) | `libs/talon/deepagents_talon/{__main__,host,runtime,config,data_lifecycle}.py`, `cron/scheduler.py` | `main`, `TalonHost`, `DeepAgentRuntime`, `PersistentCronScheduler` | `test_host.py`, `test_runtime.py`, `cron/test_scheduler.py`, or `test_data_lifecycle.py` | `cd libs/talon && make test TEST_FILE=tests/<focused_test>.py` |
| Change dcode trace metadata, turn attribution, or editable-install observability | [Deep Agents Code trace metadata](workflows/deep-agents-code.md#trace-metadata-and-editable-install-attribution) | `libs/code/deepagents_code/{config.py,tui/textual_adapter.py,client/non_interactive.py}` | `build_stream_config`, `_resolve_editable_info` | `test_coding_agent_metadata.py::TestContractCompliance`, `test_textual_adapter.py::TestBuildStreamConfig` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_coding_agent_metadata.py tests/unit_tests/tui/test_textual_adapter.py -k 'ContractCompliance or versions_contains_cli_version or versions_marks_editable_cli_version'` |
| Change dcode configuration precedence, administrator policy, managed-config startup behavior, or a remote policy descriptor | [Deep Agents Code configuration](workflows/deep-agents-code.md#configuration-and-managed-policy) | `libs/code/deepagents_code/{config_manifest.py,configuration/{providers,service,types,resolver}.py}` | `resolve_ranked`, `RemoteTomlProvider`, `get_managed_snapshot`, `require_healthy_managed_config` | `test_configuration.py::{test_remote_managed_descriptor_must_be_exclusive,test_failed_remote_reload_keeps_previous_policy}`, `test_configuration_resolver.py::test_failed_remote_refresh_keeps_policy_resolving_in_the_resolver`, `test_server_graph.py::test_managed_health_gate_runs_off_event_loop` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_configuration.py tests/unit_tests/test_configuration_resolver.py tests/unit_tests/test_server_graph.py -k 'remote_managed or failed_remote_refresh_keeps_policy_resolving_in_the_resolver or managed_health_gate_runs_off_event_loop'` |
| Change `ask_user` question/answer encoding or Auto consent evidence | [Deep Agents Code ask-user contract](workflows/deep-agents-code.md#ask-user-wire-contract) | `libs/code/deepagents_code/{_ask_user_types,ask_user,auto_mode}.py` | `encode_multi_select_answer`, `ask_user_answer_is_empty` | `test_ask_user_types.py::TestMultiSelectAnswerEncoding`, `TestAskUserAnswerIsEmpty` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/test_ask_user_types.py -k 'MultiSelectAnswerEncoding or AskUserAnswerIsEmpty'` |
| Change how sent prompts render, wrap, collapse, or copy in the dcode transcript | [Deep Agents Code](workflows/deep-agents-code.md#transcript-presentation-and-selection) | `libs/code/deepagents_code/tui/widgets/messages.py` | `UserMessage`, `_UserMessageContent`, `get_selection` | `test_messages.py::TestUserMessageAppearance` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k UserMessageAppearance` |
| Change dcode assistant-response streaming latency, batching, or flush/exit behavior | [Deep Agents Code](workflows/deep-agents-code.md#assistant-response-streaming) | `libs/code/deepagents_code/tui/{widgets/messages,textual_adapter}.py` | `AssistantMessage.append_content`, `_flush_pending_append`, `_stop_assistant_streams` | `test_messages.py::TestAssistantMessageStreamCoalescing` | `cd libs/code && uv run --group test pytest -q --disable-socket --allow-unix-socket tests/unit_tests/tui/widgets/test_messages.py -k TestAssistantMessageStreamCoalescing` |
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
- Tool authority follows the configured backend and middleware. The root README's security model is **trust the LLM**: enforce containment at tool/sandbox boundaries rather than treating model intent as a security control.
- Deep Agents Code adds approval UX and policy, but approval is not containment. For untrusted repositories, use a remote sandbox; read [Deep Agents Code](workflows/deep-agents-code.md) before changing approval/MCP behavior.
- Real model/Harbor evaluations have separate credentials, costs, and semantics from unit tests; they are documented in [Evaluation and release](workflows/evaluation-and-release.md).

## Current repository context

The recorded wiki `gitHead` (`dfde21e379201c833da4162444ef4a13b46980fd`) is unavailable in this shallow checkout, so Git cannot produce a commit range. The reachable HEAD is `9f3a1dd38f8b5dbd828d2f366e9b03dad359fadf` (`feat(code): load managed config from a remote source (#5776)`); current source and tests are the evidence for this update.

A managed-policy file can now be an exclusive fixed local descriptor for a bounded HTTPS TOML policy. The descriptor remains the trust anchor; remote fetches fail closed during startup and retain a last enforceable in-process policy on a failed refresh. This extends the configuration system described in [Deep Agents Code](workflows/deep-agents-code.md#configuration-and-managed-policy), not the core agent runtime.

## Backlog

- **Partner implementations** — `libs/partners/{daytona,modal,quickjs,runloop,vercel}`; catalogued above but not individually documented because each is an integration package with its own boundary and should be expanded when modified.
- **Examples** — `examples/README.md`; examples are intentionally navigated from their own READMEs and were not duplicated into the maintainer wiki.
