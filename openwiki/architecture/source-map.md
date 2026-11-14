---
type: architecture source map
title: Repository Source Map
description: System-oriented change routing from public dcode and SDK surfaces into graph construction, subagent behavior, Talon runtime, and dependency release maintenance. Use it to locate the state owner, enforcement boundary, and focused test before changing behavior.
tags: [deepagents, talon, dcode, source-map, middleware, runtime, subagents, dependencies]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-13T08:03:55.116Z
sources:
  - id: openwiki-source-594b8b7a84e0fb527fbafd52
    resource: repo://.github/scripts/checks/raise_langchain_minimums.py
  - id: openwiki-source-1bc32c2b33ab6e58128b0319
    resource: repo://.github/scripts/tests/checks/test_raise_langchain_minimums.py
  - id: openwiki-source-bc54cf7ca3addab1b243d7a6
    resource: repo://.github/workflows/raise_langchain_minimums.yml
  - id: openwiki-source-05106e66a949150d557266a2
    resource: repo://libs/code/deepagents_code/agent.py
  - id: openwiki-source-074ce96a8baea27a6c43328b
    resource: repo://libs/code/deepagents_code/client/launch/server.py
  - id: openwiki-source-4c023d2e012063c0083af48c
    resource: repo://libs/code/deepagents_code/git_trace.py
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-ea1089f0d7536fbc96c64866
    resource: repo://libs/code/deepagents_code/offload_api.py
  - id: openwiki-source-7ba50bd13eb62341a2061ef9
    resource: repo://libs/code/pyproject.toml
  - id: openwiki-source-68cd21ba4b9746c7605b0c89
    resource: repo://libs/code/tests/unit_tests/test_git_trace.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-b93c32bc33a8fa17b52b8a0e
    resource: repo://libs/deepagents/deepagents/middleware/rubric.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-023a5fe4719690fecce2bbf1
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_rubric_middleware.py
  - id: openwiki-source-6ce85b02eabe462f99e0c912
    resource: repo://libs/deepagents/tests/unit_tests/test_async_subagents.py
  - id: openwiki-source-10e4084b6aa57e5cc82620b3
    resource: repo://libs/deepagents/tests/unit_tests/test_end_to_end.py
  - id: openwiki-source-917b61d56e652536ba962f3a
    resource: repo://libs/deepagents/tests/unit_tests/test_subagent_refusal.py
  - id: openwiki-source-ca8183c87e6002c442ee2d62
    resource: repo://libs/deepagents/tests/unit_tests/test_subagents.py
  - id: openwiki-source-995d5d95882808a64071f617
    resource: repo://libs/talon/deepagents_talon/archive_saver.py
  - id: openwiki-source-cd45145a8c3a51b52eab3c2b
    resource: repo://libs/talon/deepagents_talon/background.py
  - id: openwiki-source-d8846ad7008aa3a1566511a7
    resource: repo://libs/talon/deepagents_talon/hook_transcripts.py
  - id: openwiki-source-cbf86ab9c145113d54045954
    resource: repo://libs/talon/deepagents_talon/hooks.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-111101dcd1462ff54277b1fc
    resource: repo://libs/talon/deepagents_talon/mcp_config.py
  - id: openwiki-source-82cac27adeecff8a900a40fa
    resource: repo://libs/talon/deepagents_talon/mcp.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-d41d7ef7f9d2f53eb0bd40e8
    resource: repo://libs/talon/deepagents_talon/speech.py
  - id: openwiki-source-444dd041f0001b6042aeee4c
    resource: repo://libs/talon/deepagents_talon/subagent_control.py
  - id: openwiki-source-2d1f686d24d8182f60108ae7
    resource: repo://libs/talon/deepagents_talon/subagents.py
  - id: openwiki-source-686a5e2ba1fe4ce0f98b9bf2
    resource: repo://libs/talon/pyproject.toml
  - id: openwiki-source-4d6726e17c8a0c78539a7d33
    resource: repo://libs/talon/tests/test_runtime.py
  - id: openwiki-source-c996df77875d3c6b30ca07cf
    resource: repo://libs/talon/tests/unit_tests/test_archive_saver.py
  - id: openwiki-source-82dab853903c3a574614fd1e
    resource: repo://libs/talon/tests/unit_tests/test_background.py
  - id: openwiki-source-c0f076507fca2b1a9eaae92f
    resource: repo://libs/talon/tests/unit_tests/test_hook_transcripts.py
  - id: openwiki-source-7dcd7e37c18e77553a46cce3
    resource: repo://libs/talon/tests/unit_tests/test_hooks.py
  - id: openwiki-source-e2be45e59936bfba43c18816
    resource: repo://libs/talon/tests/unit_tests/test_mcp_config.py
  - id: openwiki-source-24a3ab902230428c874f14d8
    resource: repo://libs/talon/tests/unit_tests/test_speech.py
  - id: openwiki-source-cf14bb91e6251d6d6d4a9413
    resource: repo://libs/talon/tests/unit_tests/test_subagent_control.py
generated: { by: "openwiki/0.4.2", at: "2026-11-13T08:03:55.116Z" }
---

# Repository Source Map

Route a change to the component that owns its state and lifetime—not merely the API that exposes it. This map follows public dcode and SDK construction surfaces into graph construction, Talon's long-lived runtime and delegation, and release dependency maintenance. See [Architecture Overview](./overview.md), [Middleware Stack](./middleware-stack.md), [development](../operations/development.md), [quickstart](../quickstart.md), and the [testing guide](../testing/testing-guide.md).

## Change routing

| Change affects… | Start at | Owner and boundary | Focused tests |
| --- | --- | --- | --- |
| SDK tools, profiles, middleware order, state schema, or injected persistence | `libs/deepagents/deepagents/graph.py` — `create_deep_agent` | The compiled graph; callers retain checkpointer, store, and cache ownership | `libs/deepagents/tests/unit_tests/test_graph.py` |
| Synchronous or forked SDK subagent input, private-state filtering, result projection, or refusal handling | `libs/deepagents/deepagents/middleware/subagents.py` — `SubAgentMiddleware` and its `task` tool | The task boundary translates the child result into parent state | `libs/deepagents/tests/unit_tests/test_subagents.py`, `test_subagent_refusal.py` |
| Remote async SDK subagent lifecycle or notifications | `middleware/async_subagents.py` | Persisted `async_tasks` state and Agent Protocol lifecycle tools | `test_async_subagents.py` |
| dcode CLI launch, command dispatch, or managed-policy admission | `libs/code/deepagents_code/main.py` — `cli_main` | CLI owns process-level bootstrap and selects TUI, ACP, non-interactive, or command paths | `libs/code/tests/unit_tests/test_main.py`, `test_main_args.py`, `test_main_acp_mode.py` |
| dcode graph assembly, shell/approval policy, skills, model selection, or checkout attribution | `agent.py` and `git_trace.py` | dcode's configured graph and per-turn trace reconciliation | `test_agent.py`, `test_git_trace.py`, `test_git_trace_integration.py` |
| dcode graph serving, offload operations, or local server lifecycle | `client/launch/server.py`; `offload_api.py` | Generated `langgraph.json`, private Starlette boundary, and `ServerProcess` | dcode server-launch and offload API tests |
| Talon graph wiring, tool replacement, approvals, MCP reload, or shutdown | `libs/talon/deepagents_talon/runtime.py` — `DeepAgentRuntime` | Runtime graph snapshot and invocation context variables | `libs/talon/tests/test_runtime.py`, `test_subagent_reload.py`, `test_mcp_config.py` |
| MCP configuration or loading | `mcp_config.py`, then `mcp.py` | Redacted revisioned configuration and provider refresh revisions | `test_mcp_config.py`, `test_mcp_enabled.py`, `test_mcp_adapter.py` |
| Fresh local children, task attachments, approvals, or follow-ups | `subagents.py`, `subagent_control.py` | Isolated child graph plus in-memory worker control | `test_research_subagents.py`, `test_subagent_control.py` |
| Detached task capacity, delivery, cancellation, or scheduled delegation | `background.py` | Process-memory job table keyed by owner conversation | `test_background.py`, `test_background_runtime.py` |
| Conversation persistence, hook transcripts, voice, or host lifecycle | `archive_saver.py`, `hooks.py`, `speech.py`, then `host.py` | Caller-owned stores, trusted hook snapshot, inbound preparation, and host lifecycle | `test_archive_saver.py`, `test_hooks.py`, `test_speech.py`, `test_host.py` |
| LangChain-ecosystem lower-bound bumps and locks | `.github/scripts/checks/raise_langchain_minimums.py`, then package `pyproject.toml` | The release manifest owns its declared compatibility floor; the workflow owns resolution filtering and lock regeneration | `.github/scripts/tests/checks/test_raise_langchain_minimums.py` |

## SDK graph and subagent boundary

`create_deep_agent` is the SDK construction seam: it accepts caller-provided backend, middleware, subagents, checkpointer, store, and cache dependencies, then passes persistence and state schema to the compiled graph. Its default `DeepAgentState.messages` uses `DeltaChannel` to avoid quadratic checkpoint growth. Treat middleware order and middleware names as behavioral API: the main stack begins with filesystem and optional synchronous subagent middleware, then parent notification, summarization, patched tool calls, and optional remote async delegation; profile extras, skills, prompt caching, optional memory, approval interrupts, unsupported-content handling, and refusal handling form the tail. A profile exclusion is validated against the assembled main and general-purpose stacks, and tool exclusion runs last so custom middleware cannot restore an excluded tool.

Caller middleware with a matching name replaces that stack slot; a novel name is appended. Private fields declared by any participating state schema are added to `SubAgentMiddleware`'s exclusion set only after caller replacements, so custom task middleware must preserve the private-state contract.

```mermaid
flowchart TD
  Input["Caller configuration"] --> Build["create_deep_agent"]
  Build --> Core["Filesystem and task core"]
  Core --> Tail["Profile, skills, cache, memory, approvals"]
  Tail --> Guard["Unsupported content and refusal"]
  Guard --> Custom["Caller replacements and additions"]
  Custom --> Exclude["Validated exclusions and final tool filter"]
  Exclude --> Graph["Compiled graph"]
```

*The graph assembly path makes ordering, replacement, and final filtering explicit.*

`SubAgentMiddleware` owns synchronous child execution. It validates names and modes, builds the `task` tool, compiles declarative specifications, and creates a child input. Ordinary children receive a single `HumanMessage` containing the task; forked children inherit eligible parent state and a reconstructed message history, but cannot delegate recursively. On completion the middleware applies non-private, changed child state and projects a single parent `ToolMessage`: structured response JSON takes priority, followed by a terminal refusal notice, then the last non-empty AI text. Missing `messages` from a compiled child is an error. The sync and async task paths both convert model-refusal errors and refusal-bearing structured-output validation errors, but re-raise other validation failures.

Use `test_graph.py` for assembly invariants—custom state schema propagation, private keys, required/excluded middleware, replacement position, and skills/cache order. Use `test_subagents.py` for actual parent/child state behavior: sync versus remote tool routing, result projection, parallel calls, fork behavior, and private-state non-propagation. Keep `test_subagent_refusal.py` when changing the refusal projection contract.

`ModelRefusalMiddleware` is tail middleware after unsupported-content handling, so a caller fallback sees a refusal first. By default a top-level refusal raises; excluding it leaves the refusal message observable. Remote async SDK tasks are separate from Talon's workers: `AsyncSubAgentMiddleware` persists `async_tasks`, exposes lifecycle tools, and polls unfinished tasks on later async invocations. `RubricMiddleware` is inactive without a rubric; otherwise private state drives post-agent grading and revision without replacing the final visible response on grading failure.

## dcode execution boundary

`cli_main()` is the dcode process entrypoint. It keeps `--version` and lightweight diagnostic/help paths ahead of expensive bootstrap, installs terminating-signal handling before launch, dispatches self-contained commands separately, and refuses policy-dependent commands when managed configuration is unhealthy.

`create_cli_agent()` is dcode's construction seam. It turns CLI/server snapshots into a Deep Agents graph and composite backend, resolves local subagents and approval policy, and composes feature middleware. Treat its main and declarative-subagent stacks as coupled: policy or telemetry middleware installed only on one path can make child behavior diverge. When enabled, `GitTraceMiddleware` is installed in both stacks; it ignores known read-only operations and enriches only the initiating turn's trace, never tool results.

`ServerProcess` owns the `langgraph dev` subprocess. Generated LangGraph configuration always registers the agent graph, while only dcode's built-in graph adds the authenticated, thread-scoped offload Starlette application. The lifespan wakes queued runs, prebuilds a default runtime during health polling, cancels that work before extension shutdown, and makes a bounded best-effort trace flush.

## Talon runtime: compile first, swap after success

`DeepAgentRuntime.start()` resolves subagents and approvals before building the initial graph. `_create_graph()` builds the current tool catalog, resolves local names, prepares subagents, then provides Talon middleware and dependencies to `create_deep_agent`. Invocation refreshes tools before the tool lock, reconstructs for an approval change under that lock, and snapshots graph and approvals into context variables before binding authorization, background results, cron origin, breakout handling, and hook snapshot. An in-flight turn consequently keeps its graph if a later reload succeeds.

```mermaid
sequenceDiagram
  participant Host as Talon host
  participant Runtime as DeepAgentRuntime
  participant Provider as MCP provider
  participant Graph as Deep Agents graph
  Host->>Runtime: invoke request
  Runtime->>Provider: refresh when revision changes
  Runtime->>Runtime: snapshot graph and approvals
  Runtime->>Runtime: bind turn context and hooks
  Runtime->>Graph: invoke with conversation thread
  Graph-->>Runtime: text or interrupt
  Runtime-->>Host: result and consumed result ids
```

*Reload precedes the stable per-turn snapshot, so a turn cannot observe a mid-turn replacement.*

`_replace_runtime_tools()` compiles replacement tools before assigning `tools` and `_graph`; failed explicit or automatic reloads therefore retain the old graph. `MCPToolProvider` uses refresh revisions to defer loading until a later turn or forced reload. `MCPConfigStore` serves redacted, revisioned configuration and atomically replaces a validated server under a sidecar lock; automatic approval cannot reuse a redacted secret while changing a field that could redirect it.

`ConversationSaver` bridges caller-owned checkpoint and archive stores. It checkpoints before appending changed top-level scoped messages, leaves archive failure retryable because no cross-store transaction exists, waits for both writes on cancellation, and refuses active-session deletion. Hook transcripts reside in a hashed owner directory guarded against symlink and revocation-marker races; `TalonHooks` snapshots trust per turn, assigns child identities, and validates interrupt ownership before caching responses.

## Talon delegation and host lifecycle

Local Talon subagents come from `AGENTS.md` frontmatter, resolve exact tools from the current catalog, and run fresh rather than forked. `TaskTools` is the only parent task tool: it validates per-task tool attachments and compiles an isolated child graph with only applicable approvals.

`BackgroundSubagents` detaches ordinary main-agent task calls into owner-scoped in-memory workers with capacity and ownership controls. It runs scheduled-turn delegation inline under a timeout so the same unattended turn receives the result; nested delegation is refused. Detached workers use a distinct task thread, clear authorization and approval-operator state, convert interruption, timeout, or failure into safe results, and acknowledge delivery only after parent consumption. `SubagentControl` keeps bounded metadata and coordinator receipts outside graph state; only Talon-built workers accept queued follow-ups at a later model boundary.

`TalonHost` starts runtime before channels and scheduler, reverses partial startup on failure, and during shutdown cancels host work before stopping channels, scheduler, and runtime while isolating stop failures. Voice transcription is best effort before media staging: it preserves the original message on disabled, failed, or empty transcription and appends successful text with `voice_transcribed`.

## Package compatibility and release touchpoints

The dcode and Talon manifests are not merely install metadata: they define the published compatibility envelope and the workspace edge between packages. dcode pins `deepagents` exactly while setting bounded LangChain/LangGraph family ranges; Talon consumes `deepagents` and `deepagents-code`, maps both to editable sibling sources for local development, and defines mutually exclusive `media` and `media-cpu` extras. Change a package API with the corresponding consumer floor, pin, extras, and lockfile in view.

The `raise_langchain_minimums.py` maintenance script is the change owner for range-floor automation. It considers `langchain`, `langgraph`, `langsmith`, and `deepagents` prefixed requirements in project, optional, and dependency-group tables; preserves upper bounds, extras, markers, and exact pins; and caps a locally sourced sibling requirement at the local declared version. It plans every manifest before writing, fails closed on malformed inputs, unavailable metadata, and unresolved baselines, and can retain only individually compatible bumps after resolution checks. The workflow invokes it with `--check-resolution`, then regenerates affected and reverse-dependent locks using the interpreter selected by the lockfile checker.

## Focused test plan

- For graph construction, begin with `libs/deepagents/tests/unit_tests/test_graph.py`; select the schema/private-state or middleware ordering/replacement cases rather than relying on broad end-to-end coverage.
- For child semantics, begin with `libs/deepagents/tests/unit_tests/test_subagents.py`; cover child input mode, state filtering, projection priority, and concurrent calls. Add `test_subagent_refusal.py` for refusal conversion.
- For dcode construction and observability, pair `test_agent.py` wiring coverage with `test_git_trace.py` for read-only bypass, turn isolation, failure, and backoff.
- For runtime or MCP edits, cover unsuccessful replacement and subsequent-turn activation in `libs/talon/tests/test_runtime.py`, then configuration redaction/CAS/secret-redirection in `test_mcp_config.py`.
- For detached work, use `test_background.py` for owner isolation, capacity, authorization stripping, acknowledgement/requeue, delivery failure limits, and scheduled inline semantics.
- For dependency automation, use `.github/scripts/tests/checks/test_raise_langchain_minimums.py`; retain tests for narrowing, exact/unraiseable floors, resolution filtering, and workflow-to-lockfile handoff when changing the script or workflow.
