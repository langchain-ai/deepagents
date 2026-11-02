---
type: testing guide
title: Testing Guide
description: Focused, network-free guidance for testing observable Deep Agents middleware contracts, QuickJS prompt and REPL lifecycle behavior, and Talon reasoning-effort selection, persistence, and provider translation.
tags: [testing, pytest, deepagents, quickjs, talon, middleware, prompts]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-31T08:04:00.426Z
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-b93c32bc33a8fa17b52b8a0e
    resource: repo://libs/deepagents/deepagents/middleware/rubric.py
  - id: openwiki-source-66cf9d0832d3cb55bec2b5ed
    resource: repo://libs/deepagents/deepagents/middleware/skills.py
  - id: openwiki-source-0f308f1610986e2f3ed6d53c
    resource: repo://libs/deepagents/Makefile
  - id: openwiki-source-a6f0952b514b04c99bc51705
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_pinned_skills.py
  - id: openwiki-source-023a5fe4719690fecce2bbf1
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_rubric_middleware.py
  - id: openwiki-source-8e7076889e81540fe4fcb8c5
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_summarization_middleware.py
  - id: openwiki-source-6ce85b02eabe462f99e0c912
    resource: repo://libs/deepagents/tests/unit_tests/test_async_subagents.py
  - id: openwiki-source-10e4084b6aa57e5cc82620b3
    resource: repo://libs/deepagents/tests/unit_tests/test_end_to_end.py
  - id: openwiki-source-fb60ee46c55b974b8341651c
    resource: repo://libs/DEVELOPMENT.md
  - id: openwiki-source-9015d76fe513ec6a3f55fe3d
    resource: repo://libs/partners/quickjs/langchain_quickjs/_prompt.py
  - id: openwiki-source-e93ea9e1f8eb3113683abb76
    resource: repo://libs/partners/quickjs/langchain_quickjs/middleware.py
  - id: openwiki-source-224af38a099019bb2a44f0a7
    resource: repo://libs/partners/quickjs/Makefile
  - id: openwiki-source-c61950428962577991516f6b
    resource: repo://libs/partners/quickjs/tests/unit_tests/smoke_tests/test_system_prompt.py
  - id: openwiki-source-06bd07c824730c626bd7efd5
    resource: repo://libs/partners/quickjs/tests/unit_tests/test_prompt_modes.py
  - id: openwiki-source-649816c915bd7d9b306918d3
    resource: repo://libs/partners/quickjs/tests/unit_tests/test_repl_middleware.py
  - id: openwiki-source-81698d033a5726401d48b135
    resource: repo://libs/talon/deepagents_talon/config.py
  - id: openwiki-source-9ee5b3034d091a886c8fdda4
    resource: repo://libs/talon/deepagents_talon/effort.py
  - id: openwiki-source-d8846ad7008aa3a1566511a7
    resource: repo://libs/talon/deepagents_talon/hook_transcripts.py
  - id: openwiki-source-6801a88de6305bc8cbdd259f
    resource: repo://libs/talon/deepagents_talon/host.py
  - id: openwiki-source-f04ce33d1db21a61b1e6e8b3
    resource: repo://libs/talon/deepagents_talon/model_selection.py
  - id: openwiki-source-665a21e2fbd09a89d3f13ac0
    resource: repo://libs/talon/deepagents_talon/runtime.py
  - id: openwiki-source-ba53b2ab73965694b2510a58
    resource: repo://libs/talon/Makefile
  - id: openwiki-source-022c52172483653e0c3a3eaf
    resource: repo://libs/talon/tests/unit_tests/test_effort.py
  - id: openwiki-source-c0f076507fca2b1a9eaae92f
    resource: repo://libs/talon/tests/unit_tests/test_hook_transcripts.py
  - id: openwiki-source-7dcd7e37c18e77553a46cce3
    resource: repo://libs/talon/tests/unit_tests/test_hooks.py
generated: { by: "openwiki/0.4.2", at: "2026-10-31T08:04:00.426Z" }
---

# Testing Guide

Put a network-free regression in the owning package's `tests/unit_tests/` tree and make it prove an observable contract: returned or persisted state, model-visible messages, a safe side effect, emitted notification, or cleanup result. Put a test that genuinely crosses a network boundary in `tests/integration_tests/`. Do not add `@pytest.mark.asyncio`: package configuration enables automatic asyncio handling. Warnings not explicitly accepted by the repository fail the suite. See [Development](../operations/development.md), [runtime behavior](../architecture/runtime-behavior.md), [middleware catalog](../concepts/middleware-catalog.md), and [Talon](../integrations/talon.md).

## Run the owning package command

Use the package Makefile rather than an ad hoc pytest command. It preserves the package's dependency group, coverage settings, socket policy, parallelism, and timeout.

```bash
make -C libs/deepagents test TEST_FILE=tests/unit_tests/middleware/test_rubric_middleware.py
make -C libs/deepagents test TEST_FILE=tests/unit_tests/middleware/test_pinned_skills.py
make -C libs/deepagents test TEST_FILE=tests/unit_tests/middleware/test_summarization_middleware.py
make -C libs/deepagents test TEST_FILE=tests/unit_tests/test_async_subagents.py
make -C libs/deepagents test TEST_FILE=tests/unit_tests/test_end_to_end.py PYTEST_EXTRA='-k RubricMiddleware'
make -C libs/partners/quickjs test TEST_FILE=tests/unit_tests/test_prompt_modes.py
make -C libs/partners/quickjs test TEST_FILE=tests/unit_tests/test_repl_middleware.py
make -C libs/partners/quickjs test TEST_FILE=tests/unit_tests/smoke_tests/test_system_prompt.py
make -C libs/partners/quickjs update_snapshots
make -C libs/talon test TEST_FILE=tests/test_runtime.py
make -C libs/talon test TEST_FILE=tests/unit_tests/test_effort.py
make -C libs/talon test TEST_FILE=tests/unit_tests/test_hooks.py
make -C libs/talon test TEST_FILE=tests/unit_tests/test_hook_transcripts.py
```

`libs/deepagents` defaults `TEST_FILE` to `tests/unit_tests/`, runs pytest with xdist, and disables sockets except Unix sockets; its integration aliases select `tests/integration_tests/` and set a 30-second timeout. QuickJS defaults to `tests/unit_tests/`, runs socket-disabled pytest, and provides `update_snapshots` for only the system-prompt smoke suite. Talon's `test` target first runs its WhatsApp bridge tests, then runs socket-disabled pytest with a 10-second timeout; it accepts the same `TEST_FILE` and `PYTEST_EXTRA` narrowing. Follow a behavior change with the owning package's `make lint`.

## QuickJS prompts and REPL lifecycle

Test the public model contract at two complementary seams. `test_prompt_modes.py` calls the prompt renderers directly and should assert mode-specific language in all three representations that users and models receive: the system addendum, the eval argument documentation, and the eval tool description. The contract is `thread` state across calls and turns, `turn` state only within one turn, and `call` state never across invocations. Also cover timeout omission as an observable absence: no timeout wording or malformed `None`/`inf` text when timeout is disabled.

At the middleware seam, give `CodeInterpreterMiddleware` a controlled `ModelRequest` and inspect the request handed to the handler. Assert that exactly one rendered snippet extends the system message, that its tool name, memory limit, mode, and timeout are visible, and that subagent guidance appears only when a live `task` tool is available and subagents are enabled. With PTC configured, construct the request with the actual exposed tool set; filtering occurs at model-call time, so test the model-visible prompt/API reference rather than the prompt-cache implementation or helper-call order.

```mermaid
flowchart TD
    Start["Agent starts"] --> Before["before_agent assigns slot"]
    Before --> Restore{"Thread mode and valid snapshot"}
    Restore -->|yes| Repl["Restore REPL state"]
    Restore -->|no| Repl
    Repl --> Calls["Eval calls in active slot"]
    Calls --> Finish["after_agent clears per-slot prompt bindings"]
    Finish --> Thread{"Thread mode"}
    Thread -->|yes| Snapshot["Create and persist snapshot then evict slot"]
    Thread -->|no| Evict["Evict slot"]
```

*QuickJS allocates a slot lazily; only `thread` mode carries a snapshot through the next agent run.*

Use `test_repl_middleware.py` for this lifecycle and its failure boundaries. Drive JavaScript through the REPL to prove state persistence or isolation, not registry internals: a second evaluation sees a binding in the same REPL, a different thread does not, and a registry reset removes globals. For middleware state, assert the slot/state update and externally visible behavior: `thread` snapshots are restored only when admissible, while `turn` and `call` evict state at the end of the agent run. Include a signed-snapshot negative case when changing snapshot authentication: a missing or tampered tag must prevent restoration and clear stale persisted data. Keep sync and async hook paths aligned.

### Prompt snapshots and the SDK-only CI seam

The smoke suite builds a real `create_deep_agent` with `CodeInterpreterMiddleware`, controlled fake model, and representative no-tool/mixed-tool inputs, then captures the first system message. It covers every persistence mode and a no-timeout configuration. The committed files in `tests/unit_tests/smoke_tests/snapshots/` are a drift detector for wording QuickJS composes with SDK-owned prompt material; they are not behavioral authority. Preserve direct assertions for semantic requirements such as mode lifecycle, resource wording, tool availability, and timeout omission. When an intentional SDK or prompt change changes the rendered text, review the full captured prompt and semantic tests first, then regenerate only with:

```bash
make -C libs/partners/quickjs update_snapshots
```

Do not update snapshots merely to make a diff pass. The Make target invokes the smoke file with `--update-snapshots`; normal execution compares with committed files and creates a missing snapshot only to fail and request a rerun.

CI runs the full QuickJS test job when QuickJS changes (and on pushes). On a non-push pull request that changes `libs/deepagents/**` but not QuickJS, `test-quickjs-sdk-smoke` installs the QuickJS package against its editable local SDK and runs only `tests/unit_tests/smoke_tests/test_system_prompt.py` with sockets disabled. Its explicit path-existence check makes a renamed smoke file fail rather than silently collecting no tests. Keep this test path and its `ci.yml` guard synchronized when relocating the suite.

## Talon reasoning effort: persistence, turn isolation, and provider payloads

Treat `/effort` as a model-keyed durable selection, not a chat-local mutable setting. The host loads `effort.json`, captures the selected model's effort into each `AgentRequest` when it starts a turn, and applies changes from the next turn across chats. Test through `TalonHost` with an operator channel and temporary configuration: set different efforts before different turns and assert each recorded request's `effort`; switch models, query `/effort`, restart the host, and verify selections reload by model. Assert the file's JSON contents and owner-only mode, and that `default` removes only the active model override. Unauthorized or unsupported commands must neither create state nor change a request; a write failure must report the failure and retain the prior in-memory selection.

```mermaid
sequenceDiagram
    participant Operator
    participant Host
    participant StateFile
    participant Runtime
    participant Middleware
    participant Provider
    Operator->>Host: /effort level
    Host->>StateFile: persist model keyed selection
    Operator->>Host: message
    Host->>Runtime: AgentRequest with captured effort
    Runtime->>Middleware: set turn scoped effort settings
    Middleware->>Provider: merged provider payload
    Runtime->>Middleware: reset turn scoped settings
```

*The selection is persisted by model, but its provider settings are scoped to one request and reset after that turn.*

Use `test_effort.py` in three layers. First, verify `effort_levels` and `effort_settings` reject unsupported values and map the selected model family to its real invocation shape: DeepSeek toggles `extra_body.thinking` and uses `reasoning_effort` except for `none`; Qwen uses its thinking budget; GLM toggles native thinking; ordinary supported models use `reasoning_effort`. Provider-profile metadata can be absent, so exercise documented fallback levels where relevant.

Second, serialize settings through the provider adapter rather than asserting the translator's private branches. Use the adapter payload builder for native providers and `httpx.MockTransport` for OpenAI-compatible calls. Assert the emitted wire JSON: Baseten reasoning effort is top-level without wrapper fields, Anthropic uses `output_config.effort`, and the OpenAI Responses API emits `reasoning.effort` rather than `reasoning_effort`. This detects translation regressions at the integration boundary without a network call.

Finally, invoke `DeepAgentRuntime` concurrently with a recording fake model. Assert each turn receives its own settings, a later effort-less turn receives none, and `ACTIVE_EFFORT` is reset. When extending middleware behavior, also assert unrelated `model_settings` and nested `extra_body` fields survive the effort merge; do not assert context-variable token operations or middleware helper ordering.

## Choose the seam that exposes the contract

```mermaid
flowchart TD
    Change["Changed behavior"] --> Boundary{"Can controlled inputs expose it"}
    Boundary -->|Middleware state or message| Unit["Focused unit test with fake model or backend"]
    Boundary -->|Graph wiring| Graph["create_deep_agent with InMemorySaver"]
    Boundary -->|Remote task client| Transport["httpx MockTransport or mocked SDK client"]
    Boundary -->|Hook process and file safety| Talon["Real Talon graph with temporary home"]
    Boundary -->|External service itself| Integration["Focused integration test"]
    Unit --> Make["Owning Makefile target"]
    Graph --> Make
    Transport --> Make
    Talon --> Make
    Integration --> Make
```

*Use controlled local seams until the external boundary is the behavior being changed; assertions should survive an implementation refactor that preserves the contract.*

## Rubric revisions: hook semantics and graph wiring

`RubricMiddleware` does nothing without a rubric. A new rubric, or reuse after a terminal attempt, resets its private attempt bookkeeping. At a natural agent stop, the after-agent hook grades the transcript: a revision verdict injects feedback and returns the graph to `model`; a terminal evaluation is recorded and allowed to finish. `GraphBubbleUp` is control flow, not a grader error, and propagates.

The rubric grader freezes the first nonempty criterion list for an attempt. If a later response under-covers it, the middleware requests one correction. A `satisfied` response that remains under-covered is downgraded so it cannot falsely end the revision loop.

Test at two levels:

- In `tests/unit_tests/middleware/test_rubric_middleware.py`, stub the grading boundary for constructor validation, state reset or stickiness, feedback, cap behavior, callbacks/events, sync/async parity, and exception propagation. Stub the grader invocation—not a sequence of incidental helper calls—when testing the coverage retry.
- In `tests/unit_tests/test_end_to_end.py`, combine `create_deep_agent`, scripted fake chat models, and `InMemorySaver`. Assert visible revision turns and the persisted rubric state with `get_state`; cover immediate success, loop-and-revise, cap termination, and recovery after a corrected full criterion response.

## Pinned skills: assert the conversation boundary

`SkillsMiddleware` loads skill metadata once per thread; resetting `skills_metadata` to `None` requests a reload. `pinned_skills` is different from metadata discovery: before the next model call, each named readable skill is added as a separate `HumanMessage` marked `lc_source="pinned_skill"`, so the model sees its instructions. Skill tools listed by a pinned skill remain available only while the pin remains in context.

`tests/unit_tests/middleware/test_pinned_skills.py` is the focused graph-level contract suite. Build an agent over a temporary skills directory with `InMemorySaver` and a fake model, then inspect model call history and checkpointed messages. Cover these observables:

- pins follow the user message in supplied order and duplicate names are emitted once;
- a pin is a stored snapshot, survives later turns, and is omitted from the immediate result payload;
- unknown, deleted, empty, oversized, or non-UTF-8 skills are skipped without preventing readable pins;
- parallel tool calls that request pins produce both pinned messages; a subagent result cannot pin a parent; and compaction removes the pin's tool disclosure.

Exercise both `invoke` and `ainvoke` where behavior is shared. Do not assert that a private pinning helper was called; assert the model-visible messages, metadata, bound tools, and persisted transcript.

## Summarization: preserve recoverability and media references

The summarization middleware turns a threshold-crossing message history into a summary plus retained recent messages, offloading evicted history to the backend. The resulting summary is marked `lc_source="summarization"`, refers to the history file only when persistence succeeded, and retains a session identifier in state so later summaries append to the same file. Independent states sharing a backend receive distinct session files.

Inline media is a key safety boundary. Tests use the local `MockBackend` and mock summarizer to verify that data URLs become content-addressed backend files and archive references, identical data is uploaded once, and image URLs survive. A per-item decode or upload failure must be represented by a placeholder and warning rather than silently preserving raw data or losing the fact that media existed. History-write failure still permits a summary, but its event has no nonexistent file path.

Use `tests/unit_tests/middleware/test_summarization_middleware.py` for threshold/no-op, message and notification shape, archive/session identity, media rewrite and failure behavior, token counting, truncation, and sync/async parity. Prefer a fake backend that records writes over assertions about internal formatting calls. The tests use `pytest.mark.anyio` where needed; do not introduce `pytest.mark.asyncio`.

## Async subagent follow-ups: task state is the contract

`AsyncSubAgentMiddleware` supplies tools to start, check, update, cancel, and list remote Agent Protocol tasks. Task identifiers and status live in `async_tasks`, allowing them to survive compaction. An update creates a run on the same remote thread and retains the task ID. The default `interrupt` strategy replaces the current run; `enqueue` leaves the follow-up pending behind it and records the superseded run IDs. Only one follow-up to a task may be sent in a parent turn.

```mermaid
flowchart TD
    Existing["Tracked task"] --> Update["update_async_task"]
    Update --> Strategy{"Update strategy"}
    Strategy -->|interrupt| NewRun["New remote run and running task"]
    Strategy -->|enqueue| Queued["Pending follow-up and superseded run IDs"]
    NewRun --> Poll["Async hooks poll current run"]
    Queued --> Poll
    Poll --> Outcome{"Observed outcome"}
    Outcome -->|success| Notice["Notification with result"]
    Outcome -->|waiting| Notice
    Outcome -->|error or cancelled| Notice
    Notice --> State["Persist refreshed async_tasks state"]
```

*The tracked task follows the newest run; an older completion must not be treated as completion of an enqueued follow-up.*

In `tests/unit_tests/test_async_subagents.py`, mock the SDK client or use `httpx.MockTransport`; no server is necessary. Assert the outgoing remote payload and resulting `Command` state/message, not client call order. Include both sync and async tool entrypoints for update strategy behavior, status polling, queued task listing, cancellation of queued and earlier active runs, and rejection of parallel follow-ups. Parent-pull tests should show that completion is injected before a model call or causes one additional turn after a final answer, while timeouts and unreadable remote state do not block or fabricate a result.

## Talon runtime, hooks, and transcript erasure

`DeepAgentRuntime` owns the Talon graph wiring: it supplies the backend, checkpointer, tools, skills, memory, subagents, middleware, and per-conversation graph configuration. `tests/test_runtime.py` uses recording graphs and local temporary workspaces to test those observable wiring and lifecycle contracts: replacement tools are transactional, the active graph remains stable while an approval is outstanding, repeated conversation IDs retain the thread, and interrupted work is repaired with a matching tool result before its interruption marker.

For hook behavior, run a real local Talon graph with a fake chat model and a temporary `hooks.json`, as `tests/unit_tests/test_hooks.py` does. Assert wire payloads and effects:

- `PreToolUse` can deny execution before the side effect, but hook allow does not bypass Talon's separate approval decision;
- post-failure hooks receive failed execution, configuration reloads on later turns, and workspace hook files are not trusted as home configuration;
- concurrent conversations and subagents keep session IDs, hook snapshots, and transcript paths isolated.

Hook transcripts have an erasure boundary per durable parent conversation. Erasure writes a revocation marker before deleting child projections, legacy projections, and backups; existing writers compare the marker and reject later writes. The transcript store refuses symlinked directories, markers, and paths. `tests/unit_tests/test_hook_transcripts.py` should use actual temporary files and the saver/runtime helpers to verify restart and retry behavior, in-flight materialization serialization, isolation from other owners, removal of backups, and symlink safety. These are security and data-deletion contracts—test filesystem outcomes and rejected writes, not the order of private cleanup calls.

## Completion checklist

1. Identify the observable behavior and route it to the narrowest network-free test under `tests/unit_tests/`.
2. Use fake models, fake backends, a mock transport, or a temporary Talon home to control inputs and failures.
3. For QuickJS, assert model-visible prompt text and real REPL persistence/isolation; review semantics before regenerating snapshots with `make -C libs/partners/quickjs update_snapshots`.
4. For Talon effort, assert the persisted model-keyed selection, captured turn value, reset isolation, and serialized provider payload.
5. For rubric and pinning, inspect model-visible messages and checkpoint state; use an end-to-end fake-model graph only for graph wiring.
6. For summarization, prove archive references, session isolation, warnings, and no-data-loss failure behavior.
7. For async subagents, assert persisted task state, notification content, and remote payload semantics for both update strategies.
8. For Talon hooks, prove deny/approval precedence, concurrent isolation, and transcript erasure safety with real local side effects.
9. Avoid `pytest.mark.asyncio`, sleeps, provider calls, and internal-call-order assertions; run the owning `make lint` after the focused tests pass.
