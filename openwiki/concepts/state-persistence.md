---
type: state persistence model
title: State Persistence
description: Explains checkpoint ownership, delta-reduced graph state, and the state-transfer boundary between isolated and forked subagents. Covers private middleware channels and dcode's local session persistence.
tags: [state, persistence, checkpoints, reducer, middleware, subagents, sessions, langgraph]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-03T08:05:39.427Z
sources:
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-2e03fee957625ca21a1c21af
    resource: repo://libs/code/deepagents_code/main.py
  - id: openwiki-source-620b4c9d0fcbd4c7e6aa0120
    resource: repo://libs/code/deepagents_code/resume_state.py
  - id: openwiki-source-0f8622164498a685abc913d5
    resource: repo://libs/code/deepagents_code/sessions.py
  - id: openwiki-source-cd2a5280cf3ca3ab491d7a8e
    resource: repo://libs/code/tests/unit_tests/test_sessions.py
  - id: openwiki-source-822ae989625ba99d4c7cc08b
    resource: repo://libs/deepagents/deepagents/_messages_reducer.py
  - id: openwiki-source-07f9eac13e71bcbdb4e6994b
    resource: repo://libs/deepagents/deepagents/backends/state.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-421bc4b065189ae1165ca326
    resource: repo://libs/deepagents/deepagents/middleware/_state.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-454da083c2cc29febd156c7e
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_subagent_middleware_init.py
  - id: openwiki-source-dc64f28a66d10932b86fcd61
    resource: repo://libs/deepagents/tests/unit_tests/test_messages_reducer.py
  - id: openwiki-source-ca8183c87e6002c442ee2d62
    resource: repo://libs/deepagents/tests/unit_tests/test_subagents.py
generated: { by: "openwiki/0.4.2", at: "2026-09-03T08:05:39.427Z" }
---

# State Persistence

Deep Agents separates three concerns that are easy to conflate:

1. **Graph state and checkpoints** are owned by LangGraph. A checkpointer preserves a thread's channel values, interrupts, and resumability.
2. **Message reduction** controls the checkpoint representation of one channel; it is not a separate durability policy.
3. **Subagent transfer** projects state into an invocation and projects an allowed result back. It does not make the child and parent share mutable state.

The important refinement is that a task subagent can leave a transcript in the *parent checkpointer* under a `tools:` checkpoint namespace, while its transcript is still isolated from the parent's root `messages` state and normal output projection. Checkpoint storage and state visibility are therefore different boundaries.

For context compaction, see [Context management](/openwiki/concepts/context-management.md). For backend durability choices, see [Backends](/openwiki/concepts/backends.md); for subagent configuration, see [Subagents and skills](/openwiki/concepts/subagents-skills.md).

## Checkpoint and backend ownership

`create_deep_agent` forwards `checkpointer` and `store` to LangChain's `create_agent`. The checkpointer is optional and preserves graph state between runs. A `store` is separately required for a backend route that uses a store.

| Concern | Owner | Scope |
| --- | --- | --- |
| Conversation state, interrupts, and resume | LangGraph checkpointer | A graph thread and its checkpoint namespaces |
| Files and memory data | Deep Agents backend | Determined by the selected route |
| Local session catalog | dcode `sessions.py` | Metadata and checkpoint rows in local SQLite |

The default `StateBackend` reads and queues writes to the `files` channel through LangGraph's `CONFIG_KEY_READ` and `CONFIG_KEY_SEND`. Its files are checkpointed within a thread, do not cross threads, and the backend rejects use outside graph execution. Store- and filesystem-backed routes have their own durability boundaries; a checkpointer alone does not make those resources durable.

```mermaid
flowchart LR
    Input["Agent input"] --> Root["Root graph state"]
    Root --> CP["Checkpointer by thread"]
    CP --> Resume["Resume and interrupts"]
    Root --> Files["StateBackend files channel"]
    Files --> CP
    Root --> Route["Store or filesystem route"]
```

This diagram distinguishes thread checkpointing from backend-route durability.

## DeepAgentState and delta reduction

`DeepAgentState` subclasses LangChain's `AgentState` and overrides `messages` with `DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)`. It is the default schema used when `create_deep_agent` receives no custom `state_schema`.

`DeltaChannel` persists message deltas and only periodically writes a full snapshot, every 50 pregel steps. This changes long-thread checkpoint growth from repeated full histories to linear persisted volume while bounding replay depth. `FilesystemState.files` applies the same delta-channel and snapshot-frequency pattern.

A custom schema is expected to subclass `DeepAgentState` so it retains this messages channel, but that constraint is type-only: `TypedDict` prevents an `issubclass` runtime validation. Middleware schemas are merged with the caller's base schema, which lets middleware own fields used by its hooks and tools while a custom schema supplies graph-wide application fields.

### Reducer invariants

`_messages_delta_reducer` receives a batch of writes for `DeltaChannel` and reconstructs `messages` as follows:

- It flattens list writes and coerces raw dictionaries, strings, and tuples to typed messages.
- It replaces or appends by message ID, appends `id=None` messages unchanged, and tombstones an identified message with `RemoveMessage`.
- The last `RemoveMessage(REMOVE_ALL_MESSAGES)` resets prior state and discards earlier writes in the batch.
- `state=None` is treated as an empty list, supporting replay of old threads whose earliest checkpoint did not seed `messages: []`.

Message IDs are deliberately not assigned in the reducer. LangGraph's `ensure_message_ids` stamps stable IDs before checkpoint serialization, which prevents replay from manufacturing IDs different from persisted ones. Focused tests cover non-`None`, stable IDs returned by `get_state()` for object and dictionary-style inputs across sync and async resumed invocations. The local adaptation does not coerce `BaseMessageChunk`, because Deep Agents writes full `AIMessage` values to state and streams on the output-event path.

```mermaid
flowchart TD
    Writes["Message writes"] --> Coerce["Flatten and coerce message values"]
    Coerce --> Clear{"Clear-all sentinel"}
    Clear -->|"yes"| Reset["Discard prior state and earlier writes"]
    Clear -->|"no"| Index["Index existing messages by ID"]
    Reset --> Index
    Index --> Apply["Append update or tombstone"]
    Apply --> Result["Reduced message list"]
```

This is channel-value reconstruction; `DeltaChannel` decides whether a checkpoint contains a snapshot or writes only.

## Middleware privacy and subagent state transfer

`private_state_field_names` finds `PrivateStateAttr` annotations across the assembled state schemas, and `create_deep_agent` assigns the resulting keys to `SubAgentMiddleware`. The task tool removes private keys on both input and result projection. This is a confidentiality boundary, not merely an output-schema preference.

Annotation resolution happens at runtime. If `get_type_hints` cannot resolve a schema because an annotation references a `TYPE_CHECKING`-only name, the schema is skipped with a warning. None of that schema's private fields are then protected, so they can cross the task boundary. Ensure names used by private annotations are imported at runtime.

A custom base `state_schema` is forwarded while declarative `SubAgent` specifications are compiled; a precompiled `CompiledSubAgent` and remote `AsyncSubAgent` own their schemas and do not inherit it.

### Isolated is the default; `handoff` is legacy

The current default mode is `"isolated"`. `"handoff"` remains an accepted legacy alias for the same behavior; neither mode forks the parent. For either mode, the task tool starts with permitted parent fields, removes `messages`, `todos`, `structured_response`, the fork marker, and private keys, and sets `messages` to one `HumanMessage` containing the task description.

On return, the result must contain `messages`. The tool filters the same excluded and private keys from state updates, then adds exactly one root `ToolMessage`: serialized `structured_response` when present, otherwise the last non-empty `AIMessage` text. Thus a parent receives an allowed state update and report, not the child's working transcript or todo list.

```mermaid
sequenceDiagram
    participant Parent
    participant Task as task tool
    participant Child as isolated subagent
    Parent->>Task: parent state and description
    Task->>Task: filter excluded and private keys
    Task->>Child: permitted fields and one task message
    Child-->>Task: child result state
    Task->>Task: filter result and select report
    Task-->>Parent: allowed updates and one ToolMessage
```

This is the isolated task-transfer flow. Private keys are filtered in both directions.

### Fork inherits input context but not root-state visibility

`mode="fork"` is experimental and explicitly opts into inherited context. A declarative fork receives parent state except `structured_response`, summarization event, and summarization session ID; it receives a fork marker and a message list made from the effective (including applied summarization) parent conversation plus a task preamble. It mirrors parent prompt-producing middleware, appends its own system prompt rather than replacing the inherited prompt, and cannot define `skills`.

A forked `CompiledSubAgent` is deliberately narrower because its runnable is opaque: it receives inherited non-private permitted state and the prepared conversation, but private keys remain excluded. The fork marker causes nested `task` calls to be refused, preventing recursive delegation. In either kind of fork, output projection remains the normal filtered return path.

```mermaid
flowchart TD
    ParentState["Parent runtime state"] --> Kind{"Fork specification"}
    Kind -->|"declarative"| Decl["Keep private state except fork exclusions"]
    Kind -->|"compiled"| Comp["Remove private and excluded state"]
    Decl --> History["Apply summary and append task preamble"]
    Comp --> History
    History --> Child["Forked child invocation"]
    Child --> Filter["Filter result and emit root ToolMessage"]
    Filter --> ParentResult["Parent root state"]
```

This diagram shows input inheritance by fork kind; it does not imply shared state.

### Checkpoint namespaces versus output projection

The `task` tool invokes a subagent directly rather than registering it as a graph node. Nevertheless, when the parent has a checkpointer, focused integration coverage shows the child transcript can be recovered by listing that checkpointer with the same thread configuration and selecting its `tools:` checkpoint namespace. It is not available through the parent's root-state projection: ordinary streaming excludes child intermediates unless `subgraphs=True`, and the root receives only the resulting `ToolMessage`.

Treat these facts separately when building observability or retention tooling: use checkpoint namespaces (or `subgraphs=True` streaming) to inspect child execution; use root state to model what the parent can continue from.

## dcode sessions and resume facts

Local dcode obtains an `AsyncSqliteSaver` over the hardened global `sessions.db`; startup calls `setup()` before constructing CLI graphs with it. Its thread catalog derives `ThreadInfo` from checkpoint metadata, including agent name, timestamps, Git branch, working directory, latest checkpoint ID, and optional prompt/message details. `list_threads` can filter by agent, branch, and exact `cwd`.

`ResumeStateMiddleware` contributes private, checkpointed resume facts. After a successful model turn it writes context-token usage from the latest `AIMessage`; configurable-model middleware writes effective model, parameters, and cache-request facts after successful calls. Accepted goal/rubric choices may be written by the client via `aupdate_state`; pending proposals and agent-driven status updates are graph-written. Reading a checkpoint gives the facts at that checkpoint, not a thread-wide aggregate.

A delta checkpoint can omit an inline `messages` snapshot. For message counts, dcode uses an inline list when available; otherwise it replays root-namespace message writes ordered by checkpoint, task, and index, deliberately excluding subgraph writes that share the thread ID. This keeps the thread browser's visible count about the root conversation rather than task-subagent transcripts.

Deleting a local thread removes its checkpoint and writes rows and clears relevant caches, then attempts offloaded conversation-history cleanup. The Boolean result reflects checkpoint deletion only; history cleanup is a separate best-effort operation and may fail without changing that result.

## Safe extension checklist

- Supply a checkpointer for resumable graph threads; choose a backend route independently for file durability.
- Preserve `DeepAgentState.messages` when supplying custom state, and use middleware state for middleware-local behavior.
- Mark sensitive middleware fields with `PrivateStateAttr`, import annotation names at runtime, and test both incoming and outgoing task projections.
- Treat `isolated`—and legacy `handoff`—as a fresh task context. Use experimental `fork` only when inherited conversation/state semantics are intended, and account for the declarative-versus-compiled privacy distinction.
- Do not equate root visibility with retention: child transcripts can reside in a `tools:` checkpoint namespace even though normal parent state and output only contain the final tool report.
- For dcode thread tools, distinguish a missing delta snapshot from an empty conversation and use the writes-aware root-namespace reconstruction path.
