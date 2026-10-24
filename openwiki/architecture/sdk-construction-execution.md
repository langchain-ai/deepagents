---
type: architecture
title: SDK Construction and Execution
description: How create_deep_agent constructs a Deep Agents LangChain graph, composes middleware and subagent stacks, propagates state, and configures compiled execution.
tags: [deepagents, sdk-construction, middleware, subagents, langgraph, permissions]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-24T08:05:08.866Z
sources:
  - id: openwiki-source-b93533cac55718d75277d1cf
    resource: repo://libs/deepagents/deepagents/_excluded_middleware.py
  - id: openwiki-source-822ae989625ba99d4c7cc08b
    resource: repo://libs/deepagents/deepagents/_messages_reducer.py
  - id: openwiki-source-e3efb5f3e4a9e8517eb6d8f5
    resource: repo://libs/deepagents/deepagents/backends/protocol.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-41c16d89908bf285e252530e
    resource: repo://libs/deepagents/deepagents/middleware/_parent_notifier.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-66cf9d0832d3cb55bec2b5ed
    resource: repo://libs/deepagents/deepagents/middleware/skills.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-13e0ea89aadd1a5e309ac7e3
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_skill_tools_payload.py
  - id: openwiki-source-f445d59792df76394a37a768
    resource: repo://libs/deepagents/tests/unit_tests/test_artifacts_root.py
  - id: openwiki-source-6ce85b02eabe462f99e0c912
    resource: repo://libs/deepagents/tests/unit_tests/test_async_subagents.py
  - id: openwiki-source-10e4084b6aa57e5cc82620b3
    resource: repo://libs/deepagents/tests/unit_tests/test_end_to_end.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
  - id: openwiki-source-851e3a9c96663d8db5ca3dec
    resource: repo://libs/deepagents/tests/unit_tests/test_permissions.py
  - id: openwiki-source-ca8183c87e6002c442ee2d62
    resource: repo://libs/deepagents/tests/unit_tests/test_subagents.py
generated: { by: "openwiki/0.4.2", at: "2026-10-24T08:05:08.866Z" }
---

# SDK Construction and Execution

`create_deep_agent()` is the SDK assembly boundary. It resolves the model and its harness profile, chooses a filesystem backend, partitions synchronous and remote subagent specifications, builds middleware, and delegates graph compilation to LangChain `create_agent()`. The result is a compiled LangGraph; it is not a separate Deep Agents execution loop.

## Construction flow

```mermaid
flowchart TD
    App["Application"] --> Build["create_deep_agent"]
    Build --> Resolve["Resolve model, profile, tools, backend, and prompt"]
    Resolve --> Partition["Partition subagent specifications"]
    Partition --> Inline["Build inline task subagents"]
    Partition --> Remote["Configure remote async subagents"]
    Inline --> Default["Add general-purpose subagent when enabled"]
    Default --> Stack["Build main and child middleware stacks"]
    Remote --> Stack
    Stack --> FirstFilter["Apply profile exclusions"]
    FirstFilter --> Custom["Replace or splice caller middleware"]
    Custom --> SecondFilter["Apply exclusions again"]
    SecondFilter --> Tools["Append excluded-tool filter"]
    Tools --> Compile["Compile with create_agent"]
```

Caption: construction partitions inline and Agent Protocol delegation before applying the two exclusion passes and caller middleware transformation.

### Inputs resolved by the entry point

The public entry point accepts a model, tools, prompt, middleware, subagent specifications, skills and memory sources, permissions, backend, interrupt policy, schemas, and LangGraph execution options. If no backend is passed, it creates `StateBackend()`. `BackendProtocol` supplies the common file-operation boundary; shell execution is a separate `SandboxBackendProtocol` capability, so filesystem support does not imply that `execute` can run shell commands.

A string model is resolved before profile lookup; a supplied `BaseChatModel` is retained. `model=None` remains a deprecated compatibility path and constructs `ChatAnthropic(model_name="claude-sonnet-4-6")`. The selected harness profile may rewrite supplied tool descriptions and contributes its prompt material. Profile text is appended to a string prompt, while a supplied `SystemMessage` retains its existing content blocks and receives profile text as an additional block.

A specification with `graph_id` is remote (`AsyncSubAgent`); declarative `SubAgent` specifications and mappings with `runnable` (`CompiledSubAgent`) are inline. Unless the profile disables it or an inline spec already uses the name `general-purpose`, construction prepends the default general-purpose inline agent. Remote specs alone do not create the synchronous `task` tool; conversely, remote async task tools remain available even when all inline agents are disabled.

## Middleware assembly and transformation

The main stack begins in this order:

1. `FilesystemMiddleware`;
2. `SubAgentMiddleware`, only when inline subagents exist;
3. `_ParentNotifierMiddleware`;
4. summarization middleware;
5. `PatchToolCallsMiddleware`;
6. `AsyncSubAgentMiddleware`, only when remote subagent specifications exist.

The tail contains profile middleware, optional `SkillsMiddleware`, provider prompt-caching middleware, optional `MemoryMiddleware`, generated `HumanInTheLoopMiddleware`, and `UnsupportedContentMiddleware`. Exclusions are applied once to that assembled stack, then caller middleware is transformed into it, then exclusions are applied again. A caller middleware with the name of a surviving entry replaces that entry in place; a new name is spliced after the surviving core and before the tail. If the profile excludes tools, `_ToolExclusionMiddleware` is appended last, preventing later stack composition from restoring an excluded model-visible tool.

```mermaid
flowchart LR
    Core["Core middleware"] --> Tail["Profile, skills, caching, memory, interrupts, unsupported content"]
    Tail --> ExcludeOne["First exclusion pass"]
    ExcludeOne --> Merge["Replace same-name entries or splice new entries after core"]
    Merge --> ExcludeTwo["Second exclusion pass"]
    ExcludeTwo --> ToolFilter["Excluded-tool filter last"]
```

Caption: caller middleware is merged between two profile-exclusion passes; tool exclusion is deliberately the final transformation.

### Exclusion and replacement invariants

`FilesystemMiddleware` and `SubAgentMiddleware` are protected scaffolding. A profile cannot exclude either by class or name: construction raises rather than leaving broken filesystem or task behavior. Class-form exclusions match the exact exclusion type; strings match `AgentMiddleware.name`. Every exclusion must match an assembled main or general-purpose stack, and an exclusion name that matches multiple distinct middleware classes is rejected. `_ParentNotifierMiddleware` is not protected and can be removed with a class-form exclusion.

Replacing a `FilesystemMiddleware` has an additional safety check: when the original carries agent permissions, a replacement inherits them if unconfigured; a different nonempty replacement policy is a conflict and raises `ValueError`.

The automatic general-purpose child is intentionally not a copy of the main stack. Its base is filesystem, summarization, and patching, followed by profile middleware, optional skills, and prompt caching. Caller middleware reaches that child only when it replaces an original general-purpose stack slot, limiting inheritance of main-agent-only behavior.

## State and compilation boundary

Absent a supplied schema, the main graph uses `DeepAgentState`. Its `messages` channel is a `DeltaChannel` with snapshot frequency 50 and the message-delta reducer. The reducer normalizes raw message inputs, treats absent replay state as empty, replaces or removes messages by ID, and honors the remove-all sentinel. This preserves message update semantics while avoiding repeated full-history checkpoint growth.

A supplied `state_schema` is the base main schema and is forwarded when declarative inline agents are compiled. A precompiled runnable owns its own schema. After the final main stack is assembled, private fields from the main schema and installed middleware schemas are assigned to every installed `SubAgentMiddleware`; these keys are not sent across the task boundary or returned as child state updates.

The outer `create_agent()` call receives the resolved model and prompt, rewritten caller tools, final middleware, response format, context schema, checkpointer, store, debug flag, name, cache, and either the supplied schema or `DeepAgentState`. The returned graph is configured with recursion limit `9999` and metadata identifying the Deep Agents integration, its installed version, and the supplied agent name. Checkpointer and store are forwarded at this outer compilation boundary, not separately injected into child compilation.

## Inline `task` delegation

Declarative inline agents resolve their own model and profile and receive a child stack. They inherit parent permissions and `interrupt_on` only when the spec does not provide its own values; caller tools are inherited only when `tools` is omitted. A fork also mirrors parent prompt-producing middleware so it can reconstruct the inherited prompt. Child invocation runs with ambient LangGraph configuration, so callbacks, tags, configurable values, and metadata propagate, while configuration bound by the child wins collisions.

```mermaid
sequenceDiagram
    participant Parent as Parent graph
    participant Task as task tool
    participant Child as Inline child graph
    Parent->>Task: Select subagent and task description
    Task->>Child: Isolated task or forked history
    Child-->>Task: Resulting child state
    Task-->>Parent: ToolMessage and permitted update
```

Caption: inline delegation is a blocking child-graph invocation rather than a remote background run.

An isolated task receives permitted state and a fresh task `HumanMessage`. A fork receives effective post-summarization parent history plus the task, and a private marker makes recursive task delegation fail. The bridge requires child `messages`, discards those messages from state updates, filters private or excluded keys and unchanged values, then returns one `ToolMessage`: JSON-serialized structured output when present, otherwise the last nonempty AI text. Shared custom fields that parallel task calls can modify need reducers, or the parent step can fail with `InvalidUpdateError`.

## Remote asynchronous subagents

`AsyncSubAgentMiddleware` is independent of `SubAgentMiddleware`. It validates unique names, `isolated` or experimental `fork` mode, and `interrupt` or `enqueue` update strategy. It contributes reducer-merged `async_tasks` parent state and exposes `start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`, and `list_async_tasks`.

```mermaid
sequenceDiagram
    participant Parent as Parent graph
    participant AsyncTools as Async task tools
    participant Server as Agent Protocol server
    participant Tracking as Parent async_tasks state
    Parent->>AsyncTools: start_async_task
    AsyncTools->>Server: Create thread and run
    AsyncTools->>Tracking: Store local running record
    AsyncTools-->>Parent: Return task ID
    Parent->>AsyncTools: Check, update, cancel, or list
    AsyncTools->>Server: Inspect or mutate remote run
    AsyncTools->>Tracking: Merge tracking update
```

Caption: remote threads and runs are owned by the Agent Protocol server, while the parent persists only its task-tracking projection.

Launching creates a remote thread and run, then immediately writes a parent-local record keyed by the thread ID while the parent continues. An SDK launch failure returns a tool error and records no task. The record tracks the newest run, status, timestamps, and—in enqueue mode—superseded run IDs. `interrupt` is the default update strategy. `enqueue` records the follow-up as `pending`, prevents an earlier completion from becoming the latest result, and rejects more than one queued follow-up for the same task in one model turn. Cancellation attempts to stop the tracked newest and earlier active or pending runs; cancelling a `waiting` task changes only local tracking because it has no active run.

Isolated launch sends only the task. Fork mode serializes a launch-time snapshot of effective parent history plus a preamble and task after removing parent-local artifacts and eviction markers and hydrating portable media. It shares conversation content once, not the parent's model, prompt, tools, filesystem, or state. A same-deployment launch may forward JSON-compatible run context after removing the parent reference; URL-targeted launches do not forward it. Local ASGI transport without `url` requires an asynchronous entry point such as `ainvoke`; synchronous invocation requires a reachable URL.

A successfully completed run whose thread is paused is represented as `waiting`. Checks can discover a newer run begun elsewhere, and list refreshes nonterminal cached records while preserving cached state on retrieval failure. A configured webhook is attached to starts and follow-up runs. Permissions, human approval, authorization, and persistence for remote work belong to the remote graph and Agent Protocol deployment; the parent's local filesystem permissions and interrupt configuration do not transfer.

### Polling and parent notification

Remote polling runs only through asynchronous hooks. `abefore_agent` records launch notification and pulls updates, `abefore_model` pulls before a model call, and `aafter_model` pulls after a final answer and returns to the model only for a new notification. Newly observed terminal or waiting outcomes are reported once; failed or timed-out observations are deferred to a later pull. Synchronous invocation still exposes the async tools but does not poll.

The normal SDK-built stack adds `_ParentNotifierMiddleware`. For a top-level asynchronous task run that carries parent linkage, it reports success or model failure once. It wakes an idle or errored parent only when parent thread and assistant identifiers are available; a paused task is not woken this way, and a busy parent learns through polling. URL-targeted launches and `notify_parent=False` omit the assistant reference. A remote task waiting for input or approval must be resumed on its own thread.

## Permissions, skills, and artifact paths

Filesystem enforcement and human approval are separate. `FilesystemMiddleware` independently enforces deny rules. Construction translates interrupt-mode filesystem permissions into path-sensitive `HumanInTheLoopMiddleware.interrupt_on` predicates and merges caller entries over generated entries for the same tool. The predicates distinguish exact-path from bulk operations, conservatively gate pathless or current-directory bulk calls, account for glob patterns that redirect outside the named path, and use resolved symlink targets when the backend supports resolution. Declarative inline agents inherit these controls unless overridden; remote agents do not. Resuming an interrupt requires a checkpointer.

Skills metadata is loaded from backend sources and cached per thread. Later sources override earlier entries with the same name; setting `skills_metadata` to `None` reloads it, and source failures are recorded as warnings. Skill-tool disclosure follows its associated read result without splitting a tool-result batch or moving ahead of queued user content, and uses the model selected after fallback.

`CompositeBackend.artifacts_root` defaults to `/`. Filesystem large-result offload and summarization-history offload normalize a trailing slash and derive their paths from that root.

## Verification focus

The graph tests exercise profile selection, stack order and conditional installation, caller replacement and exclusion failures, schema/private-state wiring, metadata, and general-purpose composition. Permission tests exercise generated approval predicates and deny enforcement, including conservative recursive-delete handling. The async middleware tests cover remote transport, fork snapshots, status tracking, polling, notification, webhooks, update strategies, and cancellation; subagent tests cover isolation, fork behavior, state filtering, and configuration propagation.

## Related pages

- [Middleware stack](middleware-stack.md) — hook responsibilities and ordering.
- [Runtime behavior](runtime-behavior.md) — runtime execution and operations.
- [Middleware catalog](../concepts/middleware-catalog.md) — middleware capabilities.
- [Subagents and skills](../concepts/subagents-skills.md) — delegation configuration.
- [Build a Deep Agent](../workflows/build-a-deep-agent.md) — SDK usage workflow.
