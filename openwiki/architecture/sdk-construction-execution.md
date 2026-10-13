---
type: architecture
title: SDK Construction and Execution
description: Explains how create_deep_agent classifies subagents, derives approvals from filesystem permissions, composes middleware, and compiles the configured LangGraph agent.
tags: [deepagents, sdk-construction, agent-execution, middleware, subagents, permissions, langgraph]
sources:
  - id: openwiki-source-b93533cac55718d75277d1cf
    resource: repo://libs/deepagents/deepagents/_excluded_middleware.py
  - id: openwiki-source-822ae989625ba99d4c7cc08b
    resource: repo://libs/deepagents/deepagents/_messages_reducer.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-66cf9d0832d3cb55bec2b5ed
    resource: repo://libs/deepagents/deepagents/middleware/skills.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-13e0ea89aadd1a5e309ac7e3
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_skill_tools_payload.py
  - id: openwiki-source-851e3a9c96663d8db5ca3dec
    resource: repo://libs/deepagents/tests/unit_tests/test_permissions.py
verified:
  - by: openwiki/0.4.2
    at: 2026-10-13T08:11:05.854Z
generated: { by: "openwiki/0.4.2", at: "2026-10-13T08:11:05.854Z" }
---

# SDK Construction and Execution

`create_deep_agent` is the public assembly entry point. It does not implement a separate agent loop: it resolves Deep Agents policy and components, calls LangChain `create_agent()`, and returns that compiled graph with Deep Agents state and configuration defaults.

## Construction flow

```mermaid
flowchart TD
    App["Application"] --> Builder["create_deep_agent"]
    Builder --> Resolve["Resolve model and harness profile"]
    Resolve --> Classify["Classify subagent specifications"]
    Classify --> Sync["Inline synchronous subagents"]
    Classify --> Remote["Async remote subagents"]
    Sync --> Assemble["Assemble filesystem and middleware stack"]
    Remote --> Assemble
    Assemble --> Approval["Derive approval predicates from permissions"]
    Approval --> Filter["Apply middleware and tool exclusions"]
    Filter --> Compile["LangChain create_agent"]
    Compile --> Graph["Configured compiled graph"]
    Graph --> Request["Model request"]
    Request --> Calls{"Tool calls"}
    Calls -->|"yes"| Tools["Run tools and update state"]
    Tools --> Request
    Calls -->|"no"| Finish["Return response"]
```

Caption: Construction resolves policy before compilation; the compiled LangChain/LangGraph agent then runs the model-and-tool loop.

## Builder inputs and compilation boundary

The builder accepts a model instance or model string. It retains a string specification while selecting the harness profile, resolves the actual model, rewrites supplied tool descriptions according to that profile, and creates a `StateBackend()` when no backend is supplied. `model=None` remains a deprecated compatibility path that warns and constructs `ChatAnthropic(model_name="claude-sonnet-4-6")`.

The profile contributes prompt material, tool-description overrides, extra middleware, excluded tools, the general-purpose-subagent policy, and middleware exclusions. The authored `system_prompt` is combined with profile prompt material: strings receive a separated suffix, while a supplied `SystemMessage` retains its content blocks and receives a text block. This preserves message metadata rather than flattening the caller's message.

At the boundary, the builder passes the resolved model, resulting prompt, rewritten tools, middleware, response format, schemas, checkpointer, store, debug flag, name, and cache to `create_agent()`. It selects `DeepAgentState` unless `state_schema` is supplied, then binds `recursion_limit` `9999` and LangSmith Deep Agents/version/name metadata with `.with_config()`.

`DeepAgentState.messages` is a `DeltaChannel` using `_messages_delta_reducer` and a 50-update snapshot frequency. The reducer accepts raw message-like values, treats absent replay state as empty, replaces or removes messages by ID, and honors `REMOVE_ALL_MESSAGES`. This keeps message checkpoint growth linear rather than repeatedly checkpointing an ever-growing list.

## Subagent classification and ownership

Subagent classification happens before the main stack is built:

- A mapping with `graph_id` is an `AsyncSubAgent`. It is not supplied to the synchronous `task` tool; it becomes part of `AsyncSubAgentMiddleware`.
- A mapping with `runnable` is a `CompiledSubAgent`, whose graph is retained as supplied.
- All other mappings are declarative `SubAgent` specifications. The builder resolves their model and profile and prepares their filesystem, summarization, patching, profile, skill, caching, and applicable memory middleware.

Unless the active profile disables it or an inline synchronous specification already has that name, the builder prepends the `general-purpose` declarative subagent. With no inline synchronous subagents, it omits `SubAgentMiddleware` and hence the `task` tool. Async subagents remain independently available through their own tools.

Declarative children inherit parent permissions and `interrupt_on` unless they supply those fields; a supplied permissions list replaces the parent list. They inherit parent tools only when their spec does not provide `tools`. `SubAgentMiddleware` compiles declarative specifications behind `task`; `create_sub_agent` requires a model and tools, adds `HumanInTheLoopMiddleware` when the resolved child has approvals, and ensures `UnsupportedContentMiddleware` is present. A compiled or remote graph owns its own runtime schema and approval behavior.

A `mode="fork"` declarative child receives the effective parent history after its summarization event plus a task message. Its private fork marker makes recursive `task` calls fail. Isolated children instead receive a new human message containing just the delegated description. On return, the task handler removes message, standard excluded, unchanged, and private state keys from child updates and emits a `ToolMessage` using structured output or the last nonempty AI text. The main builder collects private fields from the graph and installed middleware schemas and adds them to every installed `SubAgentMiddleware` boundary.

### Remote asynchronous subagents

`AsyncSubAgentMiddleware` targets Agent Protocol graphs via the LangGraph SDK. It provides `start_async_task`, `check_async_task`, `update_async_task`, `cancel_async_task`, and `list_async_tasks`; launch creates the remote thread/run and persists task tracking in `async_tasks`, allowing the parent to continue rather than await a report.

Async `mode="fork"` snapshots effective compacted history, clears nonportable tool artifacts, hydrates offloaded tool content, and appends the task before serializing it to the remote graph. Isolated mode sends only the task description. Forking shares history at launch, not the parent model, prompt, tools, middleware, or approvals. Omitting `url` selects local ASGI transport, which needs an async parent entry point such as `ainvoke`; synchronous invocation needs a reachable URL.

## Middleware assembly and exclusion guarantees

The main core is constructed in this order:

1. `FilesystemMiddleware`;
2. `SubAgentMiddleware` only when inline synchronous subagents exist;
3. summarization middleware;
4. `PatchToolCallsMiddleware`;
5. `AsyncSubAgentMiddleware` only when remote subagents exist.

The builder then adds profile middleware, optional `SkillsMiddleware`, prompt caching, optional `MemoryMiddleware`, optional human-in-the-loop middleware, and `UnsupportedContentMiddleware`. A caller middleware item with the same `.name` replaces its surviving stack slot in place. A new name is spliced immediately after the core, ahead of that tail. Profile exclusions run before and after this splice; profile tool exclusion middleware is appended last, so a custom model-call wrapper cannot restore a forbidden tool.

`FilesystemMiddleware` and `SubAgentMiddleware` are required scaffolding: filesystem middleware supplies built-in file tools and enforces permissions, while subagent middleware owns the synchronous task handler. A profile cannot exclude either by class or name. Exclusion class matching is exact concrete type and string matching is exact middleware name. Construction raises `ValueError` for protected or private-name exclusions, same-name collisions among distinct middleware classes, and exclusions that match no middleware across the main and general-purpose stacks.

Skills are sourced through the configured backend. When `skills` is omitted, a caller-provided middleware named `SkillsMiddleware` contributes its sources to the default slot rather than creating a second layer. Skill metadata is cached in graph state per thread; later same-name sources override earlier ones, setting `skills_metadata` to `None` requests reload, and source-loading errors are retained as warnings. At request time, skill tools can be disclosed only after the skill is read: provider payload tests verify that the disclosure is placed after the relevant tool-result batch, is not moved ahead of a queued user turn, and is generated for the model actually called after fallback.

## Filesystem-derived human approval

Filesystem permissions and approval are deliberately separate responsibilities. `FilesystemMiddleware` enforces `deny` rules and filters denied results. The builder derives `HumanInTheLoopMiddleware.interrupt_on` entries only from `interrupt` permission rules, then merges them with caller `interrupt_on`; a caller entry for the same tool wins. A nonempty merged mapping installs the human-in-the-loop middleware for the main graph and for constructed declarative children.

The generated configuration covers `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, and `grep` when an interrupt rule applies to that operation. It has path-sensitive `when` predicates rather than approving all calls to a registered tool:

- Exact-path operations interrupt only for a matching interrupt rule. A matching `deny` decision suppresses the interrupt because the filesystem tool will reject the operation.
- Bulk operations interrupt when the search subtree overlaps an interrupt-rule anchor. Missing bulk paths are treated conservatively as potentially touching any protected path, and `.` is normalized to the accessible root.
- `glob` also checks an absolute pattern or a relative pattern containing `..`, preventing the pattern from bypassing the supplied search path. When a backend can resolve symlinks, both the normalized path and resolved target participate in the decision.

Generated approval configurations allow `approve`, `edit`, `reject`, and `respond`; approval does not replace filesystem enforcement. In particular, an edited tool call re-enters the filesystem permission check, and a response decision skips execution. Configure a checkpointer when the application must persist and resume approval interrupts.

## Focused verification

`test_permissions.py` covers the security boundary: recursive delete is all-or-nothing when a denied pattern overlaps the target subtree; interrupt rules are not denials; generated mappings include only relevant filesystem operations; pathless, current-directory, redirecting-glob, and symlink cases are gated conservatively. `test_skill_tools_payload.py` stubs real Anthropic and OpenAI HTTP transports to assert the actual placement and persistence of inline skill-tool disclosures, including compaction and fallback behavior.

The graph and subagent test suites cover model/profile selection, stack placement and exclusions, general-purpose-subagent construction, task handoff, state/private-key isolation, fork behavior, and schema ownership.

## Related pages

- [Middleware stack](middleware-stack.md) — hook responsibilities and ordering.
- [Runtime behavior](runtime-behavior.md) — runtime execution and operational behavior.
- [Permissions and HITL](../concepts/permissions-hitl.md) — permission policy and approval use.
- [Subagents and skills](../concepts/subagents-skills.md) — delegation and skill concepts.
- [Build a Deep Agent](../workflows/build-a-deep-agent.md) — application-level construction workflow.
