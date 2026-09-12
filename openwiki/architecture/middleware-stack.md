---
type: architecture pattern
title: Middleware Stack and Customization Boundaries
description: How create_deep_agent assembles and filters ordered middleware stacks for the main agent and subagents. Covers profile exclusions, caller replacement and insertion, state boundaries, request repair, and context compaction.
tags: [middleware, deepagents, agent-construction, harness-profile, subagents, context-management]
sources:
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-b93533cac55718d75277d1cf
    resource: repo://libs/deepagents/deepagents/_excluded_middleware.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fc54598423086acf9d53d9fd
    resource: repo://libs/deepagents/deepagents/middleware/__init__.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-13b8cea81b8a29f0950cc836
    resource: repo://libs/deepagents/deepagents/middleware/patch_tool_calls.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
  - id: openwiki-source-454da083c2cc29febd156c7e
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_subagent_middleware_init.py
  - id: openwiki-source-3dd6a4926b92ae499082a552
    resource: repo://libs/deepagents/tests/unit_tests/test_middleware.py
verified:
  - by: openwiki/0.4.2
    at: 2026-09-12T08:04:33.168Z
generated: { by: "openwiki/0.4.2", at: "2026-09-12T08:04:33.168Z" }
---

# Middleware Stack and Customization Boundaries

`create_deep_agent()` is a harness assembler, not a separate agent runtime. It resolves the model and applicable `HarnessProfile`, constructs ordered `AgentMiddleware`, and passes the final main stack to LangChain's `create_agent()`, which owns the model/tool loop. The passed-through graph options include the system prompt, tools, response format, schemas, checkpointing, store, debugging, name, and cache. See [SDK construction and execution](/openwiki/architecture/sdk-construction-execution.md) for the runtime boundary.

Middleware is the request-time extension boundary. A `wrap_model_call()` hook intercepts every LLM request before it is sent, so it can change the effective system prompt, history, tool list, or typed cross-turn state. In contrast, a callable in `tools=` runs only if the model selects it; it cannot prepare a model request. Caller tools are additive to built-ins. Use middleware for per-request behavior, prompt/tool injection, or state; use an ordinary tool for a self-contained operation. See [middleware catalog](/openwiki/concepts/middleware-catalog.md).

## Main-agent assembly

Membership is conditional on inputs and the resolved profile. `skills`, subagent forms, memory, filesystem permissions, interrupt configuration, profile extras, installed provider integrations, and profile exclusions all affect the result.

```mermaid
flowchart TD
    Resolve["Resolve model and harness profile"] --> Core["Build conditional core stack"]
    Core --> Tail["Add profile extras cache memory and approval"]
    Tail --> FirstFilter["Apply profile middleware exclusions"]
    FirstFilter --> Merge["Merge caller middleware at core boundary"]
    Merge --> SecondFilter["Apply exclusions again"]
    SecondFilter --> ToolFilter["Append tool exclusion filter if configured"]
    ToolFilter --> StateKeys["Derive private state keys"]
    StateKeys --> Agent["Pass stack to create_agent"]
```

Diagram: verified main-stack assembly and the profile/caller customization points.

### Stack order

The **core band** is assembled in this order:

1. `SkillsMiddleware`, when `skills` is supplied.
2. `FilesystemMiddleware`.
3. `SubAgentMiddleware`, when synchronous inline subagents exist—normally because the general-purpose subagent is added.
4. Deep Agents summarization middleware.
5. `PatchToolCallsMiddleware`.
6. `AsyncSubAgentMiddleware`, when remote async specs exist.

`PatchToolCallsMiddleware` is history repair, rather than a request wrapper. At `before_agent`, it finds `AIMessage` tool calls (including invalid calls) that have no matching `ToolMessage` ID, then replaces the message list with the original history plus synthetic error results immediately after the calls. It leaves a complete history unchanged. This permits recovery after cancellation, interruption, malformed arguments, or other missing results instead of replaying dangling calls.

The **tail band** appends materialized `HarnessProfile.extra_middleware`, provider prompt-caching middleware, `MemoryMiddleware` when configured, and `HumanInTheLoopMiddleware` when the resolved interrupt mapping is non-empty. Cache middleware is before memory deliberately: profile extras run before caching, and memory's system-prompt mutations occur after the Anthropic cache prefix. Anthropic caching is always installed with unsupported models ignored. Bedrock and Fireworks variants are added only when their integration packages can be imported, and also ignore unsupported models.

The first profile-exclusion pass runs after the tail is assembled. Caller `middleware=` is then merged, exclusions run again, and `_ToolExclusionMiddleware` is appended only when the profile has `excluded_tools`. Being last matters: it sees the near-final request after tool-producing middleware and caller model wrappers, so excluded names cannot be restored by a caller wrapper.

The assembler also combines an explicit `state_schema` with middleware-contributed schemas, derives private state-field names, and assigns them to `SubAgentMiddleware`. This determines what ordinary synchronous delegation may carry across its state boundary.

### Context-management role

The default summarization component reconstructs the effective history from a prior summary event, then may truncate large arguments on older tool calls before deciding whether compaction or a budget recovery is needed. It makes one normal request when thresholds do not require summarization; a recognized provider context-overflow error falls back to compaction. A post-compaction input-budget check can reduce large trailing tool results for one smaller retry; irreducible input raises `ContextOverflowError`.

When it compacts, it offloads the to-be-summarized history to the backend (after replacing inline media with path references), creates a summary, and sends the summary plus preserved recent messages. The returned state update stores `_summarization_event`—cutoff, summary message, and offload path—and `_summarization_session_id`; the raw state history is retained rather than deleted. Sync processing offloads before generating the summary, whereas async processing offloads and summarizes concurrently. An offload failure is logged and warned: the summary still proceeds but older messages cannot be recovered. See [context management](/openwiki/concepts/context-management.md).

## Caller middleware and profile exclusions

Caller middleware is merged by `.name`, rather than blindly appended:

- A caller entry whose name remains in the base stack replaces that slot in place, retaining its position.
- A new name is inserted after the last surviving core member, ahead of profile extras, prompt caching, memory, and approval middleware.
- The first exclusion pass occurs before merging; the second removes an attempt to reintroduce an excluded name or exact class.

This gives callers a controlled way to replace a built-in behavior. For example, a caller can supply a `FilesystemMiddleware` with the same name and a narrower `tools=[...]` set to remove a filesystem tool entirely; `tools=` alone cannot do that. Custom middleware that uses a new name is intentionally not copied into the general-purpose subagent merely because it was installed on the main agent.

A `HarnessProfile` can subtract entries with `excluded_middleware`, subject to safety and coverage checks:

- `FilesystemMiddleware` and `SubAgentMiddleware` are protected scaffolding. Excluding either by class or name raises `ValueError`, rather than silently breaking built-in filesystem/permission behavior or the synchronous `task` handler.
- Class entries use exact `type`, not `isinstance`; excluding a base class does not remove a caller subclass. String entries match `AgentMiddleware.name` exactly, permitting a public alias such as `SummarizationMiddleware` to target an implementation class with a different `__name__`.
- A string exclusion that matches multiple distinct classes in one stack is ambiguous and raises `ValueError`. Every permitted entry must match somewhere; unmatched entries raise after assembly, catching typos and stale profiles.

For a main profile, match sets accumulate across the main-agent and auto-added general-purpose stacks, then coverage is checked once both have been filtered. Thus an exclusion may legitimately apply to only one of those stacks. A declarative subagent that resolves another profile performs its own validation, filtering, and coverage check.

### Tool visibility is not authorization

`excluded_tools` adds final `_ToolExclusionMiddleware`, which removes named tools from model requests and returns an error instead of executing an excluded name at the tool-call boundary. This maintains consistency between advertised and executable tools, but is explicitly not a security surface.

Filesystem permissions are enforced by `FilesystemMiddleware` at calls to its built-in tools, not by the backend. Direct backend use therefore bypasses these middleware-level permission rules. Permission rules and approval behavior are covered in [permissions and human-in-the-loop](/openwiki/concepts/permissions-hitl.md).

## Separate subagent paths

Subagent form is determined during assembly: a spec with `graph_id` becomes an `AsyncSubAgent` handled by `AsyncSubAgentMiddleware`; one with `runnable` is a `CompiledSubAgent`; otherwise it is a declarative `SubAgent`. These forms have different construction and inheritance boundaries.

### Declarative subagents

Every declarative spec resolves its own model and harness profile and builds an independent stack. A fork first adds parent `SkillsMiddleware` when top-level `skills` is configured; then the stack adds `FilesystemMiddleware`, summarization, and `PatchToolCallsMiddleware`. An isolated spec can instead add its own skills after those core entries. Profile extras and prompt caching follow, then two exclusion passes around spec middleware, coverage validation, and the final tool filter. A fork also mirrors top-level memory when configured.

A spec inherits top-level tools, permissions, and `interrupt_on` only when it omits each field. Its own permissions replace, rather than extend, parent rules. After those values are resolved, a non-empty interrupt mapping adds `HumanInTheLoopMiddleware` while compiling the declarative graph. The parent `state_schema` is supplied to this compilation; supplied compiled runnables and remote graphs own their own schemas and approval configuration.

The default mode is `isolated`: the child receives a `HumanMessage` containing the delegated task rather than the parent's conversation. `handoff` is accepted as a legacy alias for isolated behavior. Experimental `fork` instead receives the parent's effective compacted history plus a task preamble, and rebuilds the parent prompt with an optional child addendum. A declarative fork cannot specify independent skills; it retains eligible private state channels, while a forked compiled runnable has private keys stripped because its schema is opaque. A forked child is refused if it calls `task`, preventing recursive delegation. See [subagents and skills](/openwiki/concepts/subagents-skills.md).

### General-purpose, compiled, and async agents

Unless the active profile disables it or an inline synchronous spec already uses its name, the harness adds `general-purpose`. Its own stack contains filesystem, summarization, patching, optional skills, profile extras, caching, exclusion passes, and a final tool filter. It inherits only caller middleware that overrides one of its original default slots—not arbitrary main-agent-only middleware.

A `CompiledSubAgent` runnable is used as supplied. It does not inherit the parent state schema or top-level approval rules, and must return a state with `messages`. The parent returns a `ToolMessage` containing a JSON-serialized structured response when present, otherwise the last non-empty AI text, and merges eligible non-private state updates.

An `AsyncSubAgent` runs through the LangGraph SDK against an Agent Protocol server as a background task. `AsyncSubAgentMiddleware` returns and tracks task IDs in middleware state instead of blocking the parent task call; graph schema and approval behavior belong to the remote graph. An async spec with no `url` uses local ASGI transport and consequently requires an async parent entrypoint such as `ainvoke`; synchronous `invoke` needs a reachable URL.

## Safe changes and focused tests

Ordering changes alter what the model sees and what tools can execute. Test assembled stacks, not only middleware constructors. Focus tests on replacement versus insertion, both exclusion passes, final request/tool-call filtering, protected-scaffolding and ambiguous-name failures, and coverage across main and general-purpose stacks. Also test the separate declarative, compiled, async, isolated, and fork paths—especially private-state treatment, fork prompt/history construction, recursive-delegation refusal, and the structured-response fallback.

For request repair, test no-op behavior for complete history, synthetic errors for both ordinary and invalid missing tool calls, their placement after the originating `AIMessage`, and replacement of the full messages channel. For compaction, cover threshold and overflow paths, one-retry exhaustion, summary-event reconstruction, offload/media failure warnings, and sync versus async ordering. The unit suite exercises the subagent `task` tool and the patcher; graph tests cover profile exclusion behavior and subagent stack assembly.
