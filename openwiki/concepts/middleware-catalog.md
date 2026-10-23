---
type: capability reference
title: Middleware Catalog and Extension Rules
description: Classifies Deep Agents middleware by lifecycle responsibility, graph ordering, protected invariants, and focused tests for safe changes.
tags: [middleware, deepagents, filesystem, summarization, compaction, async-tasks, skills, subagents, permissions]
verified:
  - by: openwiki/0.4.2
    at: 2026-10-23T08:06:06.503Z
sources:
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fc54598423086acf9d53d9fd
    resource: repo://libs/deepagents/deepagents/middleware/__init__.py
  - id: openwiki-source-9841bc6daf811e4615c54a88
    resource: repo://libs/deepagents/deepagents/middleware/_message_eviction.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
  - id: openwiki-source-46a23efe78a78f9b3cd75d00
    resource: repo://libs/deepagents/deepagents/middleware/memory.py
  - id: openwiki-source-13b8cea81b8a29f0950cc836
    resource: repo://libs/deepagents/deepagents/middleware/patch_tool_calls.py
  - id: openwiki-source-b93c32bc33a8fa17b52b8a0e
    resource: repo://libs/deepagents/deepagents/middleware/rubric.py
  - id: openwiki-source-66cf9d0832d3cb55bec2b5ed
    resource: repo://libs/deepagents/deepagents/middleware/skills.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
  - id: openwiki-source-837c84a3f3120bc778033547
    resource: repo://libs/deepagents/deepagents/middleware/unsupported_content.py
  - id: openwiki-source-8698160542ebcd20ba6341ac
    resource: repo://libs/deepagents/deepagents/profiles/harness/_openai_codex.py
  - id: openwiki-source-69c260914d2dd18b60ad5900
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py
  - id: openwiki-source-530f1194888c9c15f9c71332
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_summarization_factory.py
  - id: openwiki-source-6ce85b02eabe462f99e0c912
    resource: repo://libs/deepagents/tests/unit_tests/test_async_subagents.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
generated: { by: "openwiki/0.4.2", at: "2026-10-23T08:06:06.503Z" }
---

# Middleware Catalog and Extension Rules

`deepagents.middleware` is the public import surface for middleware that prepares a model request, intercepts tool execution, or owns cross-turn state. It re-exports filesystem permissions, memory, skills and `SkillToolResolver`, summarization, synchronous and asynchronous subagents, rubric evaluation, and unsupported-content filtering. Use those public classes and factories as extension points. Graph-only helpers—prompt caching, parent notification, tool-call repair, and tool exclusion—are assembly details rather than consumer APIs.

Middleware differs from a plain `tools=[]` callable because it can run at agent, model, and tool hooks, shape the request on every model call, and persist typed state. Use a plain tool for a self-contained consumer operation; use middleware when the capability needs prompt/tool shaping, policy enforcement, or lifecycle state.

| Responsibility | Public entrypoint | Lifecycle / boundary |
| --- | --- | --- |
| Backend files and execution | `FilesystemMiddleware` | Supplies filesystem tools through `BackendProtocol`; adds execution only for `SandboxBackendProtocol`; evicts oversized results to backend storage. |
| File access policy and approval | `FilesystemPermission` | Matches read/write path rules for allow, deny, or interruption. This—not profile visibility—is the authorization boundary. |
| Durable instructions | `MemoryMiddleware` | Loads configured sources into private state before the agent and appends formatted memory on model requests. |
| Progressive workflow discovery | `SkillsMiddleware`, `SkillToolResolver` | Loads skill metadata, injects discovery context, and discloses declared tools only when their skill is read or pinned. |
| Context management | `SummarizationMiddleware`, `SummarizationToolMiddleware` | Projects effective history and automatically compacts, or exposes explicit `compact_conversation`. |
| Blocking delegation | `SubAgentMiddleware` | The `task` tool invokes a local declarative or compiled worker and returns a parent tool result. |
| Background delegation | `AsyncSubAgentMiddleware` | Launches, polls, updates, cancels, and lists remote Agent Protocol work while the parent continues. |
| Completion review | `RubricMiddleware` | Optionally grades a natural completion and can return control to the model for revision. |
| Model compatibility | `UnsupportedContentMiddleware` | Makes an outbound copy containing replacements for blocks unsupported by the resolved model. |

```mermaid
flowchart TD
    Begin["Agent invocation"] --> AgentHook["Agent hooks load or update state"]
    AgentHook --> Request["Model wrappers shape messages prompts and tools"]
    Request --> Model["Model call"]
    Model --> Decision{"Tool calls"}
    Decision -->|"yes"| Tool["Tool wrappers validate and execute"]
    Tool --> Request
    Decision -->|"no"| Complete["After-model middleware may review or resume"]
```

This is the relevant middleware lifecycle: model wrappers apply on every turn, while tool wrappers are the last interception point before dispatch.

## Default assembly is a policy order

`create_deep_agent()` builds a core stack of filesystem, optional synchronous subagents, parent notification, automatic summarization, tool-call repair, and optional asynchronous subagents. Profile middleware, optional skills, provider caching, optional memory, HITL, and unsupported-content filtering form the tail. Caller middleware replaces a same-named entry in place or otherwise lands after the core; profile-controlled tool exclusion is appended last.

```mermaid
flowchart LR
    Fs["Filesystem"] --> Sync["Sync subagents when present"]
    Sync --> Notify["Parent notifier"]
    Notify --> Summary["Automatic summarization"]
    Summary --> Repair["Tool-call repair"]
    Repair --> Async["Async subagents when configured"]
    Async --> Custom["New caller middleware"]
    Custom --> Tail["Profile skills caching memory HITL content adaptation"]
    Tail --> Exclude["Tool exclusion last"]
```

Optional layers are omitted. Same-name replacement preserves the existing slot, whereas novel middleware is inserted ahead of the profile/caching/memory tail. The general-purpose subagent has its own filesystem/summarization/repair/profile/skills/caching stack; only overrides of its original middleware slots are inherited. `TodoListMiddleware` is not in either default stack, but may be enabled explicitly or injected through a harness profile's `extra_middleware`.

`_ToolExclusionMiddleware` is deliberately last: it removes excluded tools from model requests and returns an error for a matching call at the tool boundary. It keeps advertised and dispatchable tools consistent; it is explicitly not a security surface. Protect filesystem operations with permissions and HITL instead.

## Filesystem and content invariants

`FilesystemMiddleware` provides filesystem tools through `BackendProtocol`, conditionally adds execution for `SandboxBackendProtocol`, and can evict oversized tool results to backend storage so model context is not saturated. The shared internal message-eviction helper offloads a large tool result only after the backend write succeeds, replaces text with a line-numbered head-and-tail preview while preserving non-text blocks, and otherwise leaves the original message intact. This helper is also used by summarization overflow recovery.

`FilesystemPermission` supports allow, deny, and interrupt effects for read/write path rules. Permission paths must be absolute and cannot contain `..` or `~`; on real filesystem backends, checks cover both the requested path and its symlink-resolved target. Do not treat profile exclusion as an authorization substitute.

## Memory, skills, and caching

`MemoryMiddleware` loads configured sources only when private `memory_contents` is absent. Missing files are skipped, other download errors raise, and formatted memory is injected into each model request unless its system prompt is disabled.

`SkillsMiddleware` caches source metadata in state per thread, with later duplicate names overriding earlier sources. Set `skills_metadata` to `None` to request reload. Pinned skills append their readable instructions before the next model call. A skill's declared tools are disclosed only after its skill is read or pinned, with disclosure recorded in state; the middleware resolves eligible tools again at the tool boundary and leaves undisclosed calls invalid. A resolver should therefore be stable for the thread and support the invocation mode being used.

Prompt-caching assembly always appends Anthropic caching with unsupported-model behavior ignored, and conditionally appends Bedrock and Fireworks caching when their optional middleware imports are available.

## Context compaction: effective history, not log mutation

The summarization factory chooses fraction-based trigger, retention, and argument-truncation defaults when the resolved model profile supplies `max_input_tokens`, and falls back to fixed token and message thresholds otherwise. It rejects a string rather than a resolved `BaseChatModel`.

Automatic summarization retains raw message state and records private event and session state. A later request reconstructs effective history from the summary message and cutoff, while evicted messages are appended to a session-specific backend history file. It first projects raw messages through any prior event, may truncate older large tool-call arguments, and persists the new event and session ID while leaving raw messages intact. Archive failure does not block compaction, but produces an event with no recovery path.

```mermaid
flowchart TD
    Raw["Raw state messages"] --> Effective["Apply prior summary event"]
    Effective --> Budget{"Over trigger or input budget"}
    Budget -->|"no"| Send["Send model request"]
    Budget -->|"yes"| Prepare["Select older history"]
    Prepare --> Media["Offload inline media to backend references"]
    Media --> Archive["Append recoverable history"]
    Media --> Summarize["Generate summary"]
    Archive --> Event["Record cutoff summary and session"]
    Summarize --> Event
    Event --> Send
```

The default Deep Agents summary prompt explains that offloaded-media references must be preserved and may be read later; automatic compaction converts inline media to backend references before archive and summary generation.

`SummarizationToolMiddleware` registers `compact_conversation` but never compacts automatically. It shares automatic-engine event state, gates execution until roughly half the automatic trigger, and reports no-op or compaction failure through `ToolMessage` without updating the event. The manual-compaction factory resolves a model string, constructs a model-aware engine, and registers only `SummarizationToolMiddleware`; automatic compaction requires separately registering `SummarizationMiddleware`—which `create_deep_agent()` already adds.

## Delegation and completion controls

The synchronous `task` tool runs a selected declarative or compiled subagent and returns one parent `ToolMessage`, serializing `structured_response` when present or otherwise selecting the last non-empty `AIMessage` text. Forked declarative subagents receive effective compacted history and refuse recursive `task` delegation.

An isolated async launch sends only its description. Experimental fork mode serializes the parent’s effective summarized conversation and task after removing artifacts and rehydrating offloaded media; the remote agent owns its prompt and tools. A URL-less local ASGI transport requires asynchronous parent invocation.

`AsyncSubAgentMiddleware` provides launch, check, update, cancel, and list tools for remote Agent Protocol work. It persists merge-reduced `async_tasks` records containing remote identifiers, status, timestamps, and queued-run lineage; construction rejects empty or duplicate specifications and unsupported modes or update strategies. Its asynchronous lifecycle hooks poll unfinished work and give the model another turn after a newly observed outcome; synchronous hooks do not poll.

```mermaid
stateDiagram-v2
    [*] --> running: start task
    running --> success: completed run
    running --> error: failed run
    running --> timeout: timed out run
    running --> waiting: child needs input
    running --> pending: queued update
    pending --> running: newest run starts
    waiting --> running: update or child resumes
    running --> cancelled: cancel
    pending --> cancelled: cancel
    waiting --> cancelled: cancel
    success --> [*]
    error --> [*]
    timeout --> [*]
    cancelled --> [*]
```

This shows the parent’s persisted view of remote work. `waiting` is nonterminal because the child awaits input on its own thread.

`RubricMiddleware` is inactive without a rubric in invocation state. Otherwise it initializes private iteration state, grades at natural completion, can jump back to the model for revision, and leaves the last model response unchanged for failed, grader-error, or max-iteration outcomes. `UnsupportedContentMiddleware` runs late against the resolved request model and copies only human and tool messages whose blocks need replacement, preserving the original thread for a later capable model.

## Protected invariants and safe extension

- Preserve stack order when adding a built-in. A model wrapper can only observe transformations inside it; tool-injecting middleware must precede final exclusion.
- Do not import or configure underscored helpers as public extensions. Extend through `create_deep_agent(..., middleware=...)`, public middleware classes, documented profile middleware, and public `SkillToolResolver`.
- Keep private state keys, reducers, and raw-versus-effective history semantics intact. They carry checkpointed lifecycle data and are especially important for summaries, skills, and async tasks.
- Treat tool exclusion as UX consistency only. Permission rules and `HumanInTheLoopMiddleware` enforce filesystem protection.
- Use async invocation when an async resolver, local ASGI async subagent, or async task polling is needed.

## Focused change coverage

Route changes to the narrow suite first, then graph integration tests when order or configuration changes are involved.

| Change | Focused tests |
| --- | --- |
| Stack construction, caller replacement, exclusions, profile-supplied todo middleware | `tests/unit_tests/test_graph.py`, `tests/unit_tests/middleware/test_tool_exclusion.py` |
| Filesystem initialization, execution routing, blobs/video and permissions | `tests/unit_tests/middleware/test_filesystem_middleware_init.py`, `test_execute_route_prompt.py`, `test_blob_offload.py`, `test_filesystem_video.py` |
| Memory loading and request injection | `tests/unit_tests/middleware/test_memory_middleware.py`, `test_memory_middleware_async.py` |
| Metadata/reload, pinned skills, declared-tool resolution | `tests/unit_tests/middleware/test_skills_middleware.py`, `test_skills_middleware_async.py`, `test_pinned_skills.py`, `test_skill_tools.py`, `test_skill_tool_resolver.py` |
| Factory defaults, automatic recovery, explicit compaction | `tests/unit_tests/middleware/test_summarization_factory.py`, `test_summarization_middleware.py`, `test_compaction_recovery.py`, `test_compact_tool.py` |
| Local subagent configuration and remote async lifecycle | `tests/unit_tests/middleware/test_subagent_middleware_init.py`, `tests/unit_tests/test_async_subagents.py` |
| Rubric loops and content compatibility | `tests/unit_tests/middleware/test_rubric_middleware.py` and the relevant filesystem/graph integration tests |
