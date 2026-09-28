---
type: capability reference
title: Middleware Catalog
description: Catalog of Deep Agents middleware by lifecycle hook, state and prompt effects, public exports, ordering constraints, and the boundary between middleware capabilities and caller-provided tools.
tags: [middleware, deepagents, filesystem, context-management, memory, skills, subagents, permissions]
sources:
  - id: openwiki-source-a1549ea98d425efea270be93
    resource: repo://libs/deepagents/deepagents/backends/composite.py
  - id: openwiki-source-c972622237a22631e36f3625
    resource: repo://libs/deepagents/deepagents/backends/utils.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fc54598423086acf9d53d9fd
    resource: repo://libs/deepagents/deepagents/middleware/__init__.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-9841bc6daf811e4615c54a88
    resource: repo://libs/deepagents/deepagents/middleware/_message_eviction.py
  - id: openwiki-source-64b92f60456305edc143f48a
    resource: repo://libs/deepagents/deepagents/middleware/_overflow_clip.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-454ab6b822ad87c53f679f58
    resource: repo://libs/deepagents/deepagents/middleware/_video.py
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
generated: { by: "openwiki/0.4.2", at: "2026-09-25T08:06:00.203Z" }
verified:
  - by: openwiki/0.4.2
    at: 2026-09-25T08:06:00.203Z
---

# Middleware Catalog

`deepagents.middleware` is the public import surface for the SDK middleware and its supporting types. Middleware subclasses `AgentMiddleware` and participates in lifecycle hooks: it can initialize state before a run, shape every model request, wrap a tool call, and decide what happens at a natural agent stop. A caller-provided callable in `tools=` instead runs only after the model selects it. Use a plain tool for isolated, consumer-specific work; use middleware when the behavior must alter prompts, the advertised tool set, messages, state, or loop control.

For stack placement, see [Middleware stack](../architecture/middleware-stack.md). The detailed feature contracts live in [Context management](context-management.md), [Subagents and skills](subagents-skills.md), and [Filesystem tools](tools-filesystem.md).

## Public catalog

| Capability | Public entrypoint | Hooks, state, and effect |
| --- | --- | --- |
| Filesystem and optional shell access | `FilesystemMiddleware`, `FilesystemPermission` | Supplies filesystem tools; shapes requests and can evict oversized results after a tool call. |
| Automatic compaction | `SummarizationMiddleware` | Reconstructs effective history, summarizes at its trigger or after a context-overflow response, and stores a private summary event. |
| Model-requested compaction | `SummarizationToolMiddleware`, `create_summarization_tool_middleware` | Adds `compact_conversation`; the tool layer does not itself run automatic compaction. |
| Persistent instructions | `MemoryMiddleware` | Loads `AGENTS.md` sources in `before_agent`; adds formatted memory to each model request. |
| Progressive-disclosure skills | `SkillsMiddleware`, `SkillMetadata`, `SkillsState` | Discovers metadata in `before_agent`; prompts the model to read the complete skill only when needed. |
| Blocking delegation | `SubAgentMiddleware`, `SubAgent`, `CompiledSubAgent` | Adds `task`; the parent waits for the child result. |
| Background delegation | `AsyncSubAgentMiddleware`, `AsyncSubAgent` | Adds launch, monitoring, update, cancellation, and listing tools for remote Agent Protocol work. |
| Definition-of-done review | `RubricMiddleware` and rubric result types | Grades at a natural stop and can jump back to the model with revision feedback. |
| Model-compatible multimodal requests | `UnsupportedContentMiddleware` | Replaces unsupported human or tool content blocks in the request while retaining originals in thread state. |

`PatchToolCallsMiddleware` is installed by graph construction but intentionally is not in `deepagents.middleware.__all__`. The underscore-prefixed modules are assembly and implementation helpers, rather than the normal consumer import API.

## Lifecycle: why middleware is not a plain tool

```mermaid
flowchart TD
    Begin["Run begins"] --> Before["before_agent loaders and history repair"]
    Before --> Request["Model wrappers shape messages prompts and tools"]
    Request --> Model["Model call"]
    Model --> Calls{"Tool calls"}
    Calls -->|"yes"| Tool["Tool wrappers execute or transform result"]
    Tool --> Request
    Calls -->|"no"| Review{"Rubric enabled"}
    Review -->|"needs revision"| Feedback["Add feedback and jump to model"]
    Feedback --> Request
    Review -->|"terminal"| Done["Run ends"]
```

This depicts the lifecycle boundary: loaders initialize typed state once per run; `wrap_model_call` and `awrap_model_call` affect every request; `wrap_tool_call` can mediate execution or results; and `after_agent` can continue an otherwise finished loop. A plain callable is available only at the tool-execution step, so cannot reliably provide the preceding request-wide effects.

State that must not enter a subagent is annotated with `PrivateStateAttr`. During graph construction, `private_state_field_names` resolves those annotations and configures subagent middleware to strip them. An unresolvable schema is warned about and skipped, so its intended private fields may cross the boundary; annotations must therefore be resolvable at runtime.

## Filesystem, permissions, and large results

`FilesystemMiddleware` builds an allowlisted set from `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`. An explicit allowlist must include `read_file`; omitted names do not even reach the dispatchable tool node. `execute` and `delete` are still subject to backend capability checks. The default backend is `StateBackend()`, while backend factories, nonpositive execution timeouts, and invalid `grep_max_count` values are rejected.

Filesystem denial and approval are deliberately separate. The middleware's tool implementations enforce matching `deny` permissions. During graph assembly, `_fs_interrupt` translates `interrupt` rules into `HumanInTheLoopMiddleware` mappings with path-aware predicates; approval does not grant authorization. With an execution-capable backend, unscoped permissions are rejected because execute-level permissions are not implemented.

### Storage and eviction

A `CompositeBackend` owns `artifacts_root`, defaulting to `/`. Filesystem and summarization derive `/large_tool_results` and `/conversation_history` beneath that root. A generic successful offload writes complete text under a sanitized, bounded tool-call identifier and replaces the tool result with a line-numbered head-and-tail preview while retaining non-text blocks. Failed writes leave the original message in place. The storage component replaces dots and path separators; components longer than 128 UTF-8 bytes are replaced with a SHA-256-derived name, while notices abbreviate IDs longer than 32 characters.

Filesystem performs proactive eviction only after a tool returns, and excludes `ls`, `glob`, `grep`, `read_file`, `edit_file`, `write_file`, and `delete`. Summarization uses tail clipping only as input-budget or provider-overflow recovery. For a qualifying `read_file` result whose source call has a nonempty `file_path`, it keeps a head slice and points back to that file; other trailing results are offloaded and stubbed. Recovery allows only one strictly smaller retry before raising `ContextOverflowError` with guidance to reduce request size.

The optional video boundary is also part of filesystem reads. `_video` imports PyAV lazily, so installations without the `[video]` extra remain lightweight. For a video read, `offset` and `limit` mean seconds and the result interleaves timestamp text with sampled image frames.

## Context and instruction middleware

`SummarizationMiddleware` retains raw messages and tracks compaction with a private summary event. It persists older history before creating a summary and request suffix; a persistence failure warns but does not prevent summarization. It also handles recognized context-overflow errors by summarizing and attempting constrained tail recovery. `create_summarization_middleware` creates the automatic layer with model-aware defaults; it is used internally by `create_deep_agent`. `create_summarization_tool_middleware` creates only the manual `compact_conversation` layer, resolving a model string if needed and refusing compaction before approximately half the automatic trigger.

`MemoryMiddleware` loads its configured sources into private `memory_contents` once, ignores missing files, removes HTML comments when rendering, and injects its prompt fragment on every request by default. `system_prompt=None` suppresses prompt injection but not loading. When graph-created memory runs with an Anthropic request model, it can mark the final system block as an ephemeral cache breakpoint.

`SkillsMiddleware` implements progressive disclosure entirely through backend APIs. It loads skill metadata once per thread; a checkpointed list, including an empty list, prevents rediscovery, while `skills_metadata=None` requests reload. Sources are processed in order and the last duplicate skill name wins. The system prompt lists locations and metadata and tells the model to use `read_file` for full instructions; load failures are logged and rendered as explicitly untrusted diagnostics.

Provider caching is assembled before optional memory: `append_prompt_caching_middleware` always adds Anthropic caching and adds Bedrock or Fireworks middleware only when the corresponding package is installed. This ordering lets memory add its Anthropic cache breakpoint.

`UnsupportedContentMiddleware` is a request-only compatibility guard. It consults the active request model's profile and replaces rejected image, audio, video, PDF, or tool-message block types with a text notice; it preserves the original thread message so a later compatible model can receive the original content. Put it last when assembling a middleware list manually, because it must inspect the final selected model. `create_deep_agent` adds it automatically.

## Delegation and quality control

`SubAgentMiddleware` requires at least one named subagent, advertises available agents in its optional prompt fragment, and exposes one synchronous `task` tool. A structured child response is JSON-serialized; otherwise the parent receives its final nonempty AI text. The child call blocks from the parent perspective, although independent task calls can execute concurrently. Private state is removed at this boundary.

`AsyncSubAgentMiddleware` is a distinct remote contract, not a nonblocking form of `task`. It starts Agent Protocol runs without waiting, persists task records in `async_tasks`, and provides tools to start, check, update, cancel, and list work. It rejects an empty or duplicate-named configuration.

`RubricMiddleware` is a no-op until invocation state contains a rubric. At a natural stop it invokes a separate grader agent on a bounded, sanitized transcript. A `needs_revision` result adds a `HumanMessage` and jumps back to the model; grading repeats until `satisfied`, `failed`, a grader error, or the iteration cap. The grader can emit `satisfied`, `needs_revision`, and `failed`; `max_iterations_reached` and `grader_error` are middleware-generated terminal statuses. Consumers that need to react to a non-satisfied result must inspect rubric state, its callback, or its stream event rather than the final AI message.

`PatchToolCallsMiddleware` repairs resumed history in `before_agent`. Any valid or invalid AI tool call with an ID but no matching `ToolMessage` receives an appended error result, differentiating incomplete calls from malformed or truncated arguments, and then the middleware rewrites history.

## Assembly and ordering invariants

`create_deep_agent` builds its core stack from configured skills, filesystem, inline synchronous subagents, automatic summarization, and tool-call repair; it then adds configured asynchronous subagents. Profile middleware, provider caching, optional memory, and optional HITL follow. `UnsupportedContentMiddleware` is added after HITL; profile and caller middleware can then be applied according to their configured placement. Finally, `_ToolExclusionMiddleware` is appended so excluded tool names are removed after every tool-injecting request wrapper and rejected at the execution boundary. This is consistency behavior, not a security boundary.

When changing middleware, preserve sync/async hook parity and test the focused filesystem, summarization, skills, memory, subagent, and rubric middleware tests. Stack order is behavior: moving a prompt wrapper, a model selector, or tool exclusion can change the effective request even when the individual middleware remains correct.
