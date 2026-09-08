---
type: capability reference
title: Middleware Capability Catalog
description: Catalog of the Deep Agents SDK middleware domains, their responsibilities, lifecycle boundaries, configuration constraints, and supported extension points. Use it to choose a capability while deferring stack ordering and request-flow detail to the architecture pages.
tags: [middleware, deepagents, catalog, filesystem, context-management, subagents, skills, permissions]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-07T08:06:36.835Z
sources:
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
  - id: openwiki-source-421bc4b065189ae1165ca326
    resource: repo://libs/deepagents/deepagents/middleware/_state.py
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
generated: { by: "openwiki/0.4.2", at: "2026-09-07T08:06:36.835Z" }
---

# Middleware Capability Catalog

`deepagents.middleware` is the public import surface for the SDK’s filesystem, context, delegation, and quality-control middleware. Middleware is the right integration boundary when a capability must alter every model request—for example by adding or filtering tools, extending the system message, rewriting messages, or retaining typed state across turns. A consumer-supplied callable in `tools=` is instead invoked only after the model chooses it.

This is a capability reference, not an assembly or request-flow specification. See [Middleware stack](../architecture/middleware-stack.md) for construction and ordering, [Context management](context-management.md) for compaction behavior, [Permissions and HITL](permissions-hitl.md) for approval policy, [Subagents and skills](subagents-skills.md) for delegation, and [Filesystem tools](tools-filesystem.md) for tool contracts.

## Public capabilities

| Domain | Public entrypoints | What it owns | Key configuration and boundary |
| --- | --- | --- | --- |
| Filesystem | `FilesystemMiddleware`, `FilesystemPermission` | Filesystem tool implementations, backend access, result shaping, and bounded context for large filesystem interactions. | Takes an initialized backend (default `StateBackend()`), an optional tool allowlist, custom descriptions/prompt, token limits, execution timeout, grep cap, and private `_permissions`. |
| Conversation context | `SummarizationMiddleware`, `SummarizationToolMiddleware`, `create_summarization_tool_middleware` | Automatic and on-demand conversation compaction and recoverable history offload. | The automatic layer uses threshold/keep settings; the tool layer exposes `compact_conversation` and composes with the automatic engine. |
| Persistent memory | `MemoryMiddleware` | Loading configured `AGENTS.md` sources into private state and, by default, adding the resulting reference context to requests. | Sources are ordered; setting `system_prompt=None` still loads state but suppresses prompt injection. |
| Skills | `SkillsMiddleware` | Skill discovery metadata and the prompt instructions that enable progressive disclosure. | Sources may be a path or `(path, label)`; later duplicate names override earlier ones. |
| Synchronous delegation | `SubAgentMiddleware`, `SubAgent`, `CompiledSubAgent` | The `task` tool and parent-to-child delegation conventions. | A task call waits for the child result; private state keys are stripped before invocation. |
| Asynchronous delegation | `AsyncSubAgentMiddleware`, `AsyncSubAgent` | Tools for launching and monitoring remote background Agent Protocol runs. | Requires at least one uniquely named async subagent; launch returns a task id rather than waiting. |
| Completion quality | `RubricMiddleware` and `Rubric*`/`Grader*`/`Criterion*` types | A separate structured-output grader and controlled revision loop when the main agent has no more tool calls. | The caller supplies a rubric and can bound revision attempts with `max_iterations`. |
| History repair | `PatchToolCallsMiddleware` | Valid message history at run start when a prior turn left calls unanswered. | Runs in `before_agent`; it replaces the complete messages collection only when repair is necessary. |

The package re-exports these public classes, supporting types, and the default summary/grader constants from `deepagents.middleware`; private underscore modules below are internal assembly helpers rather than a stable consumer import surface.

## Filesystem capability

`FilesystemMiddleware` builds an allowlisted set from `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`. `read_file` is mandatory whenever an explicit allowlist is supplied. `execute` is exposed only when the backend supports sandbox execution, and `delete` is subject to backend capability checks; naming either unsupported tool is therefore a no-op. This corrects the older description of the filesystem surface as an unconditional fixed tool set.

The middleware validates operational limits early: command timeouts and a configured grep cap must be positive; a backend factory is rejected in favor of an initialized `BackendProtocol` implementation. Its per-request hook applies its prompt, removes unsupported tools, normalizes media-result placement, replaces unsupported multimodal blocks with safe placeholders, and can evict an oversized latest `HumanMessage`. Tool-call handling separately offloads oversized text results to the backend and substitutes a head-and-tail preview with a retrieval path, preserving non-text blocks. The defaults are 20,000 tokens for tool results, 50,000 for human messages, a one-hour maximum command timeout, and a 1,000-match grep cap.

Filesystem deny rules are enforced by the tool implementations, but approval is deliberately outside this middleware. `_fs_interrupt` converts interrupt-mode `FilesystemPermission` rules into `HumanInTheLoopMiddleware` predicates during graph assembly. Do not assume execute is individually governed by those rules: unscoped filesystem permissions combined with an execution-capable backend are rejected because execute-level permissions are not implemented. See [Permissions and HITL](permissions-hitl.md) before extending policy.

## Context and prompt capabilities

`SummarizationMiddleware` automatically summarizes when its configured token trigger is crossed and offloads evicted history to a backend. The history is appended per invocation under `/conversation_history/{session_id}.md`; inline base64 media is stored separately and referenced from that markdown. `SummarizationToolMiddleware` offers the same engine as the `compact_conversation` tool, and the factory creates both layers with model-aware defaults.

The overflow fallback is intentionally recoverable. After `ContextOverflowError`, it clips a large trailing batch of `ToolMessage`s: a `read_file` result is head-sliced and points to the original path, whereas another tool result is written beneath `large_tool_results/{tool_call_id}` and replaced by a pointer preview. Shared message-eviction code implements the same head-and-tail preview strategy used by filesystem result offload.

`MemoryMiddleware` differs from skills in lifetime and intent: it concatenates available ordered `AGENTS.md` sources as persistent reference context rather than advertising workflows to load on demand. Missing sources are skipped, other download errors fail the load, and HTML comments are removed before formatting. Its private `memory_contents` state avoids returning file content in final agent state. For Anthropic requests, `add_cache_control=True` tags the final system block as an ephemeral cache breakpoint; it does nothing for other model wrapper types.

Prompt caching is assembled internally: `append_prompt_caching_middleware` always appends Anthropic caching configured to ignore unsupported models, and opportunistically adds Bedrock or Fireworks middleware only when their integration packages can be imported. This is a provider optimization, not a substitute for the context-management controls.

## Skills and delegation

A skill is a backend directory whose `SKILL.md` supplies YAML frontmatter and instructions. `SkillsMiddleware` uses only backend APIs, so it can work with state, filesystem, or remote storage backends. Before an agent run it populates private skill metadata once; checkpointed state prevents a repeat discovery. It records and logs discovery errors, and injects progressive-disclosure documentation into the system prompt on model calls. Since sources are processed in order into a name-keyed map, the last source wins for duplicate skill names.

`SubAgentMiddleware` injects a `task` tool and optional descriptions of available child types. Synchronous task delegation blocks for completion. Graph assembly calculates private state fields across schemas and gives those keys to the middleware so they are stripped from parent state before child invocation. `AsyncSubAgentMiddleware` is a distinct remote integration: it validates nonempty, uniquely named definitions, builds tools that initiate remote Agent Protocol runs through the LangGraph SDK, and returns promptly with a task identifier that can be monitored.

## Rubric and repair lifecycle

When a main-agent response has no tool calls, `RubricMiddleware` sends a bounded transcript to a separate grader sub-agent. A structured `needs_revision` response is represented to the main agent as a tagged `HumanMessage` containing actionable feedback, so the ordinary agent loop continues. `satisfied` and `failed` finish the grading run; reaching the cap produces `max_iterations_reached`, and a grader exception produces `grader_error`. Those latter two terminal statuses are synthesized by middleware, not emitted by the grader. The grader treats transcript contents as untrusted observation and the rubric as the definition of done.

`PatchToolCallsMiddleware` protects resumed or interrupted work before the agent begins. It finds every valid or invalid `AIMessage` tool call without a matching `ToolMessage`, appends a synthetic cancellation response (or malformed-arguments response for invalid calls), and rewrites history. This repair means downstream model and tool code sees a complete call/result pairing rather than a dangling call.

## Internal extension and enforcement helpers

| Helper | Responsibility and safe-use implication |
| --- | --- |
| `_fs_interrupt` | Generates path-aware HITL `when` predicates for interrupt-mode filesystem rules. Exact tools match the call path; bulk tools protect intersecting subtrees, including pathless bulk calls. |
| `_message_eviction` | Writes oversized tool text and creates a line-numbered head-and-tail preview while retaining media blocks. If the backend write fails, callers retain the original message. |
| `_overflow_clip` | Implements the summarization overflow fallback described above and preserves original message ids for reducer replacement. |
| `_prompt_caching` | Installs provider middleware conditionally; add another provider here only with an optional-import boundary and unsupported-model behavior. |
| `_state` | Finds `PrivateStateAttr` fields at runtime. An unresolvable schema is warned about and skipped, which can allow its nominally private fields to cross subagent boundaries. |
| `_tool_exclusion` | Applies harness-profile exclusions both to advertised tools and attempted calls. It is intentionally appended after custom middleware so a later request wrapper cannot restore an excluded tool; it is consistency control, not authorization. |
| `_video` | Optional `read_file` support for video: lazy PyAV/Pillow discovery keeps the `[video]` extra optional, while reads use offset/limit as seconds and emit timestamped sampled image blocks. |
| `_utils` | Shared immutable system-message append helper used by prompt-injecting middleware. |

## Change and test guidance

Choose middleware—not a plain tool—when a new capability must participate in request shaping, persist middleware state, or be uniformly available to SDK consumers. Keep backend I/O behind `BackendProtocol`; make state that must not propagate to child agents `PrivateStateAttr`; and preserve synchronous/asynchronous hook parity. Keep policy enforcement separate from presentation filtering: tool exclusions are not a security boundary, and filesystem approval belongs in HITL assembly.

Focused coverage lives in `tests/unit_tests/middleware/`: filesystem initialization and video behavior, memory and skills sync/async behavior, compaction and summarization, rubric iteration, subagent initialization, tool exclusion, and tool schemas. Broader tests in `tests/unit_tests/test_permissions.py`, `test_subagents.py`, `test_async_subagents.py`, and `test_middleware.py` exercise integration boundaries. Run the focused module that matches the capability being changed, then the relevant broader integration test; changes to graph assembly also need stack/profile coverage.
