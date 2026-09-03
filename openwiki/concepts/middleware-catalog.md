---
type: reference-index
title: Middleware Catalog
description: Index of the built-in deepagents SDK middleware modules — what each contributes, its responsibility, and where the deeper concept page lives.
tags: [middleware, deepagents, catalog, filesystem, summarization, subagents, skills, rubric, permissions]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T21:35:57.774Z
sources:
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
  - id: openwiki-source-a0b5b06aa03bfdbc6e7e6664
    resource: repo://libs/deepagents/deepagents/middleware/_utils.py
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
  - id: openwiki-source-f01b7478b818ecc507f2ed5d
    resource: repo://libs/deepagents/deepagents/middleware/permissions.py
  - id: openwiki-source-b93c32bc33a8fa17b52b8a0e
    resource: repo://libs/deepagents/deepagents/middleware/rubric.py
  - id: openwiki-source-66cf9d0832d3cb55bec2b5ed
    resource: repo://libs/deepagents/deepagents/middleware/skills.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T21:35:57.774Z"}
---

# Middleware Catalog

This page is a reference index of the middleware modules that ship in
[`deepagents.middleware`](repo://libs/deepagents/deepagents/middleware/__init__.py).
Each module either subclasses `AgentMiddleware` or provides glue that installs
one. Their common purpose is to shape LLM requests before they are sent —
injecting tools, editing the system prompt, transforming message history, or
maintaining typed cross-turn state — which is what distinguishes middleware from
a plain `tools=` callable that runs only *after* the model chooses to call it.

This is an index, not a behavior reference. For the ordering and assembly of the
stack see [middleware-stack.md](repo://openwiki/architecture/middleware-stack.md);
for context-window mechanics see
[context-management.md](repo://openwiki/concepts/context-management.md); for
delegation see [subagents-skills.md](repo://openwiki/concepts/subagents-skills.md);
and for the tool surface see
[tools-filesystem.md](repo://openwiki/concepts/tools-filesystem.md). Where a
trace shows a middleware firing or not firing for a given model, defer to
[runtime-behavior.md](repo://openwiki/runtime-behavior.md).

## Public middleware

The package `__init__.py` re-exports the public middleware classes and their
supporting types — the surface a consumer imports from `deepagents.middleware`.

| Module | Exports | Responsibility | Deeper page |
| --- | --- | --- | --- |
| `filesystem` | `FilesystemMiddleware`, `FilesystemPermission` | Provides the `ls`/`read_file`/`write_file`/`edit_file`/`delete`/`glob`/`grep`/`execute` tools, resolves the backend, enforces permission deny rules, scrubs unsupported multimodal content, and proactively offloads oversized tool results. The largest, most complex module. | [tools-filesystem.md](repo://openwiki/concepts/tools-filesystem.md) |
| `summarization` | `SummarizationMiddleware`, `SummarizationToolMiddleware`, `create_summarization_tool_middleware`, `DEEPAGENTS_DEFAULT_SUMMARY_PROMPT` | Automatically compacts the conversation when token usage crosses a threshold, offloads evicted history to a backend, and exposes an on-demand `compact_conversation` tool. One of the largest modules. | [context-management.md](repo://openwiki/concepts/context-management.md) |
| `skills` | `SkillsMiddleware` | Loads Anthropic-style agent skills from backend sources and injects progressive-disclosure instructions into the system prompt. | [subagents-skills.md](repo://openwiki/concepts/subagents-skills.md) |
| `subagents` | `SubAgentMiddleware`, `SubAgent`, `CompiledSubAgent` | Exposes a `task` tool that delegates work to synchronous child agents that run to completion inline. | [subagents-skills.md](repo://openwiki/concepts/subagents-skills.md) |
| `async_subagents` | `AsyncSubAgentMiddleware`, `AsyncSubAgent` | Launches background subagent runs on remote Agent Protocol servers via the LangGraph SDK, returning a task id immediately for monitoring. | [subagents-skills.md](repo://openwiki/concepts/subagents-skills.md) |
| `memory` | `MemoryMiddleware` | Loads AGENTS.md files from configured sources and always injects that persistent context into the system prompt. | [context-management.md](repo://openwiki/concepts/context-management.md) |
| `rubric` | `RubricMiddleware` (+ `Rubric*`/`Grader*`/`Criterion*` types) | Runs a grader sub-agent whenever the agent would finish; injects revision feedback and resumes the loop until satisfied, failed, or `max_iterations`. One of the largest modules. | — |
| `permissions` | `FilesystemPermission` | Backward-compatible re-export of `FilesystemPermission` from `filesystem`; no logic of its own. | [tools-filesystem.md](repo://openwiki/concepts/tools-filesystem.md) |
| `patch_tool_calls` | `PatchToolCallsMiddleware` | Repairs dangling/unanswered tool calls in the message history before the agent runs. | — |

## Private helpers

These modules are underscore-prefixed and not exported from the package. They
back the public middleware or are installed by graph-assembly code.

| Module | Role |
| --- | --- |
| `_fs_interrupt` | Glue that turns `FilesystemPermission` interrupt rules into an `interrupt_on` mapping for `HumanInTheLoopMiddleware`. |
| `_message_eviction` | Shared helper for evicting/clipping large message content to a head+tail preview; used by filesystem offload and summarization overflow. |
| `_overflow_clip` | Read-side tail-clipping for the summarization-on-`ContextOverflowError` fallback path. |
| `_prompt_caching` | Factory that appends provider-specific prompt-caching middleware (Anthropic always; Bedrock/Fireworks when installed). |
| `_state` | Helpers for Deep Agents state schemas, e.g. discovering `PrivateStateAttr` field names. |
| `_tool_exclusion` | `_ToolExclusionMiddleware` that strips harness-profile-excluded tools before the model and rejects them at the call boundary. |
| `_video` | Optional boundary to the PyAV video backend; decodes a time window of a video into sampled frame content blocks for `read_file`. |
| `_utils` | Small shared utilities, e.g. `append_to_system_message`. |

## Relative size as a complexity signal

The modules are far from uniform. `filesystem` is by far the largest — it wires
many distinct tools plus permission, multimodal, and eviction handling — while
`summarization` and `rubric` are the next-largest. By contrast `permissions` is
a one-line re-export and `_utils`/`_state` are thin helper modules. Reading time
should be budgeted accordingly.
