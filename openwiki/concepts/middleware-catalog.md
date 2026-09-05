---
type: reference index
title: Middleware Catalog
description: Compact index of built-in deepagents middleware, its assembly boundaries, and the configuration points that change the tools and context an agent receives.
tags: [middleware, deepagents, catalog, filesystem, context-management, subagents, skills, permissions]
sources:
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-fc54598423086acf9d53d9fd
    resource: repo://libs/deepagents/deepagents/middleware/__init__.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
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
  - id: openwiki-source-f913f8fa643e6c2796621ca5
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_filesystem_middleware_init.py
  - id: openwiki-source-8e42d656da29326d4189414d
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_tool_exclusion.py
verified:
  - by: openwiki/0.4.2
    at: 2026-09-05T08:05:02.390Z
generated: { by: "openwiki/0.4.2", at: "2026-09-05T08:05:02.390Z" }
---

# Middleware Catalog

`deepagents.middleware` is the public import surface for built-in middleware and
its supporting types. Middleware intercepts model requests, so it can change the
advertised tool set, system message, message history, or typed agent state before
the LLM runs; a callable supplied through `tools=` runs only after the model has
chosen it.

This is an index, not a substitute for implementation guides. See
[Middleware stack](repo://openwiki/architecture/middleware-stack.md) for ordering
and [SDK construction and execution](repo://openwiki/architecture/sdk-construction-execution.md)
for assembly. Detailed behavior belongs in [Filesystem tools](repo://openwiki/concepts/tools-filesystem.md),
[Context management](repo://openwiki/concepts/context-management.md),
[Subagents and skills](repo://openwiki/concepts/subagents-skills.md), and
[Permissions and HITL](repo://openwiki/concepts/permissions-hitl.md).

## Built-in catalog

| Component | What it contributes | Configuration and boundary |
| --- | --- | --- |
| `FilesystemMiddleware` | Filesystem tools: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, plus optional `execute`; request-time capability filtering, media cleanup, and large-result/history offload. | Give it an initialized `BackendProtocol` instance (default: `StateBackend()`), optional prompt/description overrides and eviction limits. `tools=` is an allowlist, requires `read_file`, and **does not register omitted names in the dispatchable tool node**. `delete` and `execute` are additionally removed from the model request when the backend lacks their capability. |
| `SummarizationMiddleware` | Threshold-triggered conversation compaction, persisted evicted history, and overflow fallback. `SummarizationToolMiddleware` supplies the on-demand `compact_conversation` tool. | Construct with the summarization model and backend; configure trigger, retention, token counter, summary prompt, and optional old-tool-argument truncation. |
| `SkillsMiddleware` | Anthropic-style progressive disclosure: discovers skill metadata before an agent run and presents locations, load diagnostics, and the index in the system message. | Sources load once per session; later sources win on duplicate skill names. Suppressing its prompt template still leaves metadata in state. |
| `MemoryMiddleware` | Loads configured `AGENTS.md` sources into state and appends persistent memory to the system message. | Loads once per session; missing files are skipped while other download errors fail the run. The prompt template can be disabled without disabling state loading. |
| `SubAgentMiddleware` | A synchronous `task` tool for delegated child-agent work. | The task waits for the child result. Declarative subagents may be isolated or use experimental `fork` mode. |
| `AsyncSubAgentMiddleware` | Background remote Agent Protocol runs managed through the LangGraph SDK. | Returns a task ID immediately for monitoring or update; local ASGI transport requires an async parent entrypoint. |
| `RubricMiddleware` | A finish-time grader loop. | A `needs_revision` verdict injects feedback and resumes; satisfied, failed, grader error, or iteration limit ends it. |
| `PatchToolCallsMiddleware` | Repairs incomplete tool-call history before execution. | Adds synthetic tool results for unanswered valid or malformed calls, then atomically replaces the message list. |
| `permissions` | Compatibility import only. | Re-exports `FilesystemPermission` from `filesystem`; it does not enforce anything itself. |

## Filesystem selection, dispatch, and authorization

Three similarly named mechanisms have different safety and execution meanings:

1. **Filesystem `tools=` selection is construction-time removal.** A name not in
   the allowlist is never created in `FilesystemMiddleware.tools`, so it cannot
   reach that middleware's dispatchable tool node. This is stronger than merely
   hiding a schema, but it is a product-surface choice rather than an
   authorization decision.
2. **Capability filtering is request-time advertising.** The middleware builds
   its configured tools, then `wrap_model_call` removes unsupported `delete` or
   `execute` tools from that particular model request. This prevents a model from
   being offered a capability that the resolved backend cannot serve.
3. **Permissions and HITL are downstream enforcement.** A dispatchable
   filesystem tool validates the path and applies deny rules before touching the
   backend. Separately, graph construction converts interrupt-mode filesystem
   rules into `HumanInTheLoopMiddleware` predicates. Therefore an omitted tool
   never dispatches, whereas a permission rule governs a tool that is present;
   neither harness tool exclusion nor prompt visibility should be treated as a
   security boundary.

`create_deep_agent()` owns integration: it supplies the shared backend, builds
filesystem/summarization/patching middleware for the main and default subagent
stacks, conditionally adds skills, subagent, async-subagent, caching, memory, and
HITL middleware, and appends harness `_ToolExclusionMiddleware` after custom
middleware. That final placement strips excluded injected tools from the model
request and rejects an excluded name if a model emits it anyway, but the tool
executor otherwise still dispatches registered names.

## Supporting helpers

| Helper | Role |
| --- | --- |
| `_fs_interrupt` | Adapts interrupt-mode `FilesystemPermission` rules into path-aware `interrupt_on` predicates for `HumanInTheLoopMiddleware`. |
| `_message_eviction` and `_overflow_clip` | Shared head-and-tail preview/offload utilities for filesystem result eviction and summarization's context-overflow path. |
| `_prompt_caching` | Appends Anthropic caching middleware and optional Bedrock/Fireworks adapters when those integrations are installed. |
| `_tool_exclusion` | Last-mile harness-profile visibility and call-boundary rejection, not a security control. |
| `_state`, `_utils`, `_video` | State-schema discovery, system-message append utility, and lazy optional PyAV-backed video decoding for `read_file`. |

## Focused verification

The middleware tests cover configuration boundaries as well as behavior. In
particular, filesystem initialization tests reject backend factories (callers
must pass initialized backend instances), verify when state-backed file state is
installed, and cover description/offload guidance. Tool-exclusion tests compile
an agent whose model deliberately emits an excluded `write_file` call and verify
that it returns an error without creating the file, while an allowed call still
runs. Use the focused middleware tests when changing construction or dispatch;
use the linked concept pages for broader behavior tests.
