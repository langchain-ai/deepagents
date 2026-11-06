---
type: architecture pattern
title: SDK Middleware Assembly
description: Explains how Deep Agents assembles, filters, and compiles middleware for the main agent and its synchronous and remote subagents. Covers profile exclusions, custom insertion, refusal handling, tool filtering, state isolation, and approval behavior.
tags: [middleware, deepagents, agent-construction, harness-profile, subagents, tool-surface]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-06T08:04:19.814Z
sources:
  - id: openwiki-source-b93533cac55718d75277d1cf
    resource: repo://libs/deepagents/deepagents/_excluded_middleware.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-46a23efe78a78f9b3cd75d00
    resource: repo://libs/deepagents/deepagents/middleware/memory.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-f763e99e439a1356866a7aa4
    resource: repo://libs/deepagents/deepagents/middleware/summarization.py
  - id: openwiki-source-837c84a3f3120bc778033547
    resource: repo://libs/deepagents/deepagents/middleware/unsupported_content.py
  - id: openwiki-source-59612eea63cbfafbd628feda
    resource: repo://libs/deepagents/deepagents/profiles/harness/harness_profiles.py
  - id: openwiki-source-13e0ea89aadd1a5e309ac7e3
    resource: repo://libs/deepagents/tests/unit_tests/middleware/test_skill_tools_payload.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
  - id: openwiki-source-851e3a9c96663d8db5ca3dec
    resource: repo://libs/deepagents/tests/unit_tests/test_permissions.py
  - id: openwiki-source-917b61d56e652536ba962f3a
    resource: repo://libs/deepagents/tests/unit_tests/test_subagent_refusal.py
generated: { by: "openwiki/0.4.2", at: "2026-11-06T08:04:19.814Z" }
---

# SDK Middleware Assembly

`create_deep_agent()` resolves the requested model and `HarnessProfile`, separates synchronous from remote subagent specifications, builds their middleware, and passes the finalized main list to LangChain `create_agent()`. Middleware is therefore the graph's request-time extension boundary: it may alter model requests, prompts, messages, tools, or graph state. The `tools=` argument supplies callable tools after a model selects one.

Prompt construction precedes compilation. The main prompt is caller `system_prompt`, followed by the active profile's optional base prompt and suffix, separated by blank lines. The final `create_agent()` call receives that prompt, profile-rewritten caller tools, the effective middleware, and graph options such as schemas, checkpointing, store, cache, debug, and name.

## Main stack: construction and order

The effective main stack is assembled in three bands: built-in **core**, caller middleware spliced into or immediately after that core, then the profile-dependent **tail**. Profile exclusions run both before and after caller merging, so this is an effective stack rather than simply the order in which constructors appear.

```mermaid
flowchart TD
    R["Resolve model and HarnessProfile"] --> V{"Protected scaffolding excluded"}
    V -- "yes" --> E1["ValueError"]
    V -- "no" --> B["Build core and tail"]
    B --> X1["Apply profile exclusions"]
    X1 --> M["Merge caller middleware by name"]
    M --> X2["Apply profile exclusions again"]
    X2 --> T["Append tool exclusion if configured"]
    T --> P["Collect private state and verify exclusion coverage"]
    P --> A["Compile with create_agent"]
    P --> E2["ValueError for unmatched exclusion"]
```

Diagram: main-stack assembly, including the two exclusion passes that prevent caller middleware from restoring a profile-excluded member.

### Core and tail

The core is, in order:

1. `FilesystemMiddleware`;
2. `SubAgentMiddleware` only when synchronous inline children exist;
3. `_ParentNotifierMiddleware`;
4. model-aware summarization middleware;
5. `PatchToolCallsMiddleware`; and
6. `AsyncSubAgentMiddleware` only when remote `AsyncSubAgent` specs exist.

The tail consists of materialized profile extras, optional `SkillsMiddleware`, provider prompt-caching middleware, optional `MemoryMiddleware`, optional `HumanInTheLoopMiddleware`, `UnsupportedContentMiddleware`, and `ModelRefusalMiddleware`. A profile's `_ToolExclusionMiddleware` is appended last, after both exclusions and caller merging.

`ModelRefusalMiddleware` is deliberately in the tail, after caller middleware. Thus a caller-installed `ModelFallbackMiddleware` can catch a refusal before the default refusal middleware raises `ModelRefusalError`. The default can itself be profile-excluded; without it, a refusal AI message remains in the main result rather than being raised.

Prompt caching precedes memory because memory changes the system prompt. Assembly always adds a lazy Anthropic cache adapter, which imports its delegate only for an Anthropic request; Bedrock and Fireworks adapters are added only when their integration packages are available. `MemoryMiddleware` loads sources into private state, adds formatted memory to the system message for model calls, and can put an Anthropic-only ephemeral cache-control breakpoint on the final system block.

`UnsupportedContentMiddleware` is late because it must evaluate the effective request model after routing or fallback middleware. It copies and filters only outgoing `HumanMessage` and `ToolMessage` content: explicitly unsupported multimodal blocks become explanatory text for that request, while stored thread content remains intact for a later compatible model.

### Custom middleware and exclusions

Caller `middleware=` is merged by `.name`, not blindly appended:

- A matching name replaces the assembled member in place; when several caller entries target that name, the last replacement wins.
- A novel name is inserted after the last surviving core member and before profile extras, skills, caching, memory, approval, compatibility, and refusal handling. Novel entries retain caller order.
- Replacing a `FilesystemMiddleware` retains the agent permissions if the replacement has none; a non-empty, different permission set is rejected.

`HarnessProfile.excluded_middleware` is separate and subtractive. It is validated before assembly, applied before the merge, and applied again after it. `FilesystemMiddleware` and `SubAgentMiddleware` are protected scaffolding: profiles cannot exclude them by class or name. Class exclusions use exact type (or a lazy adapter's declared `excluded_as` type); name exclusions require an exact `.name`. A name matching different concrete classes in one stack is rejected as ambiguous.

Every configured exclusion must match. Main-profile matches are accumulated across the main and automatic general-purpose stacks, then verified once; a declarative child independently verifies its own profile. This catches stale profile policy and typos. After the final main stack survives filtering, private fields declared by its state schemas are supplied to each surviving `SubAgentMiddleware`, so synchronous task dispatch excludes them from child state.

## Child stacks and execution boundaries

### Automatic general-purpose child

The automatic `general-purpose` child is added only when its profile enables it and no supplied synchronous child has that name. If it is disabled and there are no other synchronous children, no `SubAgentMiddleware` or `task` tool is present; remote async tools remain independent.

Its own stack is filesystem, summarization, patch-tool calls, profile extras, optional skills, prompt caching, and `ModelRefusalMiddleware`, followed by the same two exclusion passes and final tool exclusion. It inherits only caller middleware whose names match one of its original slots; novel middleware intended only for the main agent is not propagated. When compiled as a declarative child, its resolved approval configuration adds HITL as needed and `create_sub_agent()` adds unsupported-content filtering if absent.

### Declarative, forked, compiled, and remote children

Each declarative `SubAgent` resolves a model and harness profile of its own. Its pre-compilation stack is filesystem, summarization, patch-tool calls, profile extras, optional skills, provider caching, optional forked memory, and `ModelRefusalMiddleware`. It applies exclusions, merges spec middleware, applies exclusions again, verifies coverage, and finally applies tool exclusion. For a fork, parent caller middleware is combined with spec middleware first, with same-name spec middleware winning.

A declarative child inherits top-level tools, permissions, and `interrupt_on` only when its spec omits that field. Its own permissions replace—not extend—the parent rules. Resolved filesystem interrupt rules are merged with explicit `interrupt_on`, with explicit same-tool entries winning; `create_sub_agent()` appends `HumanInTheLoopMiddleware` only when the resolved configuration is non-empty. Compiled and remote children own their own schemas and approval policy.

The default synchronous child mode is isolated: it receives a delegated `HumanMessage` plus eligible non-private state. Unchanged returned values are discarded, preventing reducer duplication and parallel-update conflicts. A fork receives the parent's effective compacted history and a task preamble. Declarative forks retain non-excluded parent state, including private channels needed by the matching graph shape; compiled forks strip private channels because their runnable schemas are opaque. A fork that invokes `task` recursively is refused.

A synchronous `task` invocation converts a child refusal into a tool result rather than failing the parent task loop. It recognizes a refusal in the child's final AI message or catches `ModelRefusalError` and structured-output validation errors that represent one, and returns a notice with available category/explanation and a retry warning. This applies to sync and async task calls. Remote `AsyncSubAgent` specifications, identified by `graph_id`, are separated before inline construction and use `AsyncSubAgentMiddleware`; they launch background work and return a task ID rather than blocking.

## Tools, permissions, and approval

`_ToolExclusionMiddleware` removes named tools from model requests and rejects calls to those names at the tool-call boundary. Since it is last, it also removes tools injected earlier in the request path. It maintains consistency between the advertised and executable tool surface; it is explicitly not a security boundary.

Filesystem permissions are applied by `FilesystemMiddleware` to built-in filesystem tool calls, not direct backend use. Rules are evaluated in declaration order; the first matching rule wins and unmatched calls are allowed. Assembly turns `interrupt` permissions into per-tool HITL configurations, then overlays explicit `interrupt_on` values. Filesystem middleware itself enforces `deny`; HITL is responsible for approval. Generated predicates conservatively gate bulk operations, including intersecting subtrees, pathless searches, dot-root aliases, escaping glob patterns, and resolved symlink targets where available.

## Change and regression focus

Safe changes must preserve ordering and test the observable boundary, not just list contents: same-name replacement versus novel insertion, both exclusion passes, protected and unmatched exclusion failures, general-purpose override-only inheritance, child-specific profiles, fork state behavior, final tool filtering, and approval predicates.

In particular, refusal tests cover main-agent default raising, caller fallback before the default refusal layer, profile exclusion of that layer, declarative and compiled child refusals becoming task results in synchronous and asynchronous execution, structured-output refusal handling, and remote-result formatting. Provider-payload tests pin skill disclosure placement after a relevant read result without splitting tool-result batches or preceding queued user input, including relocation after compaction. Permission tests cover recursive-delete denial plus generated approval behavior for bulk and path-sensitive operations.

See also [SDK construction and execution](/openwiki/architecture/sdk-construction-execution.md), [middleware catalog](/openwiki/concepts/middleware-catalog.md), [permissions and HITL](/openwiki/concepts/permissions-hitl.md), [profiles and models](/openwiki/concepts/profiles-models.md), and [subagents and skills](/openwiki/concepts/subagents-skills.md).
