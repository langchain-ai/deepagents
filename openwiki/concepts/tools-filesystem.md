---
type: concept
title: Tool Surface & Filesystem Tools
description: How the model's visible tool set is assembled from middleware, caller tools, backend capability, and profile exclusions, and how the built-in filesystem tools and their permission checks behave.
tags: [tools, filesystem, middleware, permissions, backends, tool-visibility]
verified:
  - by: openwiki/0.4.0
    at: 2026-08-26T21:35:57.774Z
sources:
  - id: openwiki-source-44654f7b6bdd46e6f9dd122c
    resource: repo://libs/code/deepagents_code/_constants.py
  - id: openwiki-source-3300d75e0c132882e2e3b4ce
    resource: repo://libs/code/deepagents_code/tool_catalog.py
  - id: openwiki-source-e7c7a0d6e6f2fa82362f1c56
    resource: repo://libs/deepagents/deepagents/_tools.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-0fb4155c19dd248acd3ffe4f
    resource: repo://libs/deepagents/deepagents/middleware/_fs_interrupt.py
  - id: openwiki-source-8b1aaf77fc0430fd00711a73
    resource: repo://libs/deepagents/deepagents/middleware/_tool_exclusion.py
  - id: openwiki-source-fed4b84a38685f37e58018c5
    resource: repo://libs/deepagents/deepagents/middleware/filesystem.py
generated: {by: "openwiki/0.4.0", at: "2026-08-26T21:35:57.774Z"}
---

# Tool Surface & Filesystem Tools

The set of tools the model can call is not a single hand-maintained list. It is
*assembled* from several independent layers, each of which can add or remove
tools. Understanding those layers is what lets you answer two very different
questions: "why can the model not see a tool?" and "why does a tool the model
*can* see keep failing?" These have different causes and different fixes.

A second, orthogonal distinction runs through this whole page: **visibility is
not permission.** A tool can be advertised to the model, chosen by the model,
and still be denied, interrupted for human approval, or rejected at the
execution boundary. Visibility decides what the model is *offered*; permission
and backend capability decide what actually *runs*.

## The layers that produce the visible tool set

The tools bound to a compiled agent come from these layers, applied roughly in
this order:

1. **Built-in middleware tools.** Middleware injects tools as part of the base
   stack: the filesystem tools from
   [`FilesystemMiddleware`][repo], `write_todos` from the todo middleware,
   `task`/subagent-delegation tools from the subagent middleware, and so on.
   The base stack and its ordering are documented on
   [`create_deep_agent`](../architecture/middleware-stack.md).
2. **Caller-supplied `tools=`.** Tools passed by the embedder are added to the
   set. `deepagents` can copy and rewrite their descriptions (via description
   overrides) without mutating the caller's own tool objects — dict tools and
   `BaseTool` instances are rewritten in place, while plain callables are
   returned unchanged because safely replacing their descriptions would require
   wrapping them in new objects.
3. **Backend capability gating.** Capability-gated filesystem tools are dropped
   at request time when the resolved backend cannot serve them: `execute`
   requires a backend implementing `SandboxBackendProtocol` (a shell), and
   `delete` requires delete support. These are removed from the request rather
   than advertised and left to fail.
4. **Profile `excluded_tools`.** A harness profile can name tools to strip from
   the visible surface. `_ToolExclusionMiddleware` filters them out of the model
   request last, so a custom middleware cannot re-add them.
5. **Permission enforcement.** Even a visible, model-chosen tool call is subject
   to `FilesystemPermission` rules (`allow`/`deny`/`interrupt`) and to the
   exclusion middleware's own tool-call-boundary rejection.

[repo]: repo://libs/deepagents/deepagents/middleware/filesystem.py

<!-- openwiki: mermaid parse failed and this diagram was converted to a text fence so it does not break rendering. Fix the diagram source and restore the mermaid fence. Parser error: Heuristic: an unescaped angle bracket inside a label breaks rendering; rephrase the label. -->
```text
flowchart TD
  MW["Built-in middleware tools<br/>(filesystem, todo, subagent/task)"] --> SET["Candidate tool set"]
  CALLER["Caller tools= (with optional<br/>description overrides)"] --> SET
  SET --> CAP["Backend capability gate<br/>(drop execute w/o shell, delete w/o support)"]
  CAP --> EXCL["Profile excluded_tools<br/>(_ToolExclusionMiddleware, applied last)"]
  EXCL --> VISIBLE["Tools advertised to the model"]
  VISIBLE --> CALL["Model emits a tool call"]
  CALL --> PERM["Permission + exclusion enforcement<br/>allow / deny / interrupt / reject"]
  PERM --> EXEC["Backend executes the tool"]
```

## Debugging split: missing vs. visible-but-failing

Because the layers have distinct owners, the debugging path splits cleanly:

- **A tool is missing** (the model never sees it) → look at the *visibility*
  layers: the middleware that would inject it is absent or excluded, the
  filesystem allowlist (`tools=`) omits it, or a profile's `excluded_tools`
  strips it. `_ToolExclusionMiddleware` and `FilesystemMiddleware`'s allowlist
  both *omit* a tool entirely rather than merely hiding its schema.
- **A tool is visible but failing** (the model calls it and gets an error) → look
  at the *backend/permission* layers: the backend does not implement the needed
  capability, a `FilesystemPermission` denies the path, a HITL interrupt is
  pending, or the exclusion middleware rejects the name at the call boundary.

## `_ToolExclusionMiddleware`: excluded means both hidden and rejected

`_ToolExclusionMiddleware` is wired into the tail of the stack whenever the
resolved profile has a non-empty `excluded_tools`, and appended *after* custom
middleware so excluded names cannot be restored by a custom `wrap_model_call`.
It enforces exclusion at two boundaries:

- In `wrap_model_call`/`awrap_model_call` it removes excluded tools from
  `request.tools` before the model sees them.
- In `wrap_tool_call`/`awrap_tool_call` it rejects any call naming an excluded
  tool with an `Error: <name> is not available.` `ToolMessage`, because the
  executor still has the tool registered and dispatches on the name the model
  emits.

The middleware's own docstring is explicit that this is a consistency mechanism,
not a security surface: it keeps execution consistent with what was advertised,
and exclusions resolve per model. Excluding tools is also how the harness turns
off capabilities like `execute` at the profile level (for example a profile that
sets `excluded_tools=frozenset({"execute"})`).

## Enumerating the real tool set

The tool catalog (`dcode tools list` and the `/tools` slash command) never keeps
a parallel list of tool names. It compiles the agent with an offline placeholder
chat model — one that binds tools but is never invoked, so no credentials or
network are needed — and reads the bound tool node, so displayed names and
descriptions cannot drift from what the model actually sees. Built-in tools are
collected this way; MCP tools are discovered through the same path the app and
server use. Because `FilesystemMiddleware` omits disallowed filesystem tools
from the node entirely, forwarding the allowlist narrows the enumeration by
itself; a defensive backstop still checks for any disallowed filesystem tool
that leaked through and logs loudly (returning the unfiltered list) rather than
scrubbing it, so a broken allowlist is visible instead of hidden.

## The built-in filesystem tools

`FilesystemMiddleware` is the largest middleware module in `deepagents`, and it
owns the file tools. The enumerated names are fixed by the `FsToolName` literal:
`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and
`execute`.

| Tool | Purpose |
| --- | --- |
| `ls` | List directory entries at an absolute path. |
| `read_file` | Read file content, with `offset`/`limit` pagination. Required in any allowlist. |
| `write_file` | Create or replace a file. |
| `edit_file` | Apply exact string replacements to an existing file. |
| `delete` | Remove a file or directory recursively (backend-gated). |
| `glob` | Find files matching a glob pattern, returning absolute paths. |
| `grep` | Search for a **literal** text pattern (not regex) across files. |
| `execute` | Run a shell command in a sandbox (backend-gated on `SandboxBackendProtocol`). |

Several behaviors matter to callers:

- **`read_file` is mandatory.** Passing a `tools=` list that omits `read_file`
  raises `ValueError`; the middleware requires it.
- **Allowlisting omits, not hides.** When `tools=` is a list, disallowed tool
  factories are never instantiated, so the name never reaches the dispatchable
  tool node. `tools="all"` (or the unset default) enables every name subject to
  backend capability.
- **Capability checks still apply.** Listing `execute` or `delete` when the
  backend does not support them is a no-op; those tools are filtered from the
  request at model-call time and, if somehow reached, fail gracefully with an
  explanatory error.
- **Search is literal and bounded.** `grep` matches the pattern verbatim — regex
  metacharacters are ordinary characters — and its description points at
  `execute` with `rg` for genuine regex (that fallback line is dropped when
  `execute` is not active). Results are capped by `grep_max_count` (default
  1000), overridable per call via `max_count`, or disabled with `None`.
- **`glob` supports** `*`, `**`, `?`, `[abc]` sets, and `{a,b}` alternatives,
  with anchoring rules based on whether the pattern contains `/`.
- **Large results are managed.** Oversized tool results are evicted to the
  filesystem under the artifacts root (default `/large_tool_results/`), except
  for tools that truncate their own output or never grow large
  (`ls`, `glob`, `grep`, `read_file`, `edit_file`, `write_file`, `delete`).

At request time, `_filter_unsupported_tools_and_apply_prompt` runs on both the
sync and async model-call paths: it drops capability-gated tools the backend
cannot serve, reconciles the `grep` and `execute` descriptions to whether
`execute` and the search tools are actually active, and appends the
virtual-vs-host path routing prompt when `execute` is active over a composite
backend.

## Filesystem permissions: visibility's separate gate

`FilesystemPermission` rules are enforced *inside* the tool implementations, not
by removing tools. Each rule pairs a set of `operations`, a set of absolute
`paths` (glob patterns), and a `mode`:

- `"allow"` (default): the call proceeds.
- `"deny"`: the tool returns a permission-denied error `ToolMessage`.
- `"interrupt"`: the call is paused for human approval via
  `HumanInTheLoopMiddleware`.

`_check_fs_permission` matches the operation and path against the rules with
wcmatch globbing and returns the effect. Exact-path tools (`read_file`,
`write_file`, `edit_file`) fire on their single path; **bulk** tools
(`ls`, `glob`, `grep`, `delete`) fire whenever their search subtree could
overlap an anchored rule prefix — and a pathless bulk call
(`grep(path=None)`) fires unconditionally for any interrupt-mode rule, because
it can touch anything. `FilesystemMiddleware` itself only enforces `deny` and
filters denied results; the graph-assembly code translates interrupt-mode rules
into an `interrupt_on` mapping for `HumanInTheLoopMiddleware`. See
[Permissions & HITL](permissions-hitl.md) for the interrupt flow.

Permission path patterns are validated at construction: they must start with
`/` and must not contain `..` or `~`. Permissions are also not yet supported
alongside execution-capable backends unless every rule path is scoped to routes,
because tool-level permissions for `execute` are not implemented.

## OBSERVED per-tool usage

The following figures are scoped to **this pull's** captured OpenWiki
initialization run (`repo://conversation_history/session_5ec334d0.md`); they are
sample-specific and not steady-state guarantees. That session is itself a
concrete trace of the built-in tool surface in use, exercising the filesystem
tools plus subagent delegation and the wiki's own authoring tool.

- **`read_file`, `ls`, `glob`, `grep`** dominate the trace: the agent opens each
  turn with wide fan-out batches of read-only tools (e.g. an initial batch of
  `ls` + `read_file` + `glob`, and repeated multi-file `read_file` batches),
  matching the pattern where independent reads are issued together in one turn.
- **`execute`** appears for shell-only work the file tools cannot do — for
  example running `rg --files ...` for repository-wide manifest discovery — which
  is exactly the backend-gated path that only exists when a shell backend is
  present.
- **`write_file` / `edit_file`** appear later, once authoring begins (plan file,
  then page bodies), consistent with these being confirmation-only tools that are
  never evicted for size.
- **Retries / churn:** the trace shows the authoring tool being re-issued many
  times in a row with near-identical payloads, illustrating that a visible,
  permitted tool can still be repeatedly re-attempted at the model's discretion —
  a visibility-vs-outcome distinction rather than a tool-availability one.

For per-run latency, token totals, and call-count aggregates across the whole
session, see [runtime behavior](../runtime-behavior.md), which owns the
trace-derived metrics; this section only characterizes which tools were used and
why.

## Related pages

- [Middleware stack](../architecture/middleware-stack.md) — where the tool-injecting middleware sit and in what order.
- [Backends](backends.md) — which backends provide shell/delete capability that gates `execute` and `delete`.
- [Permissions & HITL](permissions-hitl.md) — the deny/interrupt enforcement path in depth.
- [Runtime behavior](../runtime-behavior.md) — aggregate trace metrics for this pull.
