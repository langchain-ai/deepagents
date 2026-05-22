# 05_PR_CANDIDATES.md — PR Opportunities in deepagents

## Context

These are upstream open PRs and issues that represent real, reviewable contribution opportunities for the `okwn/deepagents` fork.

---

## Candidate 1 — `feat(sdk): add regex support to the grep tool`

| Field | Value |
|---|---|
| **Issue / PR** | Issue #3547 (open, labeled `external`, `deepagents`) |
| **Age** | 2026-05-22 (created today — fresh request) |
| **Scope** | SDK (`libs/deepagents`) |
| **Size estimate** | M |
| **Path** | `deepagents/middleware/filesystem.py` |

### What It Is
Request to add a `regex: bool = True/False` parameter to the `grep` tool, allowing agents to search using regex patterns instead of (or in addition to) literal strings. Currently the grep tool uses `wcmatch` glob-style matching and ripgrep for literal string matching only.

### Why It's a Good Candidate
- Explicitly labeled `external` (community-facing, not internal)
- Clean, bounded scope: add one parameter to `GrepSchema`, thread through `sync_grep`/`async_grep` and into the backend protocol
- No architectural changes needed
- GrepMatch, GrepResult types already exist
- Existing literal-search implementation can be extended (e.g., pass `--no-fixed-strings` flag to ripgrep when `regex=True`)

### Implementation Sketch
1. Add `regex: bool = Field(default=False, description="...")` to `GrepSchema`
2. In `sync_grep`/`async_grep`, branch on `regex` flag before calling `backend.grep(...)` or `backend.agrep(...)`
3. Update `BackendProtocol.grep`/`agrep` signatures to accept a `regex: bool = False` kwarg
4. Propagate to `FilesystemBackend`, `CompositeBackend`, `SandboxBackend` implementations
5. When `regex=True`, use ripgrep's `--no-fixed-strings` flag instead of `--fixed-strings`

---

## Candidate 2 — `feat(sdk): add `delete_file` tool to filesystem middleware`

| Field | Value |
|---|---|
| **PR** | #3066 (merged: 2026-05-01) |
| **Scope** | SDK (`libs/deepagents`) |
| **Size estimate** | M |
| **Path** | `deepagents/middleware/filesystem.py` |

### What It Is
Already merged, but it shows a clear pattern for how to add new tools. The PR added `delete_file` to the filesystem middleware. The same pattern can be used to add other tools (e.g., `move_file`, `copy_file`, `mkdir`).

### Why It's a Good Reference
- Acts as a template for adding new filesystem tools
- Shows permission model integration, backend protocol extension, schema definition
- Already reviewed and merged — clean implementation to model

---

## Candidate 3 — Coverage gaps in `middleware/_overflow_clip.py` (70%)

| Field | Value |
|---|---|
| **Type** | Test improvement / code quality |
| **Scope** | SDK (`libs/deepagents`) |
| **Size estimate** | S–M |
| **Path** | `libs/deepagents/tests/unit_tests/` + `libs/deepagents/deepagents/middleware/_overflow_clip.py` |

### What It Is
`_overflow_clip.py` has 70% coverage — missing tests for:
- `_derive_overflow_clip_threshold_tokens` with both `"fraction"` and `"tokens"` kinds
- `_find_tail_tool_message_batch` with empty / single-message edge cases
- `_build_tool_call_index` edge cases
- `_slice_read_file_tm` with content < 4000 chars and content > 4000 chars
- `_clip_one_tail_message` generic (non-read_file) path
- `_clip_overflow_tail` with no tail batch and with multiple tail messages
- Exception/error paths during offload (though the file has no explicit `raise`/`except`)

### Why It's a Good Candidate
- Self-contained file, ~200 lines — isolated test work
- Pure unit test work (no network, no mocking of LLMs)
- Coverage improvement is always valued by maintainers
- Good intro to the codebase since overflow clipping is a key reliability feature

---

## Candidate 4 — `feat(runloop): surface Runloop blueprint/snapshot API`

| Field | Value |
|---|---|
| **Issue / PR** | #3540 (open, labeled `help wanted`, `internal`, `runloop`) |
| **Age** | 2026-05-21 |
| **Scope** | SDK (`libs/deepagents`) |
| **Size estimate** | M |
| **Path** | Likely `deepagents/graph.py` or new `runloop.py` module |

### What It Is
Feature request to expose the LangChain Runloop blueprint/snapshot API in `langchain-runloop`. This would allow agents to serialize/napshot their runloop state and restore it.

### Why It's a Good Candidate
- Labeled `help wanted` — explicitly seeking external contributions
- Relates to core agent state management (`graph.py`)
- Would involve understanding the checkpointing/snapshot mechanism in LangGraph

### Caveat
The issue is tagged `runloop` which may be a separate LangChain library. Verify this is actually in the `deepagents` codebase before starting.

---

## Candidate 5 — `feat(daytona): surface Daytona snapshot/image API`

| Field | Value |
|---|---|
| **Issue / PR** | #3539 (open, labeled `help wanted`, `internal`, `daytona`) |
| **Age** | 2026-05-21 |
| **Scope** | Partners / SDK |
| **Size estimate** | M |
| **Path** | `libs/partners/daytona/` |

### What It Is
Feature request to expose Daytona sandbox snapshot/image functionality in the `langchain-daytona` partner package.

### Why It's a Good Candidate
- Labeled `help wanted`
- Self-contained in the `partners/daytona/` package
- Snapshot/image APIs are well-defined sandbox features
- Good entry point if someone wants to work on the partner integration layer

---

## Summary Table

| # | Type | Scope | Size | Good First? |
|---|---|---|---|---|
| 1 — grep regex support | Feature | SDK | M | Yes (bounded scope) |
| 2 — delete_file PR reference | Reference | SDK | — | N/A |
| 3 — overflow_clip coverage | Test improvement | SDK | S–M | Yes (isolated) |
| 4 — Runloop API | Feature | SDK | M | Maybe (needs investigation) |
| 5 — Daytona snapshot API | Feature | Partners | M | Yes (self-contained) |