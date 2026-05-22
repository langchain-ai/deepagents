# 06_SELECTED_5_PR_PLAN.md — Selected PRs Implementation Plan

## Selected PRs for `okwn/deepagents` Fork

The following five opportunities were selected based on: external relevance, bounded scope, testability, and alignment with the repo's existing patterns.

---

## PR 1 — `feat(sdk): add regex support to the grep tool` ⭐ Primary

### Links
- **Upstream issue**: #3547 (`feat(sdk): add regex support to the grep tool`)
- **Labels**: `external`, `deepagents`, `feature`

### Problem Statement
The `grep` tool in `FilesystemMiddleware` currently only supports literal string matching. Users and agents need the ability to search using regular expressions, which is a fundamental developer tool capability.

### Implementation Plan

#### Step 1 — Add `regex` parameter to `GrepSchema`
**File**: `libs/deepagents/deepagents/middleware/filesystem.py`

```python
# In GrepSchema class, add:
regex: bool = Field(
    default=False,
    description="If True, treat pattern as a regular expression. "
    "If False (default), use literal string matching.",
)
```

#### Step 2 — Update BackendProtocol.grep / agrep signatures
**File**: `libs/deepagents/deepagents/backends/protocol.py`

```python
def grep(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    *,
    regex: bool = False,  # NEW
) -> GrepResult: ...

async def agrep(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    *,
    regex: bool = False,  # NEW
) -> GrepResult: ...
```

#### Step 3 — Update FilesystemBackend.grep / agrep
**File**: `libs/deepagents/deepagents/backends/filesystem.py`

In `FilesystemBackend`, the `grep` method delegates to ripgrep (`rg`). When `regex=False`, pass `--fixed-strings`; when `regex=True`, pass `--no-ignored` (normal behavior) and omit `--fixed-strings` so ripgrep interprets the pattern as a regex.

#### Step 4 — Update CompositeBackend.grep / agrep
**File**: `libs/deepagents/deepagents/backends/composite.py`

Forward the `regex` kwarg to all underlying backends and merge results.

#### Step 5 — Update Middleware grep tool functions
**File**: `libs/deepagents/deepagents/middleware/filesystem.py`

In `sync_grep` and `async_grep`, pass `regex` to `backend.grep(...)` and `backend.agrep(...)`.

#### Step 6 — Add unit tests
**File**: `libs/deepagents/tests/unit_tests/test_file_system_tools.py`

- `test_grep_regex_pattern_matches` — verify regex matching
- `test_grep_literal_default` — verify literal matching is still default
- `test_grep_regex_ignores_glob_patterns` — verify that patterns like `file_.*\.py` work as regex, not glob

#### Verification
```bash
cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_file_system_tools.py
make lint  # should pass
```

---

## PR 2 — Improve coverage of `middleware/_overflow_clip.py` (70% → 90%+)

### Problem Statement
`_overflow_clip.py` handles the fallback path when context windows overflow. At 70% coverage, it's one of the lowest-covered files in the middleware layer, representing a risk for edge-case bugs.

### Implementation Plan

#### Step 1 — Add tests for `_derive_overflow_clip_threshold_tokens`
**File**: New or add to existing middleware test file

```python
def test_derive_overflow_clip_threshold_tokens_fraction():
    assert _derive_overflow_clip_threshold_tokens(("fraction", 0.5), max_input_tokens=1000) == 500

def test_derive_overflow_clip_threshold_tokens_tokens():
    assert _derive_overflow_clip_threshold_tokens(("tokens", 3000), None) == 3000

def test_derive_overflow_clip_threshold_tokens_fallback():
    assert _derive_overflow_clip_threshold_tokens(("messages", 10), None) == 5000
```

#### Step 2 — Add tests for `_find_tail_tool_message_batch`
```python
def test_find_tail_batch_empty():
    assert _find_tail_tool_message_batch([]) is None

def test_find_tail_batch_single_tm():
    tm = ToolMessage(content="ok", tool_call_id="x", name="tool")
    assert _find_tail_tool_message_batch([tm]) == (0, [tm])

def test_find_tail_batch_mixed_then_tm():
    hm = HumanMessage(content="hi")
    tm = ToolMessage(content="ok", tool_call_id="x", name="tool")
    ai = AIMessage(content="", tool_calls=[{"id": "x", "name": "tool", "args": {}}])
    result = _find_tail_tool_message_batch([hm, ai, tm])
    assert result is not None
    assert result[0] == 2
```

#### Step 3 — Add tests for `_slice_read_file_tm`
```python
def test_slice_read_file_short_content():
    tm = ToolMessage(content="short", tool_call_id="x", name="read_file")
    result = _slice_read_file_tm(tm, "/path/to/file")
    assert result.content == "short"  # no truncation needed

def test_slice_read_file_long_content():
    long_content = "x" * 10_000
    tm = ToolMessage(content=long_content, tool_call_id="x", name="read_file")
    result = _slice_read_file_tm(tm, "/path/to/file")
    assert len(result.content) < len(long_content)
    assert "/path/to/file" in result.content
```

#### Step 4 — Add tests for `_clip_one_tail_message` generic path
```python
def test_clip_one_tail_message_generic():
    # Non-read_file tool message, should call _offload_tool_message_content
    tm = ToolMessage(content="large output", tool_call_id="x", name="some_tool")
    ...
```

#### Step 5 — Run coverage
```bash
cd libs/deepagents && uv run --group test pytest -vvv tests/unit_tests/test_overflow_clip.py --cov=deepagents.middleware._overflow_clip --cov-report=term-missing
```

---

## PR 3 — `feat(sdk): add interrupt mode to filesystem permissions` (Issue #3505)

### Links
- **Upstream PR**: #3505 (`feat(sdk): add interrupt mode to filesystem permissions`)
- **Labels**: `feature`, `dependencies`, `deepagents`, `internal`, `size: M`

### Problem Statement
Filesystem permission checks currently allow or deny operations. An "interrupt" mode would pause execution and allow a human-in-the-loop operator to approve or deny the operation — a key safety feature.

### Implementation Plan

#### Step 1 — Understand current permission model
**File**: `libs/deepagents/deepagents/middleware/permissions.py`

Current permissions support `allow`, `deny`, and implicitly `None` (unrestricted). An interrupt mode would add a third explicit state.

#### Step 2 — Add `interrupt` to permission check logic
**Files**: `middleware/permissions.py`, `middleware/filesystem.py`

Add a `permission_mode: Literal["allow", "deny", "interrupt"] = "allow"` config to `FilesystemMiddleware`.

When `interrupt`:
- Tool execution is paused
- A `PermissionRequest` is emitted to the state
- Operator approves/denies via HITL

#### Step 3 — Unit tests
**File**: `libs/deepagents/tests/unit_tests/test_permissions.py`

Add tests for interrupt mode behavior in `test_permissions.py`.

#### Step 4 — Verify
```bash
cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_permissions.py
```

---

## PR 4 — `feat(quickjs): add swarm task tool` (PR #3472)

### Links
- **Upstream PR**: #3472 (`feat(quickjs): add swarm task tool`)
- **Labels**: `feature`, `deepagents`, `internal`, `size: XL`, `quickjs`

### Problem Statement
The quickjs package needs a "swarm task" tool for coordinating multi-agent tasks. This is in the `langchain-quickjs` package, which is related to deepagents.

### Implementation Plan

#### Step 1 — Locate the quickjs package
**Path**: Likely in `libs/quickjs/` or referenced as a dependency

The `quickjs` package is related to the `deepagents` monorepo. Check if it's in `libs/partners/` or if it's an external dep.

#### Step 2 — Follow PR #3472's implementation
Review the open PR's diff to understand the swarm task tool implementation and its integration points.

#### Note
This PR is marked `size: XL` — it's a significant change. A smallerScoped approach would be to add unit tests or documentation for the swarm task tool once it's merged, or to fix related bugs.

---

## PR 5 — `feat(sdk): runtime-resolved summarization model` (PR #3494)

### Links
- **Upstream PR**: #3494 (`feat(sdk): runtime-resolved summarization model`)
- **Labels**: `feature`, `deepagents`, `internal`, `size: XL`, `open-swe`

### Problem Statement
Currently the summarization model is fixed at agent creation time. This PR enables runtime resolution, allowing the model to be selected dynamically (e.g., cheaper model for short summaries, stronger model for complex summaries).

### Implementation Plan

#### Step 1 — Understand the current summarization middleware
**File**: `libs/deepagents/deepagents/middleware/summarization.py`

Find where the summarization model is configured and how it's used.

#### Step 2 — Modify model resolution
Change from a static model string to a callable that returns the model at runtime based on context (e.g., message count, context size).

#### Step 3 — Add tests
**File**: `libs/deepagents/tests/unit_tests/test_summarization.py` (or add to existing)

Test that:
- A static model string still resolves correctly (backwards compat)
- A callable model is invoked with the right context
- The resolved model is used for summarization

#### Step 4 — Verify
```bash
cd libs/deepagents && make test TEST_FILE=tests/unit_tests/test_summarization.py
```

---

## Execution Order

| Priority | PR | Rationale |
|---|---|---|
| 1 | Grep regex (#3547) | Fresh external issue, clear scope, easy to validate |
| 2 | overflow_clip coverage | Self-contained test work, immediate value |
| 3 | Filesystem permissions interrupt | Clear need, bounded scope |
| 4 | Summarization model runtime | Medium scope, good for understanding middleware |
| 5 | Swarm task tool | Larger scope, lower priority |

## Notes

- All PRs must follow the Conventional Commits format: `type(scope): description`
- All changes require unit tests (no network tests needed for these)
- Run `make lint` and `make type` before submitting
- PR descriptions should reference the upstream issue using `Closes #N` or `Related to #N`