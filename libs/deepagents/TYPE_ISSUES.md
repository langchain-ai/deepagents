# Type Checking Issues

This document tracks type safety issues discovered during `ty` type checking of the `deepagents/` source code. Issues are categorized by root cause and whether they require upstream fixes in langchain/langgraph.

## Summary

| Category | Count | Status |
|----------|-------|--------|
| Unsuppressed errors | 17 | Needs fixes (upstream or local) |
| Suppressed with `# ty: ignore` | 13 | Tracked, awaiting proper solutions |
| **Total** | **30** | |

### By error rule

| Rule | Unsuppressed | Suppressed | Total |
|------|-------------|------------|-------|
| `invalid-argument-type` | 3 | 4 | 7 |
| `missing-typed-dict-key` | 6 | 0 | 6 |
| `invalid-type-arguments` | 4 | 0 | 4 |
| `invalid-method-override` | 4 | 0 | 4 |
| `unresolved-attribute` | 0 | 2 | 2 |
| `no-matching-overload` | 0 | 1 | 1 |
| Untyped (`# ty: ignore`) | 0 | 6 | 6 |

---

## Unsuppressed Errors (17)

### 1. `AgentMiddleware` Type Arguments — `invalid-type-arguments` (4 errors)

**Files:**
- `middleware/filesystem.py:388`
- `middleware/memory.py:157`
- `middleware/skills.py:595`
- `middleware/subagents.py:482`

**Issue:** Middleware classes parameterize `AgentMiddleware` with 3 type arguments (`AgentMiddleware[StateT, ContextT, ResponseT]`), but the upstream base class only accepts 0–2 type parameters.

**Example:**
```python
class FilesystemMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
#                                                           ^^^^^^^^^ too many
```

**Requires:** Upstream langchain fix — `AgentMiddleware` needs a third type parameter (`ResponseT`), or the middleware generics need to be restructured.

---

### 2. `before_agent` / `abefore_agent` Signature Mismatch — `invalid-method-override` (4 errors)

**Files:**
- `middleware/memory.py:305` (`before_agent`)
- `middleware/memory.py:334` (`abefore_agent`)
- `middleware/skills.py:722` (`before_agent`)
- `middleware/skills.py:757` (`abefore_agent`)

**Issue:** The base class defines:
```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
```

But subclasses add `config: RunnableConfig` as a third parameter:
```python
def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None
```

This violates the Liskov Substitution Principle. The `config` parameter is needed to create `ToolRuntime` for backend factories.

**Requires:** Upstream langchain fix — either:
1. Add `config: RunnableConfig` to the base class signature
2. Provide config via `runtime.config`
3. Use `get_config()` internally (workaround used in `SummarizationMiddleware`)

---

### 3. TypedDict Spread Loses Required Keys — `missing-typed-dict-key` (6 errors)

**Files:**
- `graph.py:174` (3 errors: missing `name`, `description`, `system_prompt`)
- `graph.py:214` (3 errors: missing `name`, `description`, `system_prompt`)

**Issue:** When spreading a `TypedDict` with `{**GENERAL_PURPOSE_SUBAGENT, "model": model, ...}`, the type checker cannot infer that the spread source provides the required keys:
```python
general_purpose_spec: SubAgent = {
    **GENERAL_PURPOSE_SUBAGENT,  # ty can't see these keys through spread
    "model": model,
    "tools": tools or [],
    "middleware": gp_middleware,
}
```

**Fix options:**
1. Construct the `SubAgent` explicitly with all keys
2. Use `TypedDict` unpacking (`Unpack`) if supported
3. This is a known ty limitation — may improve in future versions

---

### 4. `BackendContext` Type Variance — `invalid-argument-type` (2 errors)

**Files:**
- `backends/store.py:151` (2 errors: `state` and `runtime` args)

**Issue:** `BackendContext(state=state, runtime=self.runtime)` fails because:
- `state` is `Any | None` but `StateT` is bounded by `StateLike`
- `self.runtime` is `Unknown | ToolRuntime[...]` but expected `Runtime[None]`

**Fix options:**
1. Add a `cast()` for the `state` parameter
2. Narrow the `runtime` type with an `isinstance` check

---

### 5. Legacy Subagent API Null Safety — `invalid-argument-type` (1 error)

**File:** `middleware/subagents.py:592`

**Issue:** `default_model` can be `None` but `_get_subagents_legacy` expects `str | BaseChatModel`:
```python
subagent_specs = _get_subagents_legacy(
    default_model=default_model,  # Could be None
    ...
)
```

**Fix:** Add a None check before the call, or make the parameter optional in `_get_subagents_legacy`.

---

## Suppressed Issues (13 `# ty: ignore` comments)

### 6. `BackendFactory` Generic Type Variance — `invalid-argument-type` (3 suppressions)

**Files:**
- `middleware/memory.py:213`
- `middleware/skills.py:663`
- `middleware/summarization.py:243`

**Issue:** `self._backend` is typed as `BackendProtocol | BackendFactory` where `BackendFactory = Callable[[ToolRuntime], BackendProtocol]`. When calling the factory with a specific state type (e.g., `ToolRuntime[None, MemoryState]`), ty complains about type variance.

**Fix:** Make `BackendFactory` generic or use `Callable[[ToolRuntime[Any, Any]], BackendProtocol]`.

---

### 7. TypedDict Spread Operator Loses Type Information (6 suppressions)

**Files:**
- `backends/composite.py:244, 261, 284, 301, 322, 344`

**Issue:** Spreading a `GrepMatch` or `FileInfo` TypedDict with `{**m, "path": new_path}` produces `dict[str, Unknown]` instead of the TypedDict type:
```python
return [{**m, "path": f"{prefix}{m['path']}"} for m in raw]  # ty: ignore
```

**Fix options:**
1. Use explicit TypedDict construction: `GrepMatch(path=..., line=m["line"], ...)`
2. Accept as a known ty/type-system limitation

---

### 8. `BackendProtocol` → `SandboxBackendProtocol` Type Narrowing — `unresolved-attribute` (2 suppressions)

**Files:**
- `middleware/filesystem.py:844` (`execute`)
- `middleware/filesystem.py:877` (`aexecute`)

**Issue:** After `_supports_execution(resolved_backend)` returns `True`, ty doesn't narrow the type from `BackendProtocol` to `SandboxBackendProtocol`, so `execute()`/`aexecute()` are unresolved.

**Fix:** Use `TypeGuard` for proper narrowing:
```python
def _supports_execution(backend: BackendProtocol) -> TypeGuard[SandboxBackendProtocol]:
    ...
```

---

### 9. `SystemMessage` Constructor Overload Mismatch — `no-matching-overload` (1 suppression)

**File:** `graph.py:266`

**Issue:** `SystemMessage(content=new_content)` where `new_content` is built from `content_blocks` doesn't match any constructor overload.

**Requires:** Upstream langchain-core fix — make `content_blocks` return type compatible with the constructor.

---

### 10. `ToolCall` vs `dict[str, Any]` — `invalid-argument-type` (1 suppression)

**File:** `middleware/summarization.py:483`

**Issue:** `_truncate_tool_call` expects `dict[str, Any]` but receives `ToolCall` (a TypedDict). Structurally compatible but ty is strict.

**Fix:** Change the method signature to accept `ToolCall | dict[str, Any]`.

---

## Files Summary

| File | Unsuppressed | Suppressed | Total |
|------|-------------|------------|-------|
| `graph.py` | 6 | 1 | 7 |
| `middleware/skills.py` | 3 | 1 | 4 |
| `middleware/memory.py` | 3 | 1 | 4 |
| `middleware/filesystem.py` | 1 | 2 | 3 |
| `middleware/subagents.py` | 2 | 0 | 2 |
| `middleware/summarization.py` | 0 | 2 | 2 |
| `backends/store.py` | 2 | 0 | 2 |
| `backends/composite.py` | 0 | 6 | 6 |

## What Needs Upstream Fixes

The following require changes in langchain/langgraph before they can be resolved here:

1. **`AgentMiddleware` type parameters** — needs `ResponseT` support (issues #1)
2. **`before_agent`/`abefore_agent` signatures** — needs `config` parameter (issue #2)
3. **`SystemMessage` constructor overloads** — needs `content_blocks` compatibility (issue #9)

## What Can Be Fixed Locally

1. **TypedDict spread in `graph.py`** — explicit construction instead of spread (issue #3)
2. **`BackendContext` types in `store.py`** — add casts or narrowing (issue #4)
3. **Null check in `subagents.py`** — guard `default_model` against `None` (issue #5)
4. **`BackendFactory` variance** — widen the type alias (issue #6)
5. **TypedDict spread in `composite.py`** — explicit construction (issue #7)
6. **`TypeGuard` for execution support** — use `TypeGuard[SandboxBackendProtocol]` (issue #8)
7. **`_truncate_tool_call` signature** — accept `ToolCall | dict[str, Any]` (issue #10)
