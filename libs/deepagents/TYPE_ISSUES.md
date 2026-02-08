# Type Issues Requiring Upstream Fixes

This document tracks type safety issues discovered during ty type checking that need proper fixes (likely in langchain or type definitions).

## Summary

- **4 errors** requiring langchain fixes (method override issues)
- **15 suppressions** (`# ty: ignore`) in deepagents needing proper solutions

## Recent Changes

Middlewares have been updated to be properly generic on `ContextT` and `ResponseT` from `langchain.agents.middleware.types`:
- `MemoryMiddleware[MemoryState, ContextT, ResponseT]`
- `SkillsMiddleware[SkillsState, ContextT, ResponseT]`
- `FilesystemMiddleware[FilesystemState, ContextT, ResponseT]`
- `SubAgentMiddleware[Any, ContextT, ResponseT]`

This improves type safety for `wrap_model_call`/`awrap_model_call` methods that now properly type `ModelRequest[ContextT]` and `ModelResponse[ResponseT]`.

---

## Errors (Requiring langchain Fixes)

### 1. `AgentMiddleware.before_agent` / `abefore_agent` Signature Mismatch

**Files affected:** `memory.py:305,334`, `skills.py:593,628`

**Issue:** The base class `AgentMiddleware` defines:
```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None
```

But subclasses need an additional `config: RunnableConfig` parameter to create `ToolRuntime` for backend factories:
```python
def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None
```

**Note:** The `Runtime` class intentionally does not include `config` (see `langgraph/runtime.py:34-38`). Config must be accessed via `get_config()` or injected as a parameter.

**Proposed fix in langchain:** Either:
1. Add `config: RunnableConfig` to the base class signature
2. Provide `config` via `runtime` (e.g., `runtime.config`)
3. Or use `get_config()` in the implementation (current workaround in `SummarizationMiddleware`)

---

## Suppressions (Needing Proper Type Fixes)

### 2. `BackendFactory` Generic Type Variance

**Files affected:**
- `memory.py:213`
- `skills.py:540`
- `summarization.py:196`

**Issue:** `self._backend` is typed as `BackendProtocol | BackendFactory` where `BackendFactory = Callable[[ToolRuntime], BackendProtocol]`. When calling the factory with a specific state type like `ToolRuntime[None, MemoryState]`, the type checker complains because the factory expects `ToolRuntime[None, dict[Unknown, Unknown]]`.

**Root cause:** The `BACKEND_TYPES` type alias in `protocol.py` uses a narrow type for the callable:
```python
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
```

**Proposed fix:** Make `BackendFactory` generic or use a more flexible type:
```python
BackendFactory: TypeAlias = Callable[[ToolRuntime[Any, Any]], BackendProtocol]
```

---

### 3. `Runtime` vs `ToolRuntime` Type Mismatch

**Files affected:** `filesystem.py:896,944`

**Issue:** `request.runtime` from `ModelRequest` is typed as `Runtime[ContextT]` but `_get_backend` expects `ToolRuntime[Any, Any]`.

**Proposed fix in langchain:** Either:
- Type `ModelRequest.runtime` as `ToolRuntime` instead of `Runtime`
- Or make `_get_backend` accept `Runtime[Any, Any]`

---

### 4. `BackendProtocol` vs `SandboxBackendProtocol` Type Narrowing

**Files affected:** `filesystem.py:819,852`

**Issue:** After a runtime check `_supports_execution(resolved_backend)`, the type checker doesn't narrow `resolved_backend` from `BackendProtocol` to `SandboxBackendProtocol`, so `execute()`/`aexecute()` methods are unresolved.

**Proposed fix:** Use `isinstance` check instead of custom function, or use `TypeGuard`:
```python
def _supports_execution(backend: BackendProtocol) -> TypeGuard[SandboxBackendProtocol]:
    ...
```

---

### 5. TypedDict Spread Operator Loses Type Information

**Files affected:** `composite.py:250,267,290,307,328,350`

**Issue:** When spreading a `GrepMatch` or `FileInfo` TypedDict with `{**m, "path": new_path}`, the type checker infers `dict[str, Unknown]` instead of preserving the TypedDict type.

**Example:**
```python
# This loses the GrepMatch type:
return [{**m, "path": f"{prefix}{m['path']}"} for m in raw]
```

**Proposed fix:** This is a fundamental limitation of TypedDict spread. Options:
1. Use explicit construction: `GrepMatch(path=..., line=m["line"], text=m["text"])`
2. Or accept this as a known limitation and keep the suppression

---

### 6. `SystemMessage` Constructor Overload Mismatch

**Files affected:** `graph.py:235`

**Issue:** `SystemMessage(content=new_content)` where `new_content` includes items from `content_blocks` property doesn't match any overload signature because `content_blocks` returns a complex union type.

**Proposed fix in langchain-core:** Either:
- Make `content_blocks` return type compatible with the constructor
- Or add an overload that accepts the `content_blocks` return type

---

### 7. `ToolCall` vs `dict[str, Any]` Compatibility

**Files affected:** `summarization.py:436`

**Issue:** `_truncate_tool_call` expects `dict[str, Any]` but receives `ToolCall` (a TypedDict). While structurally compatible, the type checker is strict about this.

**Proposed fix:** Change the method signature to accept `ToolCall | dict[str, Any]` or make `ToolCall` a proper subtype.

---

## Files Changed

The following files have `# ty: ignore` comments that should be addressed:

| File | Lines | Issue |
|------|-------|-------|
| `deepagents/middleware/memory.py` | 213 | BackendFactory variance |
| `deepagents/middleware/skills.py` | 540 | BackendFactory variance |
| `deepagents/middleware/summarization.py` | 196, 436 | BackendFactory variance, ToolCall type |
| `deepagents/middleware/filesystem.py` | 819, 852, 896, 944 | execute/aexecute resolution, Runtime vs ToolRuntime |
| `deepagents/backends/composite.py` | 250, 267, 290, 307, 328, 350 | TypedDict spread |
| `deepagents/graph.py` | 235 | SystemMessage overload |
