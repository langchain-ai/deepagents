## Summary

1. Deprecate the backend factory pattern in favor of backend instances

before
```py
agent = create_deep_agent(backend=lambda rt: StateBackend(rt))
```

after
```py
agent = create_deep_agent(backend=StateBackend())
```

2. State backend writes without using `files_update` artifacts on `EditResult` and `WriteResult`

before
```py
def my_node():
    # responsibility of the caller to update state, backend.write doesn't actually write to state
    result = backend.write(...)
    if (update := result.files_update):
        return Command(update=update)
```

after
```py
def my_node():
    # backend.write writes to state
    result = backend.write(...)
```

## Core Changes

- **StateBackend**: Uses `get_config()` + `CONFIG_KEY_READ`/`CONFIG_KEY_SEND` instead of `runtime.state`. `runtime` arg deprecated with warning. Clear error when used outside graph context.
- **StoreBackend**: Uses `get_store()`/`get_runtime()` instead of `runtime.store`. New `store=` kwarg for explicit store injection. `runtime` arg deprecated.
- **WriteResult/EditResult**: `files_update` field removed — state writes go directly through `CONFIG_KEY_SEND`.
- **Middleware**: Write/edit tools return plain strings (no `Command` wrapping). Eviction writes go through `CONFIG_KEY_SEND`. `contextvars.copy_context()` fix for glob's `ThreadPoolExecutor`.
- **CompositeBackend**: Removed `files_update` sync code.
- **Summarization**: `_offload_to_backend` returns `str | None` instead of tuple.
- **Factory pattern**: `_get_backend` callable path deprecated with warning. Default backend is `StateBackend()` instance.

## Deprecations

All deprecations target **v0.7** for removal.

### Backend construction

| Deprecated | Replacement | Notes |
|---|---|---|
| `StateBackend(runtime=...)` | `StateBackend()` | State is now accessed via `get_config()` internally. The `runtime` parameter is accepted but ignored. |
| `StoreBackend(runtime=...)` | `StoreBackend()` or `StoreBackend(store=...)` | Store/context obtained via `get_store()` / `get_runtime()`. New `store=` kwarg available for explicit injection. |
| `StoreBackend()` without `namespace` | `StoreBackend(namespace=lambda ctx: (...))` | Explicit namespace required. |
| `create_deep_agent(backend=lambda rt: ...)` | `create_deep_agent(backend=StateBackend())` | Passing a callable factory as `backend` is deprecated — pass a `BackendProtocol` instance directly. |

### Backend protocol methods

| Deprecated | Replacement |
|---|---|
| `ls_info` / `als_info` | `ls` / `als` |
| `glob_info` / `aglob_info` | `glob` / `aglob` |
| `grep_raw` / `agrep_raw` | `grep` / `agrep` |

Subclass implementations that override the old names will still work (dispatched automatically), but should be renamed.

### Return types

| Deprecated | Replacement |
|---|---|
| `backend.read()` returning plain `str` | Return a `ReadResult` instead |
| `FileData` with `list[str]` content | Content should be stored as a plain `str` |
| Store items with `list[str]` content | Content should be stored as a plain `str` |

### Removed fields

| Removed | Migration |
|---|---|
| `WriteResult.files_update` | No action needed — `backend.write()` now writes to state directly via `CONFIG_KEY_SEND`. Callers no longer need to pass updates through `Command(update=...)`. |
| `EditResult.files_update` | Same as above. |

## Tests

- New `TestStateBackendConfigKeys` class with write/read/edit/ls, backward compat lambda, deprecation warnings, explicit store, `get_store` fallback, and factory deprecation tests.
- All existing tests updated to use `state_config_context()` helper or `create_deep_agent` with fake models.
