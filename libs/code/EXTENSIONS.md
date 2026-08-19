# Python extensions

Python extensions customize the agent server without modifying `dcode`. An
extension is a Python file or package exposing one factory:

```python
from deepagents.backends import StoreBackend
from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    """Mount shared agent memory.

    Args:
        d: The factory-scoped extension API.
    """
    d.register_backend_route(
        "/memories/",
        StoreBackend(namespace=lambda _runtime: ("filesystem",)),
    )
```

Place the file in `~/.deepagents/extensions/` and restart `dcode`. Filesystem
operations under `/memories/` now use the LangGraph store, so they persist
across threads. A custom `BackendProtocol` can use Postgres or another shared
store while keeping the same route contract.

## Supported registrations

| Method | Purpose |
| --- | --- |
| `register_middleware(class_or_instance)` | Add LangChain `AgentMiddleware`. Classes must have a zero-argument constructor; use an instance otherwise. |
| `register_tool(function_or_tool)` | Expose a callable or `BaseTool` to the model. |
| `register_backend_route(prefix, backend)` | Mount a `BackendProtocol` under a virtual filesystem prefix. |
| `on_shutdown(callback)` | Release session resources when the agent server stops. Sync and async callbacks are supported. |

Registration is allowed only while `extension(d)` runs. The factory itself may
be synchronous or asynchronous. Async initialization and shutdown run on the
same persistent server event loop, so loop-bound clients remain valid for the
session.

The API deliberately exposes read-only session context (`d.cwd`, `d.mode`, and
`d.path`), not mutable TUI or thread state. Custom slash commands are not part
of this API.

## Backend routes

Route prefixes are lowercase absolute paths with leading and trailing slashes,
for example `/memories/` or `/company/knowledge/`. Traversal, empty segments,
backslashes, URL query/fragment syntax, and overlaps with dcode's internal
routes are rejected.

Routed content is available through the model's filesystem tools. Shell
`execute` remains attached to the default local or sandbox backend, so shell
commands cannot see virtual route contents.

The first extension registration of a route prefix wins. Built-in tools,
middleware, and internal backend routes always take precedence over extension
units with conflicting names or paths.

## Discovery and trust

Sources load in this order; files within each directory load alphabetically:

| Source | Trust behavior |
| --- | --- |
| `~/.deepagents/extensions/` | User-controlled; loaded directly. |
| `[extensions].paths` | Explicitly configured by the user; loaded directly. |
| `.deepagents/extensions/` | Project-controlled; never scanned before trust. |

A directory scan is shallow. Direct `*.py` files are extensions. A direct
subdirectory is a package extension when it contains `__init__.py` or
`extension.py`; helper modules below that package are normal imports.

Project extensions execute arbitrary Python with the user's process privileges.
Interactive launches ask before loading them and can remember the canonical
project path. Headless or CI launches can opt in for one run with
`--trust-project-extensions`. Only trust projects you control.

## Configuration

```toml
# ~/.deepagents/config.toml
[extensions]
enabled = true
paths = ["~/work/dcode-extensions", "~/scratch/one-off.py"]
trust = "ask"  # ask | always | never
```

Environment overrides:

- `DEEPAGENTS_CODE_EXTENSIONS` enables or disables all extension loading.
- `DEEPAGENTS_CODE_EXTENSIONS_PATHS` replaces `paths` using the platform path
  separator (`:` on POSIX and `;` on Windows).
- `DEEPAGENTS_CODE_EXTENSIONS_TRUST` overrides the project trust policy.

## Failure and security behavior

Each factory is transactional. If import or initialization fails, all of that
extension's partial registrations are removed and later extensions still load.
Failures are written to the debug log.

Extension tools are appended after built-in tools and are not automatically
added to dcode's human-approval map. An extension that performs sensitive work
must enforce its own approval or policy through middleware. Backend routes can
expose persistent or shared data to model filesystem operations; choose
namespaces that maintain the tenant or user isolation your deployment needs.

See `examples/extensions/memory_store.py` for a shared memory route.
