# Python extensions

Python extensions customize the agent server without modifying `dcode`. They
are experimental and load only when `DEEPAGENTS_CODE_EXPERIMENTAL=1` is set
before starting `dcode`.

An installed plugin declares one or more Python entry files in its manifest:

```json
{
  "name": "shared-memory",
  "version": "1.0.0",
  "extensions": {
    "com.langchain.deepagents.code": {
      "pythonExtensions": "./extension.py"
    }
  }
}
```

The entry file exposes an `extension` setup function:

```python
from deepagents.backends import StoreBackend
from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    """Add shared storage under `/memories/`.

    Args:
        d: The extension setup API.
    """
    d.register_backend_route(
        "/memories/",
        StoreBackend(namespace=lambda _runtime: ("filesystem",)),
    )
```

Install and enable the plugin through the normal plugin commands, then run
`/reload`. Dcode installs each plugin into a versioned cache and reloads the
agent server from the enabled plugin set. A separately managed remote agent
server must be restarted or redeployed by its operator.

The `/memories/` route above makes shared storage available to model file
operations. It does not automatically move dcode's built-in `AGENTS.md` memory.
An extension that wants that content in the model's prompt must also register
middleware that reads the shared location.

## Supported registrations

| Method | Purpose |
| --- | --- |
| `register_middleware(class_or_instance)` | Add LangChain `AgentMiddleware`. Classes must have a zero-argument constructor; use an instance otherwise. |
| `register_tool(function_or_tool)` | Expose a callable or `BaseTool` to the model. |
| `register_backend_route(prefix, storage)` | Make a `BackendProtocol` storage provider available under a virtual path. |
| `on_shutdown(callback)` | Release session resources when the agent server stops. Sync and async callbacks are supported. |

Registration is allowed only while `extension(d)` runs. Dcode eagerly imports
every enabled extension and waits for every setup function before building the
agent and its storage routes. Setup may be synchronous or asynchronous.

Do not open long-lived connections or start background tasks during module
import. A storage provider may connect lazily on first use. If setup opens a
session resource, register an idempotent `on_shutdown` callback to release it.

The API exposes read-only session context (`d.cwd`, `d.mode`, and `d.path`), not
mutable TUI or thread state. Custom slash commands are not part of this API.

## Storage routes

`BackendProtocol` is the SDK name for a storage implementation. Route prefixes
are lowercase absolute paths with leading and trailing slashes, such as
`/memories/` or `/company/knowledge/`. Traversal, empty segments, backslashes,
URL query or fragment syntax, and overlaps with dcode's internal routes are
rejected.

Routed content is available through the model's file tools. Shell `execute`
remains attached to the default local or sandbox storage, so shell commands
cannot see virtual routed content.

The first extension registration of a route prefix wins. Built-in tools,
middleware, and internal storage routes always take precedence over extension
units with conflicting names or paths.

## Packaging, discovery, and trust

User extensions are plugin components rather than loose files. The plugin owns
their stable identity, version, installation, updates, and durable data
directory. Dcode accepts only manifest paths beginning with `./`, resolves them
inside the installed plugin snapshot, and rejects traversal, absolute paths,
symlink escapes, missing files, and non-Python files.
Python entries are ignored when the plugin manifest has no non-empty `version`.

Sources load in this order:

| Source | Version and trust behavior |
| --- | --- |
| Enabled installed plugins | Identified by `name@marketplace`, loaded from the versioned plugin cache. Installing and enabling the plugin authorizes its code. |
| `.deepagents/extensions/` | Project-controlled and normally versioned with the project; never scanned before project trust. |

The project directory is the local development escape hatch. Its scan is
shallow: direct `*.py` files are extensions, and a direct subdirectory is a
package extension when it contains `__init__.py` or `extension.py`.

Project extensions execute arbitrary Python with the user's process privileges.
Interactive launches ask before loading them and can remember the canonical
project path. Headless or CI launches can opt in for one run with
`--trust-project-extensions`. Only trust projects you control.

## Configuration

```toml
# ~/.deepagents/config.toml
[extensions]
enabled = true
trust = "ask"  # ask | always | never
```

Environment overrides:

- `DEEPAGENTS_CODE_EXTENSIONS` enables or disables all extension loading.
- `DEEPAGENTS_CODE_EXTENSIONS_TRUST` overrides the project trust policy.

There is no separate user extension path setting. Use an installed plugin for
user-wide extensions and the trusted project directory for project development.

## Failure and security behavior

Each setup function is transactional. If import or initialization fails, all of
that extension's partial registrations are removed and later extensions still
load. Failures are written to the debug log.

Extension tools are appended after built-in tools and are not automatically
added to dcode's human-approval map. An extension that performs sensitive work
must enforce its own approval or policy through middleware.

Every registration retains its plugin ID, version, installed root, and source
scope. This attribution tells dcode which extension added a storage route; it
does not prove that an arbitrary storage implementation is safe. A provider can
access the network or host files, so install plugins only from trusted sources
and choose shared namespaces that preserve tenant and user isolation.

See `examples/extensions/memory_store.py` for a shared storage route.
