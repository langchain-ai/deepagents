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

The entry file exposes an async `extension` setup function. See
[`memory_store.py`](./examples/extensions/memory_store.py) for an example that
registers a shared `/memories/` storage route.

Install and enable the plugin through the normal plugin commands, then run
`/restart`. Backend composition happens while the agent graph is built, so
backend route changes cannot be applied by `/reload`. A separately managed
remote agent server must be restarted or redeployed by its operator.

The `/memories/` route makes shared storage available to model file
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

The factory must be declared with `async def`; dcode awaits every factory before
building the agent. The registrar remains valid for callbacks that outlive the
factory. Runtime tools appear on the next model request, backend routes update
the registry but require `/restart`, and middleware also takes effect after the
next server rebuild. `/extensions` reports when a restart is required.

Do not open long-lived connections or start background tasks during module
import. A storage provider may connect lazily on first use. If setup opens a
session resource, register an idempotent `on_shutdown` callback to release it.

The API exposes read-only session context (`d.cwd`, `d.mode`, `d.has_ui`, and
`d.path`), not mutable TUI or thread state. Custom slash commands are not part
of this API. Run `/extensions` to inspect registered units and their source.

## Storage routes

`BackendProtocol` is the SDK name for a storage implementation. Route prefixes
are lowercase absolute paths with leading and trailing slashes, such as
`/memories/` or `/company/knowledge/`. Traversal, empty segments, backslashes,
URL query or fragment syntax, and overlaps with dcode's internal routes are
rejected.

Routed content is available through the model's file tools. Shell `execute`
remains attached to the default local or sandbox storage, so shell commands
cannot see virtual routed content.

Local agents may mount `FilesystemBackend` and `LocalShellBackend`. Sandboxed
agents reject direct instances of either class, including subclasses, because
they would expose host storage while `execute` still runs inside the sandbox.
Other `BackendProtocol` implementations such as `StateBackend`, `StoreBackend`,
and `ContextHubBackend` are supported. This check is intentionally shallow:
dcode does not recursively inspect custom or composite backend wrappers, whose
authors own their isolation contract.

The first extension registration of a route prefix or unit name wins. Extension
tools and middleware replace same-named built-ins. A route that is a parent or
child of dcode's artifact or conversation-history storage fails agent
construction or runtime registration immediately; internal routes cannot be
replaced.

## Packaging, discovery, and trust

For quick user-wide extensions, place Python files in
`~/.deepagents/extensions/`. Installed plugins remain the preferred
distribution mechanism: they provide stable identity, versions, updates, and a
durable data directory. Dcode resolves plugin manifest entries inside the
installed snapshot and rejects traversal, absolute paths, symlink escapes,
missing files, and non-Python files. Entries without a plugin version are
ignored. Installed Python distributions may also expose module entry points in
the `dcode.extensions` group.

Sources load in this order:

| Source | Version and trust behavior |
| --- | --- |
| `~/.deepagents/extensions/` | User-owned loose files; implicitly authorized. |
| `[extensions].extra_files` and `extra_dirs` | Explicit paths in trusted user configuration. |
| `-e/--extension PATH` | Temporary file or directory authorized for one run. |
| Enabled installed plugins | Identified by `name@marketplace`, loaded from the versioned plugin cache. Installing and enabling the plugin authorizes its code. |
| `dcode.extensions` entry points | Modules from installed Python distributions. |
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
extra_files = ["~/src/policy.py"]
extra_dirs = ["~/src/company-extensions"]
```

Environment overrides:

- `DEEPAGENTS_CODE_EXTENSIONS` enables or disables all extension loading.
- `DEEPAGENTS_CODE_EXTENSIONS_TRUST` overrides the project trust policy.

Relative extra paths resolve from `~/.deepagents/`. Repeat `-e PATH` to add
multiple temporary sources for one invocation.

## Failure and security behavior

Each setup function is transactional. If import or initialization fails, all of
that extension's partial registrations are removed and later extensions still
load. Failures are written to the debug log.

Extension tools override same-named built-ins and are not automatically added to
dcode's human-approval map. An extension that performs sensitive work must
enforce its own approval or policy through middleware.

Every registration retains its plugin ID, version, installed root, and source
scope. This attribution tells dcode which extension added a storage route; it
does not prove that an arbitrary storage implementation is safe. A provider can
access the network or host files, so install plugins only from trusted sources
and choose shared namespaces that preserve tenant and user isolation.
