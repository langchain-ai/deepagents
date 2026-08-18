# Extensions

Extensions let you add middleware, tools, and slash commands to `dcode` without
modifying its source. An extension is one Python file (or a package directory)
exposing a single factory:

```python
from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    def word_count(text: str) -> str:
        """Count the words in some text."""
        return str(len(text.split()))

    d.register_tool(word_count)
    d.register_command("hello", lambda ctx: f"Hello from {ctx.cwd}")
```

Drop that file in `~/.deepagents/extensions/` and restart `dcode`. The tool is
offered to the model and `/hello` works in the REPL.

The factory may be `async def`; it is awaited before the session starts, so
one-time setup finishes before anything else runs.

## Where extensions are loaded from

Sources resolve in this order; within a directory, entries load alphabetically.

| Source | Recipe | Trust-gated |
| --- | --- | --- |
| `~/.deepagents/extensions/` | directory scan | No |
| Paths from `[extensions].paths` | file as-is; directory scanned | No |
| `.deepagents/extensions/` (project) | directory scan | **Yes** |

The directory scan is deliberately shallow: a direct `*.py` file is an
extension, and a subdirectory is an extension when it contains `__init__.py` or
`extension.py`. There is no deeper recursion, so helper modules next to an entry
file are just imports.

## Registration verbs

| Verb | Accepts |
| --- | --- |
| `register_middleware(cls_or_instance)` | A LangChain `AgentMiddleware` subclass or instance. Interception semantics are LangChain's, so existing LangChain middleware loads unmodified. A class is instantiated by dcode; pass an instance when construction needs arguments. |
| `register_tool(fn_or_tool)` | A plain function (schema derived from its signature and docstring) or a `BaseTool` when you want to declare the schema yourself. |
| `register_command(name, handler, description=...)` | A `/name` handler receiving a `CommandContext` (`args`, `cwd`, `mode`). Sync or async. |

`d.on_shutdown(callback)` registers deterministic teardown for session-scoped
resources. `d.cwd`, `d.mode`, and `d.path` describe the session the factory is
running in.

Registrations are accepted only while `extension(d)` is running. The agent graph
is compiled from that completed registry, so retaining `d` and calling a
registration verb later raises `ExtensionError`.

Collisions resolve by load order, never by scope: the first registration of a
tool name wins (later ones are logged and dropped), and a duplicate command name
is registered under a suffixed name rather than replacing the original. Every
registered unit carries provenance (`SourceInfo`: path, source, scope, origin),
which `/extensions` displays.

## Trust

Project extensions are the first project-level resource that *executes* code, so
`.deepagents/extensions/` is scanned only after a trust decision:

1. A persisted decision for the working directory (stored under
   `~/.deepagents/.state/extension_trust.json`).
2. An interactive prompt at startup ("allow once" or "always allow").
3. The configured default policy for non-interactive runs.

`--trust-project-extensions` grants trust for one run, which is what headless and
CI invocations should use.

Extension code runs in-process with your privileges, and extension-registered
tools are not added to the approval (HITL) interrupt map — install only
extensions you trust, and gate risky work inside the extension itself (see
`examples/extensions/audit_log.py`).

## Configuration

```toml
# ~/.deepagents/config.toml
[extensions]
enabled = true
paths = ["~/work/dcode-extensions", "~/scratch/one-off.py"]
trust = "ask"  # ask | always | never
```

Environment overrides: `DEEPAGENTS_CODE_EXTENSIONS=0` disables loading,
`DEEPAGENTS_CODE_EXTENSIONS_PATHS` is a colon-separated path list, and
`DEEPAGENTS_CODE_EXTENSIONS_TRUST` overrides the trust policy.

## Failure handling

A malformed extension is reported and skipped; it never takes down the agent
loop. Run `/extensions` to see what loaded and what failed, and set
`DEEPAGENTS_CODE_DEBUG=1` for tracebacks.

## Reference extensions

See `examples/extensions/`: `audit_log.py` (permission-gate middleware),
`scratchpad.py` (a stateful tool with teardown), and `standup.py` (a command).
