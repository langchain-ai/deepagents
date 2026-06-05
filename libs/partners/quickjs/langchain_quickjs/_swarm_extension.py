"""Swarm as an interpreter extension.

A first-party ``InterpreterExtension`` that makes the swarm table API
importable from guest JS as ``import { create, run, rows } from "swarm"`` —
without the consumer wiring up any PTC tools.

The bundled swarm scripts (vendored from ``langchain-ai/langchain-skills``,
``config/skills/swarm``) call five top-level host functions: ``__swarmTask``
(subagent dispatch) and ``__swarmGlob`` / ``__swarmReadFile`` /
``__swarmWriteFile`` / ``__swarmEditFile`` (table persistence). This
extension registers those — dispatch via the shared ``build_swarm_dispatch``
closure, file ops via ``ctx.backend`` — then installs the scripts as the
``swarm`` module. No ``globalThis.tools`` involved.

Usage:

    from langchain_quickjs import CodeInterpreterMiddleware, swarm

    CodeInterpreterMiddleware(
        extensions=[swarm(subagents=[screener, classifier], default_model=model)],
        backend=store_backend,  # required for table persistence (glob/read/write)
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_quickjs._swarm_scripts import swarm_module_scope
from langchain_quickjs._swarm_task import build_swarm_dispatch

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel

    from langchain_quickjs._extensions import ExtensionContext
    from langchain_quickjs._swarm_task import SwarmDispatch, SwarmSubAgent

# Top-level host symbols the scripts call directly (they `declare function
# __swarm*`). Fixed, extension-owned identifiers — never guest-interpolated.
_DISPATCH_SYMBOL = "__swarmTask"
_GLOB_SYMBOL = "__swarmGlob"
_READ_SYMBOL = "__swarmReadFile"
_WRITE_SYMBOL = "__swarmWriteFile"
_EDIT_SYMBOL = "__swarmEditFile"

_SYSTEM_PROMPT = """\
## swarm

Process many independent items in parallel from inside the code interpreter.
`create` builds a table handle, `run` fans work out across rows to subagents
and merges results back, and `rows` reads them for aggregation.

```js
import { create, run, rows } from "swarm";
const table = await create({ tasks: records });
const stats = await run(table, { instruction: "Triage {text}" });
const all = await rows(table);
```

One row = one unit of work; swarm batches automatically. Use it instead of
dispatching tasks one at a time.
"""


def _require_dict(payload: Any, call: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        msg = f"{call} expects an options object"
        raise TypeError(msg)
    return payload


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"swarm host call: {key!r} must be a string"
        raise TypeError(msg)
    return value


def _register_swarm_host_functions(  # noqa: C901 — flat registration of 5 adapters
    ctx: ExtensionContext, dispatch: SwarmDispatch
) -> None:
    """Register the five host functions the swarm scripts read off `tools`.

    Dispatch goes to ``dispatch``; file ops go through ``ctx.backend`` (which
    raises a clear error if no backend is configured). All guest payloads are
    validated as dicts with the expected string keys before use.
    """

    def _backend() -> BackendProtocol:
        backend = ctx.backend
        if backend is None:
            msg = (
                "swarm table persistence requires a backend; pass "
                "`backend=` (or `skills_backend=`) to CodeInterpreterMiddleware"
            )
            raise RuntimeError(msg)
        return backend

    @ctx.function(name=_DISPATCH_SYMBOL)
    async def _swarm_task(payload: Any = None) -> str:
        # Scripts call with snake_case args (subagent_type, etc.).
        args = _require_dict(payload, "swarmTask")
        return await dispatch(
            args.get("description"),
            args.get("subagent_type"),
            args.get("response_schema"),
            args.get("mode"),
        )

    @ctx.function(name=_GLOB_SYMBOL)
    async def _glob(payload: Any = None) -> str:
        args = _require_dict(payload, "glob")
        result = await _backend().aglob(_require_str(args, "pattern"))
        if result.error:
            raise RuntimeError(result.error)
        # The script JSON.parses this and reads `.path` off each entry.
        return json.dumps([dict(m) for m in (result.matches or [])])

    @ctx.function(name=_READ_SYMBOL)
    async def _read_file(payload: Any = None) -> str:
        args = _require_dict(payload, "readFile")
        offset = args.get("offset", 0)
        limit = args.get("limit", 2000)
        result = await _backend().aread(
            _require_str(args, "file_path"),
            offset if isinstance(offset, int) else 0,
            limit if isinstance(limit, int) else 2000,
        )
        if result.error:
            raise RuntimeError(result.error)
        return result.file_data["content"] if result.file_data else ""

    @ctx.function(name=_WRITE_SYMBOL)
    async def _write_file(payload: Any = None) -> str:
        args = _require_dict(payload, "writeFile")
        result = await _backend().awrite(
            _require_str(args, "file_path"), _require_str(args, "content")
        )
        # The script checks for "already exists" in the returned string and
        # falls back to editFile, so surface the error verbatim, don't raise.
        if result.error:
            return result.error
        return result.path or ""

    @ctx.function(name=_EDIT_SYMBOL)
    async def _edit_file(payload: Any = None) -> str:
        args = _require_dict(payload, "editFile")
        result = await _backend().aedit(
            _require_str(args, "file_path"),
            _require_str(args, "old_string"),
            _require_str(args, "new_string"),
        )
        if result.error:
            raise RuntimeError(result.error)
        return result.path or ""


@dataclass(frozen=True)
class SwarmExtension:
    """Interpreter extension exposing the swarm table API to guest JS.

    Construct via the :func:`swarm` factory. ``on_setup`` registers the
    dispatch + file-op host functions and installs the swarm scripts as the
    ``swarm`` module — on every fresh context.
    """

    dispatch: SwarmDispatch
    system_prompt: str | None = _SYSTEM_PROMPT

    def on_setup(self, ctx: ExtensionContext) -> None:
        # Register the host functions the scripts call by name, then install
        # the scripts. The symbols are plain globals (ctx.register puts them
        # on globalThis), so the scripts' `__swarmGlob(...)` etc. resolve.
        _register_swarm_host_functions(ctx, self.dispatch)
        ctx.module(swarm_module_scope())


def swarm(
    *,
    subagents: list[SwarmSubAgent] | None = None,
    default_model: str | BaseChatModel,
) -> SwarmExtension:
    """Build a swarm interpreter extension.

    Makes ``import { create, run, rows } from "swarm"`` available in the code
    interpreter. Table persistence (glob/read/write) requires a backend on
    the middleware (``backend=`` or ``skills_backend=``); pure ``invoke``-mode
    dispatch via ``swarmTask`` does not.

    Args:
        subagents: Subagent specifications for dispatch targets.
        default_model: Default model for subagents without their own, and
            for ``invoke`` mode direct calls.

    Returns:
        A ``SwarmExtension`` for ``CodeInterpreterMiddleware(extensions=...)``.
    """
    dispatch = build_swarm_dispatch(subagents=subagents, default_model=default_model)
    return SwarmExtension(dispatch=dispatch)


__all__ = ["SwarmExtension", "swarm"]
