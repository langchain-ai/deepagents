"""Bridge Python foreign functions into QuickJS with transparent JSON round-tripping.

The QuickJS Python binding can pass primitive return values directly, but complex
Python values like lists and dicts do not automatically become JavaScript arrays
or objects. This module adds a small bridge layer that JSON-encodes complex
Python results on the way out and parses them back inside QuickJS so foreign
functions behave more naturally from JavaScript.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from typing import TYPE_CHECKING, Any, Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.tools.base import (
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from concurrent.futures import Future

    import quickjs


class _AsyncLoopThread:
    """Run coroutines on a dedicated daemon-thread event loop."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future[Any]:
        loop = self._loop
        if loop is None:
            msg = "Async loop thread was not initialized."
            raise RuntimeError(msg)
        return asyncio.run_coroutine_threadsafe(coroutine, loop)


_ASYNC_LOOP_THREAD = _AsyncLoopThread()


def _await_if_needed(value: Any) -> Any:
    """Resolve awaitable results on the background event loop when needed."""
    if inspect.isawaitable(value):
        return _ASYNC_LOOP_THREAD.submit(value).result()
    return value


def _invoke_tool(
    tool: BaseTool,
    payload: str | dict[str, Any],
    *,
    prefer_async: bool = False,
) -> Any:
    """Invoke a tool through its sync or async entrypoint as appropriate."""
    has_async = getattr(tool, "coroutine", None) is not None or (
        tool.__class__._arun is not BaseTool._arun  # noqa: SLF001
    )
    has_sync = getattr(tool, "func", None) is not None or (
        tool.__class__._run is not BaseTool._run  # noqa: SLF001
    )
    if has_async and (prefer_async or not has_sync):
        return _await_if_needed(tool.ainvoke(payload))
    return _await_if_needed(tool.invoke(payload))


def get_ptc_implementations(
    ptc: list[Callable[..., Any] | BaseTool] | None,
) -> dict[str, Callable[..., Any] | BaseTool]:
    """Return configured PTC implementations keyed by exported function name."""
    implementations: dict[str, Callable[..., Any] | BaseTool] = {}
    for implementation in ptc or []:
        if isinstance(implementation, BaseTool):
            implementations[implementation.name] = implementation
            continue
        name = getattr(implementation, "__name__", None)
        if isinstance(name, str):
            implementations[name] = implementation
    return implementations


def _get_runtime_arg_name(tool: BaseTool) -> str | None:
    """Return the injected runtime parameter name for a tool, if any."""
    for name, type_ in get_all_basemodel_annotations(tool.get_input_schema()).items():
        if name == "runtime" and _is_injected_arg_type(type_):
            return name
    return None


def _build_tool_payload(
    tool: BaseTool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    runtime: ToolRuntime | None = None,
) -> str | dict[str, Any]:
    """Convert QuickJS call arguments into a LangChain tool payload."""
    input_schema = tool.get_input_schema()
    schema_annotations = getattr(input_schema, "__annotations__", {})
    fields = [
        name
        for name, type_ in schema_annotations.items()
        if not _is_injected_arg_type(type_)
    ]
    runtime_arg_name = _get_runtime_arg_name(tool)

    if kwargs:
        payload: str | dict[str, Any] = kwargs
    elif (
        len(args) == 1 and isinstance(args[0], (str, dict)) and runtime_arg_name is None
    ):
        payload = args[0]
    elif len(args) == 1 and len(fields) == 1:
        payload = {fields[0]: args[0]}
    elif len(args) == len(fields) and fields:
        payload = dict(zip(fields, args, strict=False))
    else:
        payload = {"args": list(args)}

    if (
        runtime is not None
        and runtime_arg_name is not None
        and isinstance(payload, dict)
    ):
        return {**payload, runtime_arg_name: runtime}
    return payload


def _wrap_tool_for_js(
    tool: BaseTool,
    *,
    prefer_async: bool = False,
    runtime: ToolRuntime | None = None,
) -> Callable[..., Any]:
    """Adapt a LangChain tool into a plain sync callable for QuickJS."""

    def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = _build_tool_payload(tool, args, kwargs, runtime=runtime)
        return _invoke_tool(tool, payload, prefer_async=prefer_async)

    return tool_wrapper


def _serialize_for_js(value: Any) -> Any:
    """Convert Python return values into primitives the bridge can round-trip."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def _wrap_function_for_js(implementation: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a Python callable so complex return values are JSON-encoded."""

    def function_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _serialize_for_js(_await_if_needed(implementation(*args, **kwargs)))

    return function_wrapper


def _raw_function_name(name: str) -> str:
    """Build the hidden Python callable name used by the JS bridge shim."""
    return f"__python_{name}"


def _build_external_functions(
    implementations: dict[str, Callable[..., Any] | BaseTool] | None,
    *,
    prefer_async: bool = False,
    runtime: ToolRuntime | None = None,
) -> dict[str, Callable[..., Any]]:
    """Normalize foreign implementations into QuickJS-registerable callables."""
    external_functions: dict[str, Callable[..., Any]] = {}
    for name, implementation in (implementations or {}).items():
        callable_implementation = (
            _wrap_tool_for_js(
                implementation,
                prefer_async=prefer_async,
                runtime=runtime,
            )
            if isinstance(implementation, BaseTool)
            else implementation
        )
        external_functions[_raw_function_name(name)] = _wrap_function_for_js(
            callable_implementation
        )
    return external_functions


_EXTERNAL_FUNCTION_SHIM_TEMPLATE = """
globalThis[{name}] = (...args) => {{
    const value = globalThis[{raw_name}](...args);
    if (typeof value !== \"string\") {{ return value; }}
    const trimmed = value.trim();
    if (!trimmed) {{ return value; }}
    const first = trimmed[0];
    if (first !== \"[\" && first !== \"{{\") {{ return value; }}
    return JSON.parse(value);
}};
"""


def inject_external_function_shims(
    context: quickjs.Context, external_functions: list[str] | None
) -> None:
    """Install JavaScript shims for foreign functions inside a QuickJS context."""
    if not external_functions:
        return

    shim_lines = []
    for name in external_functions:
        raw_name = _raw_function_name(name)
        shim_lines.append(
            _EXTERNAL_FUNCTION_SHIM_TEMPLATE.format(
                name=json.dumps(name),
                raw_name=json.dumps(raw_name),
            )
        )
    context.eval("".join(shim_lines))


def install_external_functions(
    context: quickjs.Context,
    implementations: dict[str, Callable[..., Any] | BaseTool] | None,
    *,
    execution_mode: Literal["sync", "async"] = "sync",
    runtime: ToolRuntime | None = None,
) -> None:
    """Install foreign functions and JavaScript shims into a QuickJS context."""
    external_functions = _build_external_functions(
        implementations,
        prefer_async=execution_mode == "async",
        runtime=runtime,
    )
    for name, implementation in external_functions.items():
        context.add_callable(name, implementation)
    inject_external_function_shims(context, list(implementations or {}))


__all__ = ["get_ptc_implementations", "install_external_functions"]
