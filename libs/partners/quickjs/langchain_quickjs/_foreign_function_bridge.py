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
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from concurrent.futures import Future
    from typing import Literal

    import quickjs


class _AsyncLoopThread:
    """Run coroutines on a dedicated daemon-thread event loop.

    QuickJS only accepts synchronous Python callbacks via
    `quickjs.Context.add_callable()`. This helper provides a long-lived event
    loop on a background daemon thread so async foreign functions can still be
    exposed through the synchronous QuickJS callback interface.
    """

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
    """Resolve awaitable results on the background event loop when needed.

    Args:
        value: Either a plain return value or an awaitable produced by an async
            foreign function or tool.

    Returns:
        The original value for synchronous call paths, or the completed result of
        the awaitable after running it on the daemon-thread event loop.
    """
    if inspect.isawaitable(value):
        return _ASYNC_LOOP_THREAD.submit(value).result()
    return value


def _invoke_tool(
    tool: BaseTool,
    payload: str | dict[str, Any],
    *,
    prefer_async: bool = False,
) -> Any:
    """Invoke a tool through its sync or async entrypoint as appropriate.

    Args:
        tool: The tool to execute.
        payload: Input payload already normalized for the tool schema.
        prefer_async: Whether the surrounding middleware/tool call originated from
            the async path and should prefer `tool.ainvoke()` when available.

    Returns:
        The tool result, awaiting `tool.ainvoke()` on the daemon-thread event
        loop when the async path is preferred or when the tool exposes only an
        async implementation.
    """
    has_async = getattr(tool, "coroutine", None) is not None or (
        tool.__class__._arun is not BaseTool._arun  # noqa: SLF001
    )
    has_sync = getattr(tool, "func", None) is not None or (
        tool.__class__._run is not BaseTool._run  # noqa: SLF001
    )
    if has_async and (prefer_async or not has_sync):
        return _await_if_needed(tool.ainvoke(payload))
    return _await_if_needed(tool.invoke(payload))


def _build_tool_payload(
    tool: BaseTool, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | dict[str, Any]:
    """Convert QuickJS call arguments into a LangChain tool payload.

    Args:
        tool: The LangChain tool receiving the payload.
        args: Positional arguments provided by JavaScript.
        kwargs: Keyword arguments provided by JavaScript.

    Returns:
        A string or dictionary payload compatible with LangChain tool invocation.
    """
    if kwargs:
        return kwargs
    if len(args) == 1 and isinstance(args[0], (str, dict)):
        return args[0]

    input_schema = tool.get_input_schema()
    fields = list(getattr(input_schema, "__annotations__", {}))
    if len(args) == 1 and len(fields) == 1:
        return {fields[0]: args[0]}
    if len(args) == len(fields) and fields:
        return dict(zip(fields, args, strict=False))
    return {"args": list(args)}


def _wrap_tool_for_js(
    tool: BaseTool,
    *,
    prefer_async: bool = False,
) -> Callable[..., Any]:
    """Adapt a LangChain tool into a plain sync callable for QuickJS.

    Args:
        tool: The LangChain tool to expose inside the QuickJS context.
        prefer_async: Whether wrapped tool calls should prefer `tool.ainvoke()`
            when the tool supports both sync and async implementations.

    Returns:
        A Python callable that QuickJS can register via `Context.add_callable()`.
        If the wrapped tool produces an awaitable, it is executed on the
        daemon-thread event loop before the result is returned to QuickJS.
    """

    def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = _build_tool_payload(tool, args, kwargs)
        return _invoke_tool(tool, payload, prefer_async=prefer_async)

    return tool_wrapper


def _serialize_for_js(value: Any) -> Any:
    """Convert Python return values into primitives the bridge can round-trip.

    Primitive values already supported by the QuickJS binding are returned
    unchanged. More complex values are JSON-encoded so the JavaScript shim can
    parse them back into arrays or objects.

    Args:
        value: The Python value produced by a foreign function.

    Returns:
        Either the original primitive value or a JSON string representation.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value)


def _wrap_function_for_js(implementation: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a Python callable so complex return values are JSON-encoded.

    Args:
        implementation: The Python callable to expose to QuickJS.

    Returns:
        A callable that preserves invocation arguments, resolves awaitables on
        the daemon-thread event loop when necessary, and serializes any
        non-primitive return value through `_serialize_for_js()`.
    """

    def function_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _serialize_for_js(_await_if_needed(implementation(*args, **kwargs)))

    return function_wrapper


def _raw_function_name(name: str) -> str:
    """Build the hidden Python callable name used by the JS bridge shim.

    Args:
        name: The user-facing foreign function name.

    Returns:
        The internal callable name registered in QuickJS before the public shim
        is installed.
    """
    return f"__python_{name}"


def _build_external_functions(
    implementations: dict[str, Callable[..., Any] | BaseTool] | None,
    *,
    prefer_async: bool = False,
) -> dict[str, Callable[..., Any]]:
    """Normalize foreign implementations into QuickJS-registerable callables.

    This converts LangChain tools into plain callables, wraps all implementations
    so complex Python return values are JSON-encoded, and keys the resulting
    mapping by hidden raw callable names used by the JavaScript shim layer.

    Args:
        implementations: Mapping of user-facing foreign function names to either
            plain Python callables or LangChain tools.
        prefer_async: Whether wrapped LangChain tools should prefer `tool.ainvoke()`
            when both sync and async implementations are available.

    Returns:
        A mapping from internal raw callable names to callables suitable for
        registration with `quickjs.Context.add_callable()`.
    """
    external_functions: dict[str, Callable[..., Any]] = {}
    for name, implementation in (implementations or {}).items():
        callable_implementation = (
            _wrap_tool_for_js(implementation, prefer_async=prefer_async)
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
    if (typeof value !== "string") {{ return value; }}
    const trimmed = value.trim();
    if (!trimmed) {{ return value; }}
    const first = trimmed[0];
    if (first !== "[" && first !== "{{") {{ return value; }}
    return JSON.parse(value);
}};
"""


def inject_external_function_shims(
    context: quickjs.Context, external_functions: list[str] | None
) -> None:
    """Install JavaScript shims for foreign functions inside a QuickJS context.

    Each shim preserves the user-facing function name while delegating to a
    hidden Python callable name. When the Python side returns a JSON-encoded
    string that looks like an array or object, the shim parses it back into a
    native JavaScript value. Plain strings and primitive values are returned
    unchanged.

    Args:
        context: The QuickJS context receiving the callable shims.
        external_functions: The user-facing foreign function names that should be
            available inside JavaScript.
    """
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
    code = "".join(shim_lines)
    context.eval(code)


def install_external_functions(
    context: quickjs.Context,
    implementations: dict[str, Callable[..., Any] | BaseTool] | None,
    *,
    execution_mode: Literal["sync", "async"] = "sync",
) -> None:
    """Install foreign functions and JavaScript shims into a QuickJS context.

    Args:
        context: The QuickJS context receiving foreign function callables.
        implementations: Mapping of user-facing foreign function names to either
            plain Python callables or LangChain tools.
        execution_mode: Whether tool-backed foreign functions should use sync or
            async invocation when both are available.
    """
    external_functions = _build_external_functions(
        implementations,
        prefer_async=execution_mode == "async",
    )
    for name, implementation in external_functions.items():
        context.add_callable(name, implementation)
    inject_external_function_shims(context, list(implementations or {}))


__all__ = ["install_external_functions"]
