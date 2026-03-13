"""Bridge Python foreign functions into QuickJS with transparent JSON round-tripping.

The QuickJS Python binding can pass primitive return values directly, but complex
Python values like lists and dicts do not automatically become JavaScript arrays
or objects. This module adds a small bridge layer that JSON-encodes complex
Python results on the way out and parses them back inside QuickJS so foreign
functions behave more naturally from JavaScript.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import quickjs
from langchain_core.tools import BaseTool


def _wrap_tool_for_js(
    tool: BaseTool,
    payload_builder: Callable[
        [BaseTool, tuple[Any, ...], dict[str, Any]], str | dict[str, Any]
    ],
) -> Callable[..., Any]:
    """Adapt a LangChain tool into a plain sync callable for QuickJS.

    Args:
        tool: The LangChain tool to expose inside the QuickJS context.
        payload_builder: Helper that converts JavaScript positional and keyword
            arguments into the payload shape expected by `tool.invoke()`.

    Returns:
        A Python callable that QuickJS can register via `Context.add_callable()`.
    """

    def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        payload = payload_builder(tool, args, kwargs)
        return tool.invoke(payload)

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
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value)


def _wrap_function_for_js(implementation: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a Python callable so complex return values are JSON-encoded.

    Args:
        implementation: The Python callable to expose to QuickJS.

    Returns:
        A callable that preserves invocation arguments and serializes any
        non-primitive return value through `_serialize_for_js()`.
    """

    def function_wrapper(*args: Any, **kwargs: Any) -> Any:
        return _serialize_for_js(implementation(*args, **kwargs))

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


def build_external_functions(
    implementations: dict[str, Callable[..., Any] | BaseTool] | None,
    *,
    payload_builder: Callable[
        [BaseTool, tuple[Any, ...], dict[str, Any]], str | dict[str, Any]
    ],
) -> dict[str, Callable[..., Any]]:
    """Normalize foreign implementations into QuickJS-registerable callables.

    This converts LangChain tools into plain callables, wraps all implementations
    so complex Python return values are JSON-encoded, and keys the resulting
    mapping by hidden raw callable names used by the JavaScript shim layer.

    Args:
        implementations: Mapping of user-facing foreign function names to either
            plain Python callables or LangChain tools.
        payload_builder: Helper that converts JavaScript call arguments into the
            payload shape expected by LangChain tools.

    Returns:
        A mapping from internal raw callable names to callables suitable for
        registration with `quickjs.Context.add_callable()`.
    """
    external_functions: dict[str, Callable[..., Any]] = {}
    for name, implementation in (implementations or {}).items():
        callable_implementation = (
            _wrap_tool_for_js(implementation, payload_builder)
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

    Returns:
        None.
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


__all__ = ["build_external_functions", "inject_external_function_shims"]
