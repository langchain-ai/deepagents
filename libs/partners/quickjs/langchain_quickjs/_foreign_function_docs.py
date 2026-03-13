"""Render compact prompt-facing documentation for QuickJS foreign functions.

QuickJS foreign functions are Python callables or LangChain tools exposed inside
JavaScript. The model needs a concise description of their names, argument
shapes, return types, and any referenced structured types. This module converts
Python signatures and docstrings into small TypeScript-like stubs and prompt
sections that are easy for the model to consume.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from collections.abc import Callable

from langchain_core.tools import BaseTool


def _format_annotation(annotation: Any) -> str:
    """Render one Python type annotation in a TypeScript-like form.

    Args:
        annotation: The Python annotation to render.

    Returns:
        A compact string suitable for prompt-facing function signatures.
    """
    if annotation is Any or annotation is inspect.Signature.empty:
        return "any"
    if annotation is str:
        return "string"
    if annotation in (int, float):
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is type(None):
        return "null"
    if annotation is dict:
        return "Record<string, any>"
    if isinstance(annotation, type):
        return annotation.__name__

    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if origin in (list, set, frozenset):
            item = _format_annotation(args[0]) if args else "any"
            return f"{item}[]"
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return f"{_format_annotation(args[0])}[]"
            return f"[{', '.join(_format_annotation(arg) for arg in args)}]"
        if origin is dict:
            key_type = _format_annotation(args[0]) if args else "string"
            value_type = _format_annotation(args[1]) if len(args) > 1 else "any"
            return f"Record<{key_type}, {value_type}>"
        if origin is type:
            inner = _format_annotation(args[0]) if args else "any"
            return f"new (...args: any[]) => {inner}"
        if origin_name := getattr(origin, "__name__", None):
            if origin_name in {"Union", "UnionType"}:
                return " | ".join(_format_annotation(arg) for arg in args)
            formatted_args = ", ".join(_format_annotation(arg) for arg in args)
            return f"{origin_name}<{formatted_args}>"

    rendered = str(annotation).replace("typing.", "").replace("'", "")
    if rendered.endswith(" | None"):
        return f"{rendered.removesuffix(' | None')} | null"
    if "." in rendered and "[" not in rendered:
        return rendered.rsplit(".", maxsplit=1)[-1]
    return rendered


def _unwrap_typed_dict_annotation(annotation: Any) -> tuple[Any, str]:
    """Extract a TypedDict annotation from supported container types."""
    container_prefix = ""
    origin = get_origin(annotation)
    if origin not in (list, tuple, set, frozenset):
        return annotation, container_prefix

    args = get_args(annotation)
    if not args:
        return annotation, container_prefix

    unwrapped = args[0]
    container_prefix = f"Contained `{unwrapped.__name__}` structure:\n"
    return unwrapped, container_prefix


def _render_typed_dict_fields(
    annotation: type[Any], field_types: dict[str, Any]
) -> str | None:
    """Render field lines for a TypedDict annotation."""
    if not field_types:
        return None

    required_keys = getattr(annotation, "__required_keys__", frozenset())
    optional_keys = getattr(annotation, "__optional_keys__", frozenset())
    lines = [f"Return structure `{annotation.__name__}`:"]
    for key, value in field_types.items():
        marker = "required" if key in required_keys else "optional"
        if key not in required_keys and key not in optional_keys:
            marker = "field"
        lines.append(f"- {key}: {_format_annotation(value)} ({marker})")
    return "\n".join(lines)


def _format_typed_dict_structure(annotation: Any) -> str | None:
    """Render a compact field listing for a TypedDict annotation."""
    annotation, container_prefix = _unwrap_typed_dict_annotation(annotation)
    if not isinstance(annotation, type):
        return None
    if not hasattr(annotation, "__annotations__") or not hasattr(
        annotation, "__required_keys__"
    ):
        return None

    field_types = getattr(annotation, "__annotations__", {})
    with contextlib.suppress(TypeError, NameError):
        field_types = get_type_hints(annotation)

    rendered_fields = _render_typed_dict_fields(annotation, field_types)
    if rendered_fields is None:
        return None
    return container_prefix + rendered_fields


def _get_tool_doc_target(tool: BaseTool) -> Callable[..., Any] | None:
    """Choose the underlying callable that best represents a LangChain tool.

    Args:
        tool: The LangChain tool being documented.

    Returns:
        The sync function or coroutine that provides the most useful signature
        and docstring for prompt generation, or `None` if neither is available.
    """
    target = getattr(tool, "func", None)
    if callable(target):
        return target
    target = getattr(tool, "coroutine", None)
    if callable(target):
        return target
    return None


def _get_foreign_function_mode(implementation: Callable[..., Any] | BaseTool) -> str:
    """Return whether a foreign function should be treated as sync or async."""
    if isinstance(implementation, BaseTool):
        coroutine = getattr(implementation, "coroutine", None)
        return "async" if callable(coroutine) else "sync"
    return "async" if inspect.iscoroutinefunction(implementation) else "sync"


def _get_return_annotation(target: Callable[..., Any]) -> Any:
    """Resolve the return annotation for a callable, if present."""
    with contextlib.suppress(TypeError, ValueError, NameError):
        inspected_signature = inspect.signature(target)
        resolved_hints = get_type_hints(target)
        return resolved_hints.get("return", inspected_signature.return_annotation)
    return inspect.Signature.empty


def _render_jsdoc(doc: str) -> str:
    """Convert a Python docstring into a compact JSDoc block."""
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: list[tuple[str, str]] = []
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if in_args:
            if not stripped:
                continue
            if line.startswith("    ") and ":" in stripped:
                name, description = stripped.split(":", maxsplit=1)
                params.append((name.strip(), description.strip()))
                continue
            in_args = False
        if stripped:
            summary.append(stripped)

    rendered = ["/**"]
    for line in summary:
        rendered.append(f" * {line}")
    if summary and params:
        rendered.append(" *")
    for name, description in params:
        rendered.append(f" * @param {name} {description}")
    rendered.append(" */")
    return "\n".join(rendered)


def _render_function_stub(
    name: str, implementation: Callable[..., Any] | BaseTool
) -> str:
    """Render one prompt-facing function declaration for a foreign function.

    Args:
        name: The JavaScript-visible foreign function name.
        implementation: The Python callable or LangChain tool backing that name.

    Returns:
        A TypeScript-like function declaration, optionally prefixed with a small
        JSDoc block derived from the Python docstring.
    """
    function_mode = _get_foreign_function_mode(implementation)
    target = (
        _get_tool_doc_target(implementation)
        if isinstance(implementation, BaseTool)
        else implementation
    )
    if target is None:
        prefix = "async function" if function_mode == "async" else "function"
        return f"{prefix} {name}(...args: any[]): any"

    signature = "(...args: any[])"
    return_annotation = inspect.Signature.empty
    with contextlib.suppress(TypeError, ValueError, NameError):
        inspected_signature = inspect.signature(target)
        resolved_hints = get_type_hints(target)
        parameter_parts = [
            (
                f"{param.name}: "
                f"{
                    _format_annotation(resolved_hints.get(param.name, param.annotation))
                }"
            )
            if param.annotation is not inspect.Signature.empty
            or param.name in resolved_hints
            else f"{param.name}: any"
            for param in inspected_signature.parameters.values()
        ]
        signature = f"({', '.join(parameter_parts)})"
        return_annotation = resolved_hints.get(
            "return", inspected_signature.return_annotation
        )

    rendered_return = (
        _format_annotation(return_annotation)
        if return_annotation is not inspect.Signature.empty
        else "any"
    )
    if function_mode == "async":
        rendered_return = f"Promise<{rendered_return}>"
    prefix = "async function" if function_mode == "async" else "function"
    declaration = f"{prefix} {name}{signature}: {rendered_return}"
    doc = inspect.getdoc(target) or inspect.getdoc(implementation)
    if not doc:
        return declaration
    return f"{_render_jsdoc(doc)}\n{declaration}"


def _collect_referenced_types(
    implementations: dict[str, Callable[..., Any] | BaseTool],
) -> list[type[Any]]:
    """Collect unique structured return types that should be documented.

    Args:
        implementations: Mapping of JavaScript-visible function names to Python
            callables or LangChain tools.

    Returns:
        A list of TypedDict-like annotations referenced by foreign function
        return types, preserving first-seen order.
    """
    collected: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for implementation in implementations.values():
        target = (
            _get_tool_doc_target(implementation)
            if isinstance(implementation, BaseTool)
            else implementation
        )
        if target is None:
            continue
        annotation = _get_return_annotation(target)
        origin = get_origin(annotation)
        if origin in (list, tuple, set, frozenset):
            args = get_args(annotation)
            if args:
                annotation = args[0]
        if not isinstance(annotation, type):
            continue
        if not hasattr(annotation, "__annotations__") or not hasattr(
            annotation, "__required_keys__"
        ):
            continue
        if annotation not in seen:
            seen.add(annotation)
            collected.append(annotation)
    return collected


def _render_typed_dict_definition(annotation: type[Any]) -> str:
    """Render a TypeScript-like type definition for a TypedDict."""
    with contextlib.suppress(TypeError, NameError):
        field_types = get_type_hints(annotation)
        lines = [f"type {annotation.__name__} = {{"]
        for key, value in field_types.items():
            lines.append(f"  {key}: {_format_annotation(value)}")
        lines.append("}")
        return "\n".join(lines)

    field_types = getattr(annotation, "__annotations__", {})
    lines = [f"type {annotation.__name__} = {{"]
    for key, value in field_types.items():
        lines.append(f"  {key}: {_format_annotation(value)}")
    lines.append("}")
    return "\n".join(lines)


def render_foreign_function_section(
    implementations: dict[str, Callable[..., Any] | BaseTool],
) -> str:
    """Render the complete prompt section for available foreign functions."""
    function_blocks = [
        _render_function_stub(name, implementation)
        for name, implementation in implementations.items()
    ]
    sections = [
        "Available foreign functions:\n",
        (
            "These are JavaScript-callable foreign functions exposed inside QuickJS. "
            "The TypeScript-style signatures below document argument and return shapes."
        ),
        "",
        "```ts",
        "\n\n".join(function_blocks),
        "```",
    ]

    referenced_types = _collect_referenced_types(implementations)
    if referenced_types:
        type_blocks = [
            _render_typed_dict_definition(annotation) for annotation in referenced_types
        ]
        sections.extend(
            [
                "",
                "Referenced types:",
                "```ts",
                "\n\n".join(type_blocks),
                "```",
            ]
        )
    return "\n".join(sections)


def format_foreign_function_docs(
    name: str,
    implementation: Callable[..., Any] | BaseTool,
) -> str:
    """Render a compact signature and docstring block for a foreign function."""
    return _render_function_stub(name, implementation)


__all__ = ["format_foreign_function_docs", "render_foreign_function_section"]
