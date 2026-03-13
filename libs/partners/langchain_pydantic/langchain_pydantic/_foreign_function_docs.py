from __future__ import annotations

import contextlib
import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool


def format_annotation(annotation: Any) -> str:
    """Render a concise string form for a type annotation."""
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        origin_name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
        if not args:
            return origin_name
        formatted_args = ", ".join(format_annotation(arg) for arg in args)
        return f"{origin_name}[{formatted_args}]"

    if annotation is Any:
        return "Any"
    if annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, type):
        return annotation.__name__

    rendered = str(annotation).replace("typing.", "").replace("'", "")
    if "." in rendered and "[" not in rendered:
        return rendered.rsplit(".", maxsplit=1)[-1]
    return rendered


def format_typed_dict_structure(annotation: Any) -> str | None:
    """Render a compact field listing for a TypedDict annotation."""
    container_prefix = ""
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        if args:
            annotation = args[0]
            container_prefix = f"Contained `{annotation.__name__}` structure:\n"

    if not isinstance(annotation, type):
        return None
    if not hasattr(annotation, "__annotations__") or not hasattr(
        annotation, "__required_keys__"
    ):
        return None

    try:
        field_types = get_type_hints(annotation)
    except Exception:
        field_types = getattr(annotation, "__annotations__", {})

    required_keys = getattr(annotation, "__required_keys__", frozenset())
    optional_keys = getattr(annotation, "__optional_keys__", frozenset())
    if not field_types:
        return None

    lines = [f"Return structure `{annotation.__name__}`:"]
    for key, value in field_types.items():
        marker = "required" if key in required_keys else "optional"
        if key not in required_keys and key not in optional_keys:
            marker = "field"
        type_name = format_annotation(value)
        lines.append(f"- {key}: {type_name} ({marker})")
    return container_prefix + "\n".join(lines)


def get_tool_doc_target(tool: BaseTool) -> Callable[..., Any] | None:
    """Return the most useful callable to inspect for tool documentation."""
    target = getattr(tool, "func", None)
    if callable(target):
        return target
    target = getattr(tool, "coroutine", None)
    if callable(target):
        return target
    return None


def get_foreign_function_mode(implementation: Callable[..., Any] | BaseTool) -> str:
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


def _render_function_stub(
    name: str, implementation: Callable[..., Any] | BaseTool
) -> str:
    """Render a Python-like stub with attached docstring for one foreign function."""
    function_mode = get_foreign_function_mode(implementation)
    target = (
        get_tool_doc_target(implementation)
        if isinstance(implementation, BaseTool)
        else implementation
    )
    if target is None:
        prefix = "async def" if function_mode == "async" else "def"
        return f"{prefix} {name}(...)"

    signature = "(...)"
    with contextlib.suppress(TypeError, ValueError, NameError):
        inspected_signature = inspect.signature(target)
        resolved_hints = get_type_hints(target)
        parameter_parts = [
            f"{param.name}: {format_annotation(resolved_hints.get(param.name, param.annotation))}"
            if param.annotation is not inspect.Signature.empty
            or param.name in resolved_hints
            else param.name
            for param in inspected_signature.parameters.values()
        ]
        return_annotation = resolved_hints.get(
            "return", inspected_signature.return_annotation
        )
        if return_annotation is inspect.Signature.empty:
            signature = f"({', '.join(parameter_parts)})"
        else:
            signature = f"({', '.join(parameter_parts)}) -> {format_annotation(return_annotation)}"

    prefix = "async def" if function_mode == "async" else "def"
    doc = inspect.getdoc(target) or inspect.getdoc(implementation)
    if not doc:
        return f"{prefix} {name}{signature}"

    doc_lines = doc.splitlines()
    if len(doc_lines) == 1:
        return f'{prefix} {name}{signature}:\n    """{doc_lines[0]}"""'

    indented_doc = "\n".join(f"    {line}" if line else "" for line in doc_lines)
    return f'{prefix} {name}{signature}:\n    """{indented_doc.removeprefix("    ")}\n    """'


def _collect_referenced_types(
    implementations: dict[str, Callable[..., Any] | BaseTool],
) -> list[type[Any]]:
    """Collect unique referenced TypedDict-like return types from implementations."""
    collected: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for implementation in implementations.values():
        target = (
            get_tool_doc_target(implementation)
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
    """Render a Python-like TypedDict class definition."""
    try:
        field_types = get_type_hints(annotation)
    except Exception:
        field_types = getattr(annotation, "__annotations__", {})
    lines = [f"class {annotation.__name__}(TypedDict):"]
    if not field_types:
        lines.append("    pass")
        return "\n".join(lines)
    for key, value in field_types.items():
        lines.append(f"    {key}: {format_annotation(value)}")
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
        "```python",
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
                "```python",
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
