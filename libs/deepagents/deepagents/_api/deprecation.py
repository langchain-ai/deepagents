"""Adapter for `langchain_core`'s private deprecation helpers.

Centralizes the import surface so an upstream rename or move is a one-file
change.

Re-exports:
- `deprecated`: decorator for callables, classes, and properties.
- `warn_deprecated`: helper for parameter/value-level deprecations where the
    callable itself isn't being deprecated. Matches the upstream format with
    a `stacklevel` argument (the upstream version hardcodes
    `stacklevel=4`, which mis-attributes warnings emitted directly from a
    deprecated method body).
- `suppress_langchain_deprecation_warning`: context manager that silences
    emissions from this module's helpers (use sparingly).
- `LangChainDeprecationWarning`: warning class emitted by the helpers above
    (subclass of `DeprecationWarning`).
"""

from __future__ import annotations

import warnings

from langchain_core._api.deprecation import (
    _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING,
    LangChainDeprecationWarning,
    LangChainPendingDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
)

__all__ = [
    "LangChainDeprecationWarning",
    "deprecated",
    "reset_deprecation_dedupe",
    "suppress_langchain_deprecation_warning",
    "warn_deprecated",
]


def warn_deprecated(
    since: str,
    *,
    message: str = "",
    name: str = "",
    alternative: str = "",
    alternative_import: str = "",
    pending: bool = False,
    obj_type: str = "",
    addendum: str = "",
    removal: str = "",
    package: str = "",
    stacklevel: int = 2,
) -> None:
    """Emit a deprecation warning with caller-controlled stack attribution.

    Formats the upstream message locally to support `stacklevel` without
    capturing unrelated threads' warnings or changing their warning filters.

    Args:
        since: Release at which this API became deprecated.
        message: Override the default deprecation message. See upstream
            `langchain_core.warn_deprecated` for supported format specifiers.
        name: Name of the deprecated object.
        alternative: Alternative API the user may use instead.
        alternative_import: Alternative import path the user may use instead.
        pending: If `True`, uses a `PendingDeprecationWarning` instead of a
            `DeprecationWarning`. Cannot be combined with `removal`.
        obj_type: Object type label (e.g., `"function"`, `"class"`).
        addendum: Additional text appended to the final message.
        removal: Expected removal version. Cannot be combined with `pending`.
        package: Package name attribution for the deprecation message.
        stacklevel: Frames above this call to attribute the warning to,
            using the same convention as `warnings.warn` (`1` = this call,
            `2` = the caller of the method body that invoked us, etc.).
    """
    if not message:
        package_ = package or name.split(".", maxsplit=1)[0].replace("_", "-") if "." in name else "LangChain"
        message = f"The {obj_type} `{name}`" if obj_type else f"`{name}`"
        if pending:
            message += " will be deprecated in a future version"
        else:
            message += f" was deprecated in {package_} {since}"
            if removal:
                message += f" and will be removed in {removal}"
        if alternative_import:
            alt_package = alternative_import.split(".", maxsplit=1)[0].replace("_", "-")
            if alt_package == package_:
                message += f". Use {alternative_import} instead."
            else:
                alt_module, alt_name = alternative_import.rsplit(".", 1)
                message += (
                    f". An updated version of the {obj_type} exists in the {alt_package} package and should be used instead. "
                    f"To use it run `pip install -U {alt_package}` and import as `from {alt_module} import {alt_name}`."
                )
        elif alternative:
            message += f". Use {alternative} instead."
        if addendum:
            message += f" {addendum}"
    if _SUPPRESSING_LANGCHAIN_DEPRECATION_WARNING.get():
        return
    warning_cls = LangChainPendingDeprecationWarning if pending else LangChainDeprecationWarning
    warnings.warn(warning_cls(message), stacklevel=stacklevel + 1)


def reset_deprecation_dedupe(*targets: object) -> None:
    """Reset the `@deprecated` decorator's dedupe flag for testing.

    The langchain_core `@deprecated` decorator emits each warning at most once
    per process via a closure-bound `warned` flag. Tests that assert per-call
    emission must reset that flag between cases — otherwise the assertions
    become reorder-sensitive (notably under `pytest -n auto`).

    Accepts decorated functions, methods, and `property` objects (in which
    case the `fget` closure is reset). Targets without the expected `warned`
    freevar are silently skipped, so passing non-decorated callables is safe.

    Args:
        *targets: Decorated callables (or properties wrapping them) to reset.
    """
    for target in targets:
        fn = target.fget if isinstance(target, property) else target
        code = getattr(fn, "__code__", None)
        closure = getattr(fn, "__closure__", None)
        if code is None or closure is None:
            continue
        try:
            index = code.co_freevars.index("warned")
        except ValueError:
            continue
        cell = closure[index]
        try:
            current = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if isinstance(current, bool):
            cell.cell_contents = False
