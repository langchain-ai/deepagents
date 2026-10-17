"""Adapter for `langchain_core`'s private deprecation helpers.

Centralizes the import surface so an upstream rename or move is a one-file
change.

Re-exports:
- `deprecated`: decorator for callables, classes, and properties.
- `warn_deprecated`: helper for parameter/value-level deprecations where the
    callable itself isn't being deprecated. Wraps the upstream helper to
    accept a `stacklevel` argument (the upstream version hardcodes
    `stacklevel=4`, which mis-attributes warnings emitted directly from a
    deprecated method body).
- `suppress_langchain_deprecation_warning`: context manager that silences
    emissions from this module's helpers (use sparingly — it is type-wide).
- `LangChainDeprecationWarning`: warning class emitted by the helpers above
    (subclass of `DeprecationWarning`).
"""

from __future__ import annotations

import warnings

from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
    warn_deprecated as _lc_warn_deprecated,
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

    `langchain_core.warn_deprecated` formats a standard message but hardcodes
    `stacklevel=4` in its internal `warnings.warn` call. That value targets a
    decorator-wrapped frame layout; when called directly from a deprecated
    method's body the warning is attributed one frame too high (above the
    user's call site). This wrapper captures the formatted upstream warning
    and re-emits it with an explicit `stacklevel`, so the warning points at
    the user's call site.

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
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _lc_warn_deprecated(
            since,
            message=message,
            name=name,
            alternative=alternative,
            alternative_import=alternative_import,
            pending=pending,
            obj_type=obj_type,
            addendum=addendum,
            removal=removal,
            package=package,
        )
    if not captured:
        return
    record = captured[0]
    warnings.warn(record.message, category=record.category, stacklevel=stacklevel + 1)


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
