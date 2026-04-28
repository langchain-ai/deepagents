"""Adapter for `langchain_core`'s private deprecation helpers.

Centralizes the import surface so an upstream rename or move is a one-file
change.

Re-exports:
- `deprecated`: decorator for callables, classes, and properties.
- `warn_deprecated`: helper for parameter/value-level deprecations where the
    callable itself isn't being deprecated.
- `suppress_langchain_deprecation_warning`: context manager that silences
    emissions from this module's helpers (use sparingly — it is type-wide).
- `LangChainDeprecationWarning`: warning class emitted by the helpers above
    (subclass of `DeprecationWarning`).
"""

from langchain_core._api.deprecation import (
    LangChainDeprecationWarning,
    deprecated,
    suppress_langchain_deprecation_warning,
    warn_deprecated,
)

__all__ = [
    "LangChainDeprecationWarning",
    "deprecated",
    "reset_deprecation_dedupe",
    "suppress_langchain_deprecation_warning",
    "warn_deprecated",
]


def reset_deprecation_dedupe(*targets: object) -> None:
    """Reset the `@deprecated` decorator's dedupe flag for testing.

    The langchain_core `@deprecated` decorator emits each warning at most once
    per process via a closure-bound `warned` flag. Tests that assert per-call
    emission must reset that flag between cases — otherwise the assertions
    become reorder-sensitive (notably under `pytest -n auto`).

    Accepts decorated functions, methods, and `property` objects (in which
    case the `fget` closure is reset).

    Args:
        *targets: Decorated callables (or properties wrapping them) to reset.
    """
    for target in targets:
        fn = target.fget if isinstance(target, property) else target
        closure = getattr(fn, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:  # empty cell
                continue
            if isinstance(value, bool):
                cell.cell_contents = False
