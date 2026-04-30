"""Shared fixtures for unit tests."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import deepagents
from deepagents._api.deprecation import reset_deprecation_dedupe


def _is_deprecated_target(value: object) -> bool:
    """Return whether `value` (or its `fget`) carries the dedupe `warned` freevar."""
    fn = value.fget if isinstance(value, property) else value
    code = getattr(fn, "__code__", None)
    return code is not None and "warned" in code.co_freevars


def _discover_deprecated_targets() -> tuple[object, ...]:
    """Walk the `deepagents` package for `@deprecated`-wrapped callables.

    Returns every property and function whose underlying closure carries the
    `warned` freevar that langchain_core's `@deprecated` decorator installs.

    This avoids drift: any new `@deprecated` decoration is automatically
    covered by the dedupe-reset fixture, so per-call assertions stay
    reorder-safe under `pytest -n auto` without manual list maintenance.
    """
    seen: dict[int, object] = {}
    for module_info in pkgutil.walk_packages(deepagents.__path__, deepagents.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:  # noqa: BLE001, S112  # Skip optional/extras-gated modules.
            continue
        for _, value in inspect.getmembers(module):
            if _is_deprecated_target(value):
                seen.setdefault(id(value), value)
                continue
            if inspect.isclass(value):
                for _, attr in inspect.getmembers(value):
                    if _is_deprecated_target(attr):
                        seen.setdefault(id(attr), attr)
    return tuple(seen.values())


_DEDUPED_TARGETS: tuple[object, ...] = _discover_deprecated_targets()
"""Callables/properties wrapped by `@deprecated` whose dedupe flag must reset
between tests so per-call warning assertions are reorder-safe under xdist."""


@pytest.fixture(autouse=True)
def _reset_deprecation_dedupe() -> None:
    """Reset `@deprecated` dedupe flags before each test.

    The langchain_core decorator emits each warning at most once per process
    via a closure-bound flag. Without this fixture, tests asserting per-call
    emission become reorder-sensitive.
    """
    reset_deprecation_dedupe(*_DEDUPED_TARGETS)
