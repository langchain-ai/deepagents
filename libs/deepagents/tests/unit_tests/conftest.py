"""Shared fixtures for unit tests."""

from __future__ import annotations

import pytest

from deepagents._api.deprecation import reset_deprecation_dedupe
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.store import _NamespaceRuntimeCompat
from deepagents.graph import get_default_model

_DEDUPED_TARGETS: tuple[object, ...] = (
    BackendProtocol.ls_info,
    BackendProtocol.als_info,
    BackendProtocol.glob_info,
    BackendProtocol.aglob_info,
    BackendProtocol.grep_raw,
    BackendProtocol.agrep_raw,
    _NamespaceRuntimeCompat.runtime,
    _NamespaceRuntimeCompat.state,
    get_default_model,
)
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
