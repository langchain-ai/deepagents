"""Shared fixtures for integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain_wasmsh import WasmshSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def sandbox() -> Iterator[WasmshSandbox]:
    s = WasmshSandbox()
    try:
        yield s
    finally:
        s.close()
