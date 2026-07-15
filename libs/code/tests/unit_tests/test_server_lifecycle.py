"""Tests for server-owned browser resource cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepagents_code import _server_lifecycle


@pytest.fixture(autouse=True)
def _reset_cleanup_registry() -> Iterator[None]:
    _server_lifecycle._consume_browser_cleanup()
    yield
    _server_lifecycle._consume_browser_cleanup()


async def test_lifespan_awaits_registered_browser_cleanup() -> None:
    cleanup = AsyncMock()
    _server_lifecycle.register_browser_cleanup(cleanup)

    async with _server_lifecycle.app.router.lifespan_context(_server_lifecycle.app):
        cleanup.assert_not_awaited()

    cleanup.assert_awaited_once_with()


def test_duplicate_browser_cleanup_registration_is_rejected() -> None:
    _server_lifecycle.register_browser_cleanup(AsyncMock())

    with pytest.raises(RuntimeError, match="already registered"):
        _server_lifecycle.register_browser_cleanup(AsyncMock())


async def test_cleanup_failure_propagates_and_is_not_retried() -> None:
    cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    _server_lifecycle.register_browser_cleanup(cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        async with _server_lifecycle.app.router.lifespan_context(_server_lifecycle.app):
            pass

    async with _server_lifecycle.app.router.lifespan_context(_server_lifecycle.app):
        pass
    cleanup.assert_awaited_once_with()
