"""Tests for server-owned asynchronous resource cleanup."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deepagents_code import _server_lifecycle


@pytest.fixture(autouse=True)
async def _reset_server_resources() -> None:
    await _server_lifecycle.server_resources.aclose()
    yield
    await _server_lifecycle.server_resources.aclose()


async def test_lifespan_awaits_registered_cleanup() -> None:
    cleanup = AsyncMock()
    _server_lifecycle.server_resources.add_cleanup(cleanup)

    async with _server_lifecycle.app.router.lifespan_context(_server_lifecycle.app):
        cleanup.assert_not_awaited()

    cleanup.assert_awaited_once_with()


async def test_server_resources_close_callbacks_in_reverse_order() -> None:
    resources = _server_lifecycle.ServerResources()
    calls: list[str] = []
    cleanup = AsyncMock(side_effect=lambda name: calls.append(name))

    resources.add_cleanup(lambda: cleanup("first"))
    resources.add_cleanup(lambda: cleanup("second"))

    await resources.aclose()

    assert calls == ["second", "first"]


async def test_cleanup_failure_does_not_skip_earlier_resources() -> None:
    resources = _server_lifecycle.ServerResources()
    earlier_cleanup = AsyncMock()
    failing_cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    resources.add_cleanup(earlier_cleanup)
    resources.add_cleanup(failing_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await resources.aclose()

    failing_cleanup.assert_awaited_once_with()
    earlier_cleanup.assert_awaited_once_with()


async def test_consumed_cleanups_are_not_retried() -> None:
    resources = _server_lifecycle.ServerResources()
    cleanup = AsyncMock()
    resources.add_cleanup(cleanup)

    await resources.aclose()
    await resources.aclose()

    cleanup.assert_awaited_once_with()
