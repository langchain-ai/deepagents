"""Tests for bundled server lifecycle hooks."""

from __future__ import annotations

from unittest.mock import patch

from starlette.applications import Starlette

from deepagents_code.server_lifespan import _lifespan


async def test_lifespan_flushes_phoenix_on_shutdown() -> None:
    """The custom lifespan should flush after serving, not during startup."""
    with patch(
        "deepagents_code.phoenix_tracing.flush_phoenix_tracing",
        return_value=True,
    ) as flush:
        async with _lifespan(Starlette()):
            flush.assert_not_called()

    flush.assert_called_once_with(timeout_millis=1_000)
