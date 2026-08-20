"""Lifecycle hooks for the bundled LangGraph server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from starlette.applications import Starlette

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_PHOENIX_FLUSH_TIMEOUT_MILLIS = 1_000


@asynccontextmanager
async def _lifespan(_: Starlette) -> AsyncIterator[None]:
    """Flush optional tracing while the server still has time to shut down."""
    try:
        yield
    finally:
        from deepagents_code.phoenix_tracing import flush_phoenix_tracing

        try:
            flushed = await asyncio.to_thread(
                flush_phoenix_tracing,
                timeout_millis=_PHOENIX_FLUSH_TIMEOUT_MILLIS,
            )
        except Exception:
            logger.warning(
                "Failed to flush Phoenix tracing during shutdown", exc_info=True
            )
        else:
            if not flushed:
                logger.warning("Timed out flushing Phoenix tracing during shutdown")


app = Starlette(lifespan=_lifespan)
"""Empty app whose lifespan is merged into the built-in LangGraph server."""
