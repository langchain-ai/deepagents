"""LangGraph lifespan integration for server-owned extension resources."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from starlette.applications import Starlette

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _lifespan(_: Starlette) -> AsyncIterator[None]:
    """Release extensions before their server event loop closes."""
    try:
        yield
    finally:
        from deepagents_code.extensions.runtime import shutdown_server_extensions

        await shutdown_server_extensions()


app = Starlette(lifespan=_lifespan)
"""Route-free app merged into the LangGraph server for lifecycle cleanup."""
