"""LangGraph lifespan integration for server-owned extension resources."""

from __future__ import annotations

from contextlib import asynccontextmanager
from ipaddress import ip_address
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request


@asynccontextmanager
async def _lifespan(_: Starlette) -> AsyncIterator[None]:
    """Release extensions before their server event loop closes."""
    try:
        yield
    finally:
        from deepagents_code.extensions.runtime import shutdown_server_extensions

        await shutdown_server_extensions()


def _extensions(request: Request) -> JSONResponse:
    """Return extension provenance only to a loopback client."""
    from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy

    if not is_env_truthy(EXPERIMENTAL):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    host = request.client.host if request.client is not None else ""
    try:
        loopback = ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    from deepagents_code.extensions.runtime import server_extension_report

    return JSONResponse(server_extension_report())


app = Starlette(
    lifespan=_lifespan,
    routes=[Route("/extensions", _extensions, methods=["GET"])],
)
