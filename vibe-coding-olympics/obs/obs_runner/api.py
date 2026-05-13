"""FastAPI surface for the OBS compositor.

Two verbs, no state:

- `POST /scene` — switch OBS scenes.
- `POST /text`  — update an OBS text-source value.
- `GET  /healthz` — obs-websocket connection probe.

The round state machine lives in the control plane; this runner only
renders what control tells it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from obs_runner.compositor import CompositorProtocol, ObsCompositor
from obs_runner.config import Config, load_config
from obs_runner.models import HealthResponse, SceneRequest, TextRequest

logger = logging.getLogger("obs_runner")


def create_app(
    *,
    config: Config | None = None,
    compositor: CompositorProtocol | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        config: Resolved config. Loaded from env when omitted.
        compositor: Compositor to drive. Defaults to a real `ObsCompositor`
            connected on startup. Tests pass a fake to exercise the API
            without OBS.

    Returns:
        A FastAPI instance with routes and lifespan wired up.
    """
    cfg = config or load_config()
    owns_compositor = compositor is None
    comp = compositor or ObsCompositor(
        host=cfg.obs_host, port=cfg.obs_port, password=cfg.obs_password
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if owns_compositor:
            assert isinstance(comp, ObsCompositor)
            try:
                comp.connect()
            except ConnectionError as exc:
                logger.warning("OBS unreachable at startup: %s", exc)
        try:
            yield
        finally:
            if owns_compositor:
                assert isinstance(comp, ObsCompositor)
                comp.close()

    app = FastAPI(title="Vibe Olympics OBS Runner", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        connected = isinstance(comp, ObsCompositor) and comp._client is not None  # noqa: SLF001
        return HealthResponse(obs_connected=bool(connected))

    @app.post("/scene")
    async def set_scene(req: SceneRequest) -> dict[str, str]:
        try:
            comp.set_scene(req.name)
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"name": req.name}

    @app.post("/text")
    async def set_text(req: TextRequest) -> dict[str, str]:
        try:
            comp.set_text(req.source, req.value)
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"source": req.source, "value": req.value}

    return app


app = create_app()
