"""FastAPI surface for the game state machine.

Endpoints (MVP):

- `POST /transition` — advance the FSM.
- `GET  /state`      — current snapshot.
- `GET  /healthz`    — OBS connection probe + current phase.

Producers (timer, judge, `play.sh`, manual curl) fire one-shot commands.
No pub/sub, no websockets; adding those is a later pass.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from obs_runner.compositor import CompositorProtocol, ObsCompositor
from obs_runner.config import Config, load_config
from obs_runner.models import HealthResponse, StateResponse, TransitionRequest
from obs_runner.state_machine import InvalidTransitionError, StateMachine

logger = logging.getLogger("obs_runner")


def _snapshot_response(machine: StateMachine) -> StateResponse:
    """Translate the in-memory snapshot into the wire schema."""
    return StateResponse(**asdict(machine.snapshot))


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
    # Only the real compositor participates in the lifespan connect/close
    # dance; injected fakes (e.g. in tests) are assumed pre-configured.
    owns_compositor = compositor is None
    comp = compositor or ObsCompositor(
        host=cfg.obs_host, port=cfg.obs_port, password=cfg.obs_password
    )
    machine = StateMachine(compositor=comp, config=cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if owns_compositor:
            assert isinstance(comp, ObsCompositor)
            try:
                comp.connect()
                machine.prime()
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
        return HealthResponse(
            obs_connected=bool(connected),
            phase=machine.snapshot.phase,
        )

    @app.get("/state", response_model=StateResponse)
    async def get_state() -> StateResponse:
        return _snapshot_response(machine)

    @app.post("/transition", response_model=StateResponse)
    async def transition(req: TransitionRequest) -> StateResponse:
        try:
            machine.dispatch(req.event, req.payload)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _snapshot_response(machine)

    return app


app = create_app()
