"""Player-laptop HTTP relay for Deep Agents CLI socket commands."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from control_server import deepagents_config
from control_server.event_socket import send_socket_event
from control_server.project_clear import clear_round_project

logger = logging.getLogger(__name__)


class RelayCommand(BaseModel):
    """Command envelope accepted from the controller."""

    kind: Literal["command", "signal"]
    payload: str = Field(min_length=1)


def _event_socket_path() -> Path:
    """Return the configured local CLI event socket path."""
    raw = os.environ.get("VIBE_EVENT_SOCKET", "").strip()
    if not raw:
        msg = "VIBE_EVENT_SOCKET is not configured."
        raise HTTPException(status_code=503, detail=msg)
    return Path(raw)


def _round_project_path() -> Path | None:
    """Return the configured web-vibe project directory, if present."""
    raw = os.environ.get("VIBE_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _required_token() -> str:
    """Return the relay bearer token, or fail closed."""
    token = os.environ.get("VIBE_PLAYER_TOKEN", "").strip()
    if not token:
        msg = "VIBE_PLAYER_TOKEN is not configured."
        raise HTTPException(status_code=503, detail=msg)
    return token


def _authorize(authorization: str | None) -> None:
    """Require `Authorization: Bearer <VIBE_PLAYER_TOKEN>`."""
    token = _required_token()
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        msg = "Missing bearer token."
        raise HTTPException(status_code=401, detail=msg)
    if authorization[len(prefix) :] != token:
        msg = "Invalid bearer token."
        raise HTTPException(status_code=401, detail=msg)


def create_app() -> FastAPI:
    """Create the player relay FastAPI app."""
    app = FastAPI(title="Vibe Olympics Player Relay")

    @app.get("/healthz")
    async def healthz() -> dict[str, str | bool]:
        socket_path = _event_socket_path()
        return {"ok": socket_path.exists(), "socket": str(socket_path)}

    @app.post("/command")
    async def command(
        req: RelayCommand,
        authorization: str | None = Header(default=None),
    ) -> dict[str, bool]:
        _authorize(authorization)
        socket_path = _event_socket_path()
        if req.kind == "signal" and req.payload == "force-clear":
            deepagents_config.clear_recent_model()
            project_dir = _round_project_path()
            if project_dir is not None and not clear_round_project(project_dir):
                logger.warning("Could not clear round project at %s", project_dir)
        try:
            await send_socket_event(
                socket_path,
                kind=req.kind,
                payload=req.payload,
                correlation_prefix="vibe-lan",
            )
        except FileNotFoundError as exc:
            msg = "Player event socket is unavailable."
            raise HTTPException(status_code=503, detail=msg) from exc
        except (OSError, TimeoutError) as exc:
            msg = "Player event socket could not be reached."
            raise HTTPException(status_code=503, detail=msg) from exc
        except (RuntimeError, json.JSONDecodeError) as exc:
            logger.warning("Player event socket rejected relay command: %s", exc)
            msg = "Player event socket rejected the command."
            raise HTTPException(status_code=502, detail=msg) from exc
        return {"ok": True}

    return app


app = create_app()


def main() -> None:
    """Run the player relay under Uvicorn."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    host = os.environ.get("VIBE_RELAY_HOST", "127.0.0.1")
    port = int(os.environ.get("VIBE_RELAY_PORT", "9771"))
    uvicorn.run("control_server.player_relay:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
