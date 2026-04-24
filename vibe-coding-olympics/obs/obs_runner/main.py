"""Uvicorn entrypoint for `vibe-obs`."""

from __future__ import annotations

import logging
import os

import uvicorn

from obs_runner.config import load_config


def main() -> None:
    """Run the FastAPI app with Uvicorn, honoring env-driven host/port.

    Set `VIBE_OBS_RELOAD=1` to restart the server on source edits. Off
    by default because the lifespan reconnects to OBS on every reload.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # obsws-python logs a full traceback at ERROR before raising, which our
    # compositor already catches and re-logs as a one-liner. Silence the lib
    # to avoid duplicated stack traces on every missing-source warning.
    logging.getLogger("obsws_python").setLevel(logging.CRITICAL)
    cfg = load_config()
    uvicorn.run(
        "obs_runner.api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
        reload=os.environ.get("VIBE_OBS_RELOAD") == "1",
    )


if __name__ == "__main__":
    main()
