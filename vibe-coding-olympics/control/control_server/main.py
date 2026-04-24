"""Uvicorn entrypoint for `vibe-control`."""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    """Run the control-panel FastAPI app under Uvicorn."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    host = os.environ.get("VIBE_CONTROL_HOST", "127.0.0.1")
    port = int(os.environ.get("VIBE_CONTROL_PORT", "8766"))
    uvicorn.run("control_server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
