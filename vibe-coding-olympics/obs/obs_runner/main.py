"""Uvicorn entrypoint for `vibe-obs`."""

from __future__ import annotations

import logging

import uvicorn

from obs_runner.config import load_config


def main() -> None:
    """Run the FastAPI app with Uvicorn, honoring env-driven host/port."""
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
    )


if __name__ == "__main__":
    main()
