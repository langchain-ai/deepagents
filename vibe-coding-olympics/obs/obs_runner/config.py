"""Environment-driven configuration for the OBS runner.

Only obs-websocket transport + FastAPI bind config lives here. Scene
and text-source naming is a domain concern of the FSM and lives in
the control plane (`control_server.state_config`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    """Return env var `name`, stripped — falls back to `default`."""
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Config:
    """Runtime configuration resolved from the process environment.

    Attributes:
        obs_host: Host running obs-websocket.
        obs_port: obs-websocket port (OBS 28+ default is 4455).
        obs_password: obs-websocket password; empty string disables auth.
        api_host: Bind host for the FastAPI control server.
        api_port: Bind port for the FastAPI control server.
    """

    obs_host: str = "localhost"
    obs_port: int = 4455
    obs_password: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8765


def load_config() -> Config:
    """Read env vars and return an immutable `Config`."""
    return Config(
        obs_host=_env("OBS_HOST", "localhost"),
        obs_port=int(_env("OBS_PORT", "4455")),
        obs_password=os.environ.get("OBS_PASSWORD", ""),
        api_host=_env("VIBE_OBS_API_HOST", "127.0.0.1"),
        api_port=int(_env("VIBE_OBS_API_PORT", "8765")),
    )
