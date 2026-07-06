"""Configuration loading for the Talon runtime host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_ASSISTANT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_ENV_PREFIX = "DEEPAGENTS_TALON_"
_RUNTIME_ENV_PREFIXES = (_ENV_PREFIX, "AGENT_", "LANGSMITH_", "OPENAI_", "SPEECH_", "TELEGRAM_")
_RUNTIME_ENV_KEYS = frozenset({"BUILTIN_MCP_URL", "HOST_LANGCHAIN_API_URL", "TELEGRAM_BOT_TOKEN"})


class TalonConfigError(ValueError):
    """Raised when Talon runtime configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TalonConfig:
    """Runtime configuration for a single Talon assistant process.

    Args:
        assistant_id: Stable identifier used to namespace all local assistant state.
        home: Per-assistant home directory for state, manifests, sessions, and jobs.
        model: Chat model identifier supplied by the operator environment.
        fleet_dir: Optional operator-unzipped Fleet export directory to load with
            `fleet-deepagents-export`.
        env: Environment values visible to channels, providers, and future adapters.
    """

    assistant_id: str
    home: Path
    model: str | None = None
    fleet_dir: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        base_home: Path | None = None,
    ) -> TalonConfig:
        """Build runtime configuration from environment variables.

        Args:
            env: Environment mapping to read. Defaults to `os.environ`.
            base_home: Optional base directory for assistant state. Tests and
                embedding hosts can supply this to avoid the user home directory.

        Returns:
            Runtime configuration with a validated assistant id and namespaced home.

        Raises:
            TalonConfigError: If the assistant id is empty or unsafe for a path segment.
        """
        values = os.environ if env is None else env
        assistant_id = _first_present(
            values,
            "DEEPAGENTS_TALON_ASSISTANT_ID",
            "AGENT_ASSISTANT_ID",
            default="default",
        )
        if assistant_id is None:
            msg = "assistant id is required"
            raise TalonConfigError(msg)
        _validate_assistant_id(assistant_id)

        root = Path(values.get("DEEPAGENTS_TALON_HOME", "")) if base_home is None else base_home
        if not str(root):
            root = Path.home() / ".deepagents"

        model = _first_present(values, "DEEPAGENTS_TALON_MODEL", "AGENT_MODEL", default=None)
        fleet_dir = _first_present(
            values,
            "DEEPAGENTS_TALON_FLEET_DIR",
            "AGENT_FLEET_DIR",
            "FLEET_DIR",
            default=None,
        )
        return cls(
            assistant_id=assistant_id,
            home=root.expanduser() / assistant_id,
            model=model,
            fleet_dir=Path(fleet_dir).expanduser() if fleet_dir else None,
            env={key: value for key, value in values.items() if _is_runtime_env(key)},
        )

    def ensure_home(self) -> Path:
        """Create the per-assistant home directory with restrictive permissions.

        Returns:
            The created per-assistant home directory.
        """
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.home.chmod(0o700)
        for child in (self.manifest_dir, self.cron_dir, self.channel_dir, self.inbound_media_dir):
            child.mkdir(mode=0o700, parents=True, exist_ok=True)
            child.chmod(0o700)
        return self.home

    @property
    def manifest_dir(self) -> Path:
        """Directory where agent manifest files are materialized."""
        return self.home / "agent"

    @property
    def cron_dir(self) -> Path:
        """Directory reserved for scheduler state."""
        return self.home / "cron"

    @property
    def channel_dir(self) -> Path:
        """Directory reserved for channel session state."""
        return self.home / "channels"

    @property
    def inbound_media_dir(self) -> Path:
        """Directory reserved for downloaded inbound channel media."""
        return self.home / "media" / "inbound"


def _first_present(
    env: Mapping[str, str],
    *keys: str,
    default: str | None,
) -> str | None:
    for key in keys:
        if key in env:
            return env[key]
    return default


def _validate_assistant_id(assistant_id: str | None) -> None:
    if (
        not assistant_id
        or assistant_id in {".", ".."}
        or not _ASSISTANT_ID_PATTERN.fullmatch(assistant_id)
    ):
        msg = (
            "assistant id must be 1-128 characters and contain only letters, numbers, "
            "underscore, hyphen, or dot"
        )
        raise TalonConfigError(msg)


def _is_runtime_env(key: str) -> bool:
    return key in _RUNTIME_ENV_KEYS or key.startswith(_RUNTIME_ENV_PREFIXES)
