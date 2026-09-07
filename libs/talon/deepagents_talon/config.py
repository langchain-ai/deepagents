"""Configuration loading for the Talon runtime host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_talon.history_profiles import EmbeddingProfile

_ASSISTANT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_ENV_PREFIX = "DEEPAGENTS_TALON_"
_RUNTIME_ENV_PREFIXES = (_ENV_PREFIX, "AGENT_", "LANGSMITH_", "OPENAI_", "SPEECH_", "TELEGRAM_")
_RUNTIME_ENV_KEYS = frozenset(
    {
        "BUILTIN_MCP_URL",
        "HOST_LANGCHAIN_API_URL",
        "TELEGRAM_BOT_TOKEN",
        "VOYAGE_API_KEY",
        "OPENROUTER_API_KEY",
    }
)


class TalonConfigError(ValueError):
    """Raised when Talon runtime configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TalonConfig:
    """Runtime configuration for a single Talon assistant process.

    Args:
        assistant_id: Stable identifier used to namespace all local assistant state.
        home: Per-assistant home directory for state, manifests, sessions, and jobs.
        model: Chat model identifier supplied by the operator environment.
        env: Environment values visible to channels, providers, and future adapters.
    """

    assistant_id: str
    home: Path
    model: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)

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
            TalonConfigError: If the assistant id or history connection URI is invalid.
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
        _validate_history_uri(values.get("DEEPAGENTS_TALON_HISTORY_URI"))

        if base_home is None:
            configured_home = values.get("DEEPAGENTS_TALON_HOME")
            root = Path(configured_home) if configured_home else Path.home() / ".deepagents"
        else:
            root = base_home

        model = _first_present(values, "DEEPAGENTS_TALON_MODEL", "AGENT_MODEL", default=None)
        config = cls(
            assistant_id=assistant_id,
            home=root.expanduser() / assistant_id,
            model=model,
            env={key: value for key, value in values.items() if _is_runtime_env(key)},
        )

        _ = config.history_reindex
        _ = config.history_vector_search
        _ = config.history_embedding_profile
        return config

    def ensure_home(self) -> Path:
        """Create the per-assistant home directory with restrictive permissions.

        Returns:
            The created per-assistant home directory.
        """
        if not self.home.exists():
            _create_home(self.home)
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.home.chmod(0o700)
        for child in (
            self.manifest_dir,
            self.agents_dir,
            self.cron_dir,
            self.channel_dir,
            self.inbound_media_dir,
        ):
            child.mkdir(mode=0o700, parents=True, exist_ok=True)
            child.chmod(0o700)
        return self.home

    @property
    def manifest_dir(self) -> Path:
        """Directory where agent manifest files are materialized."""
        return self.home

    @property
    def agents_dir(self) -> Path:
        """Directory reserved for custom subagent definitions."""
        return self.home / "agents"

    @property
    def cron_dir(self) -> Path:
        """Directory reserved for scheduler state."""
        return self.home / "cron"

    @property
    def channel_dir(self) -> Path:
        """Directory reserved for channel session state."""
        return self.home / "channels"

    @property
    def history_uri(self) -> str | None:
        """History URI, or None for the default local SQLite archive."""
        uri = self.env.get("DEEPAGENTS_TALON_HISTORY_URI")
        _validate_history_uri(uri)
        return uri

    @property
    def checkpoint_path(self) -> Path:
        """SQLite database used for persistent LangGraph checkpoints."""
        return self._state_path("checkpoints.sqlite", "checkpoint database")

    @property
    def history_vector_search(self) -> bool:
        """Whether optional semantic history indexing is enabled."""
        value = self.env.get("DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH", "false").strip().lower()
        if value not in {"", "0", "false", "no", "off", "1", "true", "yes", "on"}:
            msg = "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH must be a boolean"
            raise TalonConfigError(msg)
        return value in {"1", "true", "yes", "on"}

    @property
    def history_reindex(self) -> bool:
        """Explicitly rebuild an incompatible vector index while retaining transcripts."""
        value = self.env.get("DEEPAGENTS_TALON_HISTORY_REINDEX", "0")
        if value not in {"0", "1"}:
            msg = "DEEPAGENTS_TALON_HISTORY_REINDEX must be 0 or 1"
            raise TalonConfigError(msg)
        return value == "1"

    @property
    def history_embedding_profile(self) -> EmbeddingProfile:
        """Validated embedding settings without importing optional provider packages."""
        from deepagents_talon.history_profiles import parse_profile  # noqa: PLC0415

        return parse_profile(self.env)

    @property
    def history_vector_path(self) -> Path:
        """Separate SQLite database so embedding cannot block checkpoint writes."""
        return self.history_generation_path("")

    def history_generation_path(self, generation: str) -> Path:
        """Resolve a vector generation within the assistant home.

        Args:
            generation: Validated profile fingerprint, or empty for the legacy index.
        """
        if generation and re.fullmatch(r"[a-f0-9]{64}", generation) is None:
            msg = "Invalid history vector generation"
            raise TalonConfigError(msg)
        suffix = f"-{generation}" if generation else ""
        return self._state_path(f"history-vectors{suffix}.sqlite", "history vector database")

    @property
    def conversation_state_path(self) -> Path:
        """JSON file used for active conversation generations."""
        return self._state_path("conversations.json", "conversation state")

    def _state_path(self, name: str, description: str) -> Path:
        home = self.home.resolve()
        expected_home = self.home.parent.resolve() / self.home.name
        state_path = (home / name).resolve()
        if home != expected_home or state_path.parent != home:
            msg = f"{description} must remain inside the assistant home"
            raise TalonConfigError(msg)
        return state_path

    @property
    def inbound_media_dir(self) -> Path:
        """Directory reserved for downloaded inbound channel media."""
        return self.home / "media" / "inbound"


def _create_home(home: Path) -> None:
    home.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".talon-", dir=home.parent) as directory:
        staged = Path(directory) / "home"
        staged.mkdir(mode=0o700)
        _install_defaults(staged)
        try:
            staged.rename(home)
        except OSError:
            if not home.is_dir():
                raise


def _install_defaults(home: Path) -> None:
    defaults = Path(__file__).with_name("defaults")
    for source in defaults.rglob("AGENTS.md"):
        contents = source.read_text(encoding="utf-8")
        target = home / source.relative_to(defaults)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(contents)


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


def _validate_history_uri(uri: str | None) -> None:
    if uri is None:
        return
    try:
        parsed = urlsplit(uri)
        if parsed.scheme and not parsed.fragment and not any(char.isspace() for char in uri):
            return
    except ValueError:
        pass
    msg = (
        "DEEPAGENTS_TALON_HISTORY_URI must be a URI with a scheme and no whitespace "
        "or fragment; unset it to use the default SQLite archive"
    )
    raise TalonConfigError(msg)
