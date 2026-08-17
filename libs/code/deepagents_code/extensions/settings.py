"""User configuration for the extension system.

Read from `[extensions]` in `~/.deepagents/config.toml`, with environment
variables taking precedence so a single run can be adjusted without editing
config:

```toml
[extensions]
enabled = true
paths = ["~/work/dcode-extensions", "~/scratch/one-off.py"]
trust = "ask"  # ask | always | never
```

- `DEEPAGENTS_CODE_EXTENSIONS=0` disables loading entirely.
- `DEEPAGENTS_CODE_EXTENSIONS_PATHS` is a colon-separated list of extra paths.
- `DEEPAGENTS_CODE_EXTENSIONS_TRUST` overrides the project-trust policy.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

ENABLED_ENV = "DEEPAGENTS_CODE_EXTENSIONS"
PATHS_ENV = "DEEPAGENTS_CODE_EXTENSIONS_PATHS"
TRUST_ENV = "DEEPAGENTS_CODE_EXTENSIONS_TRUST"

_FALSEY = {"0", "false", "no", "off"}


class TrustPolicy(StrEnum):
    """Default answer for the project-extension trust decision."""

    ASK = "ask"
    """Prompt interactively; skip project extensions when non-interactive."""

    ALWAYS = "always"
    """Load project extensions without prompting."""

    NEVER = "never"
    """Never load project extensions."""


@dataclass(frozen=True, slots=True)
class ExtensionSettings:
    """Resolved extension configuration for one run."""

    enabled: bool = True
    """Whether any extension source is loaded."""

    paths: tuple[Path, ...] = field(default_factory=tuple)
    """Extra files or directories listed in user configuration."""

    trust: TrustPolicy = TrustPolicy.ASK
    """Policy applied when a project has no persisted trust decision."""


def _read_config_section() -> dict[str, object]:
    """Read `[extensions]` from `~/.deepagents/config.toml`.

    Returns:
        The section contents, or an empty mapping when absent or unreadable.
    """
    import tomllib  # deferred: only needed when config is read

    from deepagents_code.model_config import (
        DEFAULT_CONFIG_PATH,  # deferred to keep the startup path light
    )

    try:
        with DEFAULT_CONFIG_PATH.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not read extensions config from %s",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
        return {}
    section = data.get("extensions")
    return section if isinstance(section, dict) else {}


def load_extension_settings() -> ExtensionSettings:
    """Resolve extension settings from config and environment.

    Returns:
        The effective settings, falling back to defaults for anything absent or
            malformed.
    """
    section = _read_config_section()

    enabled = section.get("enabled", True)
    env_enabled = os.environ.get(ENABLED_ENV)
    if env_enabled is not None:
        enabled = env_enabled.strip().lower() not in _FALSEY

    env_paths = os.environ.get(PATHS_ENV)
    if env_paths:
        raw_paths: list[str] = [part for part in env_paths.split(":") if part.strip()]
    else:
        configured = section.get("paths")
        raw_paths = (
            [entry for entry in configured if isinstance(entry, str)]
            if isinstance(configured, list)
            else []
        )

    raw_trust = os.environ.get(TRUST_ENV) or section.get("trust")
    try:
        trust = TrustPolicy(str(raw_trust).strip().lower())
    except ValueError:
        if raw_trust is not None:
            logger.warning("Ignoring unknown extensions trust policy %r", raw_trust)
        trust = TrustPolicy.ASK

    return ExtensionSettings(
        enabled=bool(enabled),
        paths=tuple(Path(entry).expanduser() for entry in raw_paths),
        trust=trust,
    )
