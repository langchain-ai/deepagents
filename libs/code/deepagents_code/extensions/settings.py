"""Resolve extension configuration from user config and environment."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from deepagents_code._env_vars import (
    EXTENSIONS,
    EXTENSIONS_TRUST,
    classify_env_bool,
)

logger = logging.getLogger(__name__)


class TrustPolicy(StrEnum):
    """Default decision for project-authored extension code."""

    ASK = "ask"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class ExtensionSettings:
    """Effective extension configuration for one process."""

    enabled: bool = True
    """Whether extension discovery is enabled."""

    trust: TrustPolicy = TrustPolicy.ASK
    """Fallback policy for project extension code."""

    extra_files: tuple[Path, ...] = ()
    """Explicit files from trusted user configuration."""

    extra_dirs: tuple[Path, ...] = ()
    """Explicit directories from trusted user configuration."""


def _read_config_section() -> dict[str, object]:
    import tomllib

    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

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


def parse_trust_policy(raw: object) -> TrustPolicy | None:
    """Parse a project extension trust policy.

    Args:
        raw: Value from configuration or the environment.

    Returns:
        Parsed policy, or `None` when the value is invalid.
    """
    if not isinstance(raw, str):
        return None
    try:
        return TrustPolicy(raw.strip().lower())
    except ValueError:
        return None


def _parse_paths(raw: object, *, name: str) -> tuple[Path, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        logger.warning("Ignoring [extensions].%s (expected a list of paths)", name)
        return ()
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    root = DEFAULT_CONFIG_PATH.parent
    paths: list[Path] = []
    for item in cast("list[str]", raw):
        if not (value := item.strip()):
            logger.warning("Ignoring a blank [extensions].%s path", name)
            continue
        path = Path(value).expanduser()
        paths.append(path if path.is_absolute() else root / path)
    return tuple(paths)


def load_extension_settings() -> ExtensionSettings:
    """Resolve extension settings with environment overrides.

    Invalid environment values fall through to user configuration rather than
    silently changing the runtime behavior reported by `dcode config`.

    Returns:
        Effective settings with safe defaults for malformed values.
    """
    section = _read_config_section()

    configured_enabled = section.get("enabled")
    enabled = configured_enabled if isinstance(configured_enabled, bool) else True
    env_enabled = os.environ.get(EXTENSIONS)
    if env_enabled is not None and env_enabled.strip():
        parsed_enabled = classify_env_bool(env_enabled)
        if parsed_enabled is None:
            logger.warning("Ignoring %s=%r (expected bool)", EXTENSIONS, env_enabled)
        else:
            enabled = parsed_enabled

    trust = parse_trust_policy(section.get("trust")) or TrustPolicy.ASK
    env_trust = os.environ.get(EXTENSIONS_TRUST)
    if env_trust is not None and env_trust.strip():
        parsed_trust = parse_trust_policy(env_trust)
        if parsed_trust is None:
            logger.warning(
                "Ignoring %s=%r (expected ask, always, or never)",
                EXTENSIONS_TRUST,
                env_trust,
            )
        else:
            trust = parsed_trust

    return ExtensionSettings(
        enabled=enabled,
        trust=trust,
        extra_files=_parse_paths(section.get("extra_files"), name="extra_files"),
        extra_dirs=_parse_paths(section.get("extra_dirs"), name="extra_dirs"),
    )
