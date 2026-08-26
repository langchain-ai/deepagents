"""Resolve extension configuration from user config and environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deepagents_code.configuration.resolver import USER_RANK, get_config_resolver

if TYPE_CHECKING:
    from deepagents_code.configuration.resolver import ConfigResolver

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

    extra_paths: tuple[Path, ...] = ()
    """Explicit files or directories from trusted user configuration."""


def _read_config_section(resolver: ConfigResolver) -> dict[str, object]:
    snapshot = resolver.toml_snapshot(USER_RANK)
    if snapshot is None:
        logger.error("Shared resolver has no user config provider")
        return {}
    section = snapshot.data.get("extensions")
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
    """Resolve extension settings through the shared config generation.

    Returns:
        Effective settings with safe defaults for malformed values.

    Raises:
        RuntimeError: If the extension manifest options are unavailable.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics, get_option

    resolver = get_config_resolver()
    enabled_option = get_option("extensions.enabled")
    trust_option = get_option("extensions.trust")
    if enabled_option is None or trust_option is None:
        msg = "Extension options are missing from the config manifest"
        raise RuntimeError(msg)
    enabled_resolved = resolver.get(enabled_option)
    _emit_ranked_diagnostics(enabled_option, enabled_resolved)
    trust_resolved = resolver.get(trust_option)
    _emit_ranked_diagnostics(trust_option, trust_resolved)
    enabled = cast("bool", enabled_resolved.value)
    trust = TrustPolicy(cast("str", trust_resolved.value))
    section = _read_config_section(resolver)

    return ExtensionSettings(
        enabled=enabled,
        trust=trust,
        extra_paths=_parse_paths(section.get("extra_paths"), name="extra_paths"),
    )
