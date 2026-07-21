"""Sanitized subprocess environments for Hooks v2 command handlers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from deepagents_code.config_manifest import _SECRET_NAME_MARKERS

if TYPE_CHECKING:
    from collections.abc import Mapping

_OTEL_PREFIX = "OTEL_"


def is_secret_env_name(name: str) -> bool:
    """Return whether an environment variable name looks like secret material.

    Uses the same credential-name markers as the config manifest, compared
    case-insensitively so mixed-case env names are handled consistently.

    Args:
        name: Environment variable name.

    Returns:
        `True` when the name matches the repository secret-name policy.
    """
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_NAME_MARKERS)


def sanitize_hook_environ(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an inherited environment safe to pass to hook subprocesses.

    Removes OpenTelemetry exporter variables (matching the compatible harness)
    and strips values whose names look like secrets. Hooks are user-authored
    trusted code, but secret values should not be ambiently available.

    Args:
        source: Environment to sanitize. Defaults to `os.environ`.

    Returns:
        A new environment mapping suitable for `asyncio.create_subprocess_shell`.
    """
    env = dict(os.environ if source is None else source)
    sanitized: dict[str, str] = {}
    for key, value in env.items():
        if key.startswith(_OTEL_PREFIX):
            continue
        if is_secret_env_name(key):
            continue
        sanitized[key] = value
    return sanitized
