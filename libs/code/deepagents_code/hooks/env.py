"""Sanitized subprocess environments for Hooks v2 command handlers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from deepagents_code.config_manifest import _is_secret_env

if TYPE_CHECKING:
    from collections.abc import Mapping


def sanitize_hook_environ(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an inherited environment safe to pass to hook subprocesses.

    Strips values whose names look like secrets. Hooks are user-authored trusted
    code, but secret values should not be ambiently available.

    Args:
        source: Environment to sanitize. Defaults to `os.environ`.

    Returns:
        A new environment mapping suitable for `asyncio.create_subprocess_shell`.
    """
    env = os.environ if source is None else source
    return {key: value for key, value in env.items() if not _is_secret_env(key)}
