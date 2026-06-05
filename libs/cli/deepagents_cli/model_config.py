"""Prefixed environment variable resolution for the deploy CLI.

Reads `LANGSMITH_*` / `LANGCHAIN_*` env vars (and any other canonical name
the deploy pipeline needs) with a `DEEPAGENTS_CLI_` prefix override.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENV_PREFIX = "DEEPAGENTS_CLI_"
"""Prefix that scopes env vars to the deploy CLI without colliding with
identically-named shell exports (e.g., `DEEPAGENTS_CLI_LANGSMITH_API_KEY`)."""


def resolve_env_var(name: str) -> str | None:
    """Look up an env var with `DEEPAGENTS_CLI_` prefix override.

    Checks `DEEPAGENTS_CLI_{name}` first, then falls back to `{name}`.

    If the prefixed variable is *present* in the environment (even as an
    empty string), the canonical variable is never consulted. This lets
    users set `DEEPAGENTS_CLI_X=""` to shadow a canonically-set key — the
    function returns `None` (empty strings are normalized to `None`),
    effectively suppressing the canonical value.

    If `name` already carries the prefix, the double-prefixed lookup is
    skipped to avoid nonsensical `DEEPAGENTS_CLI_DEEPAGENTS_CLI_*` reads.

    Args:
        name: The canonical environment variable name
            (e.g. `LANGSMITH_API_KEY`).

    Returns:
        The resolved value, or `None` when absent or empty.
    """
    if not name.startswith(_ENV_PREFIX):
        prefixed = f"{_ENV_PREFIX}{name}"
        if prefixed in os.environ:
            val = os.environ[prefixed]
            if not val and os.environ.get(name):
                logger.debug(
                    "%s is set but empty, blocking non-empty %s. "
                    "Unset %s to use the canonical variable.",
                    prefixed,
                    name,
                    prefixed,
                )
            if val:
                logger.debug("Resolved %s from %s", name, prefixed)
            return val or None
    return os.environ.get(name) or None
