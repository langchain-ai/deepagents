"""Lightweight hook dispatch for external tool integration.

Loads hook configuration from `~/.deepagents/hooks.json` and fires matching
commands with JSON payloads on stdin. All dispatch is fire-and-forget: commands
run in the background and failures are logged but never bubble up to the caller.

Config format (`~/.deepagents/hooks.json`):

```json
{"hooks": [{"command": ["bash", "adapter.sh"], "events": ["session.start"]}]}
```

If `events` is omitted or empty the hook receives **all** events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess  # noqa: S404
from typing import Any

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

_HOOKS_PATH = DEFAULT_CONFIG_DIR / "hooks.json"

# Cached config — loaded lazily on first dispatch.
_hooks_config: list[dict[str, Any]] | None = None


def _load_hooks() -> list[dict[str, Any]]:
    """Load and cache hook definitions from the config file.

    Returns:
        An empty list when the file is missing or malformed so that normal
            execution is never interrupted.
    """
    global _hooks_config  # noqa: PLW0603
    if _hooks_config is not None:
        return _hooks_config

    if not _HOOKS_PATH.is_file():
        _hooks_config = []
        return _hooks_config

    try:
        data = json.loads(_HOOKS_PATH.read_text())
        _hooks_config = data.get("hooks", [])
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load hooks config from %s: %s", _HOOKS_PATH, exc)
        _hooks_config = []

    return _hooks_config


def _dispatch_hook_sync(
    event: str, payload_bytes: bytes, hooks: list[dict[str, Any]]
) -> None:
    """Synchronous hook dispatch — runs in a thread to avoid blocking the event loop."""
    for hook in hooks:
        command = hook.get("command")
        if not command:
            continue

        events = hook.get("events")
        # Empty/missing events list means "subscribe to everything".
        if events and event not in events:
            continue

        try:
            subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            ).communicate(input=payload_bytes, timeout=5)
        except subprocess.TimeoutExpired:
            logger.debug("Hook command timed out for event %s: %s", event, command)
        except Exception:
            logger.debug(
                "Hook dispatch failed for event %s: %s",
                event,
                command,
                exc_info=True,
            )


async def dispatch_hook(event: str, payload: dict[str, Any]) -> None:
    """Fire matching hook commands with *payload* serialised as JSON on stdin.

    The *event* name is automatically injected into the payload under the
    `'event'` key so callers don't need to duplicate it.

    The blocking subprocess work is offloaded to a thread so the caller's event
    loop is never stalled.

    Each command is started as a detached subprocess (fire-and-forget).

    Errors are logged at debug level and never propagated.

    Args:
        event: Dotted event name (e.g. `'session.start'`).
        payload: Arbitrary JSON-serialisable dict sent on the command's stdin.
    """
    hooks = _load_hooks()
    if not hooks:
        return

    payload_bytes = json.dumps({"event": event, **payload}).encode()
    await asyncio.to_thread(_dispatch_hook_sync, event, payload_bytes, hooks)
