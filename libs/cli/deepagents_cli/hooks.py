"""Lightweight hook dispatch for external tool integration.

Loads hook configuration from `~/.deepagents/hooks.json` and fires matching
commands with JSON payloads on stdin. Subprocess work is offloaded to a
background thread so the caller's event loop is never stalled. Failures are
logged but never bubble up to the caller.

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

_HOOKS_PATH = DEFAULT_CONFIG_DIR / "hooks.json"

_hooks_config: list[dict[str, Any]] | None = None
"""Cached config — loaded lazily on first dispatch."""

_background_tasks: set[asyncio.Task[None]] = set()
"""Strong references to fire-and-forget tasks to prevent GC."""


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
        if not isinstance(data, dict):
            logger.warning(
                "Hooks config at %s must be a JSON object, got %s",
                _HOOKS_PATH,
                type(data).__name__,
            )
            _hooks_config = []
            return _hooks_config
        hooks = data.get("hooks", [])
        if not isinstance(hooks, list):
            logger.warning(
                "Hooks config 'hooks' key at %s must be a list, got %s",
                _HOOKS_PATH,
                type(hooks).__name__,
            )
            _hooks_config = []
            return _hooks_config
        _hooks_config = hooks
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load hooks config from %s: %s", _HOOKS_PATH, exc)
        _hooks_config = []

    return _hooks_config


def _run_single_hook(command: list[str], event: str, payload_bytes: bytes) -> None:
    """Execute a single hook command, writing the JSON payload to its stdin.

    Uses `subprocess.run` which automatically kills the child process on
    timeout, preventing zombie/orphan process leaks.

    Args:
        command: The command and arguments to run.
        event: Event name (for logging).
        payload_bytes: JSON payload to write to the command's stdin.
    """
    try:
        subprocess.run(  # noqa: S603
            command,
            input=payload_bytes,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Hook command timed out (>5s) for event %s: %s", event, command)
    except (FileNotFoundError, PermissionError) as exc:
        logger.warning("Hook command failed for event %s: %s — %s", event, command, exc)
    except Exception:
        logger.debug(
            "Hook dispatch failed for event %s: %s",
            event,
            command,
            exc_info=True,
        )


def _dispatch_hook_sync(
    event: str, payload_bytes: bytes, hooks: list[dict[str, Any]]
) -> None:
    """Dispatch matching hooks, running them concurrently via a thread pool.

    Iterates over all configured hooks, skipping those whose event filter
    does not match or whose `command` is missing/invalid. Matching hooks are
    executed concurrently with a 5-second timeout per command. Errors are caught
    per-hook and logged without propagating.

    Args:
        event: Dotted event name (e.g. `'session.start'`).
        payload_bytes: JSON payload to write to each command's stdin.
        hooks: List of hook definition dicts from the config file.
    """
    matching: list[list[str]] = []
    for hook in hooks:
        command = hook.get("command")
        if not isinstance(command, list) or not command:
            continue

        events = hook.get("events")
        # Empty/missing events list means "subscribe to everything".
        if events and event not in events:
            continue

        matching.append(command)

    if not matching:
        return

    if len(matching) == 1:
        _run_single_hook(matching[0], event, payload_bytes)
        return

    with ThreadPoolExecutor(max_workers=len(matching)) as pool:
        futures = [
            pool.submit(_run_single_hook, cmd, event, payload_bytes) for cmd in matching
        ]
        for future in futures:
            future.result()


async def dispatch_hook(event: str, payload: dict[str, Any]) -> None:
    """Fire matching hook commands with `payload` serialized as JSON on stdin.

    The `event` name is automatically injected into the payload under the
    `"event"` key so callers don't need to duplicate it.

    The blocking subprocess work is offloaded to a thread so the caller's
    event loop is never stalled. Matching hooks run concurrently, each with
    a 5-second timeout. Errors are logged and never propagated.

    Args:
        event: Dotted event name (e.g. `'session.start'`).
        payload: Arbitrary JSON-serializable dict sent on the command's stdin.
    """
    try:
        hooks = _load_hooks()
        if not hooks:
            return

        payload_bytes = json.dumps({"event": event, **payload}).encode()
        await asyncio.to_thread(_dispatch_hook_sync, event, payload_bytes, hooks)
    except Exception:
        logger.warning(
            "Unexpected error in dispatch_hook for event %s",
            event,
            exc_info=True,
        )


def dispatch_hook_fire_and_forget(event: str, payload: dict[str, Any]) -> None:
    """Schedule `dispatch_hook` as a background task with a strong reference.

    Use this instead of bare `create_task(dispatch_hook(...))` to prevent the
    task from being garbage collected before completion.

    Safe to call from sync code as long as an event loop is running.

    Args:
        event: Dotted event name (e.g. `'session.start'`).
        payload: Arbitrary JSON-serializable dict sent on the command's stdin.
    """
    task = asyncio.get_running_loop().create_task(dispatch_hook(event, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
