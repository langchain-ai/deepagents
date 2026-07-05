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

Onboarding emits `user.name.set` with `{"name": "...", "assistant_id": "..."}`
after the user submits a non-empty preferred name.

`tool.use` fires before a tool call once its streamed arguments parse into a
complete value *and* its tool-call id is known; a call whose arguments never
parse, or that carries no id, is skipped. `tool.result` fires after every tool
call reaches a terminal state — successful execution, failure, or HITL
rejection/cancellation:

```jsonc
{"event": "tool.use", "tool_name": "write_file", "tool_id": "toolu_abc123",
 "tool_args": {"file_path": "src/foo.py", "content": "..."}}

{"event": "tool.result", "tool_name": "write_file", "tool_id": "toolu_abc123",
 "tool_args": {"file_path": "src/foo.py", "content": "..."},
 "tool_status": "success", "tool_output": "Updated file src/foo.py"}

{"event": "tool.error", "tool_names": ["write_file"]}
```

`tool_args` is the parsed tool-call arguments; a non-object value (rare) is
wrapped as `{"value": ...}`. `tool_output` is the tool's returned content,
truncated to `HOOK_TOOL_OUTPUT_LIMIT` characters (`tool_args` is not truncated).
`tool_status` is `"success"` or `"error"`; `"error"` covers both a tool that
raised and a call the user rejected or cancelled. Whenever a `tool.result` has
`tool_status: "error"`, `tool.error` (payload `{"tool_names": [<name>]}`) fires
alongside it, so existing `tool.error` hooks are unaffected.

`tool_args` is `{}` whenever a `tool.result` cannot be correlated back to a
`tool.use` — either because the call carried no id (then `tool_id` is `null`) or
because no `tool.use` fired for it (e.g. its args never parsed), in which case
`tool_id` may still be the real string id.

Ordering: each event is dispatched fire-and-forget (see
`dispatch_hook_fire_and_forget`) and every matching hook command runs in its own
subprocess. A `tool.use` is *dispatched* before its `tool.result`, but the two
run concurrently, so a hook subscribed to both may observe them out of order, and
events from parallel tool calls interleave freely. Correlate by `tool_id` rather
than relying on arrival order — there is no cross-event delivery-ordering
guarantee.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess  # noqa: S404
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

HOOK_TOOL_OUTPUT_LIMIT = 2000
"""Max characters of `tool_output` included in `tool.result` hook payloads.

Bounds payload size (data-amplification guard) while keeping enough of the
tool's output to be useful to audit/notification hooks. Shared by both the
interactive and headless dispatch paths so the cap never drifts between them.
Only `tool_output` is capped; `tool_args` is passed through in full so hooks
that act on the arguments (e.g. a linter reading a `write_file` `content`) see
the exact value the tool received.
"""

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

    from deepagents_code.model_config import DEFAULT_CONFIG_DIR

    hooks_path = DEFAULT_CONFIG_DIR / "hooks.json"

    if not hooks_path.is_file():
        _hooks_config = []
        return _hooks_config

    try:
        data = json.loads(hooks_path.read_text())
        if not isinstance(data, dict):
            logger.warning(
                "Hooks config at %s must be a JSON object, got %s",
                hooks_path,
                type(data).__name__,
            )
            _hooks_config = []
            return _hooks_config
        hooks = data.get("hooks", [])
        if not isinstance(hooks, list):
            logger.warning(
                "Hooks config 'hooks' key at %s must be a list, got %s",
                hooks_path,
                type(hooks).__name__,
            )
            _hooks_config = []
            return _hooks_config
        _hooks_config = hooks
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load hooks config from %s: %s", hooks_path, exc)
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
        # Unexpected failure (e.g. ENOEXEC for a non-executable hook file, an
        # embedded null byte, or fd/memory exhaustion). These are the failures
        # we understand least, so surface them at warning — the expected
        # timeout / not-found / permission cases above are also warnings, and a
        # silent debug here would hide a hook that never fires.
        logger.warning(
            "Hook dispatch failed unexpectedly for event %s: %s",
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


async def dispatch_hook(event: str, payload: Mapping[str, Any]) -> None:
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


def dispatch_hook_fire_and_forget(event: str, payload: Mapping[str, Any]) -> None:
    """Schedule `dispatch_hook` as a background task with a strong reference.

    Use this instead of bare `create_task(dispatch_hook(...))` to prevent the
    task from being garbage collected before completion.

    Safe to call from sync code as long as an event loop is running.

    Args:
        event: Dotted event name (e.g. `'session.start'`).
        payload: Arbitrary JSON-serializable dict sent on the command's stdin.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No running event loop; skipping hook for %s", event)
        return
    task = loop.create_task(dispatch_hook(event, payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def has_pending_hooks() -> bool:
    """Return whether fire-and-forget hook tasks are still in flight."""
    return any(not task.done() for task in _background_tasks)


async def drain_pending_hooks() -> None:
    """Await all in-flight fire-and-forget hook tasks.

    Call this before the event loop tears down (e.g. at the end of a headless
    run driven by `asyncio.run`) so background dispatches — most importantly the
    final `tool.result` — are not cancelled mid-flight and silently dropped.
    Each task's exceptions are already swallowed inside `dispatch_hook`, and any
    stragglers are collected with `return_exceptions=True`, so this never
    raises.

    Precondition: this snapshots the in-flight set once and awaits it, so any
    hook scheduled *after* the snapshot (during the await) is not drained. Call
    it only once no further dispatches are possible — i.e. after streaming has
    ended — as both current callers do.
    """
    # Snapshot: tasks remove themselves from the set via their done-callback as
    # they finish, so iterating the live set while gathering would mutate it.
    pending = list(_background_tasks)
    if not pending:
        return
    await asyncio.gather(*pending, return_exceptions=True)
