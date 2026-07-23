"""Legacy dotted-event migration helpers for Hooks v2 configuration.

These utilities are intentionally not activated at legacy dispatch call sites.
Lifecycle wiring belongs to later tickets; this module only converts
semantically equivalent config when an explicit loader asks for it.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess  # noqa: S404  # Legacy hooks are trusted user-configured commands.
import sys
from binascii import Error as BinasciiError
from typing import TYPE_CHECKING

from deepagents_code.hooks.env import HOOK_SUBPROCESS_TIMEOUT
from deepagents_code.hooks.models.config import (
    CommandHandlerSpec,
    HooksConfig,
    MatcherGroup,
)
from deepagents_code.hooks.models.domain import HookEvent

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LEGACY_EVENT_MAP: dict[str, tuple[HookEvent, str | None]] = {
    "session.start": (HookEvent.USER_PROMPT_SUBMIT, None),
    "user.prompt": (HookEvent.USER_PROMPT_SUBMIT, None),
    "task.complete": (HookEvent.NOTIFICATION, "agent_completed"),
    "session.end": (HookEvent.SESSION_END, None),
    "context.offload": (HookEvent.PRE_COMPACT, "manual"),
    "context.compact": (HookEvent.PRE_COMPACT, "manual"),
    "input.required": (HookEvent.NOTIFICATION, "agent_needs_input"),
}
# Outer runner grace so the nested adapter timeout can fire first on Windows,
# where killing the adapter process may not reap its descendants.
_ADAPTER_OUTER_TIMEOUT_SECONDS = HOOK_SUBPROCESS_TIMEOUT + 1.0
_ADAPTER_MODULE = "deepagents_code.hooks.migration"
_ADAPTER_ARGUMENT_COUNT = 2
_THREAD_ID_EVENTS = frozenset({"session.start", "task.complete", "session.end"})


def migrate_legacy_hooks(
    legacy_hooks: Sequence[Mapping[str, object]],
) -> HooksConfig:
    """Convert legacy dotted-event hook entries into Hooks v2 configuration.

    Each distinct legacy event name becomes its own matcher group so a single
    entry subscribed to both `session.start` and `user.prompt` still runs once
    per mapped name with the matching reconstructed stdin payload.

    Args:
        legacy_hooks: Entries from the legacy `hooks.json` list form.

    Returns:
        A validated Hooks v2 configuration containing only migratable events.
    """
    grouped: dict[HookEvent, list[MatcherGroup]] = {}
    for entry in legacy_hooks:
        command = entry.get("command")
        if not isinstance(command, list) or not command:
            continue
        if not all(isinstance(part, str) for part in command):
            continue
        argv = [part for part in command if isinstance(part, str)]
        if len(argv) != len(command):
            continue
        events = entry.get("events")
        event_names: list[str]
        if events is None or events == []:
            event_names = list(_LEGACY_EVENT_MAP)
        elif isinstance(events, list):
            event_names = list(
                dict.fromkeys(name for name in events if isinstance(name, str))
            )
        else:
            continue
        for event_name in event_names:
            mapped = _LEGACY_EVENT_MAP.get(event_name)
            if mapped is None:
                continue
            event, matcher = mapped
            adapter_argv = _adapter_argv(argv, event_name)
            grouped.setdefault(event, []).append(
                MatcherGroup(
                    matcher=matcher,
                    hooks=[
                        CommandHandlerSpec(
                            type="command",
                            command=_shell_command(adapter_argv, os_name=os.name),
                            argv=adapter_argv,
                            timeout=_ADAPTER_OUTER_TIMEOUT_SECONDS,
                            inherit_environ=True,
                        )
                    ],
                )
            )
    return HooksConfig(hooks=grouped)


def _adapter_argv(argv: list[str], legacy_event: str) -> list[str]:
    encoded_argv = base64.urlsafe_b64encode(
        json.dumps(argv, separators=(",", ":")).encode()
    ).decode()
    return [sys.executable, "-m", _ADAPTER_MODULE, legacy_event, encoded_argv]


def _shell_command(argv: Sequence[str], *, os_name: str) -> str:
    if os_name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _legacy_payload(legacy_event: str, payload: Mapping[str, object]) -> bytes:
    legacy_payload: dict[str, object] = {"event": legacy_event}
    if legacy_event in _THREAD_ID_EVENTS:
        thread_id = payload.get("session_id")
        if isinstance(thread_id, str):
            legacy_payload["thread_id"] = thread_id
    return json.dumps(legacy_payload, default=str).encode()


def _decode_argv(value: str) -> list[str] | None:
    try:
        decoded: object = json.loads(base64.urlsafe_b64decode(value))
    except (BinasciiError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if (
        not isinstance(decoded, list)
        or not decoded
        or not all(isinstance(part, str) for part in decoded)
    ):
        return None
    return [part for part in decoded if isinstance(part, str)]


def _run_adapter(args: Sequence[str]) -> int:
    # Nested hook exit status is ignored (side-effect-only). Argument, decode,
    # stdin, launch, and timeout failures return nonzero for runner diagnostics.
    if len(args) != _ADAPTER_ARGUMENT_COUNT:
        return 1
    legacy_event, encoded_argv = args
    argv = _decode_argv(encoded_argv)
    if argv is None:
        return 1
    try:
        payload: object = json.loads(sys.stdin.buffer.read())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1
    if not isinstance(payload, dict):
        return 1
    wire_payload = {str(key): value for key, value in payload.items()}
    try:
        subprocess.run(  # noqa: S603  # Runs the trusted legacy hook argv directly.
            argv,
            input=_legacy_payload(legacy_event, wire_payload),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=HOOK_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return 1
    except OSError:
        return 1
    return 0


def is_legacy_hooks_document(data: object) -> bool:
    """Return whether `data` looks like the legacy list-shaped hooks document.

    Args:
        data: Parsed JSON root.

    Returns:
        `True` when `hooks` is a list of command entries rather than an event map.
    """
    if not isinstance(data, dict):
        return False
    hooks = data.get("hooks")
    if not isinstance(hooks, list):
        return False
    return all(isinstance(item, dict) for item in hooks)


if __name__ == "__main__":
    raise SystemExit(_run_adapter(sys.argv[1:]))
