"""Legacy dotted-event migration helpers for Hooks v2 configuration.

These utilities are intentionally not activated at legacy dispatch call sites.
Lifecycle wiring belongs to later tickets; this module only converts
semantically equivalent config when an explicit loader asks for it.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

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
    "task.complete": (HookEvent.STOP, None),
    "session.end": (HookEvent.SESSION_END, None),
    "context.offload": (HookEvent.PRE_COMPACT, "manual"),
    "context.compact": (HookEvent.PRE_COMPACT, "manual"),
    "input.required": (HookEvent.NOTIFICATION, "agent_needs_input"),
}
LEGACY_COMMAND_TIMEOUT_SECONDS = 5.0


def migrate_legacy_hooks(
    legacy_hooks: Sequence[Mapping[str, object]],
) -> HooksConfig:
    """Convert legacy dotted-event hook entries into Hooks v2 configuration.

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
            event_names = [name for name in events if isinstance(name, str)]
        else:
            continue
        shell_command = _observer_command(argv)
        targets = dict.fromkeys(
            mapped
            for event_name in event_names
            if (mapped := _LEGACY_EVENT_MAP.get(event_name)) is not None
        )
        for event, matcher in targets:
            grouped.setdefault(event, []).append(
                MatcherGroup(
                    matcher=matcher,
                    hooks=[
                        CommandHandlerSpec(
                            type="command",
                            command=shell_command,
                            timeout=LEGACY_COMMAND_TIMEOUT_SECONDS,
                        )
                    ],
                )
            )
    return HooksConfig(hooks=grouped)


def _observer_command(argv: list[str]) -> str:
    """Preserve legacy side-effect-only output and exit semantics.

    Args:
        argv: Legacy command and arguments.

    Returns:
        A shell command that discards output and normalizes the exit status.
    """
    command = shlex.join(argv)
    return f"({command}) >/dev/null 2>&1 || true"


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
