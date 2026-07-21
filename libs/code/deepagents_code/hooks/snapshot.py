"""Immutable runtime snapshots for Hooks v2 configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from deepagents_code.hooks.models.domain import (
    HookDiagnostic,
    HookEvent,
    NotificationEvent,
    PermissionRequestEvent,
    PostToolUseEvent,
    PreToolUseEvent,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Pattern

    from deepagents_code.hooks.models.config import HooksConfig
    from deepagents_code.hooks.models.domain import HookInvocation

_MATCHABLE_EVENTS = frozenset(
    {
        HookEvent.PERMISSION_REQUEST,
        HookEvent.NOTIFICATION,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
    }
)


@dataclass(frozen=True, slots=True)
class HookHandler:
    """One ordered command handler in a configuration snapshot."""

    id: str
    event: HookEvent
    command: str
    timeout: float | None
    status_message: str | None
    matcher: Pattern[str] | None
    matcher_text: str | None
    matcher_error: str | None = None


@dataclass(frozen=True, slots=True)
class HookMatch:
    """Matched handlers and configuration diagnostics for one invocation."""

    handlers: tuple[HookHandler, ...]
    diagnostics: tuple[HookDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class HooksSnapshot:
    """Immutable, declaration-ordered Hooks v2 runtime configuration."""

    handlers: Mapping[HookEvent, tuple[HookHandler, ...]]

    @classmethod
    def from_config(cls, config: HooksConfig) -> HooksSnapshot:
        """Build an immutable runtime snapshot from validated configuration.

        Args:
            config: Validated Hooks v2 configuration.

        Returns:
            A snapshot whose handler order and matchers cannot change in flight.
        """
        expanded: dict[HookEvent, tuple[HookHandler, ...]] = {}
        for event, groups in config.hooks.items():
            handlers: list[HookHandler] = []
            for group_index, group in enumerate(groups):
                matcher, error = _compile_matcher(group.matcher)
                for handler_index, spec in enumerate(group.hooks):
                    handlers.append(
                        HookHandler(
                            id=f"{event.value}:{group_index}:{handler_index}",
                            event=event,
                            command=spec.command,
                            timeout=spec.timeout,
                            status_message=spec.status_message,
                            matcher=matcher,
                            matcher_text=group.matcher,
                            matcher_error=error,
                        )
                    )
            expanded[event] = tuple(handlers)
        return cls(handlers=MappingProxyType(expanded))

    def match(self, invocation: HookInvocation) -> HookMatch:
        """Return handlers matching an invocation in declaration order.

        Args:
            invocation: Domain invocation to match.

        Returns:
            Matching handlers and diagnostics for invalid configured matchers.
        """
        event = invocation.event.event
        target = _match_target(invocation)
        matched: list[HookHandler] = []
        diagnostics: list[HookDiagnostic] = []
        for handler in self.handlers.get(event, ()):
            if handler.matcher_error is not None:
                diagnostics.append(
                    HookDiagnostic(
                        code="invalid_matcher",
                        severity="warning",
                        message=handler.matcher_error,
                        handler_id=handler.id,
                        field="matcher",
                    )
                )
                continue
            if (
                event not in _MATCHABLE_EVENTS
                or handler.matcher is None
                or (target is not None and handler.matcher.search(target) is not None)
            ):
                matched.append(handler)
        return HookMatch(handlers=tuple(matched), diagnostics=tuple(diagnostics))


def _compile_matcher(value: str | None) -> tuple[Pattern[str] | None, str | None]:
    if not value:
        return None, None
    try:
        return re.compile(value), None
    except re.error as exc:
        return None, f"Invalid hook matcher {value!r}: {exc}"


def _match_target(invocation: HookInvocation) -> str | None:
    event = invocation.event
    if isinstance(
        event,
        PermissionRequestEvent | PreToolUseEvent | PostToolUseEvent,
    ):
        return event.call.name
    if isinstance(event, NotificationEvent):
        return event.notification.type
    return None
