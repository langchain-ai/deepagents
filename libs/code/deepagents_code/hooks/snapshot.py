"""Immutable runtime snapshots for Hooks v2 configuration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from deepagents_code.hooks.capabilities import get_event_spec
from deepagents_code.hooks.loading import compute_snapshot_id
from deepagents_code.hooks.models.domain import (
    HookDiagnostic,
    HookEvent,
    NotificationEvent,
    PermissionRequestEvent,
    PostToolUseEvent,
    PreCompactEvent,
    PreToolUseEvent,
    SessionEndEvent,
    SessionStartEvent,
    SubagentStartEvent,
    SubagentStopEvent,
)
from deepagents_code.hooks.tools import to_wire_tool_name

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Pattern

    from deepagents_code.hooks.models.config import HooksConfig
    from deepagents_code.hooks.models.domain import HookInvocation

logger = logging.getLogger(__name__)

# Claude-compatible exact-match character set (letters, digits, _, -, spaces, ,, |).
_EXACT_MATCHER = re.compile(r"^[\w\s,\-|]+$")


@dataclass(frozen=True, slots=True)
class HookHandler:
    """One ordered command handler in a configuration snapshot."""

    id: str
    event: HookEvent
    command: str
    timeout: float | None
    status_message: str | None
    matcher: Pattern[str] | frozenset[str] | None
    matcher_text: str | None


@dataclass(frozen=True, slots=True)
class HookMatch:
    """Matched handlers for one invocation."""

    handlers: tuple[HookHandler, ...]
    diagnostics: tuple[HookDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class HooksSnapshot:
    """Immutable, declaration-ordered Hooks v2 runtime configuration."""

    handlers: Mapping[HookEvent, tuple[HookHandler, ...]]
    snapshot_id: str
    diagnostics: tuple[HookDiagnostic, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: HooksConfig,
        *,
        diagnostics: tuple[HookDiagnostic, ...] = (),
        snapshot_id: str | None = None,
    ) -> HooksSnapshot:
        """Build an immutable runtime snapshot from validated configuration.

        Invalid matcher groups are rejected at compile time with a warning
        diagnostic; their handlers are never added to the snapshot.

        Args:
            config: Validated Hooks v2 configuration.
            diagnostics: Diagnostics retained from configuration loading.
            snapshot_id: Optional precomputed canonical hash. When omitted, it
                is derived from `config`.

        Returns:
            A snapshot whose handler order, matchers, and id cannot change.

        Raises:
            ValueError: If `snapshot_id` disagrees with the canonical config.
        """
        canonical_id = compute_snapshot_id(config)
        if snapshot_id is not None and snapshot_id != canonical_id:
            msg = "Provided snapshot_id does not match canonical configuration"
            raise ValueError(msg)
        expanded: dict[HookEvent, tuple[HookHandler, ...]] = {}
        compile_diagnostics: list[HookDiagnostic] = list(diagnostics)
        for event, groups in config.hooks.items():
            matcher_field = get_event_spec(event).matcher_field
            handlers: list[HookHandler] = []
            for group_index, group in enumerate(groups):
                if matcher_field is None and group.matcher not in {None, "", "*"}:
                    message = (
                        f"Rejected hook group {event.value}:{group_index}: "
                        f"{event.value} does not support matchers"
                    )
                    logger.warning(message)
                    compile_diagnostics.append(
                        HookDiagnostic(
                            code="unsupported_matcher",
                            severity="warning",
                            message=message,
                            field="matcher",
                        )
                    )
                    continue
                matcher, error = _compile_matcher(group.matcher)
                if error is not None:
                    message = (
                        f"Rejected hook group {event.value}:{group_index}: {error}"
                    )
                    logger.warning(message)
                    compile_diagnostics.append(
                        HookDiagnostic(
                            code="invalid_matcher",
                            severity="warning",
                            message=message,
                            field="matcher",
                        )
                    )
                    continue
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
                        )
                    )
            expanded[event] = tuple(handlers)
        return cls(
            handlers=MappingProxyType(expanded),
            snapshot_id=canonical_id,
            diagnostics=tuple(compile_diagnostics),
        )

    def match(self, invocation: HookInvocation) -> HookMatch:
        """Return handlers matching an invocation in declaration order.

        Args:
            invocation: Domain invocation to match.

        Returns:
            Matching handlers. Matcher compile failures are reported on the
            snapshot itself, not per invocation.
        """
        event = invocation.event.event
        matcher_field = get_event_spec(event).matcher_field
        target = _match_target(invocation, matcher_field)
        matched = tuple(
            handler
            for handler in self.handlers.get(event, ())
            if _handler_matches(handler, matcher_field, target)
        )
        return HookMatch(handlers=matched)


def _compile_matcher(
    value: str | None,
) -> tuple[Pattern[str] | frozenset[str] | None, str | None]:
    """Compile a Claude-compatible matcher pattern.

    Omitted, empty, or `*` matches all. Exact-character matchers use exact
    string equality (with `|` / `,` alternation). Any other character switches
    to an unanchored regular expression.

    Args:
        value: Raw matcher string from configuration.

    Returns:
        `(matcher, error)` where `matcher` is `None` for match-all, a frozenset
        for exact names, or a compiled regex; `error` is set when compilation
        fails.
    """
    if not value or value == "*":
        return None, None
    if _EXACT_MATCHER.fullmatch(value):
        names = frozenset(
            part.strip() for part in re.split(r"[|,]", value) if part.strip()
        )
        return names, None
    try:
        return re.compile(value), None
    except re.error as exc:
        return None, f"Invalid hook matcher {value!r}: {exc}"


def _handler_matches(
    handler: HookHandler,
    matcher_field: str | None,
    target: str | None,
) -> bool:
    if matcher_field is None or handler.matcher is None:
        return True
    if target is None:
        return False
    matcher = handler.matcher
    if isinstance(matcher, frozenset):
        return target in matcher
    return matcher.search(target) is not None


def _match_target(
    invocation: HookInvocation,
    matcher_field: str | None,
) -> str | None:
    event = invocation.event
    if matcher_field == "tool_name" and isinstance(
        event, PermissionRequestEvent | PreToolUseEvent | PostToolUseEvent
    ):
        return to_wire_tool_name(event.call.name, mcp_server=event.call.mcp_server)
    if matcher_field == "notification_type" and isinstance(event, NotificationEvent):
        return event.notification.type
    if matcher_field == "cause" and isinstance(
        event, SessionStartEvent | SessionEndEvent
    ):
        return event.cause.value
    if matcher_field == "trigger" and isinstance(event, PreCompactEvent):
        return event.trigger.value
    if matcher_field == "agent_name" and isinstance(
        event, SubagentStartEvent | SubagentStopEvent
    ):
        return event.agent.name
    return None
