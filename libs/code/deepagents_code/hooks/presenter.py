"""User-facing presentation for Hooks v2 execution.

`HookPresenter` is the single place that turns hook results into something a
person sees. It is owned by `HooksManager`, handed to every runtime that
manager loads, and kept alive across reloads so its output sinks can be
rebound once a UI exists without any other object holding its own copy.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable

    from deepagents_code.hooks.models.domain import (
        HookDecision,
        HookDiagnostic,
        HookEvent,
        PermissionEffect,
    )

logger = logging.getLogger(__name__)

HookNoticeSeverity: TypeAlias = Literal["information", "warning", "error"]
DiagnosticKey: TypeAlias = tuple[str, str, str, str | None, str | None]


class HookNoticeCallback(Protocol):
    """Callable that surfaces a user-visible hook notice."""

    def __call__(self, message: str, severity: HookNoticeSeverity) -> None:
        """Present one notice to the user.

        Args:
            message: User-facing notice text.
            severity: Presentation severity for interactive clients.
        """


class HookStatusCallback(Protocol):
    """Callable that updates hook-owned transient status text."""

    def __call__(self, message: str) -> None:
        """Set or clear the hook-owned status message.

        Args:
            message: Status text to display, or empty string to release.
        """


@dataclass(frozen=True, slots=True)
class HookProgress:
    """Lifecycle update for one running hook handler."""

    operation_id: str
    handler_id: str
    event: HookEvent
    active: bool
    message: str = ""
    """Handler-authored status text. Empty when the handler supplied none."""


@dataclass(slots=True)
class HookPresenter:
    """Present hook output consistently across interactive and headless clients."""

    notice: HookNoticeCallback | None = None
    status: HookStatusCallback | None = None
    _active_statuses: dict[str, str] = field(default_factory=dict)

    def attach(
        self,
        *,
        notice: HookNoticeCallback | None,
        status: HookStatusCallback | None = None,
    ) -> None:
        """Rebind the output sinks without replacing the presenter.

        Lets a client that loaded hooks before its UI existed start surfacing
        output, while every runtime and service keeps the same instance.

        Args:
            notice: Sink for user-visible notices.
            status: Sink for transient hook-owned status text.
        """
        self.notice = notice
        self.status = status

    def present_decision(self, decision: HookDecision) -> None:
        """Present common side effects from a reduced hook decision.

        Args:
            decision: Reduced event-specific hook decision.
        """
        self.present_diagnostics(decision.diagnostics)
        for notice in decision.user_notices:
            self._notify(notice, "information")
        for sequence in decision.terminal_sequences:
            sys.stdout.write(sequence)
        if decision.terminal_sequences:
            sys.stdout.flush()

    def present_diagnostics(self, diagnostics: Iterable[HookDiagnostic]) -> None:
        """Log diagnostics and surface each warning or error once per invocation.

        Deduplication is scoped to a single presentation call so a recurring
        diagnostic is still shown on later invocations. A notice is marked
        delivered only after the sink accepts it, so a failed delivery stays
        eligible for retry.

        Args:
            diagnostics: Structured diagnostics to present.
        """
        delivered: set[DiagnosticKey] = set()
        for diagnostic in diagnostics:
            _log_diagnostic(diagnostic)
            if diagnostic.severity == "debug":
                continue
            key = (
                diagnostic.code,
                diagnostic.severity,
                diagnostic.message,
                diagnostic.handler_id,
                diagnostic.field,
            )
            if key in delivered:
                continue
            severity: HookNoticeSeverity = (
                "error" if diagnostic.severity == "error" else "warning"
            )
            if self._notify(f"Hook {severity}: {diagnostic.message}", severity):
                delivered.add(key)

    def update_progress(self, progress: HookProgress) -> None:
        """Update the currently visible hook-owned status.

        Concurrent handlers share one status slot. The most recently activated
        handler wins until it completes; when the last active handler finishes,
        the slot is released with an empty message.

        Args:
            progress: Handler lifecycle update.
        """
        if progress.active:
            self._active_statuses[progress.operation_id] = _status_text(progress)
        else:
            self._active_statuses.pop(progress.operation_id, None)
        message = next(reversed(self._active_statuses.values()), "")
        self._set_status(message)

    def present_permission(
        self,
        tool_name: str,
        permission: PermissionEffect,
    ) -> None:
        """Attribute a hook-owned permission decision to the hook.

        This text is user-facing only. Model-visible HITL rejection payloads must
        carry the raw hook reason without this attribution prefix.

        Args:
            tool_name: Display name of the affected tool.
            permission: Normalized permission effect.
        """
        target = tool_name or "tool request"
        if permission.behavior == "allow":
            self._notify(
                f"PermissionRequest hook allowed {target}.",
                "information",
            )
        elif permission.behavior == "deny":
            suffix = f": {permission.reason}" if permission.reason else "."
            self._notify(
                f"PermissionRequest hook denied {target}{suffix}",
                "warning",
            )

    def _notify(self, message: str, severity: HookNoticeSeverity) -> bool:
        if self.notice is None:
            logger.warning("Hook notice (no sink attached): %s", message)
            return True
        try:
            self.notice(message, severity)
        except Exception:
            logger.warning("Failed to surface hook notice", exc_info=True)
            return False
        return True

    def _set_status(self, message: str) -> None:
        if self.status is None:
            return
        try:
            self.status(message)
        except Exception:
            logger.warning("Failed to update hook status", exc_info=True)


def _status_text(progress: HookProgress) -> str:
    if progress.message:
        return progress.message
    from deepagents_code.config import get_glyphs

    return f"Running {progress.event.value} hook{get_glyphs().ellipsis}"


def _log_diagnostic(diagnostic: HookDiagnostic) -> None:
    message = "Hook diagnostic %s: %s"
    if diagnostic.severity == "error":
        logger.error(message, diagnostic.code, diagnostic.message)
    elif diagnostic.severity == "warning":
        logger.warning(message, diagnostic.code, diagnostic.message)
    else:
        logger.debug(message, diagnostic.code, diagnostic.message)
