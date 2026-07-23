"""Client-owned Hooks v2 lifecycle facade."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.domain import (
    DcodeNotification,
    DcodeNotificationKind,
    HookContext,
    HookDecision,
    HookDiagnostic,
    HookDomainEvent,
    HookEvent,
    HookInvocation,
    NotificationDecision,
    NotificationEvent,
    PermissionEffect,
    PermissionRequestDecision,
    PermissionRequestEvent,
    SessionEndCause,
    SessionEndDecision,
    SessionEndEvent,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    ToolCallData,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Protocol

    class _ClientHooksRuntime(Protocol):
        @property
        def cwd(self) -> Path: ...

        def configured_events(self) -> frozenset[HookEvent]: ...

        async def invoke(self, invocation: HookInvocation) -> HookDecision: ...


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClientHookContext:
    """Client state required to create a domain hook invocation."""

    thread_id: str
    approval_mode: ApprovalMode
    prompt_id: UUID | None = None

    @classmethod
    def create(
        cls,
        *,
        thread_id: str,
        approval_mode: ApprovalMode | str,
        prompt_id: str | UUID | None = None,
    ) -> ClientHookContext:
        """Build validated client hook context.

        Args:
            thread_id: Active conversation thread.
            approval_mode: Current client approval policy.
            prompt_id: Optional current prompt identifier.

        Returns:
            Validated context for client-owned hook events.
        """
        approval = (
            approval_mode
            if isinstance(approval_mode, ApprovalMode)
            else ApprovalMode(approval_mode)
        )
        parsed_prompt = (
            prompt_id
            if isinstance(prompt_id, UUID)
            else UUID(prompt_id)
            if prompt_id
            else None
        )
        return cls(
            thread_id=thread_id,
            approval_mode=approval,
            prompt_id=parsed_prompt,
        )


@dataclass(slots=True)
class ClientHookService:
    """Execute client-owned events and apply their common side effects."""

    runtime: _ClientHooksRuntime
    notice: Callable[[str], None] | None = None
    _session_context: dict[str, list[str]] = field(default_factory=dict)

    async def session_start(
        self,
        context: ClientHookContext,
        cause: SessionStartCause,
        *,
        model: str | None = None,
    ) -> SessionStartDecision:
        """Invoke `SessionStart` and retain context for the next model turn.

        Args:
            context: Current client session context.
            cause: Lifecycle boundary that started the session.
            model: Active model identifier when available.

        Returns:
            Aggregated session-start decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.SESSION_START):
            return SessionStartDecision(event=HookEvent.SESSION_START)
        decision = await self._invoke(
            context,
            SessionStartEvent(
                event=HookEvent.SESSION_START,
                cause=cause,
                model=model,
            ),
        )
        if not isinstance(decision, SessionStartDecision):
            msg = f"Expected SessionStartDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        if decision.context:
            self._session_context.setdefault(context.thread_id, []).extend(
                decision.context
            )
        return decision

    async def session_end(
        self,
        context: ClientHookContext,
        cause: SessionEndCause,
    ) -> SessionEndDecision:
        """Invoke `SessionEnd` for the outgoing thread.

        Args:
            context: Outgoing client session context.
            cause: Reason the session ended.

        Returns:
            Aggregated session-end decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.SESSION_END):
            self._session_context.pop(context.thread_id, None)
            return SessionEndDecision(event=HookEvent.SESSION_END)
        decision = await self._invoke(
            context,
            SessionEndEvent(event=HookEvent.SESSION_END, cause=cause),
        )
        if not isinstance(decision, SessionEndDecision):
            msg = f"Expected SessionEndDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        self._session_context.pop(context.thread_id, None)
        return decision

    async def permission_request(
        self,
        context: ClientHookContext,
        call: ToolCallData,
    ) -> PermissionRequestDecision:
        """Invoke `PermissionRequest` before client approval resolution.

        Args:
            context: Current client session context.
            call: Tool action awaiting approval.

        Returns:
            Aggregated permission decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.PERMISSION_REQUEST):
            return PermissionRequestDecision(
                event=HookEvent.PERMISSION_REQUEST,
                permission=PermissionEffect(behavior="none"),
            )
        decision = await self._invoke(
            context,
            PermissionRequestEvent(event=HookEvent.PERMISSION_REQUEST, call=call),
        )
        if not isinstance(decision, PermissionRequestDecision):
            msg = f"Expected PermissionRequestDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return decision

    async def notification(
        self,
        context: ClientHookContext,
        kind: DcodeNotificationKind,
        message: str,
        *,
        title: str | None = None,
    ) -> NotificationDecision:
        """Invoke one explicitly supported dcode notification event.

        Args:
            context: Current client session context.
            kind: Supported dcode notification kind.
            message: User-facing notification text.
            title: Optional notification title.

        Returns:
            Aggregated notification decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.NOTIFICATION):
            return NotificationDecision(event=HookEvent.NOTIFICATION)
        decision = await self._invoke(
            context,
            NotificationEvent(
                event=HookEvent.NOTIFICATION,
                notification=DcodeNotification(
                    type=kind,
                    message=message,
                    title=title,
                ),
            ),
        )
        if not isinstance(decision, NotificationDecision):
            msg = f"Expected NotificationDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return decision

    def take_session_context(self, thread_id: str) -> tuple[str, ...]:
        """Consume context accumulated for the thread's next model turn.

        Args:
            thread_id: Thread whose pending context should be consumed.

        Returns:
            Ordered context strings, removed from the service.
        """
        return tuple(self._session_context.pop(thread_id, ()))

    def has_handlers(self, event: HookEvent) -> bool:
        """Return whether the runtime has handlers for an event.

        Args:
            event: Lifecycle event to inspect.

        Returns:
            Whether at least one handler was configured.
        """
        return event in self.runtime.configured_events()

    async def _invoke(
        self,
        context: ClientHookContext,
        event: HookDomainEvent,
    ) -> HookDecision:
        invocation = HookInvocation(
            context=HookContext(
                thread_id=context.thread_id,
                cwd=self.runtime.cwd,
                prompt_id=context.prompt_id,
                approval_mode=context.approval_mode,
            ),
            event=event,
        )
        decision = await self.runtime.invoke(invocation)
        self._apply_common_effects(decision)
        return decision

    def _apply_common_effects(self, decision: HookDecision) -> None:
        for diagnostic in decision.diagnostics:
            _log_diagnostic(diagnostic)
        for notice in decision.user_notices:
            if self.notice is None:
                logger.warning("Hook user notice: %s", notice)
                continue
            try:
                self.notice(notice)
            except Exception:
                logger.warning("Failed to surface hook user notice", exc_info=True)
        for sequence in decision.terminal_sequences:
            sys.stdout.write(sequence)
        if decision.terminal_sequences:
            sys.stdout.flush()


def _log_diagnostic(diagnostic: HookDiagnostic) -> None:
    message = "Hook diagnostic %s: %s"
    if diagnostic.severity == "error":
        logger.error(message, diagnostic.code, diagnostic.message)
    elif diagnostic.severity == "warning":
        logger.warning(message, diagnostic.code, diagnostic.message)
    else:
        logger.debug(message, diagnostic.code, diagnostic.message)
