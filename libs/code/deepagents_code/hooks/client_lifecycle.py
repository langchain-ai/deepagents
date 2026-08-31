"""Client-owned Hooks v2 lifecycle facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.domain import (
    CompactTrigger,
    DcodeNotification,
    DcodeNotificationKind,
    HookContext,
    HookDecision,
    HookDomainEvent,
    HookEvent,
    HookInvocation,
    NotificationDecision,
    NotificationEvent,
    PermissionEffect,
    PermissionRequestDecision,
    PermissionRequestEvent,
    PreCompactDecision,
    PreCompactEvent,
    SessionEndCause,
    SessionEndDecision,
    SessionEndEvent,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    ToolCallData,
    UserPromptSubmitDecision,
    UserPromptSubmitEvent,
)
from deepagents_code.hooks.permissions import (
    PermissionHookOutcome,
    permission_hook_outcome,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Protocol

    from deepagents_code.hooks.presenter import HookPresenter

    class _ClientHooksRuntime(Protocol):
        @property
        def cwd(self) -> Path: ...

        @property
        def presenter(self) -> HookPresenter: ...

        def configured_events(self) -> frozenset[HookEvent]: ...

        async def invoke(self, invocation: HookInvocation) -> HookDecision: ...


class ClientHookStopError(RuntimeError):
    """Raised when a client-owned hook stops lifecycle processing."""


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
    """Execute client-owned events and apply their common side effects.

    User-facing output goes through the runtime's presenter, which the owning
    `HooksManager` also holds. The service never wraps or replaces it, so there
    is exactly one presenter per session.
    """

    runtime: _ClientHooksRuntime
    # SessionStart context accumulated per thread, consumed by
    # `take_session_context` for injection into the next model turn.
    _pending_context: dict[str, list[str]] = field(default_factory=dict)

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
            self._pending_context.setdefault(context.thread_id, []).extend(
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
            self._pending_context.pop(context.thread_id, None)
            return SessionEndDecision(event=HookEvent.SESSION_END)
        decision = await self._invoke(
            context,
            SessionEndEvent(event=HookEvent.SESSION_END, cause=cause),
        )
        if not isinstance(decision, SessionEndDecision):
            msg = f"Expected SessionEndDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        self._pending_context.pop(context.thread_id, None)
        return decision

    async def user_prompt_submit(
        self,
        context: ClientHookContext,
        prompt: str,
    ) -> UserPromptSubmitDecision:
        """Invoke `UserPromptSubmit` before a user turn.

        Args:
            context: Current client turn context.
            prompt: Original user prompt.

        Returns:
            Aggregated prompt decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.USER_PROMPT_SUBMIT):
            return UserPromptSubmitDecision(event=HookEvent.USER_PROMPT_SUBMIT)
        decision = await self._invoke(
            context,
            UserPromptSubmitEvent(
                event=HookEvent.USER_PROMPT_SUBMIT,
                prompt=prompt,
            ),
        )
        if not isinstance(decision, UserPromptSubmitDecision):
            msg = f"Expected UserPromptSubmitDecision, got {type(decision).__name__}"
            raise TypeError(msg)
        return decision

    async def pre_compact(
        self,
        context: ClientHookContext,
        trigger: CompactTrigger,
        *,
        custom_instructions: str = "",
    ) -> PreCompactDecision:
        """Invoke `PreCompact` through the session hook runtime.

        Args:
            context: Current client turn context.
            trigger: Manual or automatic compaction source.
            custom_instructions: Optional compaction instructions.

        Returns:
            Aggregated pre-compaction decision.

        Raises:
            TypeError: If the runtime returns a mismatched decision type.
        """
        if not self.has_handlers(HookEvent.PRE_COMPACT):
            return PreCompactDecision(event=HookEvent.PRE_COMPACT)
        decision = await self._invoke(
            context,
            PreCompactEvent(
                event=HookEvent.PRE_COMPACT,
                trigger=trigger,
                custom_instructions=custom_instructions,
            ),
        )
        if not isinstance(decision, PreCompactDecision):
            msg = f"Expected PreCompactDecision, got {type(decision).__name__}"
            raise TypeError(msg)
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

    async def resolve_permission(
        self,
        context: ClientHookContext,
        call: ToolCallData,
    ) -> PermissionHookOutcome:
        """Resolve a permission hook and present user-facing attribution once.

        The returned HITL decision carries the raw hook reason (or stop reason)
        for model-visible resume payloads. Attribution text is emitted only
        through the shared presenter.

        Args:
            context: Current client session context.
            call: Tool action awaiting approval.

        Returns:
            Shared approval, rejection, or unresolved result.
        """
        decision = await self.permission_request(context, call)
        outcome = permission_hook_outcome(decision)
        if outcome.decision is None:
            return outcome
        permission = (
            decision.permission
            if decision.continue_processing
            else PermissionEffect(
                behavior="deny",
                reason=decision.stop_reason or "Permission stopped by hook",
                interrupt=True,
            )
        )
        self.present_permission(call.name, permission)
        return outcome

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
            ClientHookStopError: If a handler stops lifecycle processing.
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
        if not decision.continue_processing:
            reason = decision.stop_reason or "Notification stopped by hook"
            raise ClientHookStopError(reason)
        return decision

    def take_session_context(self, thread_id: str) -> tuple[str, ...]:
        """Consume context accumulated for the thread's next model turn.

        Args:
            thread_id: Thread whose pending context should be consumed.

        Returns:
            Ordered context strings, removed from the service.
        """
        return tuple(self._pending_context.pop(thread_id, ()))

    def has_handlers(self, event: HookEvent) -> bool:
        """Return whether the runtime has handlers for an event.

        Args:
            event: Lifecycle event to inspect.

        Returns:
            Whether at least one handler was configured.
        """
        return event in self.runtime.configured_events()

    def present_permission(
        self,
        tool_name: str,
        permission: PermissionEffect,
    ) -> None:
        """Surface attribution for a hook-owned permission decision.

        Args:
            tool_name: Display name of the affected tool.
            permission: Normalized permission effect.
        """
        self.runtime.presenter.present_permission(tool_name, permission)

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
        self.runtime.presenter.present_decision(decision)
        return decision
