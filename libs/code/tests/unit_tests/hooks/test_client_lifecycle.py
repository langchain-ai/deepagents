"""Unit tests for client-owned Hooks v2 lifecycle integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import pytest

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.client_lifecycle import (
    ClientHookContext,
    ClientHookService,
    ClientHookStopError,
)
from deepagents_code.hooks.models.domain import (
    DcodeNotificationKind,
    HookDecision,
    HookDiagnostic,
    HookEvent,
    HookInvocation,
    NotificationDecision,
    PermissionEffect,
    PermissionRequestDecision,
    SessionEndCause,
    SessionEndDecision,
    SessionStartCause,
    SessionStartDecision,
)
from deepagents_code.hooks.permissions import permission_hook_outcome
from deepagents_code.hooks.presenter import HookNoticeSeverity, HookPresenter

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class _Runtime:
    cwd: Path
    decisions: deque[HookDecision]
    invocations: list[HookInvocation] = field(default_factory=list)
    presenter: HookPresenter = field(default_factory=HookPresenter)

    def configured_events(self) -> frozenset[HookEvent]:
        return frozenset(decision.event for decision in self.decisions)

    async def invoke(self, invocation: HookInvocation) -> HookDecision:
        self.invocations.append(invocation)
        return self.decisions.popleft()


def _context(*, prompt_id: UUID | str | None = None) -> ClientHookContext:
    return ClientHookContext.create(
        thread_id="thread-1", approval_mode=ApprovalMode.MANUAL, prompt_id=prompt_id
    )


async def test_common_effects_context_and_live_hook_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid4()
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque(
            [
                SessionStartDecision(
                    event=HookEvent.SESSION_START,
                    context=["hook context"],
                    user_notices=["visible notice"],
                    terminal_sequences=["\a"],
                    diagnostics=[
                        HookDiagnostic(
                            code="test_warning",
                            severity="warning",
                            message="diagnostic",
                        )
                    ]
                    * 2,
                )
            ]
        ),
    )
    notices: list[str] = []

    def record(message: str, severity: HookNoticeSeverity) -> None:
        del severity
        notices.append(message)

    runtime.presenter.attach(notice=record)
    service = ClientHookService(runtime)

    decision = await service.session_start(
        _context(prompt_id=prompt_id), SessionStartCause.STARTUP
    )

    invocation = runtime.invocations[0]
    assert decision.context == ["hook context"]
    assert notices == ["Hook warning: diagnostic", "visible notice"]
    assert capsys.readouterr().out == "\a"
    assert "test_warning" in caplog.text
    assert invocation.context.thread_id == "thread-1"
    assert invocation.context.cwd == tmp_path
    assert invocation.context.approval_mode is ApprovalMode.MANUAL
    assert invocation.context.prompt_id == prompt_id
    assert service.take_session_context("thread-1") == ("hook context",)
    assert service.take_session_context("thread-1") == ()


async def test_session_end_clears_context_and_notification_can_stop(
    tmp_path: Path,
) -> None:
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque(
            [
                SessionStartDecision(
                    event=HookEvent.SESSION_START, context=["pending"]
                ),
                SessionEndDecision(event=HookEvent.SESSION_END),
                NotificationDecision(
                    event=HookEvent.NOTIFICATION,
                    continue_processing=False,
                    stop_reason="stop now",
                ),
            ]
        ),
    )
    service = ClientHookService(runtime)
    context = _context()

    await service.session_start(context, SessionStartCause.RESUME)
    await service.session_end(context, SessionEndCause.RESUME)
    assert service.take_session_context("thread-1") == ()

    with pytest.raises(ClientHookStopError, match="stop now"):
        await service.notification(
            context, DcodeNotificationKind.AGENT_COMPLETED, "done"
        )


def _permission(
    behavior: Literal["allow", "deny", "none"],
    *,
    continue_processing: bool = True,
    stop_reason: str | None = None,
    reason: str | None = None,
    interrupt: bool = False,
) -> PermissionRequestDecision:
    return PermissionRequestDecision(
        event=HookEvent.PERMISSION_REQUEST,
        continue_processing=continue_processing,
        stop_reason=stop_reason,
        permission=PermissionEffect(
            behavior=behavior, reason=reason, interrupt=interrupt
        ),
    )


@pytest.mark.parametrize(
    ("decision", "expected", "interrupt"),
    [
        (
            _permission("none", continue_processing=False, stop_reason="stopped"),
            {"type": "reject", "message": "stopped"},
            True,
        ),
        (_permission("allow"), {"type": "approve"}, False),
        (
            _permission("deny", reason="blocked", interrupt=True),
            {"type": "reject", "message": "blocked"},
            True,
        ),
        (_permission("none"), None, False),
    ],
)
def test_permission_hook_outcome(
    decision: PermissionRequestDecision,
    expected: dict[str, str] | None,
    interrupt: bool,
) -> None:
    outcome = permission_hook_outcome(decision)
    assert outcome.decision == expected
    assert outcome.interrupt is interrupt
