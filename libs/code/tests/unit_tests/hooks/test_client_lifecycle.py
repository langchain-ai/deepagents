"""Unit tests for client-owned Hooks v2 lifecycle integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pytest
from rich.console import Console

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.client.non_interactive import (
    StreamState,
    _process_hitl_interrupts,
)
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
from deepagents_code.hooks.models.wire import NotificationWireInput
from deepagents_code.hooks.projection import project_hook_input
from deepagents_code.tui.textual_adapter import (
    _merge_permission_outcomes,
    _permission_hook_outcomes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain.agents.middleware.human_in_the_loop import (
        ApproveDecision,
        EditDecision,
        RejectDecision,
    )

    HITLDecision = ApproveDecision | EditDecision | RejectDecision


@dataclass(slots=True)
class _Runtime:
    cwd: Path
    decisions: deque[HookDecision]
    invocations: list[HookInvocation] = field(default_factory=list)

    def configured_events(self) -> frozenset[HookEvent]:
        return frozenset(decision.event for decision in self.decisions)

    async def invoke(self, invocation: HookInvocation) -> HookDecision:
        self.invocations.append(invocation)
        return self.decisions.popleft()


@dataclass(slots=True)
class _SessionState:
    thread_id: str
    approval_mode: ApprovalMode
    turn_id: str | None
    client_hooks: ClientHookService | None


def _context() -> ClientHookContext:
    return ClientHookContext.create(
        thread_id="thread-1",
        approval_mode=ApprovalMode.MANUAL,
    )


def _permission(
    behavior: Literal["allow", "deny", "ask", "none"],
    *,
    reason: str | None = None,
    interrupt: bool = False,
) -> PermissionRequestDecision:
    return PermissionRequestDecision(
        event=HookEvent.PERMISSION_REQUEST,
        permission=PermissionEffect(
            behavior=behavior,
            reason=reason,
            interrupt=interrupt,
        ),
    )


async def test_service_applies_common_effects_and_session_context(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    notices: list[str] = []
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
                    ],
                )
            ]
        ),
    )
    service = ClientHookService(runtime, notice=notices.append)

    decision = await service.session_start(_context(), SessionStartCause.STARTUP)

    assert decision.context == ["hook context"]
    assert notices == ["visible notice"]
    assert capsys.readouterr().out == "\a"
    assert "test_warning" in caplog.text
    assert service.take_session_context("thread-1") == ("hook context",)
    assert service.take_session_context("thread-1") == ()


@pytest.mark.parametrize(
    ("kind", "wire_type"),
    [
        (DcodeNotificationKind.PERMISSION_REQUIRED, "permission_prompt"),
        (DcodeNotificationKind.AGENT_NEEDS_INPUT, "agent_needs_input"),
        (DcodeNotificationKind.AGENT_COMPLETED, "agent_completed"),
    ],
)
async def test_notification_service_maps_supported_kinds(
    tmp_path: Path,
    kind: DcodeNotificationKind,
    wire_type: str,
) -> None:
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque([NotificationDecision(event=HookEvent.NOTIFICATION)]),
    )
    service = ClientHookService(runtime)

    await service.notification(_context(), kind, "message")

    wire = project_hook_input(
        runtime.invocations[0],
        transcript_path=tmp_path / "transcript.jsonl",
    )
    assert isinstance(wire, NotificationWireInput)
    assert wire.notification_type == wire_type


async def test_session_end_discards_pending_context(tmp_path: Path) -> None:
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque(
            [
                SessionStartDecision(
                    event=HookEvent.SESSION_START,
                    context=["pending"],
                ),
                SessionEndDecision(event=HookEvent.SESSION_END),
            ]
        ),
    )
    service = ClientHookService(runtime)
    context = _context()

    await service.session_start(context, SessionStartCause.RESUME)
    await service.session_end(context, SessionEndCause.RESUME)

    assert service.take_session_context("thread-1") == ()
    assert [invocation.event.event for invocation in runtime.invocations] == [
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
    ]


async def test_notification_stop_interrupts_client_processing(tmp_path: Path) -> None:
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque(
            [
                NotificationDecision(
                    event=HookEvent.NOTIFICATION,
                    continue_processing=False,
                    stop_reason="stop now",
                )
            ]
        ),
    )
    service = ClientHookService(runtime)

    with pytest.raises(ClientHookStopError, match="stop now"):
        await service.notification(
            _context(),
            DcodeNotificationKind.AGENT_COMPLETED,
            "done",
        )


@pytest.mark.parametrize(
    ("behavior", "hook_decision", "reviewed", "expected"),
    [
        ("allow", {"type": "approve"}, [], {"type": "approve"}),
        (
            "deny",
            {"type": "reject", "message": "blocked"},
            [],
            {"type": "reject", "message": "blocked"},
        ),
        ("none", None, [{"type": "approve"}], {"type": "approve"}),
    ],
)
async def test_tui_permission_decisions_precede_review(
    tmp_path: Path,
    behavior: Literal["allow", "deny", "none"],
    hook_decision: dict[str, str] | None,
    reviewed: list[HITLDecision],
    expected: dict[str, str],
) -> None:
    runtime = _Runtime(
        cwd=tmp_path,
        decisions=deque(
            [
                _permission(
                    behavior,
                    reason="blocked" if behavior == "deny" else None,
                )
            ]
        ),
    )
    state = _SessionState(
        thread_id="thread-1",
        approval_mode=ApprovalMode.MANUAL,
        turn_id=None,
        client_hooks=ClientHookService(runtime),
    )

    outcomes = await _permission_hook_outcomes(
        state,
        "interrupt-1",
        [{"name": "read_file", "args": {"path": "README.md"}}],
        {},
    )

    assert outcomes[0].decision == hook_decision
    assert _merge_permission_outcomes(outcomes, reviewed) == [expected]
    assert runtime.invocations[0].event.event is HookEvent.PERMISSION_REQUEST


@pytest.mark.parametrize(
    ("decisions", "expected"),
    [
        (
            deque([_permission("allow")]),
            {"type": "approve"},
        ),
        (
            deque([_permission("deny", reason="blocked")]),
            {"type": "reject", "message": "blocked"},
        ),
        (
            deque(
                [
                    _permission("none"),
                    NotificationDecision(event=HookEvent.NOTIFICATION),
                ]
            ),
            {"type": "approve"},
        ),
    ],
)
async def test_headless_permission_decisions_precede_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: deque[HookDecision],
    expected: dict[str, str],
) -> None:
    runtime = _Runtime(cwd=tmp_path, decisions=decisions)
    state = StreamState(client_hooks=ClientHookService(runtime))
    state.pending_interrupts["interrupt-1"] = {
        "action_requests": [
            {"name": "read_file", "args": {"path": "README.md"}},
        ],
        "review_configs": [],
    }
    resolution_calls = 0

    def _resolve(*_args: object) -> dict[str, str]:
        nonlocal resolution_calls
        resolution_calls += 1
        assert runtime.invocations[-1].event.event is HookEvent.NOTIFICATION
        return {"type": "approve"}

    monkeypatch.setattr(
        "deepagents_code.client.non_interactive._make_hitl_decision",
        _resolve,
    )

    await _process_hitl_interrupts(state, Console(quiet=True), "thread-1")

    assert state.hitl_response["interrupt-1"]["decisions"] == [expected]
    should_resolve = expected == {"type": "approve"} and len(runtime.invocations) == 2
    assert resolution_calls == int(should_resolve)
    assert runtime.invocations[0].event.event is HookEvent.PERMISSION_REQUEST
