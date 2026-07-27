"""Tests for shared Hooks v2 user feedback."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from deepagents_code.hooks.feedback import HookFeedback, HookProgress
from deepagents_code.hooks.models.domain import (
    HookDiagnostic,
    HookEvent,
    PermissionEffect,
    SessionEndDecision,
)

if TYPE_CHECKING:
    import pytest


def test_decision_feedback_scopes_diagnostics_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[tuple[str, str]] = []
    output = StringIO()
    monkeypatch.setattr("deepagents_code.hooks.feedback.sys.stdout", output)
    feedback = HookFeedback(
        notice=lambda message, severity: notices.append((message, severity))
    )
    diagnostic = HookDiagnostic(
        code="invalid_output",
        severity="warning",
        message="Hook output failed validation",
        handler_id="SessionEnd:0:0",
    )
    decision = SessionEndDecision(
        event=HookEvent.SESSION_END,
        user_notices=["visible notice"],
        terminal_sequences=["\x1b]9;done\x07"],
        diagnostics=[diagnostic, diagnostic],
    )

    feedback.present_decision(decision)
    feedback.present_decision(
        SessionEndDecision(
            event=HookEvent.SESSION_END,
            diagnostics=[diagnostic],
        )
    )

    assert notices == [
        ("Hook warning: Hook output failed validation", "warning"),
        ("visible notice", "information"),
        ("Hook warning: Hook output failed validation", "warning"),
    ]
    assert output.getvalue() == "\x1b]9;done\x07"


def test_failed_diagnostic_notice_remains_eligible_for_retry() -> None:
    attempts = {"count": 0}
    notices: list[str] = []

    def flaky_notice(message: str, severity: str) -> None:
        _ = severity
        attempts["count"] += 1
        if attempts["count"] == 1:
            msg = "sink unavailable"
            raise RuntimeError(msg)
        notices.append(message)

    feedback = HookFeedback(notice=flaky_notice)
    diagnostic = HookDiagnostic(
        code="invalid_output",
        severity="warning",
        message="Hook output failed validation",
    )

    feedback.present_diagnostics([diagnostic])
    feedback.present_diagnostics([diagnostic])

    assert notices == ["Hook warning: Hook output failed validation"]


def test_progress_keeps_latest_concurrent_status_visible() -> None:
    statuses: list[str] = []

    def capture_status(message: str) -> None:
        statuses.append(message)

    feedback = HookFeedback(status=capture_status)
    first = HookProgress(
        operation_id="first",
        handler_id="Stop:0:0",
        event=HookEvent.STOP,
        message="Checking output",
        active=True,
    )
    second = HookProgress(
        operation_id="second",
        handler_id="Stop:0:1",
        event=HookEvent.STOP,
        message="Running policy",
        active=True,
    )

    feedback.update_progress(first)
    feedback.update_progress(second)
    feedback.update_progress(
        HookProgress(
            operation_id=first.operation_id,
            handler_id=first.handler_id,
            event=first.event,
            message=first.message,
            active=False,
        )
    )
    feedback.update_progress(
        HookProgress(
            operation_id=second.operation_id,
            handler_id=second.handler_id,
            event=second.event,
            message=second.message,
            active=False,
        )
    )

    assert statuses == [
        "Checking output",
        "Running policy",
        "Running policy",
        "",
    ]


def test_progress_callback_raise_does_not_break_updates() -> None:
    statuses: list[str] = []

    def flaky_status(message: str) -> None:
        if message == "boom":
            msg = "status sink failed"
            raise RuntimeError(msg)
        statuses.append(message)

    feedback = HookFeedback(status=flaky_status)
    feedback.update_progress(
        HookProgress(
            operation_id="one",
            handler_id="Stop:0:0",
            event=HookEvent.STOP,
            message="boom",
            active=True,
        )
    )
    feedback.update_progress(
        HookProgress(
            operation_id="one",
            handler_id="Stop:0:0",
            event=HookEvent.STOP,
            message="recovered",
            active=True,
        )
    )

    assert statuses == ["recovered"]


def test_permission_feedback_attributes_hook_decisions() -> None:
    notices: list[tuple[str, str]] = []
    feedback = HookFeedback(
        notice=lambda message, severity: notices.append((message, severity))
    )

    feedback.present_permission("read_file", PermissionEffect(behavior="allow"))
    feedback.present_permission(
        "execute",
        PermissionEffect(behavior="deny", reason="command blocked"),
    )

    assert notices == [
        ("PermissionRequest hook allowed read_file.", "information"),
        (
            "PermissionRequest hook denied execute: command blocked",
            "warning",
        ),
    ]
