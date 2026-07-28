"""Tests for shared Hooks v2 user feedback."""

from __future__ import annotations

from deepagents_code.hooks.feedback import HookFeedback, HookProgress
from deepagents_code.hooks.models.domain import HookEvent


def _progress(operation_id: str, message: str, *, active: bool = True) -> HookProgress:
    return HookProgress(
        operation_id=operation_id,
        handler_id=f"Stop:{operation_id}",
        event=HookEvent.STOP,
        message=message,
        active=active,
    )


def test_progress_keeps_latest_concurrent_status_visible() -> None:
    statuses: list[str] = []
    feedback = HookFeedback(status=lambda message: statuses.append(message))

    for update in (
        _progress("first", "Checking output"),
        _progress("second", "Running policy"),
        _progress("first", "Checking output", active=False),
        _progress("second", "Running policy", active=False),
    ):
        feedback.update_progress(update)

    assert statuses == ["Checking output", "Running policy", "Running policy", ""]
