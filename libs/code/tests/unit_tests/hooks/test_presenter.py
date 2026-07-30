"""Tests for the shared Hooks v2 presenter."""

from __future__ import annotations

from deepagents_code.hooks.models.domain import HookEvent, PermissionEffect
from deepagents_code.hooks.presenter import (
    HookNoticeSeverity,
    HookPresenter,
    HookProgress,
)


def _progress(
    operation_id: str,
    message: str = "",
    *,
    active: bool = True,
) -> HookProgress:
    return HookProgress(
        operation_id=operation_id,
        handler_id=f"Stop:{operation_id}",
        event=HookEvent.STOP,
        message=message,
        active=active,
    )


def test_progress_keeps_latest_concurrent_status_visible() -> None:
    statuses: list[str] = []

    def record(message: str) -> None:
        statuses.append(message)

    presenter = HookPresenter(status=record)

    for update in (
        _progress("first", "Checking output"),
        _progress("second", "Running policy"),
        _progress("first", "Checking output", active=False),
        _progress("second", "Running policy", active=False),
    ):
        presenter.update_progress(update)

    assert statuses == ["Checking output", "Running policy", "Running policy", ""]


def test_progress_without_handler_message_falls_back_to_event_text() -> None:
    statuses: list[str] = []

    def record(message: str) -> None:
        statuses.append(message)

    presenter = HookPresenter(status=record)

    presenter.update_progress(_progress("only"))

    assert statuses[0].startswith("Running Stop hook")


def test_attach_rebinds_sinks_on_the_same_presenter() -> None:
    first: list[str] = []
    second: list[str] = []

    def to_first(message: str, severity: HookNoticeSeverity) -> None:
        del severity
        first.append(message)

    def to_second(message: str, severity: HookNoticeSeverity) -> None:
        del severity
        second.append(message)

    presenter = HookPresenter(notice=to_first)
    presenter.attach(notice=to_second)
    presenter.present_permission(
        "shell",
        PermissionEffect(behavior="deny", reason="nope"),
    )

    assert first == []
    assert second == ["PermissionRequest hook denied shell: nope"]
