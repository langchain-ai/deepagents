"""`DeepAgentsApp` reports its state to the terminal with `OSC 7501`."""

from __future__ import annotations

import base64
import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from deepagents_code import program_status
from deepagents_code._env_vars import TERMINAL_STATUS
from deepagents_code.app import DeepAgentsApp
from deepagents_code.program_status import TurnOutcome

if TYPE_CHECKING:
    from pathlib import Path

_REPORT_RE = re.compile(r"\x1b\]7501;(?P<body>[^\x1b]*)\x1b\\")


class _Sink:
    def __init__(self) -> None:
        self.sequences: list[str] = []

    def __call__(self, sequence: str) -> bool:
        self.sequences.append(sequence)
        return True

    def reports(self) -> list[dict[str, str]]:
        decoded = []
        for sequence in self.sequences:
            match = _REPORT_RE.fullmatch(sequence)
            assert match is not None
            pairs = dict(p.split("=", 1) for p in match["body"].split(":"))
            for key in ("title", "msg"):
                if key in pairs:
                    pairs[key] = base64.b64decode(pairs[key]).decode()
            decoded.append(pairs)
        return decoded

    def states(self) -> list[str]:
        return [
            f"{r['state']}:{r['kind']}" if "kind" in r else r["state"]
            for r in self.reports()
        ]


@pytest.fixture
def sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Sink:
    """Capture reports and keep the developer's config and env out of the test."""
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_PATH", tmp_path / "config.toml"
    )
    monkeypatch.delenv(TERMINAL_STATUS, raising=False)
    monkeypatch.setattr(program_status.atexit, "register", lambda _fn: None)
    sink = _Sink()
    monkeypatch.setattr(program_status, "_default_write", lambda: sink)
    return sink


def _approval_request() -> list[dict[str, object]]:
    return [{"name": "write_file", "args": {"file_path": "/tmp/x"}}]


async def test_reports_idle_on_mount_and_clears_on_exit(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert sink.reports()[0] == {"state": "idle", "app": "deepagents-code"}
    assert sink.states()[-1] == "clear"


async def test_turn_reports_working_then_done(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
        app._mark_thread_turn_completed()
        app._set_agent_running(False)
        assert sink.states() == ["idle", "working", "done"]


async def test_background_compaction_keeps_unseen_done(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._agent_status.set_outcome(TurnOutcome.DONE)
        app._set_agent_running(True, report_status=False)
        app._set_agent_running(False)
        assert sink.states() == ["idle", "done"]


async def test_approval_reports_blocked_permission(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
        future = await app._request_approval(_approval_request(), None)
        await pilot.pause()
        assert sink.reports()[-1] == {
            "state": "blocked",
            "kind": "permission",
            "app": "deepagents-code",
            "msg": "Approval needed: write_file",
        }
        future.set_result({"type": "approve"})
        await pilot.pause()
        assert sink.states()[-1] == "working"


async def test_ask_user_reports_blocked_question(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
        future = await app._request_ask_user(
            [{"question": "Which branch?", "type": "text"}]
        )
        await pilot.pause()
        assert sink.reports()[-1]["kind"] == "question"
        assert sink.reports()[-1]["msg"] == "Which branch?"
        future.cancel()
        await pilot.pause()
        assert sink.states()[-1] == "working"


async def test_thread_name_is_the_title(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._thread_name = "Fix login"
        app._update_terminal_title()
        assert sink.reports()[-1]["title"] == "Fix login"


async def test_preference_off_reports_nothing(
    sink: _Sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TERMINAL_STATUS, "0")
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
    assert sink.sequences == []


async def test_clear_drops_unanswered_goal_review(sink: _Sink) -> None:
    # `/clear` removes the review widget without settling its future.
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        await app._request_goal_review("Ship it", "- tests pass")
        await pilot.pause()
        await app._handle_command("/clear")
        await pilot.pause()
        assert sink.states() == ["idle", "blocked:permission", "idle"]


async def test_clear_forgets_previous_thread_outcome(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
        app._agent_status.set_outcome(TurnOutcome.ERROR, "Agent error: boom")
        app._set_agent_running(False)
        await app._handle_command("/clear")
        await pilot.pause()
        assert sink.states() == ["idle", "working", "error", "idle"]


async def test_cancelling_goal_review_after_done_reports_idle(sink: _Sink) -> None:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-7501")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._set_agent_running(True)
        app._mark_thread_turn_completed()
        app._set_agent_running(False)
        await app._request_goal_review("Ship it", "- tests pass")
        await pilot.pause()
        menu = app._pending_goal_review_widget
        assert menu is not None
        menu.action_cancel()
        await pilot.pause()
        assert sink.states() == [
            "idle",
            "working",
            "done",
            "blocked:permission",
            "idle",
        ]
