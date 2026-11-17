"""Tests for `deepagents_code.program_status` (OSC 7501)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from deepagents_code import program_status, terminal_escape
from deepagents_code.program_status import (
    MAX_MSG_BYTES,
    MAX_SEQUENCE_BYTES,
    MAX_TITLE_BYTES,
    AgentActivity,
    AgentStatusTracker,
    BlockedKind,
    Blocker,
    ProgramState,
    ProgramStatus,
    ProgramStatusReporter,
    TurnOutcome,
    activity_status,
    encode_program_status,
    exit_status,
    sanitize_text,
)

_REPORT_RE = re.compile(r"\x1b\]7501;(?P<body>[^\x1b\x07]*)\x1b\\")


def _decode(sequence: str) -> dict[str, str]:
    """Parse a report the way a terminal does, decoding base64 fields."""
    match = _REPORT_RE.fullmatch(sequence)
    assert match is not None, f"not an OSC 7501 report: {sequence!r}"
    pairs = dict(pair.split("=", 1) for pair in match["body"].split(":"))
    for key in ("title", "msg"):
        if key in pairs:
            pairs[key] = base64.b64decode(pairs[key], validate=True).decode()
    return pairs


class _Sink:
    """Records written sequences; can simulate a terminal that is unreachable."""

    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.sequences: list[str] = []

    def __call__(self, sequence: str) -> bool:
        if self.ok:
            self.sequences.append(sequence)
        return self.ok

    @property
    def decoded(self) -> list[dict[str, str]]:
        return [_decode(s) for s in self.sequences]


@pytest.fixture
def sink(monkeypatch: pytest.MonkeyPatch) -> _Sink:
    monkeypatch.setattr(program_status.atexit, "register", lambda _fn: None)
    return _Sink()


class TestEncode:
    def test_root_report_has_state_and_app(self) -> None:
        sequence = encode_program_status(ProgramStatus(ProgramState.IDLE))
        assert sequence == "\x1b]7501;state=idle:app=deepagents-code\x1b\\"

    def test_blocked_with_kind_progress_title_and_msg(self) -> None:
        status = ProgramStatus(
            ProgramState.BLOCKED,
            kind=BlockedKind.PERMISSION,
            progress=40,
            title="Fix bug",
            msg="Approve `rm -rf build`?",
        )
        assert _decode(encode_program_status(status)) == {
            "state": "blocked",
            "kind": "permission",
            "progress": "40",
            "app": "deepagents-code",
            "title": "Fix bug",
            "msg": "Approve `rm -rf build`?",
        }

    def test_values_stay_inside_the_spec_value_charset(self) -> None:
        status = ProgramStatus(
            ProgramState.ERROR, record_id="task/a.b", title="ü" * 10, msg="x?>~"
        )
        body = _REPORT_RE.fullmatch(encode_program_status(status))
        assert body is not None
        for pair in body["body"].split(":"):
            key, value = pair.split("=", 1)
            assert re.fullmatch(r"[a-z]+", key)
            assert re.fullmatch(r"[A-Za-z0-9_.,+/=-]*", value)

    def test_child_record_carries_id(self) -> None:
        status = ProgramStatus(ProgramState.WORKING, record_id="task/1")
        assert _decode(encode_program_status(status))["id"] == "task/1"

    def test_clear_omits_title_and_msg(self) -> None:
        status = ProgramStatus(ProgramState.CLEAR, title="t", msg="m")
        assert _decode(encode_program_status(status)) == {
            "state": "clear",
            "app": "deepagents-code",
        }

    def test_empty_text_fields_are_omitted(self) -> None:
        status = ProgramStatus(ProgramState.DONE, title=" \n ", msg="")
        assert set(_decode(encode_program_status(status))) == {"state", "app"}

    def test_largest_report_fits_sequence_limit(self) -> None:
        status = ProgramStatus(
            ProgramState.BLOCKED,
            kind=BlockedKind.QUESTION,
            progress=100,
            record_id="/".join(["a" * 15] * 8),
            title="界" * 500,
            msg="界" * 5000,
        )
        sequence = encode_program_status(status, app="a" * 32)
        assert len(sequence.encode()) <= MAX_SEQUENCE_BYTES
        decoded = _decode(sequence)
        assert len(decoded["title"].encode()) <= MAX_TITLE_BYTES
        assert len(decoded["msg"].encode()) <= MAX_MSG_BYTES

    @pytest.mark.parametrize("app", ["", "a" * 33, "has space", "x:y"])
    def test_rejects_invalid_app(self, app: str) -> None:
        with pytest.raises(ValueError, match="app"):
            encode_program_status(ProgramStatus(ProgramState.IDLE), app=app)


class TestValidation:
    def test_kind_requires_blocked(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            ProgramStatus(ProgramState.WORKING, kind=BlockedKind.AUTH)

    @pytest.mark.parametrize("state", [ProgramState.IDLE, ProgramState.DONE])
    def test_progress_requires_working_or_blocked(self, state: ProgramState) -> None:
        with pytest.raises(ValueError, match="progress"):
            ProgramStatus(state, progress=10)

    @pytest.mark.parametrize("progress", [-1, 101])
    def test_progress_must_be_a_percentage(self, progress: int) -> None:
        with pytest.raises(ValueError, match="0-100"):
            ProgramStatus(ProgramState.WORKING, progress=progress)

    @pytest.mark.parametrize(
        "record_id",
        ["", "a/", "/a", "a//b", "a b", "a:b", "x" * 33, "/".join(["a"] * 9)],
    )
    def test_rejects_invalid_record_id(self, record_id: str) -> None:
        with pytest.raises(ValueError, match="record id"):
            ProgramStatus(ProgramState.IDLE, record_id=record_id)


class TestSanitizeText:
    def test_replaces_control_characters_and_collapses_whitespace(self) -> None:
        assert sanitize_text("a\nb\tc\x1b[31m\x7f\x85d", 100) == "a b c [31m d"

    def test_truncates_on_a_character_boundary(self) -> None:
        assert sanitize_text("ab界", 4) == "ab"

    def test_replaces_lone_surrogates(self) -> None:
        assert sanitize_text("a\udcffb", 10) == "a?b"


class TestReporter:
    def test_reports_and_skips_exact_repeats(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink)
        assert reporter.report(ProgramStatus(ProgramState.WORKING)) is True
        assert reporter.report(ProgramStatus(ProgramState.WORKING)) is False
        assert reporter.report(ProgramStatus(ProgramState.DONE)) is True
        assert [d["state"] for d in sink.decoded] == ["working", "done"]

    def test_disabled_reporter_writes_nothing(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink, enabled=False)
        assert reporter.report(ProgramStatus(ProgramState.WORKING)) is False
        assert reporter.clear() is False
        assert sink.sequences == []

    def test_failed_write_is_retried_next_time(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink)
        sink.ok = False
        assert reporter.report(ProgramStatus(ProgramState.IDLE)) is False
        sink.ok = True
        assert reporter.report(ProgramStatus(ProgramState.IDLE)) is True

    def test_writer_exception_is_swallowed(self) -> None:
        def boom(_sequence: str) -> bool:
            raise OSError

        reporter = ProgramStatusReporter(write=boom)
        assert reporter.report(ProgramStatus(ProgramState.IDLE)) is False

    def test_writer_exception_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(_sequence: str) -> bool:
            raise TypeError

        reporter = ProgramStatusReporter(write=boom)
        with caplog.at_level(logging.DEBUG, logger=program_status.__name__):
            reporter.report(ProgramStatus(ProgramState.IDLE))
            reporter.report(ProgramStatus(ProgramState.WORKING))
        assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.DEBUG]

    def test_clear_is_noop_until_something_was_reported(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink)
        assert reporter.clear() is False
        reporter.report(ProgramStatus(ProgramState.IDLE))
        assert reporter.clear() is True
        assert reporter.clear() is False
        assert [d["state"] for d in sink.decoded] == ["idle", "clear"]

    def test_clear_subtree_forgets_children_only(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink)
        for rid in (None, "task", "task/1", "taskx"):
            reporter.report(ProgramStatus(ProgramState.WORKING, record_id=rid))
        assert reporter.clear("task") is True
        assert sink.decoded[-1] == {
            "state": "clear",
            "id": "task",
            "app": "deepagents-code",
        }
        assert reporter.clear("task/1") is False
        # Records outside the cleared subtree keep deduplicating.
        assert reporter.report(ProgramStatus(ProgramState.WORKING)) is False
        rid = "taskx"
        assert (
            reporter.report(ProgramStatus(ProgramState.WORKING, record_id=rid)) is False
        )

    def test_report_after_clear_is_sent_again(self, sink: _Sink) -> None:
        reporter = ProgramStatusReporter(write=sink)
        reporter.report(ProgramStatus(ProgramState.IDLE))
        reporter.clear()
        assert reporter.report(ProgramStatus(ProgramState.IDLE)) is True

    def test_registers_exit_clear_after_first_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hooks: list[Callable[[], None]] = []
        monkeypatch.setattr(program_status.atexit, "register", hooks.append)
        sink = _Sink()
        reporter = ProgramStatusReporter(write=sink)
        reporter.clear()
        assert hooks == []
        reporter.report(ProgramStatus(ProgramState.WORKING))
        reporter.report(ProgramStatus(ProgramState.DONE))
        assert len(hooks) == 1
        hooks[0]()
        assert sink.decoded[-1]["state"] == "clear"

    def test_no_terminal_escape_env_disables_default_writer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[bool] = []
        monkeypatch.setenv(terminal_escape.NO_TERMINAL_ESCAPE, "1")
        monkeypatch.setattr(terminal_escape, "_open_tty", lambda: opened.append(True))
        reporter = ProgramStatusReporter()
        assert reporter.report(ProgramStatus(ProgramState.WORKING)) is False
        assert opened == []


class TestActivityStatus:
    def test_starts_idle(self) -> None:
        assert activity_status(AgentActivity()) == ProgramStatus(ProgramState.IDLE)

    def test_working_wins_over_last_outcome(self) -> None:
        activity = AgentActivity(working=True, outcome=TurnOutcome.DONE, title="t")
        assert activity_status(activity) == ProgramStatus(
            ProgramState.WORKING, title="t"
        )

    def test_newest_prompt_wins_over_work(self) -> None:
        activity = AgentActivity(
            working=True,
            blockers=(
                (1, Blocker(BlockedKind.PERMISSION, "first")),
                (2, Blocker(BlockedKind.QUESTION, "second")),
            ),
        )
        assert activity_status(activity) == ProgramStatus(
            ProgramState.BLOCKED, kind=BlockedKind.QUESTION, msg="second"
        )

    def test_error_outcome_carries_detail(self) -> None:
        activity = AgentActivity(outcome=TurnOutcome.ERROR, detail="boom")
        assert activity_status(activity) == ProgramStatus(
            ProgramState.ERROR, msg="boom"
        )


class TestExitStatus:
    def test_success_is_done(self) -> None:
        assert exit_status(0).state is ProgramState.DONE

    @pytest.mark.parametrize("code", [1, 124])
    def test_failure_is_error_with_code(self, code: int) -> None:
        assert exit_status(code) == ProgramStatus(
            ProgramState.ERROR, msg=f"Exited with code {code}"
        )

    @pytest.mark.parametrize("code", [130, None])
    def test_interrupt_or_unknown_clears(self, code: int | None) -> None:
        assert exit_status(code).state is ProgramState.CLEAR


class TestAgentStatusTracker:
    @pytest.fixture
    def tracker(self, sink: _Sink) -> AgentStatusTracker:
        tracker = AgentStatusTracker(ProgramStatusReporter(write=sink))
        tracker.start()
        return tracker

    @staticmethod
    def states(sink: _Sink) -> list[str]:
        return [
            f"{d['state']}:{d['kind']}" if "kind" in d else d["state"]
            for d in sink.decoded
        ]

    def test_turn_lifecycle(self, tracker: AgentStatusTracker, sink: _Sink) -> None:
        tracker.set_working(True)
        tracker.set_outcome(TurnOutcome.DONE)
        tracker.set_working(False)
        tracker.set_working(True)
        tracker.set_working(False)
        assert self.states(sink) == ["idle", "working", "done", "working", "idle"]

    def test_error_detail_and_clear_error(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_working(True)
        tracker.set_outcome(TurnOutcome.ERROR, "Agent error: boom")
        tracker.set_working(False)
        assert sink.decoded[-1] == {
            "state": "error",
            "app": "deepagents-code",
            "msg": "Agent error: boom",
        }
        tracker.clear_error()
        assert sink.decoded[-1]["state"] == "idle"

    def test_clear_error_keeps_done(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_outcome(TurnOutcome.DONE)
        tracker.clear_error()
        assert self.states(sink) == ["idle", "done"]

    async def test_tracked_future_blocks_until_resolved(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_working(True)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        tracker.track(future, BlockedKind.PERMISSION, "Approval needed: execute")
        assert sink.decoded[-1]["msg"] == "Approval needed: execute"
        future.cancel()
        await asyncio.sleep(0)
        assert self.states(sink) == ["idle", "working", "blocked:permission", "working"]

    def test_blocked_context_unblocks_on_error(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        with (
            pytest.raises(RuntimeError),
            tracker.blocked(BlockedKind.AUTH, "Sign in"),
        ):
            raise RuntimeError
        assert self.states(sink) == ["idle", "blocked:auth", "idle"]

    def test_answering_a_prompt_marks_the_outcome_seen(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_outcome(TurnOutcome.DONE)
        with tracker.blocked(BlockedKind.AUTH, "Sign in"):
            pass
        assert self.states(sink) == ["idle", "done", "blocked:auth", "idle"]

    @pytest.mark.parametrize("cancelled", [False, True])
    def test_closing_a_prompt_keeps_an_error(
        self, tracker: AgentStatusTracker, sink: _Sink, *, cancelled: bool
    ) -> None:
        tracker.set_outcome(TurnOutcome.ERROR, "Server failed to start")
        with (
            pytest.raises(asyncio.CancelledError)
            if cancelled
            else contextlib.nullcontext(),
            tracker.blocked(BlockedKind.AUTH, "Sign in"),
        ):
            if cancelled:
                raise asyncio.CancelledError
        assert sink.decoded[-1] == {
            "state": "error",
            "app": "deepagents-code",
            "msg": "Server failed to start",
        }

    async def test_new_conversation_drops_transcript_prompts_only(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        tracker.set_outcome(TurnOutcome.ERROR, "boom")
        with tracker.blocked(BlockedKind.AUTH, "Sign in"):
            tracker.track(future, BlockedKind.PERMISSION, "Approve")
            tracker.start_new_conversation()
            assert sink.decoded[-1]["kind"] == "auth"
        assert self.states(sink)[-1] == "idle"
        future.set_result("late")
        await asyncio.sleep(0)
        assert self.states(sink)[-1] == "idle"

    def test_title_is_sent_with_every_report(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_title("Fix login")
        tracker.set_working(True)
        assert [d.get("title") for d in sink.decoded] == [
            None,
            "Fix login",
            "Fix login",
        ]

    def test_refresh_resends_current_report(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.refresh()
        assert self.states(sink) == ["idle", "idle"]

    def test_close_clears_and_stops_reporting(
        self, tracker: AgentStatusTracker, sink: _Sink
    ) -> None:
        tracker.set_working(True)
        tracker.close()
        tracker.set_working(False)
        tracker.refresh()
        tracker.close()
        assert self.states(sink) == ["idle", "working", "clear"]


def test_headless_reporter_keeps_final_report_at_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks: list[Callable[[], None]] = []
    monkeypatch.setattr(program_status.atexit, "register", hooks.append)
    reporter = ProgramStatusReporter(write=_Sink(), clear_on_exit=False)
    reporter.report(exit_status(0))
    assert hooks == []
