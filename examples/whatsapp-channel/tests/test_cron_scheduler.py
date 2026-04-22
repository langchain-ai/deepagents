"""Tests for cron.scheduler — tick loop and run/deliver helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from cron.jobs import create_job as db_create_job, get_job, load_jobs, save_jobs
from cron.scheduler import (
    SILENT_MARKER,
    _build_prompt,
    _extract_final_text,
    _run_job,
    _tick_once,
)
from cron.scheduler import start_ticker


class TestBuildPrompt:
    def test_prepends_cron_hint(self) -> None:
        out = _build_prompt("do the thing")
        assert "SYSTEM:" in out
        assert SILENT_MARKER in out
        assert out.endswith("do the thing")


class TestExtractFinalText:
    def test_extracts_string_content(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        out = {"messages": [
            HumanMessage(content="q"),
            AIMessage(content="the answer"),
        ]}
        assert _extract_final_text(out) == "the answer"

    def test_extracts_list_content_text_block(self) -> None:
        from langchain_core.messages import AIMessage

        out = {"messages": [AIMessage(content=[
            {"type": "tool_use", "id": "x", "name": "y", "input": {}},
            {"type": "text", "text": "the answer"},
        ])]}
        assert _extract_final_text(out) == "the answer"

    def test_empty_when_no_ai_message(self) -> None:
        from langchain_core.messages import HumanMessage

        assert _extract_final_text({"messages": [HumanMessage(content="q")]}) == ""

    def test_empty_on_missing_messages(self) -> None:
        assert _extract_final_text({}) == ""
        assert _extract_final_text(None) == ""


@dataclass
class FakeSendResult:
    success: bool = True
    message_id: str | None = "fake-msg"
    error: str | None = None


@dataclass
class FakeAdapter:
    """Captures send() calls; all sends succeed unless configured otherwise."""
    sent: list[dict[str, Any]] = field(default_factory=list)
    fail_next: bool = False

    async def send(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> FakeSendResult:
        self.sent.append({"chat_id": chat_id, "text": text, "reply_to": reply_to})
        if self.fail_next:
            self.fail_next = False
            return FakeSendResult(success=False, error="mocked failure")
        return FakeSendResult()


class FakeAgent:
    """Returns a pre-set AIMessage for any input."""

    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(state)
        if isinstance(self._response, Exception):
            raise self._response
        return {"messages": [*state["messages"], AIMessage(content=self._response)]}


class TestRunJob:
    async def test_success_returns_text(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="hi", schedule="30m",
                          origin={"chat_id": "1", "message_id": None})
        agent = FakeAgent("hello world")
        success, text, error = await _run_job(agent, j)
        assert success is True
        assert text == "hello world"
        assert error is None

    async def test_failure_captures_exception(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="hi", schedule="30m",
                          origin={"chat_id": "1", "message_id": None})
        agent = FakeAgent(RuntimeError("boom"))
        success, text, error = await _run_job(agent, j)
        assert success is False
        assert text == ""
        assert "boom" in (error or "")


class TestTickOnce:
    async def test_no_jobs_no_sends(self, jobs_path) -> None:
        adapter = FakeAdapter()
        agent = FakeAgent("unused")
        await _tick_once(jobs_path, adapter, agent, {})
        assert adapter.sent == []

    async def test_fires_one_shot_and_delivers(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": "msg1"})
        # Backdate so the job is due
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter()
        agent = FakeAgent("here is the report")
        await _tick_once(jobs_path, adapter, agent, {})

        assert len(adapter.sent) == 1
        assert adapter.sent[0]["chat_id"] == "chat1"
        assert adapter.sent[0]["text"] == "here is the report"
        assert adapter.sent[0]["reply_to"] == "msg1"
        # One-shot disabled after firing
        assert get_job(jobs_path, j["id"])["enabled"] is False

    async def test_silent_marker_suppresses_delivery(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": None})
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter()
        agent = FakeAgent(SILENT_MARKER)
        await _tick_once(jobs_path, adapter, agent, {})

        assert adapter.sent == []
        assert get_job(jobs_path, j["id"])["last_status"] == "ok"

    async def test_failure_delivers_error_notice(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": None})
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter()
        agent = FakeAgent(RuntimeError("nope"))
        await _tick_once(jobs_path, adapter, agent, {})

        assert len(adapter.sent) == 1
        assert adapter.sent[0]["text"].startswith("⚠️")
        assert "nope" in adapter.sent[0]["text"]
        assert get_job(jobs_path, j["id"])["last_status"] == "error"

    async def test_delivery_failure_captured_in_last_error(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": None})
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter(fail_next=True)
        agent = FakeAgent("ok")
        await _tick_once(jobs_path, adapter, agent, {})

        updated = get_job(jobs_path, j["id"])
        assert updated["last_status"] == "error"
        assert "delivery failed" in (updated["last_error"] or "")

    async def test_interval_next_run_advances(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="every 10m",
                          origin={"chat_id": "chat1", "message_id": None})
        first_next = j["next_run_at"]
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter()
        agent = FakeAgent("ok")
        await _tick_once(jobs_path, adapter, agent, {})

        updated = get_job(jobs_path, j["id"])
        assert updated["enabled"] is True
        assert updated["next_run_at"] != first_next

    async def test_advance_before_run_persists_new_next_run_at(
        self, jobs_path,
    ) -> None:
        """If the agent crashes mid-run, the stored next_run_at should already
        have advanced so the job doesn't re-fire on a subsequent tick."""

        j = db_create_job(jobs_path, prompt="p", schedule="every 10m",
                          origin={"chat_id": "chat1", "message_id": None})
        first_next = j["next_run_at"]
        _force_due(jobs_path, j["id"])

        captured: dict[str, Any] = {}

        class ReadBackAgent:
            async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
                # At the moment the agent runs, next_run_at should already be advanced
                captured["next_run_at_during_run"] = get_job(jobs_path, j["id"])["next_run_at"]
                return {"messages": [*state["messages"], AIMessage(content="ok")]}

        adapter = FakeAdapter()
        await _tick_once(jobs_path, adapter, ReadBackAgent(), {})

        assert captured["next_run_at_during_run"] != first_next

    async def test_reuses_per_chat_lock(self, jobs_path) -> None:
        """The chat_locks map must be used; a pre-held lock blocks the run."""
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": None})
        _force_due(jobs_path, j["id"])

        locks: dict[str, asyncio.Lock] = {"chat1": asyncio.Lock()}
        await locks["chat1"].acquire()

        adapter = FakeAdapter()
        agent = FakeAgent("hi")

        tick_task = asyncio.create_task(_tick_once(jobs_path, adapter, agent, locks))
        await asyncio.sleep(0.05)
        assert adapter.sent == []  # still blocked
        assert tick_task.done() is False

        locks["chat1"].release()
        await tick_task

        assert len(adapter.sent) == 1


def _force_due(jobs_path, job_id) -> None:
    """Rewrite the job's next_run_at to one minute ago so it's due."""
    from datetime import timedelta
    jobs = load_jobs(jobs_path)
    for j in jobs:
        if j["id"] == job_id:
            j["next_run_at"] = (_now_helper() - timedelta(minutes=1)).isoformat()
    save_jobs(jobs_path, jobs)


def _now_helper():
    from datetime import datetime
    return datetime.now().astimezone()


class TestStartTicker:
    async def test_fires_due_job_then_cancels_cleanly(self, jobs_path) -> None:
        j = db_create_job(jobs_path, prompt="p", schedule="30m",
                          origin={"chat_id": "chat1", "message_id": None})
        _force_due(jobs_path, j["id"])

        adapter = FakeAdapter()
        agent = FakeAgent("done")
        # Very short tick interval for the test
        task = start_ticker(
            jobs_path, adapter, agent, chat_locks={}, tick_interval=0.05,
        )

        # Wait up to 1s for the send
        for _ in range(40):
            if adapter.sent:
                break
            await asyncio.sleep(0.05)

        assert len(adapter.sent) == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_swallows_tick_errors_and_keeps_running(
        self, jobs_path, caplog,
    ) -> None:
        """A corrupt jobs.json should not kill the ticker."""
        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        jobs_path.write_text("{not json")

        adapter = FakeAdapter()
        agent = FakeAgent("x")
        task = start_ticker(
            jobs_path, adapter, agent, chat_locks={}, tick_interval=0.05,
        )
        await asyncio.sleep(0.15)
        assert task.done() is False  # still running despite the error

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
