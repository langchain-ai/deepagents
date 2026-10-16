from __future__ import annotations

import asyncio
import json
from html import escape, unescape
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import PrivateAttr

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.heads_up import _MAX_INPUT, _notice, _Review, _snapshot
from deepagents_talon.interfaces import AgentRequest, SendResult
from deepagents_talon.runtime import DeepAgentRuntime
from tests.unit_tests.test_research_subagents import ToolModel

if TYPE_CHECKING:
    from pathlib import Path


class ObserverModel(ToolModel):
    _reviews: list = PrivateAttr(default_factory=list)
    _blocked: asyncio.Event | None = PrivateAttr(default=None)
    _entered: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _cancelled: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _fail: bool = PrivateAttr(default=False)

    async def ainvoke(self, messages, *args: object, **kwargs: object):
        if not messages[-1].text.startswith("<conversation>"):
            return await super().ainvoke(messages, *args, **kwargs)
        self._reviews.append(messages)
        self._entered.set()
        if self._fail:
            msg = "private provider failure"
            raise RuntimeError(msg)
        if self._blocked is not None:
            try:
                await self._blocked.wait()
            except asyncio.CancelledError:
                self._cancelled.set()
                raise
        payload = unescape(
            messages[-1].text.removeprefix("<conversation>").removesuffix("</conversation>")
        )
        evidence = json.loads(payload)
        key = next(key for key, (_label, text) in evidence.items() if "without approval" in text)
        return AIMessage(
            content=json.dumps(
                {
                    "summary": "The proposed deadline has not been approved.",
                    "message": key,
                    "quote": "deadline without approval",
                }
            )
        )


@pytest.mark.parametrize("enabled", [False, True])
async def test_runtime_opt_in_final_review(tmp_path: Path, monkeypatch, *, enabled: bool):
    model = ObserverModel(responses=[AIMessage(content="I proposed a deadline without approval.")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        include_web_tools=False,
        env={"DEEPAGENTS_TALON_HEADS_UP": str(enabled)},
    )
    sent = []

    async def deliver(text):
        sent.append(text)
        return SendResult(success=True)

    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest("chat", "draft a proposal", message_handler=deliver)
        )
        assert result.text == "I proposed a deadline without approval."
        assert len(model._reviews) == int(enabled)
        assert len(sent) == int(enabled)
        if enabled:
            assert "Evidence (ai): deadline without approval" in sent[0]
            assert all(isinstance(m, (SystemMessage, HumanMessage)) for m in model._reviews[0])
    finally:
        await runtime.stop()


@pytest.mark.parametrize("metadata", [{"trigger": "cron"}, {"background_delivery": True}])
async def test_unattended_runs_skip_review(tmp_path: Path, monkeypatch, metadata):
    model = ObserverModel(responses=[AIMessage(content="deadline without approval")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )

    async def deliver(_text):
        pytest.fail("unattended review should not deliver")

    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest("chat", "work", metadata=metadata, message_handler=deliver)
        )
        assert model._reviews == []
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "not json",
        json.dumps(
            {
                "summary": "Unsupported finding",
                "message": "m0",
                "quote": "a fabricated excerpt",
            }
        ),
    ],
)
def test_rejects_empty_invalid_and_fabricated_findings(content):
    assert (
        _notice(AIMessage(content=content), {"m0": ("user", "deadline without approval")}) is None
    )


def test_snapshot_bounds_and_redacts_visible_evidence():
    evidence = _snapshot(
        [
            SystemMessage("PRIVATE SYSTEM"),
            HumanMessage("api_key=private-token"),
            ToolMessage("deadline without approval", tool_call_id="call", name="calendar"),
            AIMessage(
                content=[
                    {"type": "text", "text": "visible"},
                    {"type": "reasoning", "reasoning": "PRIVATE REASONING"},
                ]
            ),
        ]
    )
    assert "PRIVATE" not in str(evidence)
    assert "private-token" not in str(evidence)
    assert evidence["m2"] == ("tool calendar", "deadline without approval")
    huge = _snapshot([HumanMessage("<&😀" * 1_000) for _ in range(100)])
    assert len(escape(json.dumps(huge, ensure_ascii=False)).encode()) <= _MAX_INPUT


def test_notice_does_not_notify_slack_mentions():
    response = AIMessage(
        content=json.dumps(
            {
                "summary": "Ask <@U123> about this.",
                "message": "m0",
                "quote": "deadline without approval",
            }
        )
    )
    notice = _notice(response, {"m0": ("user", "deadline without approval")})
    assert notice is not None
    assert "<@U123>" not in notice


async def test_periodic_review_is_nonblocking_and_cancelled():
    model = ObserverModel(responses=[AIMessage(content="unused")])
    model._blocked = asyncio.Event()
    sent = []

    async def deliver(text):
        sent.append(text)
        return SendResult(success=True)

    review = _Review(deliver)
    messages = [
        HumanMessage("deadline without approval"),
        AIMessage(
            content="working",
            tool_calls=[
                {"name": "work", "args": {}, "id": "work"},
            ],
        ),
    ]
    for _ in range(6):
        await review.observe(model, messages)
    await model._entered.wait()
    assert sent == []
    assert review.pending is not None
    await review.close()
    assert model._cancelled.is_set()
    assert review.pending is None
    assert sent == []


async def test_duplicate_evidence_and_budget():
    model = ObserverModel(responses=[AIMessage(content="unused")])
    sent = []

    async def deliver(text):
        sent.append(text)
        return SendResult(success=True)

    review = _Review(deliver)
    messages = [HumanMessage("deadline without approval"), AIMessage(content="done")]
    for _ in range(10):
        await review.observe(model, messages)
    assert len(model._reviews) == 3
    assert len(sent) == 1


async def test_observer_failure_does_not_fail_main_turn(tmp_path: Path, monkeypatch):
    model = ObserverModel(responses=[AIMessage(content="done")])
    model._fail = True
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )

    async def deliver(_text):
        pytest.fail("failed review should not deliver")

    await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("chat", "work", message_handler=deliver))
        assert result.text == "done"
    finally:
        await runtime.stop()


async def test_subagent_review_excluded(tmp_path: Path, monkeypatch):
    model = ObserverModel(responses=[AIMessage(content="deadline without approval")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )

    async def deliver(_text):
        pytest.fail("subagent review should not deliver")

    await runtime.start()
    token = _IN_SUBAGENT.set(True)
    try:
        await runtime.invoke(AgentRequest("chat", "work", message_handler=deliver))
        assert model._reviews == []
    finally:
        _IN_SUBAGENT.reset(token)
        await runtime.stop()


async def test_turn_cancellation_discards_final_notice(tmp_path: Path, monkeypatch):
    model = ObserverModel(responses=[AIMessage(content="deadline without approval")])
    model._blocked = asyncio.Event()
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )
    sent = []

    async def deliver(text):
        sent.append(text)
        return SendResult(success=True)

    await runtime.start()
    try:
        task = asyncio.create_task(
            runtime.invoke(AgentRequest("chat", "work", message_handler=deliver))
        )
        await model._entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model._cancelled.is_set()
        assert sent == []
    finally:
        await runtime.stop()


async def test_concurrent_chat_reviews_are_isolated(tmp_path: Path, monkeypatch):
    model = ObserverModel(responses=[AIMessage(content="deadline without approval")])
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )
    sent = [[], []]

    async def run(index):
        async def deliver(text):
            sent[index].append(text)
            return SendResult(success=True)

        return await runtime.invoke(
            AgentRequest(str(index), f"chat-{index}", message_handler=deliver)
        )

    await runtime.start()
    try:
        results = await asyncio.gather(run(0), run(1))
        assert all(result.text == "deadline without approval" for result in results)
        assert [len(notices) for notices in sent] == [1, 1]
        assert len(model._reviews) == 2
        snapshots = [messages[-1].text for messages in model._reviews]
        assert sum("chat-0" in snapshot for snapshot in snapshots) == 1
        assert sum("chat-1" in snapshot for snapshot in snapshots) == 1
    finally:
        await runtime.stop()


async def test_review_uses_selected_chat_model(tmp_path: Path, monkeypatch):
    default = ObserverModel(responses=[AIMessage(content="default")])
    selected = ObserverModel(responses=[AIMessage(content="deadline without approval")])
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: default
    )
    runtime = DeepAgentRuntime(
        model="test:parent",
        assistant_dir=tmp_path,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_HEADS_UP": "true"},
    )
    monkeypatch.setattr(runtime.models, "resolve", lambda _spec: selected)
    sent = []

    async def deliver(text):
        sent.append(text)
        return SendResult(success=True)

    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest(
                "chat",
                "work",
                model="test:selected",
                message_handler=deliver,
            )
        )
        assert result.text == "deadline without approval"
        assert len(selected._reviews) == 1
        assert default._reviews == []
        assert len(sent) == 1
    finally:
        await runtime.stop()
