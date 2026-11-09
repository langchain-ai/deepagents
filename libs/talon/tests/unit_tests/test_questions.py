from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, AgentResult, ChannelMessage, SendResult
from deepagents_talon.questions import (
    MAX_ANSWER_CHARS,
    Answer,
    Question,
    QuestionSession,
    ask_user,
    parse_answer,
    question_scope,
)
from deepagents_talon.runtime import _question_handler_from_request
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, VoiceTranscriber, _config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

DATABASE = {
    "header": "Database",
    "question": "Which database should we use?",
    "options": [
        {"label": "SQLite", "description": "Zero setup"},
        {"label": "Postgres", "description": "Matches production", "recommended": True},
    ],
}
FEATURES = {
    "header": "Features",
    "question": "Which features ship first?",
    "multi_select": True,
    "options": [{"label": "Auth"}, {"label": "Billing"}, {"label": "Search"}],
}


def _questions(*raw: dict[str, object]) -> list[Question]:
    return [Question.model_validate(item) for item in raw]


def test_prompt_lists_recommendation_first_and_always_offers_other() -> None:
    prompt = QuestionSession(_questions(DATABASE)).prompt()

    assert "1. Postgres (Recommended) — Matches production" in prompt
    assert "2. SQLite — Zero setup" in prompt
    assert "3. Other" in prompt
    assert "Question 1 of" not in prompt


@pytest.mark.parametrize(
    ("raw", "text", "expected"),
    [
        (DATABASE, "1", Answer(("Postgres",))),
        (DATABASE, "2 - keep it local", Answer(("SQLite",), note="keep it local")),
        (DATABASE, "3 MySQL", Answer(other="MySQL")),
        (DATABASE, "other: MySQL", Answer(other="MySQL")),
        (DATABASE, "whatever the team uses", Answer(other="whatever the team uses")),
        (FEATURES, "1, 3 ship fast", Answer(("Auth", "Search"), note="ship fast")),
        (FEATURES, "2 4 SSO", Answer(("Billing",), other="SSO")),
    ],
)
def test_reply_parses_choices_notes_and_custom_answers(
    raw: dict[str, object], text: str, expected: Answer
) -> None:
    assert parse_answer(_questions(raw)[0], text) == expected


@pytest.mark.parametrize(
    ("raw", "text"),
    [(DATABASE, "1 2"), (DATABASE, "9"), (DATABASE, "3"), (FEATURES, "other"), (DATABASE, "")],
)
def test_ambiguous_replies_return_hints(raw: dict[str, object], text: str) -> None:
    assert isinstance(parse_answer(_questions(raw)[0], text), str)


def test_single_question_submits_on_answer_and_can_be_cancelled() -> None:
    session = QuestionSession(_questions(DATABASE))
    assert session.reply("1").status == "submitted"
    assert session.result == (Answer(("Postgres",)),)
    assert QuestionSession(_questions(DATABASE)).reply("CANCEL").status == "cancelled"


def test_batch_shows_progress_preserves_answers_and_requires_submit() -> None:
    session = QuestionSession(_questions(DATABASE, FEATURES))
    assert session.prompt().startswith("Question 1 of 2 · Database")

    step = session.reply("2")
    assert step.message.startswith("Question 2 of 2 · Features")
    step = session.reply("back")
    assert "Current answer: SQLite" in step.message
    session.reply("next")
    step = session.reply("1 2")
    assert step.status == "asking"
    assert "Review your answers" in step.message
    assert "submit" in step.message

    session.reply("edit 1")
    step = session.reply("1")
    assert step.status == "asking"
    assert "1. Database: Postgres" in step.message
    assert session.reply("submit").status == "submitted"
    assert session.result == (Answer(("Postgres",)), Answer(("Auth", "Billing")))


@pytest.mark.parametrize(
    "options",
    [
        [{"label": "A"}, {"label": "a"}],
        [{"label": "A"}, {"label": "Other"}],
        [{"label": "A", "recommended": True}, {"label": "B", "recommended": True}],
        [{"label": "A"}],
    ],
)
def test_question_schema_rejects_ambiguous_options(options: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError):
        Question.model_validate({"header": "H", "question": "Q?", "options": options})


async def test_tool_reports_unavailable_without_a_person_to_ask() -> None:
    args = {"questions": [DATABASE]}
    assert "none were asked" in await ask_user.ainvoke(args)
    with question_scope(None):
        assert "none were asked" in await ask_user.ainvoke(args)
    token = _IN_SUBAGENT.set(True)
    try:
        with question_scope(_fail):
            assert "unavailable to subagents" in await ask_user.ainvoke(args)
    finally:
        _IN_SUBAGENT.reset(token)


async def _fail(_questions: Sequence[Question]) -> tuple[Answer, ...] | None:
    pytest.fail("question reached a handler")


async def test_tool_result_is_never_approval_and_parallel_calls_wait_their_turn() -> None:
    active = 0
    overlap = False

    async def handler(questions: Sequence[Question]) -> tuple[Answer, ...] | None:
        nonlocal active, overlap
        active += 1
        overlap = overlap or active > 1
        await asyncio.sleep(0)
        active -= 1
        return tuple(Answer(("Postgres",)) for _ in questions)

    with question_scope(handler):
        first, second = await asyncio.gather(
            ask_user.ainvoke({"questions": [DATABASE]}),
            ask_user.ainvoke({"questions": [DATABASE]}),
        )
    assert not overlap
    assert "Answer: Postgres" in first
    assert "do not approve any action" in second.lower()


async def test_cancelled_questions_tell_the_agent_not_to_assume() -> None:
    async def cancelled(_questions: Sequence[Question]) -> tuple[Answer, ...] | None:
        return None

    with question_scope(cancelled):
        assert "did not answer" in await ask_user.ainvoke({"questions": [DATABASE]})


@pytest.mark.parametrize("metadata", [{"trigger": "cron"}, {"background_delivery": True}])
def test_unattended_runs_get_no_question_handler(metadata: dict[str, object]) -> None:
    request = AgentRequest("chat", "hi", metadata=metadata, question_handler=_fail)
    assert _question_handler_from_request(request) is None


@pytest.mark.parametrize(
    ("provider", "is_dm"),
    [("test", False), ("slack", False), ("discord", False), ("slack", True), ("discord", True)],
)
async def test_host_waits_for_the_starting_sender_without_starting_a_turn(
    tmp_path: Path, provider: str, *, is_dm: bool
) -> None:
    channel = RecordingChannel(provider)
    agent = BlockingAgent()
    host = TalonHost(
        config=_config(tmp_path),
        agent=agent,
        channels=[channel],
        voice_transcriber=VoiceTranscriber(),
    )
    await host.start()
    try:
        await host.receive_message(
            channel, ChannelMessage("chat", "block", sender_id="alice", metadata={"is_dm": is_dm})
        )
        await asyncio.sleep(0)
        handler = agent.requests[0].question_handler
        assert handler is not None
        asked = asyncio.create_task(handler(_questions(DATABASE, FEATURES)))
        await asyncio.sleep(0)
        assert channel.sent[-1][1].startswith("Question 1 of 2")
        assert ("Mention the bot" in channel.sent[-1][1]) == (provider != "test" and not is_dm)

        await host.receive_message(channel, ChannelMessage("chat", "1", sender_id="mallory"))
        assert "Only the person who started" in channel.sent[-1][1]
        callback = "http://127.0.0.1:6359/callback?code=synthetic&state=synthetic"
        await host.receive_message(channel, ChannelMessage("chat", callback, sender_id="alice"))
        assert "No matching MCP authorization" in channel.sent[-1][1]
        for reply in (
            ChannelMessage("chat", "1 note", sender_id="alice"),
            ChannelMessage("chat", "", sender_id="alice", metadata={"voice_path": "voice.ogg"}),
            ChannelMessage("chat", "submit", sender_id="alice"),
        ):
            await host.receive_message(channel, reply)

        answers = await asyncio.wait_for(asked, 2)
        assert answers == (Answer(("Postgres",), note="note"), Answer(other="transcribed voice"))
        assert len(agent.requests) == 1
    finally:
        agent.released.set()
        await host.stop()


@pytest.mark.parametrize("successful_sends", [0, 1, 2])
async def test_failed_question_delivery_unblocks_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, successful_sends: int
) -> None:
    channel = RecordingChannel()
    monkeypatch.setattr(
        channel,
        "send_message",
        AsyncMock(
            side_effect=[SendResult(success=True)] * successful_sends + [SendResult(success=False)]
        ),
    )
    host = TalonHost(config=_config(tmp_path), agent=BlockingAgent(), channels=[channel])
    asked = asyncio.create_task(
        host._ask_questions(
            channel,
            _questions(DATABASE, FEATURES),
            conversation_id="test:chat",
            message=ChannelMessage("chat", "choose", sender_id="alice"),
        )
    )
    try:
        await asyncio.sleep(0)
        for _ in range(successful_sends):
            await host.receive_message(channel, ChannelMessage("chat", "1", sender_id="alice"))
        assert await asyncio.wait_for(asked, 2) is None
        assert not host._pending_questions
    finally:
        asked.cancel()
        await asyncio.gather(asked, return_exceptions=True)


async def test_host_offers_no_questions_without_a_sender(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("chat", "block"))
        await asyncio.sleep(0)
        assert agent.requests[0].question_handler is None
    finally:
        agent.released.set()
        await host.stop()


class QuestionAgent(BlockingAgent):
    def __init__(self, *, batch: bool = False, rounds: int = 1) -> None:
        super().__init__()
        self.questions = _questions(DATABASE, FEATURES) if batch else _questions(DATABASE)
        self.rounds = rounds
        self.answers: list[tuple[Answer, ...] | None] = []
        self.finished = asyncio.Event()

    async def invoke(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        assert request.question_handler is not None
        try:
            for _ in range(self.rounds):
                self.answers.append(await request.question_handler(self.questions))
        finally:
            self.finished.set()
        return AgentResult(text="done")


@pytest.mark.parametrize(
    ("action", "transcript"),
    [("answer", "2"), ("/stop", "2"), ("/new", "2"), ("replace", "2"), ("/stop", None)],
)
async def test_voice_question_reply_leaves_threads_responsive_and_discards_stale_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str, transcript: str | None
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def transcribe(_message: ChannelMessage) -> str | None:
        started.set()
        await release.wait()
        return transcript

    transcriber = VoiceTranscriber()
    monkeypatch.setattr(transcriber, "transcribe", transcribe)
    channel, agent = RecordingChannel("slack"), QuestionAgent()
    host = TalonHost(
        config=_config(tmp_path), agent=agent, channels=[channel], voice_transcriber=transcriber
    )

    def message(text: str, chat: str = "C123:thread", *, voice: bool = False) -> ChannelMessage:
        metadata: dict[str, object] = {"is_dm": False}
        if voice:
            metadata["voice_path"] = "voice.ogg"
        return ChannelMessage(chat, text, sender_id="alice", metadata=metadata)

    await host.start()
    voice = None
    try:
        await host.receive_message(channel, message("ask"))
        await asyncio.sleep(0)
        voice = asyncio.create_task(host.receive_message(channel, message("", voice=True)))
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(host.receive_message(channel, message("/stop", "C123:other")), 2)
        assert channel.sent[-1] == ("C123:other", "No in-flight run to stop.")

        if action == "replace":
            await host.receive_message(channel, message("1"))
            await asyncio.wait_for(agent.finished.wait(), 2)
            agent.finished.clear()
            await host.receive_message(channel, message("ask again"))
            await asyncio.sleep(0)
        elif action != "answer":
            await asyncio.wait_for(host.receive_message(channel, message(action)), 2)

        release.set()
        await asyncio.wait_for(voice, 2)
        assert any("That question closed" in text for _, text in channel.sent) == (
            action != "answer"
        )
        if action == "replace":
            assert agent.answers == [(Answer(("Postgres",)),)]
            await host.receive_message(channel, message("2"))
        await asyncio.wait_for(agent.finished.wait(), 2)
        assert len(agent.requests) == (2 if action == "replace" else 1)
        expected = [(Answer(("SQLite",)),)] if action in {"answer", "replace"} else []
        if action == "replace":
            expected.insert(0, (Answer(("Postgres",)),))
        assert agent.answers == expected
    finally:
        release.set()
        if voice is not None:
            voice.cancel()
            await asyncio.gather(voice, return_exceptions=True)
        await host.stop()


@pytest.mark.parametrize(("reply", "rounds"), [("1", 1), ("submit", 1), ("cancel", 1), ("1", 2)])
async def test_follow_up_queued_behind_question_completion_reaches_current_recipient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str, rounds: int
) -> None:
    acknowledging, release = asyncio.Event(), asyncio.Event()
    channel, agent = RecordingChannel(), QuestionAgent(batch=reply == "submit", rounds=rounds)
    send = channel.send_message

    async def send_message(chat: str, text: str) -> SendResult:
        if text.startswith(("Answer sent:", "Answers sent;", "Questions cancelled;")):
            acknowledging.set()
            await release.wait()
        return await send(chat, text)

    monkeypatch.setattr(channel, "send_message", send_message)
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])

    async def receive(text: str) -> None:
        await host.receive_message(channel, ChannelMessage("chat", text, sender_id="alice"))

    await host.start()
    tasks: list[asyncio.Task[None]] = []
    try:
        await receive("ask")
        await asyncio.sleep(0)
        if reply == "submit":
            await receive("1")
            await receive("1")
        tasks.append(asyncio.create_task(receive(reply)))
        await asyncio.wait_for(acknowledging.wait(), 2)
        follow_up = "also keep it under 100 lines"
        tasks.append(asyncio.create_task(receive(follow_up)))
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        await asyncio.sleep(0)
        if rounds == 2:
            await asyncio.wait_for(agent.finished.wait(), 2)
            assert len(agent.requests) == 1
            assert agent.answers == [(Answer(("Postgres",)),), (Answer(other=follow_up),)]
        else:
            assert [request.text for request in agent.requests] == ["ask", follow_up]
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await host.stop()


def test_answer_at_character_limit_is_preserved() -> None:
    session = QuestionSession(_questions(DATABASE))
    text = "x" * MAX_ANSWER_CHARS
    assert session.reply(text).status == "submitted"
    assert session.result == (Answer(other=text),)


def test_oversized_answer_is_rejected_without_advancing_the_batch() -> None:
    session = QuestionSession(_questions(DATABASE, FEATURES))
    session.reply("2")
    step = session.reply("x" * (MAX_ANSWER_CHARS + 1))
    assert step.status == "asking"
    assert f"{MAX_ANSWER_CHARS} characters" in step.message
    session.reply("1")
    assert session.reply("submit").status == "submitted"
    assert session.result == (Answer(("SQLite",)), Answer(("Auth",)))
