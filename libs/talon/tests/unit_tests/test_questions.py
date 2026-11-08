from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.questions import (
    Answer,
    Question,
    QuestionSession,
    ask_user,
    parse_answer,
    question_scope,
)
from deepagents_talon.runtime import _question_handler_from_request
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config

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


async def test_host_waits_for_the_starting_sender_without_starting_a_turn(tmp_path: Path) -> None:
    channel = RecordingChannel()
    agent = BlockingAgent()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, ChannelMessage("chat", "block", sender_id="alice"))
        await asyncio.sleep(0)
        handler = agent.requests[0].question_handler
        assert handler is not None
        asked = asyncio.create_task(handler(_questions(DATABASE, FEATURES)))
        await asyncio.sleep(0)
        assert channel.sent[-1][1].startswith("Question 1 of 2")

        await host.receive_message(channel, ChannelMessage("chat", "1", sender_id="mallory"))
        assert "Only the person who started" in channel.sent[-1][1]
        for reply in ("1 note", "2", "submit"):
            await host.receive_message(channel, ChannelMessage("chat", reply, sender_id="alice"))

        answers = await asyncio.wait_for(asked, 2)
        assert answers == (Answer(("Postgres",), note="note"), Answer(("Billing",)))
        assert len(agent.requests) == 1
    finally:
        agent.released.set()
        await host.stop()


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
