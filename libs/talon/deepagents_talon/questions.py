"""Structured clarifying questions answered by the person who started the turn.

Questions are asked and answered as plain chat messages, so they work on every
channel. Answers clarify intent only: they never approve a gated tool call.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepagents_talon.background import _IN_SUBAGENT

MAX_QUESTIONS = 4
MAX_ANSWER_CHARS = 2000
_COMMANDS = frozenset({"back", "next", "submit", "cancel"})
_SEPARATORS = " \t\n-\u2013\u2014:;,.)"
_NUMBERS = re.compile(r"\d+(?:(?:\s*,\s*|\s+)\d+)*")
_NOT_APPROVAL = (
    "These answers clarify intent only. They do not approve any action: tools that "
    "require approval still ask for it separately."
)


class QuestionOption(BaseModel):
    """One labeled choice with its trade-off."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(min_length=1, max_length=80, description="Short choice text.")
    description: str = Field(
        default="",
        max_length=300,
        description="One line on the trade-off or consequence of choosing this option.",
    )
    recommended: bool = Field(
        default=False, description="Mark the option you recommend; it is shown first."
    )


class Question(BaseModel):
    """One decision for the user, with an automatic Other answer."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    header: str = Field(
        min_length=1, max_length=30, description="Very short topic label, e.g. `Database`."
    )
    question: str = Field(min_length=1, max_length=1000, description="The full question.")
    options: list[QuestionOption] = Field(
        min_length=2,
        max_length=4,
        description="Distinct choices. Do not add an Other option; one is always offered.",
    )
    multi_select: bool = Field(
        default=False, description="True when the user may select every option that applies."
    )

    @model_validator(mode="after")
    def _check_options(self) -> Question:
        labels = [option.label.casefold() for option in self.options]
        if len(set(labels)) != len(labels):
            msg = "Option labels must be distinct."
            raise ValueError(msg)
        if "other" in labels:
            msg = "Do not add an Other option; one is always offered."
            raise ValueError(msg)
        if not self.multi_select and sum(o.recommended for o in self.options) > 1:
            msg = "A single-choice question can recommend at most one option."
            raise ValueError(msg)
        return self

    def ordered_options(self) -> list[QuestionOption]:
        """Return options with recommended ones first, otherwise in the given order."""
        return sorted(self.options, key=lambda option: not option.recommended)


@dataclass(frozen=True, slots=True)
class Answer:
    """One answered question: picked labels, an optional custom answer, and a note."""

    selected: tuple[str, ...] = ()
    other: str | None = None
    note: str | None = None

    def describe(self) -> str:
        """Render the answer on one line for prompts and the tool result."""
        parts = [*self.selected, *([f"Other: {self.other}"] if self.other else [])]
        text = ", ".join(parts)
        return f"{text} (note: {self.note})" if self.note else text


QuestionHandler = Callable[[Sequence[Question]], Awaitable[tuple[Answer, ...] | None]]
"""Ask the turn's sender and wait; `None` means the questions were not answered."""

QUESTION_HANDLER: ContextVar[QuestionHandler | None] = ContextVar(
    "talon_question_handler", default=None
)


@contextmanager
def question_scope(handler: QuestionHandler | None) -> Generator[None]:
    """Bind a handler to one invocation, asking parallel calls one batch at a time."""
    lock = asyncio.Lock()

    async def serialized(questions: Sequence[Question]) -> tuple[Answer, ...] | None:
        async with lock:
            return await handler(questions) if handler is not None else None

    token = QUESTION_HANDLER.set(serialized if handler is not None else None)
    try:
        yield
    finally:
        QUESTION_HANDLER.reset(token)


@tool
async def ask_user(
    questions: Annotated[list[Question], Field(min_length=1, max_length=MAX_QUESTIONS)],
) -> str:
    """Ask the user 1-4 related multiple-choice questions and wait for their answers.

    Use only for a decision that materially changes the outcome and that you cannot
    resolve from context. Give 2-4 distinct options, each with a one-line trade-off,
    and mark your recommendation. The user can always give their own answer and add
    a note. Answers are not approval: gated tools still ask for approval separately.

    Args:
        questions: Related questions, asked in order and submitted together.
    """
    if _IN_SUBAGENT.get():
        return "Questions are unavailable to subagents; report the open decision instead."
    handler = QUESTION_HANDLER.get()
    if handler is None:
        return (
            "No one can answer questions in this run, so none were asked. Do not wait: "
            "proceed only where a choice is safe and state your assumption, or explain "
            "in your reply what needs deciding."
        )
    answers = await handler(questions)
    if answers is None:
        return (
            "The user did not answer. Do not assume an answer; continue only with what "
            "is already clear, or explain what you need decided."
        )
    return format_answers(questions, answers)


def format_answers(questions: Sequence[Question], answers: Sequence[Answer]) -> str:
    """Render answered questions for the model, restating that they are not approval."""
    lines = ["The user answered:"]
    for index, (question, answer) in enumerate(zip(questions, answers, strict=True), start=1):
        lines.append(f"{index}. [{question.header}] {question.question}")
        lines.append(f"   Answer: {answer.describe()}")
    lines.append(_NOT_APPROVAL)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Step:
    """Outcome of one reply: keep asking with `message`, or finish."""

    status: Literal["asking", "submitted", "cancelled"]
    message: str = ""


class QuestionSession:
    """Chat state for one batch: one question per message, then an explicit submit."""

    def __init__(self, questions: Sequence[Question]) -> None:
        """Start at the first question with nothing answered."""
        self.questions = tuple(questions)
        self.answers: list[Answer | None] = [None] * len(self.questions)
        self.index = 0
        self.reviewing = False

    @property
    def result(self) -> tuple[Answer, ...]:
        """Return every answer once all questions are answered."""
        if any(answer is None for answer in self.answers):
            msg = "Not every question is answered."
            raise ValueError(msg)
        return tuple(answer for answer in self.answers if answer is not None)

    def prompt(self) -> str:
        """Render the current question, or the review screen."""
        return self._review() if self.reviewing else self._question()

    def reply(self, text: str) -> Step:
        """Apply one chat reply from the person answering."""
        text = text.strip()[:MAX_ANSWER_CHARS]
        command = text.casefold()
        if command == "cancel":
            return Step("cancelled", "Questions cancelled; the agent will continue without them.")
        if self.reviewing:
            return self._review_reply(command)
        if command in _COMMANDS:
            return self._navigate(command)
        answer = parse_answer(self.questions[self.index], text)
        if isinstance(answer, str):
            return Step("asking", f"{answer}\n\n{self._question()}")
        self.answers[self.index] = answer
        if len(self.questions) == 1:
            return Step("submitted", f"Answer sent: {answer.describe()}")
        return self._advance()

    def _advance(self) -> Step:
        pending = [i for i, answer in enumerate(self.answers) if answer is None]
        later = [i for i in pending if i > self.index]
        if not pending:
            self.reviewing = True
        else:
            self.index = (later or pending)[0]
        return Step("asking", self.prompt())

    def _navigate(self, command: str) -> Step:
        if command == "back" and self.index > 0:
            self.index -= 1
        elif command == "next" and self.answers[self.index] is not None:
            return self._advance()
        elif command in {"back", "next"}:
            hint = "There is no earlier question." if command == "back" else "Answer this first."
            return Step("asking", f"{hint}\n\n{self._question()}")
        else:
            return Step("asking", f"Answer every question first.\n\n{self._question()}")
        return Step("asking", self._question())

    def _review_reply(self, command: str) -> Step:
        if command == "submit":
            return Step("submitted", "Answers sent; the agent is continuing.")
        number = command.removeprefix("edit").strip()
        target = len(self.questions) if command == "back" else 0
        if command.startswith("edit") and number.isdecimal():
            target = int(number)
        if 0 < target <= len(self.questions):
            self.index = target - 1
            self.reviewing = False
            return Step("asking", self._question())
        return Step("asking", f"Reply `submit`, `edit <number>`, or `cancel`.\n\n{self._review()}")

    def _question(self) -> str:
        question = self.questions[self.index]
        total = len(self.questions)
        heading = f"Question {self.index + 1} of {total} · " if total > 1 else ""
        lines = [f"{heading}{question.header}", question.question]
        if question.multi_select:
            lines.append("Select all that apply.")
        lines.append("")
        lines.extend(_option_lines(question))
        current = self.answers[self.index]
        if current is not None:
            lines.extend(["", f"Current answer: {current.describe()}"])
        lines.extend(["", _answer_hint(question), self._navigation_hint()])
        return "\n".join(lines)

    def _navigation_hint(self) -> str:
        if len(self.questions) == 1:
            return "Reply `cancel` to skip the question."
        commands = [
            *(["`back`"] if self.index > 0 else []),
            *(["`next` keeps the current answer"] if self.answers[self.index] else []),
            "`cancel`",
        ]
        return "Also: " + ", ".join(commands) + "."

    def _review(self) -> str:
        lines = ["Review your answers"]
        for index, (question, answer) in enumerate(
            zip(self.questions, self.answers, strict=True), start=1
        ):
            lines.append(f"{index}. {question.header}: {answer.describe() if answer else '-'}")
        lines.extend(
            ["", "Reply `submit` to send them, `edit <number>` to change one, or `cancel`."]
        )
        return "\n".join(lines)


def _option_lines(question: Question) -> list[str]:
    lines = []
    options = question.ordered_options()
    for number, option in enumerate(options, start=1):
        label = f"{option.label} (Recommended)" if option.recommended else option.label
        detail = f" — {option.description}" if option.description else ""
        lines.append(f"{number}. {label}{detail}")
    lines.append(f"{len(options) + 1}. Other — type your own answer")
    return lines


def _answer_hint(question: Question) -> str:
    other = len(question.options) + 1
    if question.multi_select:
        return (
            "Reply with the numbers that apply, e.g. `1 2`, optionally followed by a note. "
            f"Add `{other}` and your text, or just type, to give your own answer."
        )
    return (
        "Reply with one number, optionally followed by a note, e.g. `1 keep it small`. "
        f"Reply `{other}` and your text, or just type, to give your own answer."
    )


def parse_answer(question: Question, text: str) -> Answer | str:
    """Parse one reply into an answer, or return a hint explaining what is wrong."""
    if not text:
        return "Reply with a number or your own answer."
    lowered = text.casefold()
    if lowered == "other" or lowered.startswith(("other:", "other ")):
        custom = text[5:].strip(_SEPARATORS)
        return Answer(other=custom) if custom else "Type your answer after `other`."
    match = _NUMBERS.match(text)
    if match is None or (match.end() < len(text) and text[match.end()] not in _SEPARATORS):
        return Answer(other=text)
    numbers = list(dict.fromkeys(int(n) for n in re.findall(r"\d+", match.group())))
    rest = text[match.end() :].strip(_SEPARATORS) or None
    return _numbered_answer(question, numbers, rest)


def _numbered_answer(question: Question, numbers: list[int], rest: str | None) -> Answer | str:
    options = question.ordered_options()
    other = len(options) + 1
    if any(not 0 < number <= other for number in numbers):
        return f"Choose numbers from 1 to {other}."
    if not question.multi_select and len(numbers) > 1:
        return "Choose one option."
    selected = tuple(options[number - 1].label for number in numbers if number != other)
    if other in numbers:
        return Answer(selected, other=rest) if rest else f"Type your answer after `{other}`."
    return Answer(selected, note=rest)
