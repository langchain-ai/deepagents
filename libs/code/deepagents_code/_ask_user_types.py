"""Lightweight types for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict


class Choice(TypedDict):
    """A single choice option for a multiple choice question."""

    value: Annotated[str, Field(description="The display label for this choice.")]


class Question(TypedDict):
    """A question to ask the user."""

    question: Annotated[str, Field(description="The question text to display.")]

    type: Annotated[
        Literal["text", "multiple_choice"],
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' for "
                "predefined options."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Choice],
            Field(
                description=(
                    "Options for multiple_choice questions. An 'Other' free-form "
                    "option is always appended automatically."
                )
            ),
        ]
    ]

    required: NotRequired[
        Annotated[
            bool,
            Field(
                description="Whether the user must answer. Defaults to true if omitted."
            ),
        ]
    ]


class AskUserRequest(TypedDict):
    """Request payload sent via interrupt when asking the user questions."""

    type: Literal["ask_user"]
    """Discriminator tag, always `'ask_user'`."""

    questions: list[Question]
    """Questions to present to the user."""

    tool_call_id: str
    """ID of the originating tool call, used to route the response back."""


ASK_USER_AUTHORIZATION_METADATA_KEY = "deepagents_code_ask_user_authorization"
MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS = 4000


class AskUserAuthorizationReceipt(TypedDict):
    """Trusted same-turn authorization recorded after an ask_user response."""

    version: Literal[1]
    thread_id: str
    turn_id: str
    tool_call_id: str
    answers: list[str]


class AskUserAnswered(TypedDict):
    """Widget result when the user submits answers."""

    type: Literal["answered"]
    """Discriminator tag, always `'answered'`."""

    answers: list[str]
    """User-provided answers, one per question."""


class AskUserCancelled(TypedDict):
    """Widget result when the user cancels the prompt."""

    type: Literal["cancelled"]
    """Discriminator tag, always `'cancelled'`."""


AskUserWidgetResult = AskUserAnswered | AskUserCancelled
"""Discriminated union for the ask_user widget Future result."""


ASK_USER_NO_ANSWER = "(no answer)"
"""Placeholder recorded when a question received no answer at all."""

ASK_USER_CANCELLED_ANSWER = "(cancelled)"
"""Placeholder recorded for every question when the user cancels the prompt."""

ASK_USER_ANSWERED_SUMMARY = "User answered"
"""One-line summary shown for an answered `ask_user` row before it is expanded."""


def format_ask_user_transcript(questions: list[Question], answers: list[str]) -> str:
    r"""Render questions and answers as the `Q:`/`A:` transcript.

    This is the text the `ask_user` tool returns to the model, and the same text
    the TUI shows on the tool row, so the two cannot drift.

    Args:
        questions: Questions that were asked.
        answers: Answers, positionally matched to `questions`. A missing entry
            falls back to `ASK_USER_NO_ANSWER`.

    Returns:
        Blank-line separated `Q: ...\nA: ...` blocks, one per question.
    """
    blocks = [
        f"Q: {question.get('question', '')}\n"
        f"A: {answers[i] if i < len(answers) else ASK_USER_NO_ANSWER}"
        for i, question in enumerate(questions)
    ]
    return "\n\n".join(blocks)
