"""Lightweight types and shared rendering for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack. The answer placeholders, row summary
strings, and `format_ask_user_transcript` live here for the same reason: the
tool and the TUI both need them and neither should import the other.
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
"""Placeholder for a question with no positionally matching answer.

Recorded only when the answer list came back *shorter* than the question list (a
count mismatch, warned about in `ask_user`). A question the user deliberately
left blank renders as an empty `A:` line instead, never as this placeholder.
"""

ASK_USER_CANCELLED_ANSWER = "(cancelled)"
"""Placeholder recorded for every question when the user cancels the prompt."""

ASK_USER_ERROR_ANSWER_PREFIX = "(error: "
"""Prefix of the placeholder recorded for every question when the prompt fails.

The full placeholder is `(error: <detail>)`. Both the transcript producer in
`ask_user` and the TUI's summary classifier match on this prefix, so a failed
prompt is never summarized as answered.
"""

ASK_USER_ANSWERED_SUMMARY = "User answered"
"""One-line summary shown for an answered `ask_user` row before it is expanded.

Doubles as the `tool.result` hook payload's `tool_output` for an answered
prompt, deliberately in place of the transcript, so user-typed answers are not
forwarded to hook scripts. Rewording this string changes that hook contract as
well as the row; see `textual_adapter` and its `tool.result` tests.
"""

ASK_USER_CANCELLED_SUMMARY = "Question cancelled"
"""One-line summary shown for a cancelled `ask_user` prompt."""

ASK_USER_FAILED_SUMMARY = "Question failed"
"""One-line summary shown for an `ask_user` prompt that errored."""


def format_ask_user_transcript(questions: list[Question], answers: list[str]) -> str:
    r"""Render questions and answers as the `Q:`/`A:` transcript.

    This is the text the `ask_user` tool returns to the model, and the same text
    the TUI shows on the tool row, so the two cannot drift in *format*. The two
    call sites compute it from different inputs (the resume payload server-side,
    the interrupt's questions plus the widget's answers in the TUI), so contents
    can still differ.

    The encoding is lossy: an answer containing a blank line followed by the
    literal next `Q: <text>\nA:` header is indistinguishable from a real block
    boundary. The TUI's parser anchors on the known question text to keep that
    from fabricating an extra question, but cannot recover the split point.

    Args:
        questions: Questions that were asked. `question` is a required key, and
            `ask_user` validates it as non-empty before interrupting, so the
            empty default below is unreachable in practice.
        answers: Answers, positionally matched to `questions`. A missing entry
            falls back to `ASK_USER_NO_ANSWER`; extra entries are dropped.

    Returns:
        Blank-line separated `Q: ...\nA: ...` blocks, one per question.
    """
    blocks = [
        f"Q: {question.get('question', '')}\n"
        f"A: {answers[i] if i < len(answers) else ASK_USER_NO_ANSWER}"
        for i, question in enumerate(questions)
    ]
    return "\n\n".join(blocks)
