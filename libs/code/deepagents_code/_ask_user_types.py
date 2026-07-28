"""Lightweight types and shared rendering for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.

Only the TypedDicts are consumed by both sides. The rest is colocated here so the
whole wire format stays in one file and neither side has to import the other: the
answer placeholders and `format_ask_user_transcript` are tool-side (`ask_user`),
while the `ASK_USER_*_SUMMARY` strings are TUI-side row labels that all double as
`tool.result` hook payloads, as their docstrings note.
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

Defensive fallback for callers that pass fewer answers than questions. The
middleware validates answer counts before formatting, so its normal paths do
not produce this value. A question the user deliberately left blank renders as
an empty `A:` line instead, never as this placeholder.
"""

ASK_USER_CANCELLED_ANSWER = "(cancelled)"
"""Placeholder recorded for every question when the user cancels the prompt."""

ASK_USER_ERROR_ANSWER_PREFIX = "(error: "
"""Prefix of the placeholder recorded for every question when the prompt fails.

The full placeholder is `(error: <detail>)`. Producer-side only: nothing in-tree
matches on it. The TUI derives its row summary from the recorded tool status
instead, because the placeholder is in-band — the prefix alone cannot distinguish
a failed prompt from a user who typed `(error: ...)` as their answer.
"""

ASK_USER_ANSWERED_SUMMARY = "User answered"
"""One-line summary shown for an answered `ask_user` row before it is expanded.

Doubles as the `tool.result` hook payload's `tool_output` for an answered
prompt, deliberately in place of the transcript, so user-typed answers are not
forwarded to hook scripts. Rewording this string changes that hook contract as
well as the row; see `textual_adapter` and its `tool.result` tests.
"""

ASK_USER_CANCELLED_SUMMARY = "Question cancelled"
"""The `tool.result` hook payload's `tool_output` for the cancelled path.

Rewording it changes that hook contract. No longer rendered on a row: a live
cancel calls `set_rejected` (which records no output), and a transcript of
`(cancelled)` placeholders from a non-TUI client is summarized from the recorded
status like any other, so it reads as `ASK_USER_ANSWERED_SUMMARY`.
"""

ASK_USER_FAILED_SUMMARY = "Question failed"
"""One-line summary shown for an `ask_user` prompt that errored.

Like `ASK_USER_ANSWERED_SUMMARY`, this doubles as the `tool.result` hook
payload's `tool_output` for a failed prompt — deliberately in place of the
transcript, whose `(error: ...)` placeholders carry an arbitrary detail string.
Rewording it changes that hook contract as well as the row.
"""


def format_ask_user_error_answer(detail: str) -> str:
    """Render the placeholder answer recorded for every question on failure.

    Keeps the closing paren together with `ASK_USER_ERROR_ANSWER_PREFIX` so the
    sentinel is not split between the constant and its producer.

    Args:
        detail: Human-readable reason the prompt failed.

    Returns:
        The `(error: <detail>)` placeholder.
    """
    return f"{ASK_USER_ERROR_ANSWER_PREFIX}{detail})"


def format_ask_user_transcript(questions: list[Question], answers: list[str]) -> str:
    r"""Render questions and answers as the `Q:`/`A:` transcript.

    This is the text the `ask_user` tool returns to the model and persists in the
    thread. The TUI renders that authoritative text literally rather than trying
    to parse the unrestricted answer content back into structured data.

    Answers are interpolated unescaped, so the encoding is not unambiguously
    decodable: an answer containing a blank line followed by a literal
    `Q: <text>\nA:` header is indistinguishable from a real block boundary. Only
    the model reads it that way today. Any future decoder must anchor on the known
    question text rather than on a generic `Q: ` pattern, or a crafted answer can
    fabricate an extra question/answer pair.

    Args:
        questions: Questions that were asked. Callers must pass questions whose
            `question` text is a non-empty string; `ask_user._validate_questions`
            enforces that before interrupting. The empty default below only
            keeps a caller that skips validation from raising `KeyError`.
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
