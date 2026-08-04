"""Lightweight types and shared rendering for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.

This is the shared wire format for the tool, the TUI, and `auto_mode`, colocated
so no consumer has to import another. Each symbol's own docstring records which
side produces it and which reads it.
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

Unreachable from `_parse_answers` today: every branch there emits exactly
`len(questions)` answers, and the answered branch rejects a mismatch outright
rather than formatting it. Kept as a guard for any future caller that formats a
transcript without that check. A question the user deliberately left blank
renders as an empty `A:` line instead, never as this placeholder.
"""

ASK_USER_CANCELLED_ANSWER = "(cancelled)"
"""Placeholder recorded for every question when the user cancels the prompt."""

ASK_USER_ERROR_ANSWER_PREFIX = "(error: "
"""Prefix of the placeholder recorded for every question when the prompt fails.

The full placeholder is `(error: <detail>)`. Producer-side only: no production
code matches on it — `test_ask_user_types` is the only consumer, and it asserts
against this constant. The TUI derives its row summary from the recorded tool
status instead, because the placeholder is in-band — the prefix alone cannot
distinguish a failed prompt from a user who typed `(error: ...)` as their answer.
"""

AskUserRowSummary = Literal["User answered", "Question failed"]
"""The summaries an `ask_user` row may collapse to.

Narrows `ToolCallMessage.defer_success` so the transcript cannot be passed where
a summary belongs — the constraint that keeps user-typed answers out of
`tool.result` hook payloads.

Of the four `*_SUMMARY` constants below, only `ASK_USER_ANSWERED_SUMMARY` and
`ASK_USER_FAILED_SUMMARY` are members: the other two are hook bodies that no row
ever renders. `Literal[...]` cannot reference a plain string constant, so those
two values are restated here; both are annotated with this alias, so `ty` rejects
a reword that drifts. `test_row_summary_alias_matches_its_constants` pins it too.
"""

ASK_USER_ANSWERED_SUMMARY: AskUserRowSummary = "User answered"
"""One-line summary shown for an answered `ask_user` row before it is expanded.

Doubles as the `tool.result` hook payload's `tool_output` for an answered prompt
whose `ToolMessage` arrived, deliberately in place of the transcript, so
user-typed answers are not forwarded to hook scripts. When no `ToolMessage`
arrives the hook reports `ASK_USER_ANSWERED_NO_RESULT_SUMMARY` instead, while
the row still settles to this string. Rewording changes that hook contract as
well as the row; see `textual_adapter` and its `tool.result` tests.
"""

ASK_USER_ANSWERED_NO_RESULT_SUMMARY = "User answered (no tool result)"
"""The `tool.result` hook payload for an answered prompt that never completed.

Reported by `_dispatch_terminal_tool_result_hooks` when a teardown closes out a
row carrying a deferred success — the agent crashed, the stream ended, or the
user cancelled the turn. The guard there matches an already-settled row too, not
only one still awaiting its `ToolMessage`; see
`ToolCallMessage.deferred_success_output`.

The status stays `"success"` because the answers did reach the graph and only the
tool's own completion was lost, and `ask_user` results double as authorization
records; this distinct body is what lets an audit consumer tell "answers
delivered and the tool completed" from "answers delivered, then the turn died".
When the answers were never delivered at all the hook reports
`ASK_USER_ANSWERED_NOT_DELIVERED_SUMMARY` with `tool_status="error"` instead.
Hook contract: rewording it changes what those consumers see.
"""

ASK_USER_ANSWERED_NOT_DELIVERED_SUMMARY = "User answered (answers not delivered)"
"""The `tool.result` hook payload for answers the turn discarded before resuming.

Reported with `tool_status="error"` when a sibling question in the same batch was
cancelled: `textual_adapter` aborts the turn *before* `Command(resume=...)`, so
the resume payload — including this prompt's answers — is dropped, and the inline
widget is already unmounted, making the answers unrecoverable. The user answered,
but nothing downstream ever saw it.

Distinct from `ASK_USER_ANSWERED_NO_RESULT_SUMMARY`, and an error rather than a
success, precisely because `ask_user` results double as authorization records: a
`"success"` here would record an authorization that never took effect. Hook
contract: rewording it changes what audit consumers see.
"""

ASK_USER_CANCELLED_SUMMARY = "Question cancelled"
"""The `tool.result` hook payload's `tool_output` for the cancelled path.

Rewording it changes that hook contract. Not rendered on any row: a live cancel
calls `set_rejected` (which records no output), and a transcript of `(cancelled)`
placeholders from a non-TUI client is summarized from the recorded status like
any other, so it reads as `ASK_USER_ANSWERED_SUMMARY`. The cancel banner in
`textual_adapter` shares this wording but deliberately not this constant — it is
user-facing prose, not the hook contract.
"""

ASK_USER_FAILED_SUMMARY: AskUserRowSummary = "Question failed"
"""One-line summary shown for an `ask_user` prompt the middleware reported failed.

Also the `tool.result` hook payload's `tool_output` on that path — an answered
prompt whose `ToolMessage` came back with `status="error"` — deliberately in
place of the transcript, whose `(error: ...)` placeholders carry an arbitrary
detail string. Live widget failures are not this: they report their own error
text (see `textual_adapter`'s invalid-payload and cancel branches). Rewording
changes that hook contract as well as the row.
"""


def format_ask_user_error_answer(detail: str) -> str:
    """Render the placeholder answer recorded for every question on failure.

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
