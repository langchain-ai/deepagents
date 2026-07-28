"""Lightweight types for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, assert_never, get_args

from pydantic import Field
from typing_extensions import TypedDict

QuestionType = Literal["text", "multiple_choice", "multi_select"]
"""Supported `ask_user` question types."""

QUESTION_TYPES: frozenset[str] = frozenset(get_args(QuestionType))
"""Runtime membership view of `QuestionType`.

Derived via `get_args` so the membership checks in `ask_user` and `auto_mode`
cannot drift from the alias when a new question type is added. That drift would
be silent: an unrecognized type makes `_ask_user_question_count` return `None`,
which drops the user's answers as same-turn authorization without any error.
"""


def _requires_choices(question_type: QuestionType) -> bool:
    """Return whether `question_type` needs a non-empty `choices` list.

    Args:
        question_type: A member of `QuestionType`.

    Returns:
        True if questions of this type must define `choices`.
    """
    if question_type == "text":
        return False
    if question_type in {"multiple_choice", "multi_select"}:
        return True
    # Exhaustiveness guard: adding a `QuestionType` member without deciding here
    # whether it needs choices fails type checking rather than silently landing
    # on the wrong side of `CHOICE_QUESTION_TYPES`.
    assert_never(question_type)


CHOICE_QUESTION_TYPES: frozenset[str] = frozenset(
    question_type
    for question_type in get_args(QuestionType)
    if _requires_choices(question_type)
)
"""Question types that require a non-empty `choices` list.

Derived from `_requires_choices` rather than written out, so it cannot omit a
new choice-based `QuestionType` member. Note that requiring `choices` is not the
same as being *rendered* as a choice list: the TUI has its own exhaustive
dispatch in `_QuestionWidget.compose`.
"""

MULTI_SELECT_ANSWER_SEPARATOR = ", "
"""Separator joining the selected values of a `multi_select` answer."""

MULTI_SELECT_FORBIDDEN_IN_VALUE = ","
"""Substring a `multi_select` choice value must not contain.

Kept separate from `MULTI_SELECT_ANSWER_SEPARATOR` so validation states the
forbidden text outright instead of deriving it by stripping the separator. That
derivation only happened to work for punctuation: a separator of `" and "` would
have rejected every value containing "and".
"""


class Choice(TypedDict):
    """A single choice option for a multiple choice or multi-select question."""

    value: Annotated[
        str,
        Field(
            description=(
                "The display label for this choice. Also the text returned as "
                "the answer when this choice is selected."
            )
        ),
    ]


class Question(TypedDict):
    """A question to ask the user."""

    question: Annotated[str, Field(description="The question text to display.")]

    type: Annotated[
        QuestionType,
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' for "
                "picking exactly one predefined option, 'multi_select' for picking "
                "one or more predefined options. A 'multi_select' answer comes back "
                "as the selected values joined with ', '; if nothing is selected on "
                "an optional question the answer is an empty string."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Choice],
            Field(
                description=(
                    "Options for 'multiple_choice' and 'multi_select' questions. "
                    "Every choice needs a non-empty 'value'. For 'multiple_choice', "
                    "an 'Other' free-form option is always appended automatically; "
                    "'multi_select' has no 'Other' option, and its values must not "
                    "contain ',' because the answer joins them with ', '."
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
