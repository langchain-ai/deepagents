"""Lightweight types for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.

`pydantic.Field` is deliberately imported under `TYPE_CHECKING` only.
With `from __future__ import annotations` the `Annotated[..., Field(...)]`
expressions are stored as strings and never evaluated at class body time.
Call sites that need to resolve these annotations at runtime (`TypeAdapter`,
LangChain `@tool`) must call `ensure_field_available` first to inject
`Field` into this module's namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, NotRequired

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pydantic import Field


def ensure_field_available() -> None:
    """Inject `pydantic.Field` into this module's globals if not already present.

    This is a no-op when `Field` has already been injected.  By the time any
    caller needs to resolve the `Annotated[..., Field(...)]` string annotations,
    pydantic is already in `sys.modules` (loaded by LangChain or a `TypeAdapter`
    import), so the import here is a dict lookup.
    """
    if "Field" not in globals():
        from pydantic import Field

        globals()["Field"] = Field


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
