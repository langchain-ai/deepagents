"""Ask user middleware for interactive question-answering during agent execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from typing import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import InjectedToolCallId
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class Choice(TypedDict):
    """A single choice option for a multiple choice question."""

    value: str


class Question(TypedDict):
    """A question to ask the user.

    When `type` is `'multiple_choice'`, `choices` must be provided and
    non-empty. The tool validates this at runtime.
    """

    question: str

    type: Literal["text", "multiple_choice"]

    choices: NotRequired[list[Choice]]


class AskUserRequest(TypedDict):
    """Request payload sent via interrupt when asking the user questions."""

    type: Literal["ask_user"]

    questions: list[Question]

    tool_call_id: str


class AskUserResponse(TypedDict):
    """Response payload from the user answering questions."""

    answers: list[str]


ASK_USER_TOOL_DESCRIPTION = """Ask the user one or more questions when you need clarification or input before proceeding.

Each question can be either:
- "text": Free-form text response from the user
- "multiple_choice": User selects from predefined options (an "Other" option is always available)

For multiple choice questions, provide a list of choices. The user can pick one or type a custom answer via the "Other" option.

Use this tool when:
- You need clarification on ambiguous requirements
- You want the user to choose between multiple valid approaches
- You need specific information only the user can provide
- You want to confirm a plan before executing it

Do NOT use this tool for:
- Simple yes/no confirmations (just proceed with your best judgment)
- Questions you can answer yourself from context
- Trivial decisions that don't meaningfully affect the outcome"""

ASK_USER_SYSTEM_PROMPT = """## `ask_user`

You have access to the `ask_user` tool to ask the user questions when you need clarification or input.
Use this tool sparingly - only when you genuinely need information from the user that you cannot determine from context.

When using `ask_user`:
- Be concise and specific with your questions
- Use multiple choice when there are clear options to choose from
- Use text input when you need free-form responses
- Group related questions into a single ask_user call rather than making multiple calls
- Never ask questions you can answer yourself from the available context"""


def _validate_questions(questions: list[Question]) -> None:
    """Validate that multiple-choice questions include non-empty choices.

    Raises:
        ValueError: If a `multiple_choice` question has missing or empty `choices`.
    """
    for q in questions:
        if q.get("type") == "multiple_choice" and not q.get("choices"):
            msg = f"multiple_choice question {q.get('question')!r} requires a non-empty 'choices' list"
            raise ValueError(msg)


def _parse_answers(
    response: object,
    questions: list[Question],
    tool_call_id: str,
) -> Command[Any]:
    """Parse an interrupt response into a `Command` with a `ToolMessage`.

    Validates that the response is a dict with an `'answers'` key and logs a
    warning when the number of answers does not match the number of questions.

    Args:
        response: Raw value returned by `interrupt()`.
        questions: The questions that were asked.
        tool_call_id: Originating tool call ID for the `ToolMessage`.

    Returns:
        `Command` containing a formatted `ToolMessage` with Q&A pairs.
    """
    if not isinstance(response, dict) or "answers" not in response:
        logger.warning(
            "ask_user received malformed resume payload (expected dict with 'answers' key, got %s); treating all answers as empty",
            type(response).__name__,
        )
        answers: list[str] = []
    else:
        answers = cast("list[str]", cast("dict[str, Any]", response)["answers"])

    if len(answers) != len(questions):
        logger.warning(
            "ask_user answer count mismatch: expected %d, got %d",
            len(questions),
            len(answers),
        )

    formatted_answers = []
    for i, q in enumerate(questions):
        answer = answers[i] if i < len(answers) else "(no answer)"
        formatted_answers.append(f"Q: {q['question']}\nA: {answer}")
    result_text = "\n\n".join(formatted_answers)
    return Command(
        update={
            "messages": [ToolMessage(result_text, tool_call_id=tool_call_id)],
        }
    )


class AskUserMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that provides an ask_user tool for interactive questioning.

    This middleware adds an `ask_user` tool that allows agents to ask the user
    questions during execution. Questions can be free-form text or multiple choice.
    The tool uses LangGraph interrupts to pause execution and wait for user input.
    """

    def __init__(
        self,
        *,
        system_prompt: str = ASK_USER_SYSTEM_PROMPT,
        tool_description: str = ASK_USER_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize AskUserMiddleware.

        Args:
            system_prompt: System-level instructions injected into every LLM
                request to guide `ask_user` usage.
            tool_description: Description string passed to the `ask_user` tool
                decorator, visible to the LLM in the tool schema.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        def _ask_user(
            questions: list[Question],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command[Any]:
            """Ask the user one or more questions."""
            _validate_questions(questions)
            ask_request = AskUserRequest(
                type="ask_user",
                questions=questions,
                tool_call_id=tool_call_id,
            )
            response = interrupt(ask_request)
            return _parse_answers(response, questions, tool_call_id)

        _ask_user.name = "ask_user"
        self.tools = [_ask_user]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(content=cast("list[str | dict[str, str]]", new_system_content))
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt (async)."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(content=cast("list[str | dict[str, str]]", new_system_content))
        return await handler(request.override(system_message=new_system_message))
