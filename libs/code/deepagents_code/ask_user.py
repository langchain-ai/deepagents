"""Ask user middleware for interactive question-answering during agent execution."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, cast, override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    ToolCallRequest,
    TracePolicy,
    omit_payload,
)
from langchain.tools import InjectedToolCallId, ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from pydantic import AfterValidator, Field, ValidationError

from deepagents_code._ask_user_types import (
    ASK_USER_AUTHORIZATION_METADATA_KEY,
    ASK_USER_CANCELLED_ANSWER,
    MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS,
    AskUserAuthorizationReceipt,
    AskUserRequest,
    Question,
    ValidatedQuestion,
    _validate_questions,
    format_ask_user_error_answer,
    format_ask_user_transcript,
)

logger = logging.getLogger(__name__)


ASK_USER_TOOL_DESCRIPTION = """Ask the user one or more questions when you need clarification or input before proceeding.

Each question can be one of:
- "text": Free-form text response from the user
- "multiple_choice": User selects exactly one of the predefined options (an "Other" option is always available)
- "multi_select": User selects one or more of the predefined options (an "Other" free-form option is always available; filling one reveals an "Add another" slot for more custom values)

For "multiple_choice" and "multi_select" questions, provide a list of choices, each with a non-empty "value". For "multiple_choice" the user picks one option or types a custom answer via the "Other" option; for "multi_select" the user toggles one or more of the provided options and may also add one or more custom free-form Other values among the selected values.

A "multi_select" answer is returned as a JSON array of the selected values, e.g. ["a", "b"] (an optional question the user leaves untouched returns []). "multi_select" choice values and custom Other text may themselves contain commas, quotes, and newlines. A "multiple_choice" value is returned on its own with no escaping, so keep that one to a single line.

By default all questions are required. Set "required" to false for optional questions that the user can skip. Do not include "(required)", "(optional)", "- optional", or similar annotations in the question text — the UI renders that separately based on the "required" field.

Use this tool when:
- You need clarification on ambiguous requirements
- You want the user to choose between multiple valid approaches
- You need specific information only the user can provide
- You want to confirm a plan before executing it

Do NOT use this tool for:
- Simple yes/no confirmations (just proceed with your best judgment)
- Questions you can answer yourself from context
- Trivial decisions that don't meaningfully affect the outcome"""  # noqa: E501

ASK_USER_SYSTEM_PROMPT = """## `ask_user`

You have access to the `ask_user` tool to ask the user questions when you need clarification or input.
Use this tool sparingly - only when you genuinely need information from the user that you cannot determine from context.

When using `ask_user`:
- Be concise and specific with your questions
- Use multiple choice when there are clear options and exactly one applies
- Use multi-select when the user may legitimately pick several of the options
- Use text input when you need free-form responses
- Group related questions into a single ask_user call rather than making multiple calls
- Never ask questions you can answer yourself from the available context"""  # noqa: E501


def _context_string(context: object, name: str) -> str | None:
    value = (
        context.get(name)
        if isinstance(context, Mapping)
        else getattr(context, name, None)
    )
    return value if isinstance(value, str) and value else None


def _execution_thread_id(runtime: object) -> str | None:
    execution_info = getattr(runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    return thread_id if isinstance(thread_id, str) and thread_id else None


def _active_turn_id(runtime: object) -> str | None:
    from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY

    state = getattr(runtime, "state", None)
    messages = state.get("messages") if isinstance(state, Mapping) else None
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        metadata = message.additional_kwargs.get(USER_PROMPT_METADATA_KEY)
        if not isinstance(metadata, Mapping):
            return None
        turn_id = metadata.get("turn_id")
        return turn_id if isinstance(turn_id, str) and turn_id else None
    return None


def _parse_answers(
    response: object,
    questions: list[Question],
    tool_call_id: str,
    *,
    thread_id: str | None = None,
    turn_id: str | None = None,
) -> Command[Any]:
    """Parse an interrupt response into a `Command` with a `ToolMessage`.

    Supports explicit status signaling from the adapter:

    - `answered` (default): consume provided `answers`. An answer count that does
      not match `questions` is rejected as `error` rather than padded or
      truncated, since either would misattribute answers to questions.
    - `cancelled`: synthesize `(cancelled)` answers
    - `error`: synthesize `(error: ...)` answers

    Malformed payloads are converted into explicit error answers instead of
    silently defaulting to `(no answer)`.

    Args:
        response: Raw value returned by `interrupt()`.
        questions: The questions that were asked.
        tool_call_id: Originating tool call ID for the `ToolMessage`.
        thread_id: Trusted runtime thread identity.
        turn_id: Trusted runtime user-turn identity.

    Returns:
        `Command` containing a formatted `ToolMessage` with Q&A pairs, carrying an
            explicit `status` — `"error"` for a failed prompt, `"success"` for an
            answered or cancelled one. Consumers depend on that field; see the
            comment at the `ToolMessage` construction below.
    """
    # Untrusted: holds whatever `status` the resume payload carried until the
    # branches below normalize it to one of answered/cancelled/error.
    status: str = "answered"
    # Detail for a defect found here while trusting the payload, kept apart from
    # `client_error_text` so the two cannot clobber each other in either order.
    local_error_text: str | None = None
    # Detail supplied by a caller that declared the failure itself.
    client_error_text: str | None = None
    answers_are_strings = False
    answers: list[str]
    if not isinstance(response, dict):
        logger.error(
            "ask_user received malformed resume payload "
            "(expected dict, got %s); returning explicit error answers",
            type(response).__name__,
        )
        answers = []
        status = "error"
        local_error_text = "invalid ask_user response payload"
    else:
        response_dict = cast("dict[str, Any]", response)
        response_status = response_dict.get("status")
        if isinstance(response_status, str):
            status = response_status

        if status == "error":
            # Read before local validation can flip `status` to "error" itself:
            # a payload claiming "answered" may carry a stale `error` field, and
            # that must not end up describing a failure detected here.
            response_error = response_dict.get("error")
            if isinstance(response_error, str) and response_error:
                client_error_text = response_error

        if "answers" not in response_dict:
            if status == "answered":
                logger.error(
                    "ask_user received resume payload without 'answers'; "
                    "returning explicit error answers"
                )
                answers = []
                status = "error"
                local_error_text = "missing ask_user answers payload"
            else:
                answers = []
        else:
            raw_answers = response_dict["answers"]
            if isinstance(raw_answers, list):
                answers_are_strings = all(
                    isinstance(answer, str) for answer in raw_answers
                )
                if not answers_are_strings:
                    # Coerced rather than rejected so the model still sees
                    # something for each question, but logged: the `str()` of a
                    # non-string element is presented to the model as the user's
                    # own words, and it silently withholds the authorization
                    # receipt below (which requires `answers_are_strings`).
                    logger.warning(
                        "ask_user received non-string answer element(s) (%s); "
                        "coercing with str() and withholding the authorization "
                        "receipt",
                        ", ".join(
                            sorted(
                                {
                                    type(answer).__name__
                                    for answer in raw_answers
                                    if not isinstance(answer, str)
                                }
                            )
                        ),
                    )
                answers = [str(answer) for answer in raw_answers]
            else:
                logger.error(
                    "ask_user received non-list 'answers' payload (%s); "
                    "returning explicit error answers",
                    type(raw_answers).__name__,
                )
                answers = []
                status = "error"
                local_error_text = "invalid ask_user answers payload"

        match status:
            case "cancelled":
                answers = [ASK_USER_CANCELLED_ANSWER for _ in questions]
            case "answered":
                if len(answers) != len(questions):
                    # Treated as a failed prompt, not a partial one. A short list
                    # silently re-attributes every answer after the gap to the
                    # wrong question, and a long one drops the extras — either way
                    # the payload is untrustworthy, and a `"success"` transcript
                    # would hand the model a confident wrong Q->A pairing.
                    logger.error(
                        "ask_user answer count mismatch: expected %d, got %d; "
                        "returning explicit error answers",
                        len(questions),
                        len(answers),
                    )
                    status = "error"
                    local_error_text = (
                        f"ask_user answer count mismatch (expected "
                        f"{len(questions)}, got {len(answers)})"
                    )
            case "error":
                # Already normalized above; the detail is resolved below.
                pass
            case _:
                logger.error(
                    "ask_user received unknown status %r; returning explicit "
                    "error answers",
                    status,
                )
                answers = []
                status = "error"
                local_error_text = "invalid ask_user response status"

    if status == "error":
        # A caller that declared the failure knows the root cause; a detail
        # derived here describes a payload defect found while trusting it.
        detail = client_error_text or local_error_text or "ask_user interaction failed"
        answers = [format_ask_user_error_answer(detail) for _ in questions]

    additional_kwargs: dict[str, object] = {}
    if (
        status == "answered"
        and answers_are_strings
        and len(answers) == len(questions)
        and all(
            len(answer) <= MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS for answer in answers
        )
        and thread_id is not None
        and turn_id is not None
    ):
        receipt = AskUserAuthorizationReceipt(
            version=1,
            thread_id=thread_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            answers=list(answers),
        )
        additional_kwargs[ASK_USER_AUTHORIZATION_METADATA_KEY] = receipt

    result_text = format_ask_user_transcript(questions, answers)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    result_text,
                    name="ask_user",
                    tool_call_id=tool_call_id,
                    additional_kwargs=additional_kwargs,
                    # Consumers, so a failed prompt must not be left at the
                    # `"success"` default:
                    #   - `normalize_tool_status`, on both the live TUI stream and
                    #     the headless surface (`client.non_interactive`);
                    #   - the `case "error"` arm of `_restore_deferred_state`, on
                    #     reload;
                    #   - `auto_mode`, which refuses to mint a trusted
                    #     authorization receipt unless this reads `"success"` —
                    #     the consumer with real consequences.
                    # A cancel stays `"success"` — it is a user choice, not a tool
                    # failure — and is safe for that last consumer because the
                    # receipt above requires `status == "answered"`, so a cancelled
                    # prompt carries none to trust.
                    status="error" if status == "error" else "success",
                )
            ],
        }
    )


def _log_rejected_ask_user_call(
    request: ToolCallRequest, result: ToolMessage | Command[Any]
) -> None:
    """Log an `ask_user` call the schema rejected.

    Args:
        request: The tool call request that produced `result`.
        result: The handler's result.
    """
    if (
        request.tool_call["name"] == "ask_user"
        and isinstance(result, ToolMessage)
        and result.status == "error"
    ):
        logger.warning("ask_user rejected the model's arguments: %s", result.content)


class AskUserMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that provides an ask_user tool for interactive questioning.

    This middleware adds an `ask_user` tool that allows agents to ask the user
    questions during execution. Questions can be free-form text, multiple choice
    (pick exactly one), or multi-select (pick one or more).
    The tool uses LangGraph interrupts to pause execution and wait for user input.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

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
            questions: Annotated[
                list[ValidatedQuestion],
                AfterValidator(_validate_questions),
                Field(description="Questions to present to the user."),
            ],
            tool_call_id: Annotated[str, InjectedToolCallId],
            runtime: ToolRuntime[Any, Any],
        ) -> Command[Any]:
            """Ask the user one or more questions.

            Returns:
                `Command` containing the parsed user answers as a `ToolMessage`.

            Raises:
                RuntimeError: If the tool body raises a `ValidationError` after
                    the arguments have been validated. Re-raised as a type
                    `ToolNode` will not convert, so the fault stays fatal
                    instead of being reported to the model as bad input.
            """
            # The arguments below are already validated: the schema rejects an
            # empty list, blank question text, an unknown `type`, a non-boolean
            # `required`, blank choice values, and the cross-field `choices`
            # rules on `ValidatedQuestion`. `ToolNode` converts that rejection
            # into an error `ToolMessage` the model can correct and retry from,
            # so no handling is wired here.
            #
            # Two separate mechanisms keep the injected arguments out of that
            # message, and neither covers the other:
            #   - `runtime` is dropped by `_filter_validation_errors`, which
            #     builds its name set from state/store/runtime only.
            #   - `tool_call_id` is an `InjectedToolCallId`, which that filter
            #     does *not* know about. It stays out because
            #     `ToolInvocationError` is built from the pre-injection
            #     `call["args"]`.
            # `AskUserMiddleware.wrap_tool_call` logs the rejection, since
            # `ToolNode` logs nothing itself.
            ask_request = AskUserRequest(
                type="ask_user",
                questions=questions,
                tool_call_id=tool_call_id,
            )
            # interrupt() raises GraphInterrupt from INSIDE tool execution,
            # within ToolNode's wrap_tool_call chain. Any
            # wrap_tool_call middleware that catches exceptions MUST re-raise
            # GraphBubbleUp — a broad `except Exception` (e.g. ToolRetryMiddleware)
            # would swallow this interrupt and silently break ask_user.
            # `ToolNode` wraps the tool body in the same `try` as argument
            # parsing, so any `ValidationError` escaping from here would be
            # reported to the model as *its* bad input — naming fields that are
            # not even on the tool schema, against arguments the model wrote
            # correctly, and discarding the user's answer. Re-raise as a
            # non-`ValidationError` so it stays fatal, which is what
            # `_default_handle_tool_errors` does with every other type.
            #
            # Nothing in the body raises one today. This guards the next edit,
            # not a live fault. `GraphInterrupt` from `interrupt()` is not a
            # `ValidationError` and passes through untouched.
            try:
                response = interrupt(ask_request)
                execution_thread_id = _execution_thread_id(runtime)
                context_thread_id = _context_string(runtime.context, "thread_id")
                context_turn_id = _context_string(runtime.context, "turn_id")
                active_turn_id = _active_turn_id(runtime)
                runtime_tool_call_id = runtime.tool_call_id
                return _parse_answers(
                    response,
                    questions,
                    tool_call_id,
                    thread_id=(
                        execution_thread_id
                        if execution_thread_id == context_thread_id
                        and runtime_tool_call_id == tool_call_id
                        else None
                    ),
                    turn_id=(
                        context_turn_id if context_turn_id == active_turn_id else None
                    ),
                )
            except ValidationError as exc:
                msg = (
                    "ask_user failed internally after its arguments were "
                    "validated; this is not a model-authored error"
                )
                raise RuntimeError(msg) from exc

        _ask_user.name = "ask_user"
        self.tools = [_ask_user]

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Log a rejected `ask_user` call, then pass the result through.

        `ToolNode` converts an argument `ValidationError` into an error
        `ToolMessage` before it reaches here, and it logs nothing itself, so
        without this a model sending malformed arguments — or looping on them —
        leaves no operator-visible record at all. The user sees only a red
        `ask_user` row in the transcript.

        The result type is the discriminant: `_ask_user` always returns a
        `Command`, so a `ToolMessage` here means the call never entered the tool
        body. That keeps this off the `_parse_answers` error path, which reports
        a malformed *resume payload* inside a `Command` and logs itself.

        Nothing is caught. An exception from the body must stay fatal, and
        `GraphBubbleUp` from `interrupt()` must keep bubbling.

        Args:
            request: The tool call request.
            handler: Callable that executes the tool.

        Returns:
            The handler's result, unchanged.
        """
        result = handler(request)
        _log_rejected_ask_user_call(request, result)
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async twin of `wrap_tool_call`.

        Defined so the async path keeps executing tools asynchronously. With
        only the sync wrapper present, `ToolNode` falls back to running the tool
        through `_execute_tool_sync`.

        Args:
            request: The tool call request.
            handler: Awaitable callable that executes the tool.

        Returns:
            The handler's result, unchanged.
        """
        result = await handler(request)
        _log_rejected_ask_user_call(request, result)
        return result

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt.

        Returns:
            Model response from the wrapped handler.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject the ask_user system prompt (async).

        Returns:
            Model response from the wrapped handler.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
