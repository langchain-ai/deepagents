"""Middleware for providing subagents to an agent via a `task` tool."""

import asyncio
import contextlib
import dataclasses
import json
import logging
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.agents.structured_output import ResponseFormat
from langchain.tools import BaseTool, ToolRuntime
from langchain_core._api.beta_decorator import warn_beta
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langsmith.run_helpers import get_tracing_context, tracing_context
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import TypeIs

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._skill_tools import _SKILL_TOOLS_DISCLOSED_KEY
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SUMMARIZATION_SESSION_ID_KEY,
    SummarizationEvent,
    _DeepAgentsSummarizationMiddleware,
)
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware

if TYPE_CHECKING:
    from pydantic_core import InitErrorDetails

logger = logging.getLogger(__name__)

SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"
"""Configurable key used by task-tool callers to request dynamic response format."""

_FORK_EXCLUDED_STATE_KEYS = frozenset(
    {"structured_response", "pinned_skills", SUMMARIZATION_EVENT_KEY, SUMMARIZATION_SESSION_ID_KEY, _SKILL_TOOLS_DISCLOSED_KEY}
)
"""State a fork must not resume.

The summarization event is folded into the fork's messages instead. Dropping the
session ID lets the subagent generate its own, and dropping a prior structured response
ensures it cannot be mistaken for the fork's result. The disclosed skill tools
describe the parent's last model call; the fork records its own. Skills the parent
has yet to pin are the parent's own.
"""

_FORKED_CONTEXT_KEY = "_deepagents_forked_context"
"""Set on a forked subagent's own initial state; never on the parent's.

Lets `task`/`atask` refuse recursive delegation at call time instead of
omitting the tool -- see `_ForkTaskToolMiddleware` for why.
"""

_FORK_RECURSION_REFUSAL = (
    "You are a subagent and cannot delegate to another subagent. Complete this task yourself instead of calling this tool again."
)


class SubAgent(TypedDict):
    """Specification for a declarative subagent.

    By default the subagent is isolated: it receives only the delegated task
    description. Setting `mode="fork"` makes it continue the parent's
    conversation instead.

    !!! warning "Experimental"

        `mode="fork"` is experimental and may change in a future release.

    When using `create_deep_agent`, subagents automatically receive
    a default middleware stack before any custom `middleware` specified in
    this spec.

    `name` and `description` are required; all other fields are optional.
    See the individual attributes for configuration details.
    """

    name: str
    """Unique identifier used by the main agent when calling the `task()` tool."""

    description: str
    """What this subagent does.

    Be specific and action-oriented: the main agent uses this to decide when to delegate.
    """

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use.

    If not specified, inherits tools from the main agent via `default_tools`.
    """

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model.

    Use `'provider:model-name'` format.
    """

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior, logging, or rate limiting.

    To restrict filesystem tools, include a `FilesystemMiddleware(tools=...)`
    instance here to replace the subagent's default filesystem middleware.
    """

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools. Requires a checkpointer."""

    skills: NotRequired[list[str]]
    """Skill source paths for `SkillsMiddleware`. Forbidden under `mode="fork"`.

    For example, `["/skills/user/", "/skills/project/"]`.
    """

    permissions: NotRequired[list[FilesystemPermission]]
    """List of `FilesystemPermission` rules for this subagent.

    If omitted, inherits the parent agent's permissions. If specified, replaces
    the parent's permissions entirely for this subagent.

    Rules are evaluated in declaration order; the first match wins.
    `FilesystemMiddleware` enforces these rules for the built-in filesystem
    tools on the subagent stack.
    """

    response_format: NotRequired[ResponseFormat[Any] | type | dict[str, Any]]
    """Structured output response format for the subagent.

    When specified, the subagent will produce a `structured_response` conforming
    to the given schema. The structured response is JSON-serialized and returned
    as the `ToolMessage` content to the parent agent, replacing the default
    last-message extraction.

    Accepted formats (from `langchain.agents.structured_output`):

    - `ToolStrategy(schema)`: Use tool calling to extract structured output from the model.
    - `ProviderStrategy(schema)`: Use the model provider's native structured output mode.
    - `AutoStrategy(schema)`: Automatically select the best strategy.
    - A bare Python `type`: A Pydantic `BaseModel` subclass, `dataclass`,
        or `TypedDict` class.

        Equivalent to `AutoStrategy(schema)`.
    - `dict[str, Any]`: A JSON schema dictionary
        (e.g., `{"type": "object", "properties": {...}, "required": [...]}`).

    Example:
        ```python
        from pydantic import BaseModel

        class Findings(BaseModel):
            findings: str
            confidence: float

        analyzer: SubAgent = {
            "name": "analyzer",
            "description": "Analyzes data and returns structured findings",
            "system_prompt": "Analyze the data and return your findings.",
            "model": "openai:gpt-5.5",
            "tools": [],
            "response_format": Findings,
        }
        ```
    """

    system_prompt: NotRequired[str]
    """Instructions for the subagent. Uses an empty prompt when omitted.

    Under `mode="fork"` this is appended to the inherited prompt rather than
    replacing it.
    """

    mode: NotRequired[Literal["isolated", "fork"]]
    """Context mode. Defaults to `isolated`, where the subagent only sees the delegated task.

    Under `fork`, the subagent receives the parent's effective conversation
    history and state, and mirrors the parent's prompt-producing middleware so
    it rebuilds the same system prompt. It cannot define `skills`, which would
    diverge from the parent's. `tools` isn't restricted the same way -- a fork's
    own tools work normally; the tradeoff is cache misses.
    """


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec.

    !!! note

        The `runnable`'s state schema must include a 'messages' key.

        This is required for the subagent to communicate results back to
        the main agent.

    !!! note

        `CompiledSubAgent` runnables are used as provided. They do not
        inherit `create_deep_agent(state_schema=...)`; if the runnable
        needs custom state fields, compile it with a compatible state
        schema yourself.

    When the subagent completes, the parent reads the returned state:
    if `structured_response` is non-`None`, it is JSON-serialized and used as
    the `ToolMessage` content; otherwise, the last non-empty `AIMessage`
    text is used.

    Examples:
        Using `create_agent` with `response_format`:

        ```python
        from pydantic import BaseModel
        from langchain.agents import create_agent


        class Findings(BaseModel):
            summary: str
            confidence: float


        researcher: CompiledSubAgent = {
            "name": "researcher",
            "description": "Researches a topic and returns findings.",
            "runnable": create_agent(
                "openai:gpt-5.5",
                tools=[],  # your tools here
                response_format=Findings,
            ),
        }
        ```

        Custom `langgraph` graph (write `structured_response` directly):

        ```python
        def node(state):
            return {
                "messages": [...],
                "structured_response": Findings(summary="...", confidence=0.9),
            }
        ```
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    runnable: Runnable
    """A custom agent implementation.

    Create a custom agent using either:

    1. LangChain's [`create_agent()`](https://docs.langchain.com/oss/python/langchain/quickstart)
    2. A custom graph using [`langgraph`](https://docs.langchain.com/oss/python/langgraph/quickstart)

    If you're creating a custom graph, make sure the state schema includes
    a 'messages' key. This is required for the subagent to communicate
    results back to the main agent.
    """

    mode: NotRequired[Literal["isolated", "fork"]]
    """Use `fork` to inherit the parent's conversation without changing the runnable prompt.

    A declarative [`SubAgent`][deepagents.middleware.subagents.SubAgent] fork
    inherits the full state, including private keys.
    """


_SubAgentSpec = SubAgent | CompiledSubAgent


class BackgroundTask(TypedDict):
    """A subagent the `task` tool started with `run_in_background`.

    !!! warning "Experimental"

        Background subagents are in beta; this shape may change.
    """

    task_id: str
    """ID the `task` tool returned to the model when the subagent started."""

    subagent_type: str
    """Name of the subagent that ran."""

    description: str
    """The task the model gave the subagent."""

    status: Literal["running", "success", "error"]
    """`success` or `error` by the time the report reaches the callback."""

    result: str
    """The subagent's final message, or the error it raised."""

    config: RunnableConfig
    """The parent run's config without its per-run internals.

    Carries the parent's `thread_id` and metadata, so the callback can notify
    the parent, for example by queuing a run on the same thread.
    """


BackgroundTaskCallback = Callable[[BackgroundTask], Awaitable[None]]
"""Receives each background subagent's report when it finishes."""


def _validate_subagent_mode(spec: _SubAgentSpec) -> None:
    """Reject unsupported context modes before a subagent can run."""
    mode = spec.get("mode")
    if mode not in (None, "isolated", "fork", "handoff"):  # "handoff" is legacy alias for "isolated"
        msg = f"SubAgent '{spec['name']}' has invalid mode '{mode}'; expected 'isolated' or 'fork'"
        raise ValueError(msg)
    if mode == "fork" and spec.get("skills"):
        msg = f"SubAgent '{spec['name']}' cannot set skills under mode='fork'; the parent's skills are inherited instead."
        raise ValueError(msg)


def _validate_unique_subagent_names(subagents: Sequence[_SubAgentSpec]) -> None:
    """Reject subagent specs that share a name.

    The model selects a subagent purely by name, so a collision has no
    coherent meaning — it would otherwise silently resolve to whichever
    spec happens to be last.
    """
    seen: set[str] = set()
    for spec in subagents:
        name = spec["name"]
        if name in seen:
            msg = f"Duplicate subagent name '{name}'; each subagent must have a unique name."
            raise ValueError(msg)
        seen.add(name)


def _is_forked_subagent(spec: _SubAgentSpec) -> TypeIs[SubAgent]:
    """Return whether a declarative subagent inherits its parent's state."""
    return "runnable" not in spec and spec.get("mode") == "fork"


def _is_compiled_subagent(spec: _SubAgentSpec) -> TypeIs[CompiledSubAgent]:
    """Return whether a spec provides an already-compiled runnable."""
    return "runnable" in spec


def _is_forked_compiled_subagent(spec: _SubAgentSpec) -> TypeIs[CompiledSubAgent]:
    """Return whether a compiled subagent inherits its parent's state."""
    return "runnable" in spec and spec.get("mode") == "fork"


# A forked subagent replays the parent's exact history, including the human
# message that told the parent to delegate -- it can mistake that as a fresh
# request aimed at itself. This marks it as already-happened, and separately
# nudges toward grounding in facts already in that history instead of
# answering generically when the task description itself is sparse.
_FORK_TASK_PREAMBLE = (
    "[The messages above are a prior conversation you are continuing as the "
    "subagent that was just invoked. Any mention in them of delegating to a "
    "subagent already happened — you are that subagent, not the one being "
    "asked to delegate further. If you try to delegate to another subagent "
    "yourself, it will be refused — complete this task directly. Use the "
    "specific facts, figures, and identifiers already established in that "
    "conversation when completing the task below — do not answer "
    "generically when exact details are already available above. Your "
    "actual task is below.]\n\n"
)


def _fork_messages(
    messages: Sequence[AnyMessage],
    event: SummarizationEvent | None,
    description: str,
) -> list[AnyMessage]:
    """Build a fork's history: the parent's effective conversation, then the task.

    Applies summarization event so that a compiled subagent sees the
    compacted conversation instead of replaying what the parent already evicted.
    """
    history = list(messages)
    if history and isinstance(history[-1], AIMessage) and history[-1].tool_calls:
        history.pop()
    effective = _DeepAgentsSummarizationMiddleware._apply_event_to_messages(history, event)
    return [*effective, HumanMessage(content=_FORK_TASK_PREAMBLE + description)]


DEFAULT_SUBAGENT_PROMPT = """In order to complete the objective that the user asks of you, you have access to a number of standard tools.

The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final
response contains the complete answer."""

_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
    "skills_metadata",
    "pinned_skills",
    _SKILL_TOOLS_DISCLOSED_KEY,
    _FORKED_CONTEXT_KEY,
}
"""State keys that are excluded when passing state to subagents and when
returning updates from subagents.

When returning updates:

1. The messages key is handled explicitly to ensure only the final message
    is included
2. The todos and `structured_response` keys are excluded as they do not have
    a defined reducer and no clear meaning for returning them from a subagent
    to the main agent.
3. `skills_metadata` is excluded so a subagent loads skills from its own
    sources rather than reusing the parent's list, and cannot replace the
    parent's list with its own. `_skill_tools_disclosed` is excluded because it
    describes one agent's latest model call, and `pinned_skills` because a
    pending pin belongs to the agent it was requested for.
4. Agent-private fields on middleware state schemas are excluded from both
    subagent output and subagent inputs.
"""


_TASK_TOOL_INJECTED_ARGS = frozenset({"runtime"})
"""Arguments the tool node injects into the `task` call alongside the model's."""


class TaskToolSchema(BaseModel):
    """Input schema for the `task` tool."""

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )

    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: object) -> object:
        """Refuse keys the schema does not define.

        Models sometimes put the real instructions under an invented key (e.g.
        `prompt`) and leave a short label in `description`. Ignoring the extra key
        dispatched the subagent with the label alone and reported success; a
        validation error goes back to the model as a tool error naming the key.

        `extra="forbid"` cannot be used: the tool node validates the arguments
        after injecting the `runtime` parameter, which is not a schema field.
        """
        if isinstance(data, dict):
            arguments = {str(key): value for key, value in data.items()}
            unknown = sorted(arguments.keys() - set(cls.model_fields) - _TASK_TOOL_INJECTED_ARGS)
            if unknown:
                # LangGraph filters out errors without a field location.
                errors: list[InitErrorDetails] = []
                for key in unknown:
                    msg = f"Unexpected argument {key!r}; put all instructions for the subagent in `description`."
                    errors.append({"type": "value_error", "loc": (key,), "input": arguments[key], "ctx": {"error": ValueError(msg)}})
                raise ValidationError.from_exception_data(cls.__name__, errors)
        return data


class BackgroundTaskToolSchema(TaskToolSchema):
    """Input schema for the `task` tool when background subagents are enabled."""

    run_in_background: bool = Field(
        default=False,
        description="Return at once with a task ID instead of waiting for the subagent's report.",
    )


TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle a complex, multi-step task.

Available agent types and the tools they have access to:
{available_agents}

Specify subagent_type to select the agent. Usage notes:
- Launch multiple agents concurrently when their tasks are independent, using a single message with multiple tool calls.
- Each invocation is stateless by default: the agent sees only the prompt you give it and returns a single final report. Put full detail in the prompt and state exactly what it should return — unless an agent type below says it inherits your conversation instead.
- The agent's report is not shown to the user; relay a summary yourself.
- Tell the agent whether to create content, analyze, or only research, since it can't necessarily see the user's intent unless it inherits your conversation, as noted per agent type below.
- If an agent's description says to use it proactively, do so without waiting to be asked.
- When only general-purpose is available, use it for any complex, context-heavy task; it has the same capabilities as the main agent."""  # noqa: E501

_BACKGROUND_TASK_TOOL_NOTE = """
- Set `run_in_background` for long work you don't need to wait on: the call returns a task ID at once, and the agent's report arrives in a later message when it finishes. Don't poll for it."""  # noqa: E501

_BACKGROUND_SYNC_REFUSAL = "Background subagents need the async entrypoint (`ainvoke` or `astream`); call `task` without `run_in_background`."

_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
"""Running background subagents, held so the event loop does not drop them."""

_PER_RUN_CONFIG_PREFIXES = ("__pregel_", "checkpoint_")

_FORKED_SUBAGENT_TOOL_NOTE = " (inherits your full conversation and system prompt — no need to restate context here)"
"""Appended to a forked subagent's line in the task tool's listing.

Load-bearing: without it, the "Each invocation is stateless" line above is
the model's only signal, and it can refuse to delegate to a forked subagent
even when told the subagent inherits the conversation.
"""


def _describe_subagent_for_tool(name: str, description: str, *, forked: bool) -> str:
    """Render one subagent's listing line for the task tool description."""
    suffix = _FORKED_SUBAGENT_TOOL_NOTE if forked else ""
    return f"- {name}: {description}{suffix}"


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
"""Base spec for general-purpose subagent (caller adds model, tools, middleware)."""


class _ForkedContextState(TypedDict):
    """Private state marking a graph as running as a forked subagent.

    The key needs a declared schema: an undeclared key on a subagent's initial
    state is not tracked as a real channel, so `task()` would never see it via
    `runtime.state.get(...)`.
    """

    _deepagents_forked_context: NotRequired[Annotated[bool, OmitFromSchema(input=False, output=True)]]


class _ForkTaskToolMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Gives a forked subagent the real `task` tool, guarded against recursion."""

    state_schema = _ForkedContextState

    def __init__(self, task_tool: BaseTool) -> None:
        self.tools = [task_tool]


@contextlib.contextmanager
def _subagent_tracing_context() -> Generator[None, None, None]:
    """Context manager that tags subagent runs with `ls_agent_type="subagent"`.

    Sets `ls_agent_type` on the langsmith tracing context `metadata`, which is
    propagated to LangSmith runs. This mirrors
    langchain's `ls_agent_type="root"` tagging behavior.

    Forwards all other current tracing-context fields (parent, client, tags,
    etc.) unchanged so this wrapper does not clobber the enclosing context.
    """
    current = get_tracing_context()

    merged_metadata = {**(current.get("metadata") or {}), "ls_agent_type": "subagent"}
    # Pass every field from the current tracing context through to
    # `tracing_context` so we don't accidentally clobber fields that may be
    # added to langsmith in the future. The only change is `metadata`.

    kwargs: dict[str, Any] = {**current, "metadata": merged_metadata}

    with tracing_context(**kwargs):
        yield


def _subagent_report(result: dict[str, Any]) -> str:
    """Return a subagent's report: its structured response, else its last non-empty AI text."""
    structured = result.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            return structured.model_dump_json()
        if dataclasses.is_dataclass(structured) and not isinstance(structured, type):
            return json.dumps(dataclasses.asdict(structured))
        return json.dumps(structured)
    # Walk back to the last AIMessage with non-empty text. Anthropic
    # occasionally emits a trailing empty `end_turn` AIMessage after a
    # successful final tool call, which would otherwise be forwarded
    # as an empty ToolMessage.
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            text = msg.text.rstrip() if msg.text else ""
            if text:
                return text
    return ""


def _detached_config(config: RunnableConfig) -> RunnableConfig:
    """Return the parent's config without the internals of a run that may have ended.

    A background subagent can outlive the parent's run, so it must not write to
    that run's checkpoints or report to its callbacks. The thread ID and metadata
    stay, so thread-scoped backends and tools resolve the parent's thread.
    """
    configurable = {key: value for key, value in (config.get("configurable") or {}).items() if not key.startswith(_PER_RUN_CONFIG_PREFIXES)}
    detached = {key: value for key, value in config.items() if key not in {"configurable", "callbacks"}}
    return cast("RunnableConfig", {**detached, "configurable": configurable})


def _start_background_task(subagent: Runnable, state: dict[str, Any], report: BackgroundTask, on_complete: BackgroundTaskCallback) -> str:
    """Start `subagent` on the running event loop and return the `task` tool's reply."""
    handle = asyncio.get_running_loop().create_task(_run_background_task(subagent, state, report, on_complete))
    _BACKGROUND_TASKS.add(handle)
    handle.add_done_callback(_BACKGROUND_TASKS.discard)
    return f"Started `{report['subagent_type']}` in the background as {report['task_id']}. Its report arrives in a later message when it finishes."


async def _run_background_task(subagent: Runnable, state: dict[str, Any], report: BackgroundTask, on_complete: BackgroundTaskCallback) -> None:
    """Run a background subagent under its parent's detached config, then hand over its report."""
    var_child_runnable_config.set(report["config"])
    try:
        with _subagent_tracing_context():
            result = await subagent.ainvoke(state, {"configurable": {"ls_agent_type": "subagent"}})
        report.update(status="success", result=_subagent_report(result))
    except Exception as error:  # noqa: BLE001  # the report tells the parent how its subagent failed
        report.update(status="error", result=f"{type(error).__name__}: {error}")
    try:
        await on_complete(report)
    except Exception:
        logger.exception("`on_background_complete` failed for background task %s", report["task_id"])


def create_sub_agent(
    spec: SubAgent,
    *,
    state_schema: type | None = None,
    response_format: ResponseFormat[Any] | type | dict[str, Any] | None = None,
) -> Runnable:
    """Create a runnable agent from a raw `SubAgent` spec.

    This is the shared entrypoint for the `create_agent` path used by
    raw subagent specs. Pre-compiled `CompiledSubAgent` runnables are already
    created by the caller and are handled separately by `SubAgentMiddleware`.

    Args:
        spec: Subagent spec to compile. Must specify `model` and `tools`.
        state_schema: Base graph state schema forwarded to `create_agent` for
            the subagent.
        response_format: Optional response format override for this compiled
            subagent instance.

    Returns:
        Runnable agent ready for task-tool invocation.

    Raises:
        ValueError: If `spec` is missing `model` or `tools`.
    """
    if "model" not in spec:
        msg = f"SubAgent '{spec['name']}' must specify 'model'"
        raise ValueError(msg)
    if "tools" not in spec:
        msg = f"SubAgent '{spec['name']}' must specify 'tools'"
        raise ValueError(msg)

    from deepagents._models import resolve_model  # noqa: PLC0415

    model = resolve_model(spec["model"])
    middleware: list[AgentMiddleware] = list(spec.get("middleware", []))

    interrupt_on = spec.get("interrupt_on")
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    if not any(m.name == UnsupportedContentMiddleware.__name__ for m in middleware):
        middleware.append(UnsupportedContentMiddleware())

    selected_response_format = response_format if response_format is not None else spec.get("response_format")
    create_agent_kwargs: dict[str, Any] = {
        "system_prompt": spec.get("system_prompt", ""),
        "tools": spec["tools"],
        "middleware": middleware,
        "name": spec["name"],
        "response_format": selected_response_format,
    }
    if state_schema is not None:
        create_agent_kwargs["state_schema"] = state_schema

    return create_agent(model, **create_agent_kwargs)


def _get_subagent_response_format(
    runtime: ToolRuntime,
) -> ResponseFormat[Any] | type | dict[str, Any] | None:
    """Return the response format carried in this task tool call's config."""
    config = runtime.config
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        return None
    value = configurable.get(SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY)
    if value is None:
        return None
    return value


def _build_task_tool(  # noqa: C901, PLR0915
    subagents: Sequence[_SubAgentSpec],
    task_description: str | None = None,
    *,
    private_state_keys: frozenset[str] = frozenset(),
    state_schema: type | None = None,
    on_background_complete: BackgroundTaskCallback | None = None,
) -> BaseTool:
    """Create a task tool from subagent specs.

    Args:
        subagents: List of raw or compiled subagent specs.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.
        private_state_keys: State keys marked with `PrivateStateAttr` that
            should be stripped from parent state before invoking subagents.
        state_schema: Base graph state schema forwarded to raw subagent specs.
        on_background_complete: Receives background subagents' reports. When
            set, the tool accepts `run_in_background`.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    _validate_unique_subagent_names(subagents)
    for spec in subagents:
        _validate_subagent_mode(spec)

    # Computed early (from raw specs) so the mirrored `task` tool below has
    # this exact string for cache hits.
    subagent_description_str = "\n".join(
        _describe_subagent_for_tool(
            s["name"],
            s["description"],
            forked=_is_forked_subagent(s) or _is_forked_compiled_subagent(s),
        )
        for s in subagents
    )
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=subagent_description_str)
    else:
        description = task_description
    args_schema = TaskToolSchema
    if on_background_complete is not None:
        description += _BACKGROUND_TASK_TOOL_NOTE
        args_schema = BackgroundTaskToolSchema

    def _resolved_declarative_spec(spec: SubAgent) -> SubAgent:
        """Resolve the inherited prompt for a declarative fork."""
        if not _is_forked_subagent(spec):
            return spec
        resolved = {key: value for key, value in spec.items() if key != "mode"}
        fork_task_tool = StructuredTool.from_function(
            name="task",
            func=task,
            coroutine=atask,
            description=description,
            infer_schema=False,
            args_schema=args_schema,
        )
        fork_middleware = list(spec.get("middleware", []))
        # The parent's `task` sits right after its filesystem tools; matching that
        # position keeps both tools blocks in the same order. Tests assert that tool
        # order matches.
        fs_index = next((i for i, m in enumerate(fork_middleware) if isinstance(m, FilesystemMiddleware)), -1)
        fork_middleware.insert(fs_index + 1, _ForkTaskToolMiddleware(fork_task_tool))
        resolved["middleware"] = fork_middleware
        return cast("SubAgent", resolved)

    def _compile_spec(
        spec: _SubAgentSpec,
        *,
        response_format: ResponseFormat[Any] | type | dict[str, Any] | None = None,
    ) -> CompiledSubAgent:
        """Compile one raw spec or configure one provided runnable."""
        if "runnable" in spec:
            if response_format is not None:
                msg = f'response_schema cannot be used with compiled subagent "{spec["name"]}"; dynamic schemas require a raw SubAgent spec.'
                raise ValueError(msg)

            compiled = cast("CompiledSubAgent", spec)
            runnable = compiled["runnable"].with_config(
                {
                    "metadata": {"lc_agent_name": spec["name"]},
                    "run_name": spec["name"],
                }
            )
            return {
                "name": spec["name"],
                "description": spec["description"],
                "runnable": runnable,
            }
        return {
            "name": spec["name"],
            "description": spec["description"],
            "runnable": create_sub_agent(
                _resolved_declarative_spec(spec),
                state_schema=state_schema,
                response_format=response_format,
            ),
        }

    # Defined before compiling subagents so _resolved_declarative_spec's
    # mirror tool (above) can reference these directly -- everything they
    # call is only looked up when actually invoked, long after this
    # function returns, so those don't need to exist yet.
    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        # Validate that the result contains a 'messages' key
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS and k not in private_state_keys}
        content = _subagent_report(result)

        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
            }
        )

    def _select_subagent(
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> Runnable:
        """Return the runnable to use for this task invocation."""
        response_format = _get_subagent_response_format(runtime)
        if response_format is not None:
            new_spec = _compile_spec(
                subagents_by_name[subagent_type],
                response_format=response_format,
            )
            return new_spec["runnable"]

        return subagent_graphs[subagent_type]

    def _validate_and_prepare_state(
        subagent_type: str,
        description: str,
        runtime: ToolRuntime,
    ) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = subagents_by_name[subagent_type]
        subagent_runnable = _select_subagent(subagent_type, runtime)

        if _is_forked_subagent(subagent) or _is_forked_compiled_subagent(subagent):
            # The event is folded into the message list above to support compiled subagents
            if _is_forked_subagent(subagent):
                # A declarative fork runs the same graph shape as its parent, so
                # private channels carry over and its own middleware can rebuild
                # what the parent built from them.
                inherited = {key: value for key, value in runtime.state.items() if key not in _FORK_EXCLUDED_STATE_KEYS}
                inherited[_FORKED_CONTEXT_KEY] = True
            else:
                # A compiled runnable is opaque -- it may not declare these
                # channels, and internal state isn't ours to hand it.
                inherited = {key: value for key, value in runtime.state.items() if key not in _EXCLUDED_STATE_KEYS | private_state_keys}
            subagent_state = {
                **inherited,
                "messages": _fork_messages(
                    runtime.state.get("messages", []),
                    runtime.state.get(SUMMARIZATION_EVENT_KEY),
                    description,
                ),
            }
        else:
            subagent_state = {key: value for key, value in runtime.state.items() if key not in _EXCLUDED_STATE_KEYS | private_state_keys}
            subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent_runnable, subagent_state

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        *,
        run_in_background: bool = False,
    ) -> str | Command:
        if run_in_background:
            return _BACKGROUND_SYNC_REFUSAL
        if runtime.state.get(_FORKED_CONTEXT_KEY):
            return _FORK_RECURSION_REFUSAL
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
        )
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        subagent_config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = subagent.invoke(subagent_state, subagent_config)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        *,
        run_in_background: bool = False,
    ) -> str | Command:
        if runtime.state.get(_FORKED_CONTEXT_KEY):
            return _FORK_RECURSION_REFUSAL
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
        )
        if run_in_background and on_background_complete is not None:
            report: BackgroundTask = {
                "task_id": f"bg_{runtime.tool_call_id}",
                "subagent_type": subagent_type,
                "description": description,
                "status": "running",
                "result": "",
                "config": _detached_config(runtime.config),
            }
            return _start_background_task(subagent, subagent_state, report, on_background_complete)
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        subagent_config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = await subagent.ainvoke(subagent_state, subagent_config)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    compiled_subagents = [_compile_spec(spec) for spec in subagents]
    subagents_by_name = {spec["name"]: spec for spec in subagents}

    # Build the graphs dict from the unified spec list
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in compiled_subagents}

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=args_schema,
    )


class SubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for providing subagents to an agent via a `task` tool.

    This middleware adds a `task` tool to the agent that can be used
    to invoke subagents.

    Subagents are useful for handling complex tasks that require multiple steps,
    or tasks that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks,
    and then return a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require
    a narrower subset of tools and focus.

    Args:
        backend: Backend for file operations and execution.
        subagents: List of fully-specified subagent configs.

            Each SubAgent must specify `model` and `tools`.

            Optional `interrupt_on` on individual subagents is respected.
        system_prompt: Instructions appended to main agent's system prompt
            about how to use the task tool.
        task_description: Custom description for the task tool.
        state_schema: Base graph state schema forwarded to raw `SubAgent`
            specs when their runnables are compiled.

            Leave unset to use `create_agent`'s default. `CompiledSubAgent`
            entries are unaffected — callers own those runnables' schemas.
        on_background_complete: Receives each background subagent's report.

            !!! warning "Experimental"

                Setting it lets the model pass `run_in_background` to `task`:
                the subagent starts on the running event loop under the
                parent's thread config, and the call returns at once. The
                callback decides how the parent hears about the result, for
                example by queuing a run on the parent's thread. Only the
                report comes back; state updates such as `StateBackend` files
                are not merged into the parent. Requires the async entrypoint.

            On an Agent Server, queue a run on the parent's thread:

            ```python
            from deepagents.middleware.subagents import BackgroundTask
            from langgraph_sdk import get_client


            async def notify_parent(task: BackgroundTask) -> None:
                config = task["config"]
                await get_client().runs.create(
                    config["configurable"]["thread_id"],
                    config["metadata"]["assistant_id"],
                    input={"messages": [{"role": "user", "content": f"Task {task['task_id']} finished: {task['result']}"}]},
                    multitask_strategy="enqueue",
                )
            ```

    Example:
        ```python
        from deepagents.middleware import SubAgentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                SubAgentMiddleware(
                    backend=my_backend,
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "Research agent",
                            "system_prompt": "You are a researcher.",
                            "model": "openai:gpt-5.5",
                            "tools": [search_tool],
                        }
                    ],
                )
            ],
        )
        ```

    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = None,
        task_description: str | None = None,
        private_state_keys: frozenset[str] | None = None,
        state_schema: type | None = None,
        on_background_complete: BackgroundTaskCallback | None = None,
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()

        if not subagents:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)
        self._backend = backend
        self._subagents = subagents
        self._private_state_keys = private_state_keys or frozenset()
        self._task_description = task_description
        self._state_schema = state_schema
        self._on_background_complete = on_background_complete
        if any(_is_forked_subagent(spec) or _is_forked_compiled_subagent(spec) for spec in subagents):
            warn_beta(name="forked subagents", obj_type="feature")
        if on_background_complete is not None:
            warn_beta(name="background subagents", obj_type="feature")
        self.subagent_names: frozenset[str] = frozenset(spec["name"] for spec in subagents)
        """Declared subagent names. Public so streamers can discover them
        without introspecting the `task` tool's closure."""

        task_tool = _build_task_tool(
            self._subagents,
            task_description,
            private_state_keys=self._private_state_keys,
            state_schema=self._state_schema,
            on_background_complete=on_background_complete,
        )

        # Build system prompt with available agents
        if system_prompt and subagents:
            agents_desc = "\n".join(
                _describe_subagent_for_tool(
                    s["name"],
                    s["description"],
                    forked=_is_forked_subagent(s) or _is_forked_compiled_subagent(s),
                )
                for s in subagents
            )
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        self.tools = [task_tool]

    @property
    def private_state_keys(self) -> frozenset[str]:
        """State keys stripped from parent state before invoking subagents."""
        return self._private_state_keys

    @private_state_keys.setter
    def private_state_keys(self, value: frozenset[str]) -> None:
        self._private_state_keys = value
        task_tool = _build_task_tool(
            self._subagents,
            task_description=self._task_description,
            private_state_keys=value,
            state_schema=self._state_schema,
            on_background_complete=self._on_background_complete,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            request = request.override(system_message=new_system_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            request = request.override(system_message=new_system_message)
        return await handler(request)
