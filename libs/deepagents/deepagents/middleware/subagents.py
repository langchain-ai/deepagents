"""Middleware for providing subagents to an agent via a `task` tool."""

import contextlib
import dataclasses
import json
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from typing import Annotated, Any, Literal, Never, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    ResponseT,
)
from langchain.agents.structured_output import ResponseFormat
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langsmith.run_helpers import get_tracing_context, tracing_context
from pydantic import BaseModel, Field
from typing_extensions import TypeIs

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import FilesystemPermission
from deepagents.middleware.summarization import SummarizationEvent, _apply_summarization_event

SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"
"""Configurable key used by task-tool callers to request dynamic response format."""

_PARENT_SYSTEM_MESSAGE_KEY = "_deepagents_parent_system_message"


class _ParentSystemMessageState(TypedDict):
    """Private state used to carry the parent's effective system message."""

    _deepagents_parent_system_message: NotRequired[Annotated[SystemMessage | None, OmitFromSchema(input=False, output=True)]]


class _SubAgentBase(TypedDict):
    """Fields shared by declarative subagent specifications.

    When using `create_deep_agent`, subagents automatically receive
    a default middleware stack before any custom `middleware` specified in
    this spec.

    Required fields:
        name: Unique identifier for the subagent.

            The main agent uses this name when calling the `task()` tool.
        description: What this subagent does.

            Be specific and action-oriented. The main agent uses this
            to decide when to delegate.
    Optional fields:
        tools: Tools the subagent can use.

            If not specified, inherits tools from the main agent
            via `default_tools`.
        model: Override the main agent's model.

            Use the format `'provider:model-name'` (e.g., `'openai:gpt-5.5'`).
        middleware: Additional middleware for custom behavior, logging,
            or rate limiting. To restrict filesystem tools, include a
            `FilesystemMiddleware(tools=...)` instance here — it
            will be used as the subagent's filesystem middleware instead of
            the default one.
        interrupt_on: Configure human-in-the-loop for specific tools.

            Requires a checkpointer.
        skills: Skill source paths for `SkillsMiddleware`.

            List of paths to skill directories
            (e.g., `["/skills/user/", "/skills/project/"]`).
        permissions: Filesystem permission rules for this subagent.

            If omitted, inherits the parent agent's permissions. If provided,
            replaces the parent agent's rules entirely for this subagent.

            Rules are evaluated in declaration order; the first match wins.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use.

    If not specified, inherits from main agent.
    """

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model.

    Use `'provider:model-name'` format.
    """

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""

    skills: NotRequired[list[str]]
    """Skill source paths for `SkillsMiddleware`."""

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


class SubAgent(_SubAgentBase):
    """Specification for an isolated declarative subagent.

    The subagent receives only the delegated task description. Use
    [`ForkedSubAgent`][deepagents.middleware.subagents.ForkedSubAgent] to inherit
    the parent's conversation and system prompt.
    """

    system_prompt: NotRequired[str]
    """Instructions for the subagent. Uses an empty prompt when omitted."""

    mode: NotRequired[Literal["handoff"]]
    """Context mode. Declarative `SubAgent` instances are always isolated."""


class ForkedSubAgent(_SubAgentBase):
    """Specification for a subagent that inherits its parent's context.

    !!! warning "Experimental"

        Forked subagents are experimental and may change in a future release.

    A forked subagent receives the parent's effective conversation history and
    exact system prompt. It cannot define a separate `system_prompt`.
    """

    mode: Literal["fork"]
    """Required discriminator that enables conversation forking."""

    system_prompt: NotRequired[Never]
    """Forked subagents always use the parent's system prompt."""


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

    mode: NotRequired[Literal["handoff", "fork"]]
    """Use `fork` to inherit parent history without changing the runnable prompt."""


_SubAgentSpec = SubAgent | ForkedSubAgent | CompiledSubAgent


def _validate_subagent_mode(spec: _SubAgentSpec) -> None:
    """Reject unsupported context modes before a subagent can run."""
    mode = spec.get("mode")
    if mode not in (None, "handoff", "fork"):
        msg = f"SubAgent '{spec['name']}' has invalid mode '{mode}'; expected 'handoff' or 'fork'"
        raise ValueError(msg)
    if mode == "fork" and spec.get("system_prompt") is not None:
        msg = f"ForkedSubAgent '{spec['name']}' cannot set system_prompt; it always inherits the parent's."
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


def _is_forked_subagent(spec: _SubAgentSpec) -> TypeIs[ForkedSubAgent]:
    """Return whether a subagent inherits parent conversation history."""
    return spec.get("mode") == "fork"


def _fork_messages(state: Mapping[str, object], description: str) -> list[AnyMessage]:
    """Build fork input without replaying the in-flight tool-call message."""
    messages = list(cast("Sequence[AnyMessage]", state.get("messages", [])))
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        messages.pop()
    event = cast("SummarizationEvent | None", state.get("_summarization_event"))
    effective_messages = _apply_summarization_event(messages, event)
    return [*effective_messages, HumanMessage(content=description)]


DEFAULT_SUBAGENT_PROMPT = """In order to complete the objective that the user asks of you, you have access to a number of standard tools.

The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final
response contains the complete answer."""

_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
    _PARENT_SYSTEM_MESSAGE_KEY,
}
"""State keys that are excluded when passing state to subagents and when
returning updates from subagents.

When returning updates:

1. The messages key is handled explicitly to ensure only the final message
    is included
2. The todos and `structured_response` keys are excluded as they do not have
    a defined reducer and no clear meaning for returning them from a subagent
    to the main agent.
3. Agent-private fields on middleware state schemas are excluded from both
    subagent output and subagent inputs.
"""


class TaskToolSchema(BaseModel):
    """Input schema for the `task` tool."""

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )

    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))


TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle a complex, multi-step task in an isolated context window.

Available agent types and the tools they have access to:
{available_agents}

Specify subagent_type to select the agent. Usage notes:
- Launch multiple agents concurrently when their tasks are independent, using a single message with multiple tool calls.
- Each invocation is stateless: the agent sees only the prompt you give it and returns a single final report. Put full detail in the prompt and state exactly what it should return.
- The agent's report is not shown to the user; relay a summary yourself.
- Tell the agent whether to create content, analyze, or only research, since it cannot see the user's intent.
- If an agent's description says to use it proactively, do so without waiting to be asked.
- When only general-purpose is available, use it for any complex, context-heavy task; it has the same capabilities as the main agent."""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
"""Base spec for general-purpose subagent (caller adds model, tools, middleware)."""


class _ParentSystemMessageMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    state_schema = _ParentSystemMessageState

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ExtendedModelResponse[ResponseT]:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={_PARENT_SYSTEM_MESSAGE_KEY: request.system_message}),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ExtendedModelResponse[ResponseT]:
        response = await handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={_PARENT_SYSTEM_MESSAGE_KEY: request.system_message}),
        )


class _ForkSystemMessageMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    state_schema = _ParentSystemMessageState

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        parent_message = request.state.get(_PARENT_SYSTEM_MESSAGE_KEY)
        if parent_message is not None:
            request = request.override(system_message=parent_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        parent_message = request.state.get(_PARENT_SYSTEM_MESSAGE_KEY)
        if parent_message is not None:
            request = request.override(system_message=parent_message)
        return await handler(request)


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
    parent_system_prompt: str | SystemMessage | None = None,
) -> BaseTool:
    """Create a task tool from subagent specs.

    Args:
        subagents: List of raw or compiled subagent specs.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.
        private_state_keys: State keys marked with `PrivateStateAttr` that
            should be stripped from parent state before invoking subagents.
        state_schema: Base graph state schema forwarded to raw subagent specs.
        parent_system_prompt: Static prompt inherited by declarative forked subagents.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    _validate_unique_subagent_names(subagents)
    for spec in subagents:
        _validate_subagent_mode(spec)

    def _resolved_declarative_spec(spec: SubAgent | ForkedSubAgent) -> SubAgent:
        """Resolve the inherited prompt for a declarative fork."""
        if not _is_forked_subagent(spec):
            return spec
        resolved = {key: value for key, value in spec.items() if key not in {"mode", "system_prompt"}}
        resolved["system_prompt"] = parent_system_prompt if parent_system_prompt is not None else ""
        resolved["middleware"] = [*spec.get("middleware", []), _ForkSystemMessageMiddleware()]
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

    compiled_subagents = [_compile_spec(spec) for spec in subagents]
    subagents_by_name = {spec["name"]: spec for spec in subagents}
    fork_mode_names = {name for name, spec in subagents_by_name.items() if _is_forked_subagent(spec)}

    # Build the graphs dict and descriptions from the unified spec list
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in compiled_subagents}

    subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in compiled_subagents)

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=subagent_description_str)
    else:
        description = task_description

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

        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump_json"):
                content: str = structured.model_dump_json()
            elif dataclasses.is_dataclass(structured) and not isinstance(structured, type):
                content = json.dumps(dataclasses.asdict(structured))
            else:
                content = json.dumps(structured)
        else:
            # Walk back to the last AIMessage with non-empty text. Anthropic
            # occasionally emits a trailing empty `end_turn` AIMessage after a
            # successful final tool call, which would otherwise be forwarded
            # as an empty ToolMessage.
            content = ""
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    text = msg.text.rstrip() if msg.text else ""
                    if text:
                        content = text
                        break

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
        effective_system_message: SystemMessage | None,
    ) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = _select_subagent(subagent_type, runtime)
        forked = subagent_type in fork_mode_names
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state = {k: v for k, v in subagent_state.items() if k not in private_state_keys}
        if forked and "runnable" not in subagents_by_name[subagent_type] and effective_system_message is not None:
            subagent_state[_PARENT_SYSTEM_MESSAGE_KEY] = effective_system_message
        if forked:
            subagent_state["messages"] = _fork_messages(runtime.state, description)
        else:
            subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        effective_system_message = runtime.state.get(_PARENT_SYSTEM_MESSAGE_KEY)
        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
            effective_system_message,
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
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        effective_system_message = runtime.state.get(_PARENT_SYSTEM_MESSAGE_KEY)
        subagent, subagent_state = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
            effective_system_message,
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
            result = await subagent.ainvoke(subagent_state, subagent_config)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema,
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
        parent_system_prompt: Prompt inherited by declarative `ForkedSubAgent`
            entries. Compiled subagents keep their own prompt.

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

    state_schema = _ParentSystemMessageState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        subagents: Sequence[SubAgent | ForkedSubAgent | CompiledSubAgent],
        system_prompt: str | None = None,
        task_description: str | None = None,
        private_state_keys: frozenset[str] | None = None,
        state_schema: type | None = None,
        parent_system_prompt: str | SystemMessage | None = None,
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
        self._parent_system_prompt = parent_system_prompt
        self.subagent_names: frozenset[str] = frozenset(spec["name"] for spec in subagents)
        """Declared subagent names. Public so streamers can discover them
        without introspecting the `task` tool's closure."""

        task_tool = _build_task_tool(
            self._subagents,
            task_description,
            private_state_keys=self._private_state_keys,
            state_schema=self._state_schema,
            parent_system_prompt=self._parent_system_prompt,
        )

        # Build system prompt with available agents
        if system_prompt and subagents:
            agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)
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
            parent_system_prompt=self._parent_system_prompt,
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
