from deepagents.sub_agent import (
    _create_task_tool,
    _create_sync_task_tool,
    SubAgent,
    CustomSubAgent,
)
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import LanguageModelLike
from deepagents.interrupt import create_interrupt_hook, ToolInterruptConfig
from langgraph.types import Checkpointer
from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage."""


def _agent_builder(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent | CustomSubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    builtin_tools: Optional[list[str]] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
    main_agent_tools: Optional[list[str]] = None,
    is_async: bool = False,
):
    prompt = instructions + base_prompt

    all_builtin_tools = [write_todos, write_file, read_file, ls, edit_file]

    if builtin_tools is not None:
        tools_by_name = {}
        for tool_ in all_builtin_tools:
            if not isinstance(tool_, BaseTool):
                tool_ = tool(tool_)
            tools_by_name[tool_.name] = tool_
        # Only include built-in tools whose names are in the specified list
        built_in_tools = [tools_by_name[_tool] for _tool in builtin_tools]
    else:
        built_in_tools = all_builtin_tools

    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState

    # Should never be the case that both are specified
    if post_model_hook and interrupt_config:
        raise ValueError(
            "Cannot specify both post_model_hook and interrupt_config together. "
            "Use either interrupt_config for tool interrupts or post_model_hook for custom post-processing."
        )
    elif post_model_hook is not None:
        selected_post_model_hook = post_model_hook
    elif interrupt_config is not None:
        selected_post_model_hook = create_interrupt_hook(interrupt_config)
    else:
        selected_post_model_hook = None

    if not is_async:
        task_tool = _create_sync_task_tool(
            list(tools) + built_in_tools,
            instructions,
            subagents or [],
            model,
            state_schema,
            selected_post_model_hook,
        )
    else:
        task_tool = _create_task_tool(
            list(tools) + built_in_tools,
            instructions,
            subagents or [],
            model,
            state_schema,
            selected_post_model_hook,
        )
    if main_agent_tools is not None:
        passed_in_tools = []
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = tool(tool_)
            if tool_.name in main_agent_tools:
                passed_in_tools.append(tool_)
    else:
        passed_in_tools = list(tools)
    all_tools = built_in_tools + passed_in_tools + [task_tool]

    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        post_model_hook=selected_post_model_hook,
        config_schema=config_schema,
        checkpointer=checkpointer,
    )


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent | CustomSubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    builtin_tools: Optional[list[str]] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
    main_agent_tools: Optional[list[str]] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict settings)
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        builtin_tools: If not provided, all built-in tools are included. If provided,
            only the specified built-in tools are included.
        interrupt_config: Optional Dict[str, HumanInterruptConfig] mapping tool names to interrupt configs.
        config_schema: The schema of the deep agent.
        post_model_hook: Custom post model hook
        checkpointer: Optional checkpointer for persisting agent state between runs.
        main_agent_tools: Optional list of tool names that the main agent should have. If not provided,
            will have access to all tools. Note that built-in tools (for filesystem and todo and subagents) are
            always included - this filtering only applies to passed in tools.
    """
    return _agent_builder(
        tools=tools,
        instructions=instructions,
        model=model,
        subagents=subagents,
        state_schema=state_schema,
        builtin_tools=builtin_tools,
        interrupt_config=interrupt_config,
        config_schema=config_schema,
        checkpointer=checkpointer,
        post_model_hook=post_model_hook,
        main_agent_tools=main_agent_tools,
        is_async=False,
    )


def async_create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent | CustomSubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    builtin_tools: Optional[list[str]] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
    main_agent_tools: Optional[list[str]] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict settings)
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        builtin_tools: If not provided, all built-in tools are included. If provided,
            only the specified built-in tools are included.
        interrupt_config: Optional Dict[str, HumanInterruptConfig] mapping tool names to interrupt configs.
        config_schema: The schema of the deep agent.
        post_model_hook: Custom post model hook
        checkpointer: Optional checkpointer for persisting agent state between runs.
        main_agent_tools: Optional list of tool names that the main agent should have. If not provided,
            will have access to all tools. Note that built-in tools (for filesystem and todo and subagents) are
            always included - this filtering only applies to passed in tools.
    """
    return _agent_builder(
        tools=tools,
        instructions=instructions,
        model=model,
        subagents=subagents,
        state_schema=state_schema,
        builtin_tools=builtin_tools,
        interrupt_config=interrupt_config,
        config_schema=config_schema,
        checkpointer=checkpointer,
        post_model_hook=post_model_hook,
        main_agent_tools=main_agent_tools,
        is_async=True,
    )
