from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Dict
from langchain_core.tools import BaseTool
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


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    exclude_builtin_tools: bool = False,
    include_builtin_tool_names: Optional[list[str]] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
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
        exclude_builtin_tools: If True, exclude all built-in tools from the agent.
        include_builtin_tool_names: If specified, only include built-in tools whose names 
            match those in this list. If not specified, include all built-in tools 
            (unless exclude_builtin_tools is True).
        interrupt_config: Optional Dict[str, HumanInterruptConfig] mapping tool names to interrupt configs.
        config_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
    """
    
    prompt = instructions + base_prompt
    
    # Define all built-in tools with their names
    all_builtin_tools = [write_todos, write_file, read_file, ls, edit_file]
    
    # Filter built-in tools based on parameters
    if exclude_builtin_tools:
        built_in_tools = []
    elif include_builtin_tool_names is not None:
        # Only include built-in tools whose names are in the specified list
        builtin_tool_names = {tool.name if hasattr(tool, 'name') else tool.__name__ for tool in all_builtin_tools}
        built_in_tools = [
            tool for tool in all_builtin_tools 
            if (tool.name if hasattr(tool, 'name') else tool.__name__) in include_builtin_tool_names
        ]
    else:
        # Include all built-in tools (default behavior)
        built_in_tools = all_builtin_tools
    
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + list(tools) + [task_tool]
    
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
    
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        post_model_hook=selected_post_model_hook,
        config_schema=config_schema,
        checkpointer=checkpointer,
    )
