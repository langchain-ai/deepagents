from deepagents.prompts import BASE_PROMPT
from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from typing import Sequence, Union, Callable, Any, TypeVar, Type, Optional, Dict
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import LanguageModelLike
from deepagents.interrupt import create_interrupt_hook, ToolInterruptConfig
from langgraph.types import Checkpointer
from langgraph.prebuilt import create_react_agent

StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    base_prompt_override: Optional[str] = None,
    builtin_tools_override: Optional[Sequence[Union[BaseTool, Callable, dict[str, Any]]]] = None,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    builtin_tools: Optional[list[str]] = None,
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
        base_prompt_override: If provided, this will override the default base prompt.
            If not provided, the default base prompt will be used. Your base prompt should include
            the instructions for the `write_todos` tool and the `task` tool.
        builtin_tools_override: If provided, this will override the default built-in tools for each tool specified.
            Built-in tools are: write_todos, write_file, read_file, ls, edit_file.
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
        checkpointer: Optional checkpointer for persisting agent state between runs.
    """
    base_prompt = base_prompt_override or BASE_PROMPT
    prompt = instructions + base_prompt
    
    all_builtin_tools = [write_todos, write_file, read_file, ls, edit_file]
    
    if builtin_tools is not None:
        tools_by_name = {}
        for tool_ in all_builtin_tools:
            if not isinstance(tool_, BaseTool):
                tool_ = tool(tool_)
            tools_by_name[tool_.name] = tool_
        # Only include built-in tools whose names are in the specified list
        built_in_tools = [ tools_by_name[_tool] for _tool in builtin_tools        ]
    else:
        built_in_tools = all_builtin_tools

    # override built-in tools
    if builtin_tools_override is not None:
        tools_by_name = {}
        for tool_ in built_in_tools:
            if not isinstance(tool_, BaseTool):
                tool_ = tool(tool_)
            tools_by_name[tool_.name] = tool_
        # If a tool override was passed in, replace the built-in tool with the override
        for tool_name, tool_override in builtin_tools_override.items():
            if tool_name in tools_by_name:
                tools_by_name[tool_name] = tool_override
        built_in_tools = [tools_by_name[_tool] for _tool in tools_by_name]

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
