from deepagents.prompts import TASK_DESCRIPTION_PREFIX, TASK_DESCRIPTION_SUFFIX
from deepagents.state import DeepAgentState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.chat_models import init_chat_model
from typing import Annotated, NotRequired, Any
from langgraph.types import Command

from langgraph.prebuilt import InjectedState


class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    include_files: NotRequired[bool]
    include_todos: NotRequired[bool]
    # Optional per-subagent model configuration
    model_settings: NotRequired[dict[str, Any]]


def _create_task_tool(tools, instructions, subagents: list[SubAgent], model, state_schema):
    agents = {
        "general-purpose": create_react_agent(
            model, prompt=instructions, tools=tools, state_schema=state_schema
        )
    }
    subagent_configs = {
        "general-purpose": {"include_files": True, "include_todos": True}
    }
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_
    for _agent in subagents:
        subagent_configs[_agent["name"]] = {
            "include_files": _agent.get("include_files", False),
            "include_todos": _agent.get("include_todos", False)
        }
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            _tools = tools
        # Resolve per-subagent model if specified, else fallback to main model
        if "model_settings" in _agent:
            model_config = _agent["model_settings"]
            # Always use get_default_model to ensure all settings are applied
            sub_model = init_chat_model(**model_config)
        else:
            sub_model = model
        agents[_agent["name"]] = create_react_agent(
            sub_model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema, checkpointer=False
        )

    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]

    @tool(
        description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string)
        + TASK_DESCRIPTION_SUFFIX
    )
    async def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
        sub_agent = agents[subagent_type]
        sub_state = {"messages": [{"role": "user", "content": description}]}
        config = subagent_configs.get(subagent_type, {"include_files": False, "include_todos": False})
        if config["include_files"]:
            sub_state["files"] = state.get("files", {})
        if config["include_todos"]:
            sub_state["todos"] = state.get("todos", [])
        result = await sub_agent.ainvoke(sub_state)
        update = {
            "messages": [
                ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)
            ]
        }
        if config["include_files"]:
            update["files"] = result.get("files", {})
        if config["include_todos"]:
            update["todos"] = result.get("todos", [])
        return Command(update=update)

    return task