"""DeepAgents implemented as Middleware"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, SummarizationMiddleware
from langchain.agents.middleware.prompt_caching import AnthropicPromptCachingMiddleware
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from typing import NotRequired, Annotated
from deepagents.state import Todo, file_reducer
from deepagents.tools import write_todos, ls, read_file, write_file, edit_file
from deepagents.prompts import WRITE_TODOS_SYSTEM_PROMPT, TASK_SYSTEM_PROMPT, FILESYSTEM_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION, BASE_AGENT_PROMPT
from deepagents.types import SubAgent, CustomSubAgent


###############################
# Current Limitations
# - Prompts are segregated for each piece of middleware, no composition (write todos in subagents)
# - State Schema doesn't work right now!
# - Need to add back interrupt support!
###############################

###########################
# Planning Middleware
###########################

class PlanningState(AgentState):
    todos: NotRequired[list[Todo]]

class PlanningMiddleware(AgentMiddleware):
    state_schema = PlanningState
    tools = [write_todos]

    def modify_model_request(self, request: ModelRequest, agent_state: PlanningState) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + WRITE_TODOS_SYSTEM_PROMPT
        return request

    # TODO: Prune out more than one call of write_todos in parallel.
    # def after_model(self, state: AgentState) -> dict[str, Any] | None:
    #     return {
    #         "messages": [AIMessage(content="")]
    #     }

###########################
# Filesystem Middleware
###########################

class FilesystemState(AgentState):
    files: Annotated[NotRequired[dict[str, str]], file_reducer]

class FilesystemMiddleware(AgentMiddleware):
    state_schema = FilesystemState
    tools = [ls, read_file, write_file, edit_file]

    def modify_model_request(self, request: ModelRequest, agent_state: FilesystemState) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + FILESYSTEM_SYSTEM_PROMPT
        return request

###########################
# SubAgent Middleware
###########################

class SubAgentMiddleware(AgentMiddleware):
    def __init__(
        self,
        tools,
        subagents: list[SubAgent | CustomSubAgent],
        model,
        is_async=False,
    ) -> None:
        super().__init__()
        self.tools = [
            create_task_tool(
                tools=tools,
                subagents=subagents,
                model=model,
                is_async=is_async,
            )
        ]

    def modify_model_request(self, request: ModelRequest, agent_state: AgentState) -> ModelRequest:
        request.system_prompt = request.system_prompt + "\n\n" + TASK_SYSTEM_PROMPT
        return request

def _get_agents(
    tools,
    subagents: list[SubAgent | CustomSubAgent],
    model
):
    default_subagent_middleware = [
        PlanningMiddleware(),
        FilesystemMiddleware(),
        # AnthropicPromptCachingMiddleware(ttl="5m"),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=20000,
            messages_to_keep=20,
        ),
    ]
    agents = {
        # NOTE: The general-purpose agent gets todos and files, but not subagents.
        "general-purpose": create_agent(
            model,
            prompt=BASE_AGENT_PROMPT,
            tools=tools,
            checkpointer=False,
            middleware=default_subagent_middleware
        )
    }
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_
    for _agent in subagents:
        if "graph" in _agent:
            agents[_agent["name"]] = _agent["graph"]
            continue
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
        else:
            _tools = tools
        # Resolve per-subagent model: can be instance or dict
        if "model" in _agent:
            agent_model = _agent["model"]
            if isinstance(agent_model, dict):
                # Dictionary settings - create model from config
                sub_model = init_chat_model(**agent_model)
            else:
                # Model instance - use directly
                sub_model = agent_model
        else:
            # Fallback to main model
            sub_model = model
        agents[_agent["name"]] = create_agent(
            sub_model,
            prompt=_agent["prompt"],
            tools=_tools,
            middleware=[
                *default_subagent_middleware,
                *_agent["middleware"],
            ],
            checkpointer=False,
        )
    return agents


def _get_subagent_description(subagents: list[SubAgent | CustomSubAgent]):
    return [f"- {_agent['name']}: {_agent['description']}" for _agent in subagents]


def create_task_tool(
    tools,
    subagents: list[SubAgent | CustomSubAgent],
    model,
    is_async: bool = False,
):
    agents = _get_agents(
        tools, subagents, model
    )
    other_agents_string = _get_subagent_description(subagents)

    if is_async:
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        async def task(
            description: str,
            subagent_type: str,
            state: Annotated[AgentState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]
            state["messages"] = [{"role": "user", "content": description}]
            result = await sub_agent.ainvoke(state)
            return Command(
                update={
                    "files": result.get("files", {}),
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    else: 
        @tool(
            description=TASK_TOOL_DESCRIPTION.format(other_agents=other_agents_string)
        )
        def task(
            description: str,
            subagent_type: str,
            state: Annotated[AgentState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            if subagent_type not in agents:
                return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            sub_agent = agents[subagent_type]
            state["messages"] = [{"role": "user", "content": description}]
            result = sub_agent.invoke(state)
            # TODO: Write back all keys from the result to parent graph EXCEPT for todos
            return Command(
                update={
                    "files": result.get("files", {}),
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
    return task
