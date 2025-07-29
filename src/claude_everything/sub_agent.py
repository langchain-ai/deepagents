from claude_everything.model import model
from claude_everything.prompts import TASK_DESCRIPTION
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from .sub_agent_registry import SUBAGENT_REGISTRY, register_subagent


@register_subagent("general-purpose")
def make_general_purpose_agent(tools):
    return create_react_agent(
        model,
        prompt=TASK_DESCRIPTION,
        tools=tools or [],
    )


def create_task_tool(tools):
    @tool(description=TASK_DESCRIPTION)
    def task(description: str, subagent_type: str = None):
        if not subagent_type:
            subagent_type = "general-purpose"
        agent_factory = SUBAGENT_REGISTRY.get(subagent_type)
        if not agent_factory:
            return f"Error: unknown agent type {subagent_type}"
        sub_agent = agent_factory(tools)
        result = sub_agent.invoke(
            {"messages": [{"role": "user", "content": description}]}
        )
        return result["messages"][-1].content

    return task
