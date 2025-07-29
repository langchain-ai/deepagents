import os
from typing import Literal, Annotated

from tavily import TavilyClient

from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage, tool

from claude_everything.graph import create_deep_agent
from claude_everything.sub_agent import create_task_tool
from claude_everything.tools import write_todos
from langgraph.prebuilt.chat_agent_executor import AgentState


# Modified agent state to store the final report
class ResearchAgentState(AgentState):
    report: str


# Search tool to use to do research
def internet_search(
    query,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    search_docs = tavily_async_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


# Function that writes the report to state
def write_report(report: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """Use this to write your final report to a file.

    the `report` argument should be the whole report. Make sure it is comprehensive."""
    return Command(
        update={
            "report": report,
            "messages": [ToolMessage(f"Wrote final report", tool_call_id=tool_call_id)],
        }
    )


# Prompt prefix to steer the agent to be an expert researcher
research_prompt_prefix = """You are an expert researcher. Your job is to write a through research report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# use-case specific tools
research_tools = [internet_search]

task_tool = create_task_tool(research_tools)

agent = create_deep_agent(
    [task_tool],
    research_prompt_prefix,
    state_schema=ResearchAgentState,
    main_agent_tools=[write_report],
)
