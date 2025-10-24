from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command


def assert_all_deepagent_qualities(agent):
    assert "todos" in agent.stream_channels
    assert "files" in agent.stream_channels
    assert "write_todos" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "ls" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "read_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "write_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "edit_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()


###########################
# Mock tools and middleware
###########################

SAMPLE_MODEL = "claude-sonnet-4-20250514"


@tool(description="Use this tool to get premier league standings")
def get_premier_league_standings(runtime: ToolRuntime):
    long_tool_msg = "This is a long tool message that should be evicted to the filesystem.\n" * 300
    return Command(
        update={
            "messages": [ToolMessage(content=long_tool_msg, tool_call_id=runtime.tool_call_id)],
            "files": {"/test.txt": {"content": ["Goodbye world"], "created_at": "2021-01-01", "modified_at": "2021-01-01"}},
            "research": "extra_value",
        }
    )


@tool(description="Use this tool to get la liga standings")
def get_la_liga_standings(runtime: ToolRuntime):
    long_tool_msg = "This is a long tool message that should be evicted to the filesystem.\n" * 300
    return Command(
        update={
            "messages": [ToolMessage(content=long_tool_msg, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool(description="Use this tool to get a comprehensive report on the NBA standings")
def get_nba_standings():
    return "Sample text that is too long to fit in the token limit\n" * 10000


@tool(description="Use this tool to get a comprehensive report on the NBA standings")
def get_nfl_standings():
    return "Sample text that is too long to fit in the token limit\n" * 100


@tool(description="Use this tool to get the weather")
def get_weather(location: str):
    return f"The weather in {location} is sunny."


@tool(description="Use this tool to get the latest soccer scores")
def get_soccer_scores(team: str):
    return f"The latest soccer scores for {team} are 2-1."


@tool(description="Sample tool")
def sample_tool(sample_input: str):
    return sample_input


@tool(description="Sample tool with injected state")
def sample_tool_with_injected_state(sample_input: str, runtime: ToolRuntime):
    return sample_input + runtime.state["sample_input"]


TOY_BASKETBALL_RESEARCH = "Lebron James is the best basketball player of all time with over 40k points and 21 seasons in the NBA."


@tool(description="Use this tool to conduct research into basketball and save it to state")
def research_basketball(topic: str, runtime: ToolRuntime):
    current_research = runtime.state.get("research", "")
    research = f"{current_research}\n\nResearching on {topic}... Done! {TOY_BASKETBALL_RESEARCH}"
    return Command(update={"research": research, "messages": [ToolMessage(research, tool_call_id=runtime.tool_call_id)]})


class ResearchState(AgentState):
    research: str


class ResearchMiddlewareWithTools(AgentMiddleware):
    state_schema = ResearchState
    tools = [research_basketball]


class ResearchMiddleware(AgentMiddleware):
    state_schema = ResearchState


class SampleMiddlewareWithTools(AgentMiddleware):
    tools = [sample_tool]


class SampleState(AgentState):
    sample_input: str


class SampleMiddlewareWithToolsAndState(AgentMiddleware):
    state_schema = SampleState
    tools = [sample_tool]


class WeatherToolMiddleware(AgentMiddleware):
    tools = [get_weather]
