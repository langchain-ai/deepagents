# 🧠🤖Deep Agents

Using an LLM to call tools in a loop is the simplest form of an agent.
This architecture, however, can yield agents that are “shallow” and fail to plan and act over longer, more complex tasks.

Applications like “Deep Research”, "Manus", and “Claude Code” have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="../../deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application. For a full overview and quickstart of `deepagents`, the best resource is our [docs](https://docs.langchain.com/oss/python/deepagents/overview).

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## Installation

```bash
# pip
pip install deepagents

# uv
uv add deepagents

# poetry
poetry add deepagents
```

## Usage

> **Note:** `deepagents` requires using a LLM that supports [tool calling](https://python.langchain.com/docs/concepts/tool_calling/).

This example uses [Tavily](https://tavily.com/) as an example search provider, but you can substitute any search API (e.g., DuckDuckGo, SerpAPI, Brave Search). To run the example below, you will need to `pip install tavily-python`.

Make sure to set `TAVILY_API_KEY` in your environment. You can generate one [here](https://www.tavily.com/).

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Web search tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# Create the deep agent
agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example.

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Core Capabilities

**Planning & Task Decomposition**

Deep Agents include a built-in `write_todos` tool that enables agents to break down complex tasks into discrete steps, track progress, and adapt plans as new information emerges.

**Context Management**

File system tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) allow agents to offload large context to memory, preventing context window overflow and enabling work with variable-length tool results.

**Subagent Spawning**

A built-in `task` tool enables agents to spawn specialized subagents for context isolation. This keeps the main agent's context clean while still going deep on specific subtasks.

**Long-term Memory**

Extend agents with persistent memory across threads using LangGraph's `BaseStore`. Agents can save and retrieve information from previous conversations.

## Customizing Deep Agents

There are several parameters you can pass to `create_deep_agent` to create your own custom deep agent.

### `model`

By default, `deepagents` uses `claude-sonnet-4-5-20250929`. You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

> **Tip:** Use the `provider:model` format (e.g., `openai:gpt-5`) to quickly switch between models. See the [reference](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model(model)) for more info.

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model(model="openai:gpt-5")
agent = create_deep_agent(model=model)
```

### `system_prompt`

Deep Agents come with a built-in system prompt. This is relatively detailed prompt that is heavily based on and inspired by [attempts](https://github.com/kn1026/cc/blob/main/claudecode.md) to [replicate](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md)
Claude Code's system prompt. It was made more general purpose than Claude Code's system prompt. The default prompt contains detailed instructions for how to use the built-in planning tool, file system tools, and sub agents.

Each deep agent tailored to a use case should include a custom system prompt specific to that use case as well. The importance of prompting for creating a successful deep agent cannot be overstated.

```python
from deepagents import create_deep_agent

research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.
"""

agent = create_deep_agent(
    system_prompt=research_instructions,
)
```

### `tools`

In addition to custom tools you provide, `deepagents` include built-in tools for planning (`write_todos`), file management (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`), and subagent spawning (`task`).

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

agent = create_deep_agent(
    tools=[internet_search]
)
```

### `middleware`

`create_deep_agent` is implemented with middleware that can be customized. You can provide additional middleware to extend functionality, add tools, or implement custom hooks.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

@tool
def get_temperature(city: str) -> str:
    """Get the temperature in a city."""
    return f"The temperature in {city} is 70 degrees Fahrenheit."

class WeatherMiddleware(AgentMiddleware):
  tools = [get_weather, get_temperature]

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[WeatherMiddleware()]
)
```

### `subagents`

A main feature of Deep Agents is their ability to spawn subagents. You can specify custom subagents that your agent can hand off work to in the subagents parameter. Sub agents are useful for context quarantine (to help not pollute the overall context of the main agent) as well as custom instructions.

`subagents` should be a list of dictionaries, where each dictionary follow this schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]

class CompiledSubAgent(TypedDict):
    name: str
    description: str
    runnable: Runnable
```

**`SubAgent` fields:**

Required fields:

- `name` (`str`): Unique identifier for the subagent. The main agent uses this name when calling the `task()` tool.
- `description` (`str`): What this subagent does. Be specific and action-oriented. The main agent uses this to decide when to delegate.
- `system_prompt` (`str`): Instructions for the subagent. Include tool usage guidance and output format requirements.
- `tools` (`list[Callable]`): Tools the subagent can use. Keep this minimal and include only what's needed.

Optional fields:

- `model` (`str | BaseChatModel`): Override the main agent's model. Use the format `'provider:model-name'` (e.g., `'openai:gpt-4o'`). Defaults to `claude-sonnet-4-5-20250929`.
- `middleware` (`list[Middleware]`): Additional middleware for custom behavior, logging, or rate limiting.
- `interrupt_on` (`dict[str, bool]`): Configure human-in-the-loop for specific tools. Requires a checkpointer.

**`CompiledSubAgent` fields:**

- `name` (`str`): Unique identifier for the subagent.
- `description` (`str`): What this subagent does.
- `runnable` (`Runnable`): A compiled LangGraph graph (must call `.compile()` first).

#### Using `SubAgent`

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-4o",  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    subagents=subagents
)
```

#### Using `CompiledSubAgent`

For complex workflows, use a pre-built LangGraph graph:

```python
# Create a custom agent graph
custom_graph = create_agent(
    model=your_model,
    tools=specialized_tools,
    prompt="You are a specialized agent for data analysis..."
)

# Use it as a compiled subagent
custom_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="Specialized agent for complex data analysis tasks",
    runnable=custom_graph
)

subagents = [custom_subagent]

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[internet_search],
    system_prompt=research_instructions,
    subagents=subagents
)
```

### `interrupt_on`

The harness can pause agent execution at specified tool calls to allow human approval or modification. This feature is opt-in via the `interrupt_on` parameter.

Pass `interrupt_on` to `create_deep_agent` with a mapping of tool names to interrupt configurations. Example: `interrupt_on={"edit_file": True}` pauses before every edit.

These tool configs are passed to our prebuilt [HITL middleware](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop) so that the agent pauses execution and waits for feedback from the user before executing configured tools.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)

```

## Deep Agents Middleware

Deep Agents are built with a modular middleware architecture. As a reminder, Deep Agents have access to:

- A planning tool
- A filesystem for storing context and long-term memories
- The ability to spawn subagents

Each of these features is implemented as separate middleware. When you create a deep agent with `create_deep_agent`, we automatically attach **TodoListMiddleware**, **FilesystemMiddleware** and **SubAgentMiddleware** to your agent.

Middleware is a composable concept, and you can choose to add as many or as few middleware to an agent depending on your use case. That means that you can also use any of the aforementioned middleware independently!

### `TodoListMiddleware`

Planning is integral to solving complex problems. If you've used Claude Code recently, you'll notice how it writes out a to-do list before tackling complex, multi-part tasks. You'll also notice how it can adapt and update this to-do list on the fly as more information comes in.

`TodoListMiddleware` provides your agent with a tool specifically for updating this to-do list. Before, and while it executes a multi-part task, the agent is prompted to use the `write_todos` tool to keep track of what it's doing, and what still needs to be done.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

# TodoListMiddleware is included by default in create_deep_agent
# You can customize it if building a custom agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    # Custom planning instructions can be added via middleware
    middleware=[
        TodoListMiddleware(
            system_prompt="Use the write_todos tool to..."  # Optional: Custom addition to the system prompt
        ),
    ],
)
```

### `FilesystemMiddleware`

Context engineering is one of the main challenges in building effective agents. This can be particularly hard when using tools that can return variable length results (e.g., `web_search`, RAG), as long tool results can quickly fill up your context window.
`FilesystemMiddleware` provides four tools to your agent to interact with both short-term and long-term memory:

- `ls`: List the files in your filesystem
- `read_file`: Read an entire file, or a certain number of lines from a file
- `write_file`: Write a new file to your filesystem
- `edit_file`: Edit an existing file in your filesystem

```python
from langchain.agents import create_agent
from deepagents.middleware.filesystem import FilesystemMiddleware


# FilesystemMiddleware is included by default in create_deep_agent
# You can customize it if building a custom agent
agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[
        FilesystemMiddleware(
            backend=..., # Optional: customize storage backend
            system_prompt="Write to the filesystem when...",  # Optional custom system prompt override
            custom_tool_descriptions={
                "ls": "Use the ls tool when...",
                "read_file": "Use the read_file tool to..."
            }  # Optional: Custom descriptions for filesystem tools
        ),
    ],
)
```

### `SubAgentMiddleware`

Handing off tasks to subagents is a great way to isolate context, keeping the context window of the main (supervisor) agent clean while still going deep on a task. `SubAgentMiddleware` allows you to supply subagents through a `task` tool.

A subagent is defined with a name, description, system prompt, and tools. You can also provide a subagent with a custom model, or with additional middleware. This can be particularly useful when you want to give the subagent an additional state key to share with the main agent.

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from deepagents.middleware.subagents import SubAgentMiddleware


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="claude-sonnet-4-20250514",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-20250514",
            default_tools=[],
            subagents=[
                {
                    "name": "weather",
                    "description": "This subagent can get weather in cities.",
                    "system_prompt": "Use the get_weather tool to get the weather in a city.",
                    "tools": [get_weather],
                    "model": "gpt-4.1",
                    "middleware": [],
                }
            ],
        )
    ],
)
```

For more complex use cases, you can also provide your own pre-built LangGraph graph as a subagent.

```python
# Create a custom LangGraph graph
def create_weather_graph():
    workflow = StateGraph(...)
    # Build your custom graph
    return workflow.compile()

weather_graph = create_weather_graph()

# Wrap it in a `CompiledSubAgent`
weather_subagent = CompiledSubAgent(
    name="weather",
    description="This subagent can get weather in cities.",
    runnable=weather_graph
)

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    middleware=[
        SubAgentMiddleware(
            default_model="claude-sonnet-4-20250514",
            default_tools=[],
            subagents=[weather_subagent],
        )
    ],
)
```

## Sync vs Async

Prior versions of deepagents separated sync and async agent factories.

`async_create_deep_agent` has been folded in to `create_deep_agent`.

**You should use `create_deep_agent` as the factory for both sync and async agents**

## MCP

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**NOTE:** MCP tools are async, so you'll need to use `agent.ainvoke()` or `agent.astream()` for invocation.

(To run the example below, will need to `pip install langchain-mcp-adapters`)

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    # Collect MCP tools
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    # Create agent
    agent = create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```
