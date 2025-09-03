# 🧠🤖Deep Agents

Using an LLM to call tools in a loop is the simplest form of an agent. 
This architecture, however, can yield agents that are “shallow” and fail to plan and act over longer, more complex tasks. 
Applications like “Deep Research”, "Manus", and “Claude Code” have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## Installation

```bash
pip install deepagents
```

## Usage

(To run the example below, will need to `pip install tavily-python`)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Search tool to use to do research
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


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "what is langgraph?"}]})
```

See [examples/research/research_agent.py](examples/research/research_agent.py) for a more complex example.

The agent created with `create_deep_agent` is just a LangGraph graph - so you can interact with it (streaming, human-in-the-loop, memory, studio)
in the same way you would any LangGraph agent.

## Creating a custom deep agent

There are three parameters you can pass to `create_deep_agent` to create your own custom deep agent.

### `tools` (Required)

The first argument to `create_deep_agent` is `tools`.
This should be a list of functions or LangChain `@tool` objects.
The agent (and any subagents) will have access to these tools.

### `instructions` (Required)

The second argument to `create_deep_agent` is `instructions`.
This will serve as part of the prompt of the deep agent.
Note that there is a [built in system prompt](src/deepagents/prompts.py) as well, so this is not the *entire* prompt the agent will see.

### `subagents` (Optional)

A keyword-only argument to `create_deep_agent` is `subagents`.
This can be used to specify any custom subagents this deep agent will have access to.
You can read more about why you would want to use subagents [here](#sub-agents)

`subagents` should be a list of dictionaries, where each dictionary follow this schema:

```python
class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    model_settings: NotRequired[dict[str, Any]]
```

- **name**: This is the name of the subagent, and how the main agent will call the subagent
- **description**: This is the description of the subagent that is shown to the main agent
- **prompt**: This is the prompt used for the subagent
- **tools**: This is the list of tools that the subagent has access to. By default will have access to all tools passed in, as well as all built-in tools.
- **model_settings**: Optional dictionary for per-subagent model configuration (inherits the main model when omitted).

To use it looks like:

```python
research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "prompt": sub_research_prompt,
}
subagents = [research_subagent]
agent = create_deep_agent(
    tools,
    prompt,
    subagents=subagents
)
```

### `model` (Optional)

By default, `deepagents` uses `"claude-sonnet-4-20250514"`. You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

### `builtin_tools` (Optional)

By default, a deep agent will have access to a number of [built-in tools](#builtintools--optional-).
You can change this by specifying the tools (by name) that the agent should have access to with this parameter.

Example:
```python
# Only give agent access to todo tool, none of the filesystem tools
builtin_tools = ["write_todos"]
agent = create_deep_agent(..., builtin_tools=builtin_tools, ...)
```

#### Example: Using a Custom Model

Here's how to use a custom model (like OpenAI's `gpt-oss` model via Ollama):

(Requires `pip install langchain` and then `pip install langchain-ollama` for Ollama models)

```python
from deepagents import create_deep_agent

# ... existing agent definitions ...

model = init_chat_model(
    model="ollama:gpt-oss:20b",  
)
agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    model=model,
    ...
)
```

#### Example: Per-subagent model override (optional)

Use a fast, deterministic model for a critique sub-agent, while keeping a different default model for the main agent and others:

```python
from deepagents import create_deep_agent

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Critique the final report",
    "prompt": "You are a tough editor.",
    "model_settings": {
        "model": "anthropic:claude-3-5-haiku-20241022",
        "temperature": 0,
        "max_tokens": 8192
    }
}

agent = create_deep_agent(
    tools=[internet_search],
    instructions="You are an expert researcher...",
    model="claude-sonnet-4-20250514",  # default for main agent and other sub-agents
    subagents=[critique_sub_agent],
)
```

## Deep Agent Details

The below components are built into `deepagents` and helps make it work for deep tasks off-the-shelf.

### System Prompt

`deepagents` comes with a [built-in system prompt](src/deepagents/prompts.py). This is relatively detailed prompt that is heavily based on and inspired by [attempts](https://github.com/kn1026/cc/blob/main/claudecode.md) to [replicate](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md)
Claude Code's system prompt. It was made more general purpose than Claude Code's system prompt.
This contains detailed instructions for how to use the built-in planning tool, file system tools, and sub agents.
Note that part of this system prompt [can be customized](#instructions-required)

Without this default system prompt - the agent would not be nearly as successful at going as it is.
The importance of prompting for creating a "deep" agent cannot be understated.

### Planning Tool

`deepagents` comes with a built-in planning tool. This planning tool is very simple and is based on ClaudeCode's TodoWrite tool.
This tool doesn't actually do anything - it is just a way for the agent to come up with a plan, and then have that in the context to help keep it on track.

### File System Tools

`deepagents` comes with four built-in file system tools: `ls`, `edit_file`, `read_file`, `write_file`.
These do not actually use a file system - rather, they mock out a file system using LangGraph's State object.
This means you can easily run many of these agents on the same machine without worrying that they will edit the same underlying files.

Right now the "file system" will only be one level deep (no sub directories).

These files can be passed in (and also retrieved) by using the `files` key in the LangGraph State object.

```python
agent = create_deep_agent(...)

result = agent.invoke({
    "messages": ...,
    # Pass in files to the agent using this key
    # "files": {"foo.txt": "foo", ...}
})

# Access any files afterwards like this
result["files"]
```

### Sub Agents

`deepagents` comes with the built-in ability to call sub agents (based on Claude Code).
It has access to a `general-purpose` subagent at all times - this is a subagent with the same instructions as the main agent and all the tools that is has access to.
You can also specify [custom sub agents](#subagents-optional) with their own instructions and tools.

Sub agents are useful for ["context quarantine"](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html#context-quarantine) (to help not pollute the overall context of the main agent)
as well as custom instructions.

### Built In Tools

By default, deep agents come with five built-in tools:

- `write_todos`: Tool for writing todos
- `write_file`: Tool for writing to a file in the virtual filesystem
- `read_file`: Tool for reading from a file in the virtual filesystem
- `ls`: Tool for listing files in the virtual filesystem
- `edit_file`: Tool for editing a file in the virtual filesystem

These can be disabled via the [`builtin_tools`](#builtintools--optional-) parameter.

### Human-in-the-Loop

`deepagents` supports human-in-the-loop approval for tool execution. You can configure specific tools to require human approval before execution using the `interrupt_config` parameter, which maps tool names to `HumanInterruptConfig`.

`HumanInterruptConfig` is how you specify what type of human in the loop patterns are supported. 
It is a dictionary with four specific keys:

- `allow_ignore`: Whether the user can skip the tool call
- `allow_respond`: Whether the user can add a text response
- `allow_edit`: Whether the user can edit the tool arguments
- `allow_accept`: Whether the user can accept the tool call

Currently, `deepagents` does NOT support `allow_ignore`

Currently, `deepagents` only support interrupting one tool at a time. If multiple tools are called in parallel, each requiring interrupts, then the agent will error.

Instead of specifying a `HumanInterruptConfig` for a tool, you can also just set `True`. This will set `allow_ignore`, `allow_respond`, and `allow_edit` to be `True`.

In order to use human in the loop, you need to have a checkpointer attached.
Note: if you are using LangGraph Platform, this is automatically attached.

Example usage:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

# Create agent with file operations requiring approval
agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your instructions here",
    interrupt_config={
        # You can specify a dictionary for fine grained control over what interrupt options exist
        "tool_1": {
            "allow_ignore": False,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept":True,
        },
        # You can specify a boolean for shortcut
        # This is a shortcut for the same functionality as above
        "tool_2": True,
    }
)

checkpointer= InMemorySaver()
agent.checkpointer = checkpointer
```

#### Approve

To "approve" a tool call means the agent will execute the tool call as is.

This flow shows how to approve a tool call (assuming the tool requiring approval is called):

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
for s in agent.stream(Command(resume=[{"type": "accept"}]), config=config):
    print(s)

```

#### Edit

To "edit" a tool call means the agent will execute the new tool with the new arguments. You can change both the tool to call or the arguments to pass to that tool.

The `args` parameter you pass back should be a dictionary with two keys:

- `action`: maps to a string which is the name of the tool to call
- `args`: maps to a dictionary which is the arguments to pass to the tool

This flow shows how to edit a tool call (assuming the tool requiring approval is called):

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
# Replace the `...` with the tool name you want to call, and the arguments
for s in agent.stream(Command(resume=[{"type": "edit", "args": {"action": "...", "args": {...}}}]), config=config):
    print(s)

```

#### Respond

To "respond" to a tool call means that tool is NOT called. Rather, a tool message is appended with the content you respond with, and the updated messages list is then sent back to the model.

The `args` parameter you pass back should be a string with your response.

This flow shows how to respond to a tool call (assuming the tool requiring approval is called):

```python
config = {"configurable": {"thread_id": "1"}}
for s in agent.stream({"messages": [{"role": "user", "content": message}]}, config=config):
    print(s)
# If this calls a tool with an interrupt, this will then return an interrupt
# Replace the `...` with the response to use all the ToolMessage content
for s in agent.stream(Command(resume=[{"type": "response", "args": "..."}]), config=config):
    print(s)

```
## Async

If you are passing async tools to your agent, you will want to `from deepagents import async_create_deep_agent`
## MCP

The `deepagents` library can be ran with MCP tools. This can be achieved by using the [Langchain MCP Adapter library](https://github.com/langchain-ai/langchain-mcp-adapters).

**NOTE:** will want to use `from deepagents import async_create_deep_agent` to use the async version of `deepagents`, since MCP tools are async

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
    agent = async_create_deep_agent(tools=mcp_tools, ....)

    # Stream the agent
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is langgraph?"}]},
        stream_mode="values"
    ):
        if "messages" in chunk:
            chunk["messages"][-1].pretty_print()

asyncio.run(main())
```

## Configurable Agent

Configurable agents allow you to control the agent via a config passed in.

```python
from deepagents import create_configurable_agent

agent_config = {"instructions": "foo", "subagents": []}

build_agent = create_configurable_agent(
    agent_config['instructions'],
    agent_config['subagents'],
    [],
    agent_config={"recursion_limit": 1000}
)
```
You can now use `build_agent` in your `langgraph.json` and deploy it with `langgraph dev`

For async tools, you can use `from deepagents import async_create_configurable_agent`


## Roadmap
- [ ] Allow users to customize full system prompt
- [ ] Code cleanliness (type hinting, docstrings, formating)
- [ ] Allow for more of a robust virtual filesystem
- [ ] Create an example of a deep coding agent built on top of this
- [ ] Benchmark the example of [deep research agent](examples/research/research_agent.py)
