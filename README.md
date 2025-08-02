# 🧠🤖Deep Agents

**Note: This is a modified version of [hwchase17/deepagents](https://github.com/hwchase17/deepagents?ref=blog.langchain.com) with additional features including local Ollama support and improved output handling.**

📚 **[DeepWiki Documentation](https://deepwiki.com/Cam10001110101/deepagents)** - Interactive documentation for this repository

Using an LLM to call tools in a loop is the simplest form of an agent. 
This architecture, however, can yield agents that are "shallow" and fail to plan and act over longer, more complex tasks. 
Applications like "Deep Research", "Manus", and "Claude Code" have gotten around this limitation by implementing a combination of four things:
a **planning tool**, **sub agents**, access to a **file system**, and a **detailed prompt**.

<img src="deep_agents.png" alt="deep agent" width="600"/>

`deepagents` is a Python package that implements these in a general purpose way so that you can easily create a Deep Agent for your application.

**Acknowledgements: This project was primarily inspired by Claude Code, and initially was largely an attempt to see what made Claude Code general purpose, and make it even more so.**

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package and project manager:

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-username/deepagents.git
cd deepagents

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment (optional, UV handles this automatically)
source .venv/bin/activate
```

### Using pip

```bash
pip install deepagents
```

## Model Context Protocol (MCP) Integration 🆕

DeepAgents can now use tools from external MCP servers! This allows your agents to access tools from:
- Claude Desktop's tool ecosystem
- File system operations via MCP servers
- Database queries, web search, and more

MCP integration is available as a separate package for clean dependency management:

```bash
# Install MCP integration
pip install deepagents[mcp]

# Or from source
cd extensions/mcp && pip install -e .
```

**Quick Example:**
```bash
# Run with MCP tools
python -m deepagents --mcp-config examples/mcp/mcp_config_all.json
```

See `extensions/mcp/` directory for full documentation, and `examples/mcp/` for examples.

For a complete example of using MCP tools with DeepAgents, see [examples/mcp/deepagents_with_mcp.py](examples/mcp/deepagents_with_mcp.py).

## Usage

(To run the example below, will need to `pip install tavily-python`)

```python
import os
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_async_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return tavily_async_client.search(
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

### Running the Research Agent Example

To run the research agent example:

#### Using UV (Recommended)

```bash
# Create and configure .env file
cp .env.example .env  # Then edit with your API keys
# Or set environment variables directly:
export TAVILY_API_KEY="your-tavily-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Optional: Enable LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="your-project-name"

# Run the research agent with UV
uv run python run_research_agent.py "Your research question here"

# Example:
uv run python run_research_agent.py "What are the latest developments in quantum computing?"
```

#### Using pip

```bash
# Install dependencies
pip install deepagents tavily-python

# Set your API keys
export TAVILY_API_KEY="your-tavily-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # or configure your preferred LLM

# Optional: Enable LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="your-project-name"

# Run the research agent
python run_research_agent.py "Your research question here"

# Example:
python run_research_agent.py "What are the latest developments in quantum computing?"
```

### Running with Local Ollama Models

You can also run the research agent using local Ollama models instead of cloud-based APIs:

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull a model
ollama pull qwen3:latest

# Configure Ollama settings (or add to .env)
export OLLAMA_MODEL="qwen3:latest"              # Model to use
export OLLAMA_HOST="http://localhost:11434" # Ollama server URL
export OLLAMA_TEMPERATURE="0.7"             # Temperature (0.0-1.0)

# Run the local research agent (with UV)
uv run python run_research_agent_local.py "Your research question here"

# Or without UV:
python run_research_agent_local.py "Your research question here"
```

The local version uses the same research capabilities but runs entirely on your machine using Ollama.

### Running Local Research Agent with MCP Integration

For the most advanced local research experience with full MCP (Model Context Protocol) integration:

```bash
# Ensure Ollama is running and model is pulled
ollama pull llama3.1:latest  # or your preferred model

# Set environment variable for the model
export OLLAMA_MODEL="llama3.1:latest"

# Run the research agent with MCP Phase 5 integration
uv run python run_research_agent_local_mcp.py "Your research question here"

# Example research queries:
python run_research_agent_local_mcp.py "Analyze the enterprise integration landscape for AI platforms in 2024"
python run_research_agent_local_mcp.py "What are the best practices for implementing multi-cloud service orchestration?"
python run_research_agent_local_mcp.py "Compare enterprise CRM platforms and their API integration capabilities"
```

This version includes **MCP Phase 5 Integration** which provides:
- **Phase 1 Foundation**: Filesystem, DuckDuckGo search, time utilities
- **Phase 2 Knowledge & Memory**: Enhanced filesystem for knowledge storage and retrieval
- **Phase 3 Development & Code**: GitHub integration and code analysis
- **Phase 4 AI & Research**: Advanced search capabilities and AI tools
- **Phase 5 Integration & Services**: Enterprise cloud services, databases, APIs, workflow automation, business intelligence, CRM/ERP integrations, and real-time messaging

The research output is automatically saved to `output-examples/` directory with timestamps.

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
Note that there is a [built in system prompt](#built-in-prompt) as well, so this is not the *entire* prompt the agent will see.

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
```

- **name**: This is the name of the subagent, and how the main agent will call the subagent
- **description**: This is the description of the subagent that is shown to the main agent
- **prompt**: This is the prompt used for the subagent
- **tools**: This is the list of tools that the subagent has access to. By default will have access to all tools passed in, as well as all built-in tools.

To use it looks like:

```python
research_sub_agent = {
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

By default, `deepagents` will use `"claude-sonnet-4-20250514"`. If you want to use a different model,
you can pass a [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

## Deep Agent Details

The below components are built into `deepagents` and helps make it work for deep tasks off-the-shelf.

### System Prompt

`deepagents` comes with a [built-in system prompt](src/deepagents/prompts.py). This is relatively detailed prompt that is heavily based on and inspired by [attempts](https://github.com/kn1026/cc/blob/main/claudecode.md) to [replicate](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-code.md)
Claude Code's system prompt. It was made more general purpose than Claude Code's system prompt.
This contains detailed instructions for how to use the built-in planning tool, file system tools, and sub agents.
Note that part of this system prompt [can be customized](#promptprefix--required-)

Without this default system prompt - the agent would not be nearly as successful at going as it is.
The importance of prompting for creating a "deep" agent cannot be understated.

### Planing Tool

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
You can also specify [custom sub agents](#subagents--optional-) with their own instructions and tools.

Sub agents are useful for ["context quarantine"](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html#context-quarantine) (to help not pollute the overall context of the main agent)
as well as custom instructions.

## Roadmap
[] Allow users to customize full system prompt
[] Code cleanliness (type hinting, docstrings, formating)
[] Allow for more of a robust virtual filesystem
[] Create an example of a deep coding agent built on top of this
[] Benchmark the example of [deep research agent](examples/research/research_agent.py)
[] Add human-in-the-loop support for tools

## LangSmith Integration

Deep Agents automatically supports LangSmith tracing when the appropriate environment variables are set:

```bash
# Enable LangSmith tracing
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_PROJECT="your-project-name"
```

Once configured, all agent executions will be traced in LangSmith, allowing you to:
- Debug agent reasoning and tool calls
- Monitor performance and costs
- Analyze agent behavior patterns
- Share traces with your team
