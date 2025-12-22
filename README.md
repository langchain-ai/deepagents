# ChATLAS Agents

This repository is a fork of the LangChain `deepagents` library, modified to integrate with ChATLAS. Extends the functionality of deep agents in the following ways:
- **Native MCP Support**: MCPMiddleware for seamless integration with Model Context Protocol servers without modifying upstream packages
- **ChATLAS MCP** search ChATLAS vector stores by connecting to the MCP server.
- **ATLAS software** compatible through SetupATLAS (on Lxplus).
- **HTCondor integration** submit agent sandboxes to the HTCondor batch farm.

ChATLAS-specific features can be found in `libs/chatlas-agents`. 

## MCP Server Integration

DeepAgents v0.3.0 does not provide native MCP server support. We've extended it with a **middleware-based approach** that:
- ✅ Requires **zero changes** to upstream packages (deepagents, deepagents-cli)
- ✅ Provides **full lifecycle integration** with tool loading, system prompt injection, and state management
- ✅ Is **composable** with other middleware (Skills, Memory, Shell)
- ✅ Maintains **forward compatibility** with future deepagents versions

### Quick Start

```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Create MCP middleware
mcp_config = MCPServerConfig(url="https://chatlas-mcp.app.cern.ch/mcp", timeout=60)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent with MCP support
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
)
```

## ChATLAS CLI Usage

The `chatlas` command-line interface provides an interactive AI assistant with access to ChATLAS MCP tools for searching ATLAS documentation and resources.

### Installation

```bash
# Install from repository
cd libs/chatlas-agents
pip install -e .

# Or install with uv (recommended)
uv pip install -e .
```

### Quick Start

Simply run `chatlas` to start an interactive session:

```bash
chatlas
```

This launches an interactive agent session with:
- **ChATLAS MCP tools** for searching ATLAS documentation
- **DeepAgents capabilities** (file operations, planning, sub-agents)
- **Skills system** for custom tools
- **Memory** for conversation persistence
- **Human-in-the-loop** approval for destructive operations

### Configuration

Initialize a configuration file with your API keys:

```bash
chatlas init
```

This creates a `.env` file. Edit it to add your API keys:

```bash
# .env
CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
CHATLAS_MCP_TIMEOUT=120

CHATLAS_LLM_PROVIDER=openai
CHATLAS_LLM_MODEL=gpt-5-mini

OPENAI_API_KEY=your-api-key-here
```

Load the configuration:

```bash
export $(cat .env | xargs)
chatlas
```

### Usage Examples

**Basic interactive session:**
```bash
chatlas
```

**Use a custom agent name (for separate memory):**
```bash
chatlas --agent my-research-agent
```

**Override MCP server:**
```bash
chatlas --mcp-url https://custom-mcp.example.com/mcp
```

**Use a different model:**
```bash
chatlas --model gpt-5-mini
```

**Enable Docker sandbox for isolated code execution:**
```bash
chatlas --sandbox docker
```

**Use Apptainer sandbox (for HPC environments like lxplus):**
```bash
chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim
```

**Auto-approve all tool calls (non-interactive mode):**
```bash
chatlas --auto-approve
```

**Enable verbose logging:**
```bash
chatlas --verbose
```

**Use YAML configuration file:**
```bash
chatlas --config my-config.yaml
```

### Sandbox Execution

ChATLAS supports isolated code execution in containers:

- **Docker sandbox**: Uses Docker containers for code execution
- **Apptainer sandbox**: Uses Apptainer/Singularity (ideal for HPC environments like CERN lxplus)

Sandbox execution provides:
- Isolated environment for running code
- Secure execution boundaries
- Support for custom container images
- File upload/download capabilities

Example with Apptainer on lxplus:
```bash
# SSH to lxplus
ssh lxplus.cern.ch

# Run ChATLAS with Apptainer sandbox
chatlas --sandbox apptainer --sandbox-image docker://python:3.13-slim
```

### CLI Commands

- **`chatlas`** - Start interactive session (default)
- **`chatlas init`** - Create configuration file
- **`chatlas version`** - Show version information
- **`chatlas --help`** - Show help for all options

### Documentation

**For Developers & AI Agents:**
- **[AGENTS.md](AGENTS.md)** - Quick reference for coding agents working on this repository
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - GitHub Copilot specific guidance

**Technical Documentation:**
- **[.github/MCP_INTEGRATION.md](.github/MCP_INTEGRATION.md)** - Comprehensive guide to MCP integration approaches and architecture
- **[.github/MCP_APPROACHES_COMPARISON.md](.github/MCP_APPROACHES_COMPARISON.md)** - Quick comparison of different integration strategies
- **[.github/DEPENDENCY_ANALYSIS.md](.github/DEPENDENCY_ANALYSIS.md)** - Module dependency analysis and setup
- **[.github/IMPLEMENTATION_SUMMARY_MCP.md](.github/IMPLEMENTATION_SUMMARY_MCP.md)** - MCP implementation summary

**Examples:**
- **[examples/mcp_middleware_example.py](libs/chatlas-agents/examples/mcp_middleware_example.py)** - Working example with deepagents
- **[examples/mcp_cli_integration_example.py](libs/chatlas-agents/examples/mcp_cli_integration_example.py)** - CLI integration patterns

**Module Documentation:**
- **[libs/chatlas-agents/README.md](libs/chatlas-agents/README.md)** - ChATLAS agents module documentation
- **[libs/chatlas-agents/SETUP.md](libs/chatlas-agents/SETUP.md)** - Detailed setup instructions 

### ATLAS Software Tools Skills

ChATLAS includes specialized skills for working with ATLAS experiment software tools on LXPlus:

- **[AMI Query](libs/chatlas-agents/examples/skills/ami-query/SKILL.md)** - Query ATLAS Metadata Interface for dataset information and metadata
- **[Rucio Management](libs/chatlas-agents/examples/skills/rucio-management/SKILL.md)** - Download and manage ATLAS grid data using Rucio DDM
- **[ATLAS Run Query](libs/chatlas-agents/examples/skills/atlas-runquery/SKILL.md)** - Query run information, data quality, and luminosity records

**Overview:** See [ATLAS_SKILLS.md](libs/chatlas-agents/examples/skills/ATLAS_SKILLS.md) for detailed documentation on using these skills.

These skills provide guidance for:
- Finding and downloading ATLAS datasets from the grid
- Querying dataset metadata and production information
- Managing data quality and run selection for physics analysis
- Working with distributed data management (Rucio)

**Prerequisites:** Users must initialize their ATLAS environment in their shell **before** starting the agent:
```bash
setupATLAS
lsetup pyami              # For AMI queries
localSetupRucioClients    # For Rucio data management
voms-proxy-init -voms atlas
```

**Note:** Not all commands are needed for all skills. See individual skill prerequisites for details.

The skills are designed to work on the CERN LXPlus cluster with the full ATLAS software stack available via CVMFS. The agent can verify it's on LXPlus by checking `echo $HOSTNAME` (should match `lxplus*.cern.ch`).

## TODO list for ChatLAS Agents
### v0.3
- [x] Fix timeout issues with MCP server -- increased timeout client side and provided more pods on the server. Should be able to handle many concurrent requests now and return answers more quickly.
- [ ] Fix known bugs:
  - [ ] Agent seems to get stuck sometimes when using MCP tools in interactive mode. Needs investigation.
  - [x] Not all tools seem to be available / configured properly with the chatlas agent. Web search tool seems to be missing, for example. Fixed by modifying MCPMiddleware and adding web search tools to CLI.
- [ ] Properly set up docker and apptainer sandbox. 
  - [x] Sandboxes set up with new CLI and MCP middleware.
  - [ ] Need to understand how to handle file transfers between host and sandbox. Implement this. 
  - [ ] Set up and test HTCondor submission.
  - [x] Alternative container solution: set up registry with chatlas-deepagents packages pre-installed, mount workdir into sandbox & tell agent to copy files there. -> Docker container has been set up on gitlab (`gitlab-registry.cern.ch/asopio/chatlas-deepagents/chatlas_deepagents`). Can be run with either docker (`docker runn -it`) or apptainer (`apptainer shell --docker-login`).
- [ ] Interface with ATLAS software stack. Create local MCP, tools for ATLAS data sources: AMI, Rucio, Upcoming indico meetings
  - [x] Simple, preliminary solution: use deepagents skills to wrap command line tools that access ATLAS data sources.
  - [ ] Longer term: create proper MCP server with tools for ATLAS data sources (can interface this with other agent providers eg. Copilot).

### v0.4+
- [ ] Add GitLab remote. Set up CI/CD. Would be cool to have agents running in GitLab runners, eg. to produce automated reviews of paper latex sources.


--- 

# 🚀🧠 Deep Agents

Agents can increasingly tackle long-horizon tasks, [with agent task length doubling every 7 months](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)! But, long horizon tasks often span dozens of tool calls, which present cost and reliability challenges. Popular agents such as [Claude Code](https://code.claude.com/docs) and [Manus](https://www.youtube.com/watch?v=6_BcCthVvb8) use some common principles to address these challenges, including **planning** (prior to task execution), **computer access** (giving the agent access to a shell and a filesystem), and **sub-agent delegation** (isolated task execution). `deepagents` is a simple agent harness that implements these tools, but is open source and easily extendable with your own custom tools and instructions.

<img src=".github/images/deepagents_banner.png" alt="deep agent" width="100%"/>

## 📚 Resources

- **[Documentation](https://docs.langchain.com/oss/python/deepagents/overview)** - Full overview and API reference
- **[Quickstarts Repo](https://github.com/langchain-ai/deepagents-quickstarts)** - Examples and use-cases
- **[CLI](libs/deepagents-cli/)** - Interactive command-line interface with skills, memory, and HITL workflows

## 🚀 Quickstart

You can give `deepagents` custom tools. Below, we'll optionally provide the `tavily` tool to search the web. This tool will be added to the `deepagents` build-in tools (see below).

```bash
pip install deepagents tavily-python
```

Set `TAVILY_API_KEY` in your environment ([get one here](https://www.tavily.com/)):

```python
import os
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5):
    """Run a web search"""
    return tavily_client.search(query, max_results=max_results)

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt="Conduct research and write a polished report.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph?"}]})
```

The agent created with `create_deep_agent` is compiled [LangGraph StateGraph](https://docs.langchain.com/oss/python/langgraph/overview), so it can used it with streaming, human-in-the-loop, memory, or Studio just like any LangGraph agent. See our [quickstarts repo](https://github.com/langchain-ai/deepagents-quickstarts) for more examples.

## Customizing Deep Agents

There are several parameters you can pass to `create_deep_agent`.

### `model`

By default, `deepagents` uses `"claude-sonnet-4-5-20250929"`. You can customize this by passing any [LangChain model object](https://python.langchain.com/docs/integrations/chat/).

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model("openai:gpt-5-mini")
agent = create_deep_agent(
    model=model,
)
```

### `system_prompt`

You can provide a `system_prompt` parameter to `create_deep_agent()`. This custom prompt is **appended to** default instructions that are automatically injected by middleware.

When writing a custom system prompt, you should:

- ✅ Define domain-specific workflows (e.g., research methodology, data analysis steps)
- ✅ Provide concrete examples for your use case
- ✅ Add specialized guidance (e.g., "batch similar research tasks into a single TODO")
- ✅ Define stopping criteria and resource limits
- ✅ Explain how tools work together in your workflow

**Don't:**

- ❌ Re-explain what standard tools do (already covered by middleware)
- ❌ Duplicate middleware instructions about tool usage
- ❌ Contradict default instructions (work with them, not against them)

```python
from deepagents import create_deep_agent
research_instructions = """your custom system prompt"""
agent = create_deep_agent(
    system_prompt=research_instructions,
)
```

See our [quickstarts repo](https://github.com/langchain-ai/deepagents-quickstarts) for more examples.

### `tools`

Provide custom tools to your agent (in addition to [Built-in Tools](#built-in-tools)):

```python
from deepagents import create_deep_agent

def internet_search(query: str) -> str:
    """Run a web search"""
    return tavily_client.search(query)

agent = create_deep_agent(tools=[internet_search])
```

You can also connect MCP tools via [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters):

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()
    agent = create_deep_agent(tools=mcp_tools)

    async for chunk in agent.astream({"messages": [{"role": "user", "content": "..."}]}):
        chunk["messages"][-1].pretty_print()
```

### `middleware`

Deep agents use [middleware](https://docs.langchain.com/oss/python/langchain/middleware) for extensibility (see [Built-in Tools](#built-in-tools) for defaults). Add custom middleware to inject tools, modify prompts, or hook into the agent lifecycle:

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

class WeatherMiddleware(AgentMiddleware):
    tools = [get_weather]

agent = create_deep_agent(middleware=[WeatherMiddleware()])
```

### `subagents`

The main agent can delegate work to sub-agents via the `task` tool (see [Built-in Tools](#built-in-tools)). You can supply custom sub-agents for context isolation and custom instructions:

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "research-agent",
    "description": "Used to research in-depth questions",
    "prompt": "You are an expert researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5-mini",  # Optional, defaults to main agent model
}

agent = create_deep_agent(subagents=[research_subagent])
```

For complex cases, pass a pre-built LangGraph graph:

```python
from deepagents import CompiledSubAgent, create_deep_agent

custom_graph = create_agent(model=..., tools=..., prompt=...)

agent = create_deep_agent(
    subagents=[CompiledSubAgent(
        name="data-analyzer",
        description="Specialized agent for data analysis",
        runnable=custom_graph
    )]
)
```

See the [subagents documentation](https://docs.langchain.com/oss/python/deepagents/subagents) for more details.

### `interrupt_on`

Some tools may be sensitive and require human approval before execution. Deepagents supports human-in-the-loop workflows through LangGraph’s interrupt capabilities. You can configure which tools require approval using a checkpointer.

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

See the [human-in-the-loop documentation](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) for more details.

### `backend`

Deep agents use pluggable backends to control how filesystem operations work. By default, files are stored in the agent's ephemeral state. You can configure different backends for local disk access, persistent cross-conversation storage, or hybrid routing.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

Available backends include:

- **StateBackend** (default): Ephemeral files stored in agent state
- **FilesystemBackend**: Real disk operations under a root directory
- **StoreBackend**: Persistent storage using LangGraph Store
- **CompositeBackend**: Route different paths to different backends

See the [backends documentation](https://docs.langchain.com/oss/python/deepagents/backends) for more details.

### Long-term Memory

Deep agents can maintain persistent memory across conversations using a `CompositeBackend` that routes specific paths to durable storage.

This enables hybrid memory where working files remain ephemeral while important data (like user preferences or knowledge bases) persists across threads.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

Files under `/memories/` will persist across all conversations, while other paths remain temporary. Use cases include:

- Preserving user preferences across sessions
- Building knowledge bases from multiple conversations
- Self-improving instructions based on feedback
- Maintaining research progress across sessions

See the [long-term memory documentation](https://docs.langchain.com/oss/python/deepagents/long-term-memory) for more details.

## Built-in Tools

<img src=".github/images/deepagents_tools.png" alt="deep agent" width="600"/>

Every deep agent created with `create_deep_agent` comes with a standard set of tools:

| Tool Name | Description | Provided By |
|-----------|-------------|-------------|
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows | TodoListMiddleware |
| `read_todos` | Read the current todo list state | TodoListMiddleware |
| `ls` | List all files in a directory (requires absolute path) | FilesystemMiddleware |
| `read_file` | Read content from a file with optional pagination (offset/limit parameters) | FilesystemMiddleware |
| `write_file` | Create a new file or completely overwrite an existing file | FilesystemMiddleware |
| `edit_file` | Perform exact string replacements in files | FilesystemMiddleware |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) | FilesystemMiddleware |
| `grep` | Search for text patterns within files | FilesystemMiddleware |
| `execute`* | Run shell commands in a sandboxed environment | FilesystemMiddleware |
| `task` | Delegate tasks to specialized sub-agents with isolated context windows | SubAgentMiddleware |

The `execute` tool is only available if the backend implements `SandboxBackendProtocol`. By default, it uses the in-memory state backend which does not support command execution. As shown, these tools (along with other capabilities) are provided by default middleware:

See the [agent harness documentation](https://docs.langchain.com/oss/python/deepagents/harness) for more details on built-in tools and capabilities.

## Built-in Middleware

`deepagents` uses middleware under the hood. Here is the list of the middleware used.

| Middleware | Purpose |
|------------|---------|
| **TodoListMiddleware** | Task planning and progress tracking |
| **FilesystemMiddleware** | File operations and context offloading (auto-saves large results) |
| **SubAgentMiddleware** | Delegate tasks to isolated sub-agents |
| **SummarizationMiddleware** | Auto-summarizes when context exceeds 170k tokens |
| **AnthropicPromptCachingMiddleware** | Caches system prompts to reduce costs (Anthropic only) |
| **PatchToolCallsMiddleware** | Fixes dangling tool calls from interruptions |
| **HumanInTheLoopMiddleware** | Pauses execution for human approval (requires `interrupt_on` config) |

## Built-in prompts

The middleware automatically adds instructions about the standard tools. Your custom instructions should **complement, not duplicate** these defaults:

#### From [TodoListMiddleware](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/middleware/todo.py)

- Explains when to use `write_todos` and `read_todos`
- Guidance on marking tasks completed
- Best practices for todo list management
- When NOT to use todos (simple tasks)

#### From [FilesystemMiddleware](libs/deepagents/deepagents/middleware/filesystem.py)

- Lists all filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*)
- Explains that file paths must start with `/`
- Describes each tool's purpose and parameters
- Notes about context offloading for large tool results

#### From [SubAgentMiddleware](libs/deepagents/deepagents/middleware/subagents.py)

- Explains the `task()` tool for delegating to sub-agents
- When to use sub-agents vs when NOT to use them
- Guidance on parallel execution
- Subagent lifecycle (spawn → run → return → reconcile)
