# ChATLAS Agents

AI agent framework for the ChATLAS AI RAG system using DeepAgents and LangChain.

## Features

- 🚀 **DeepAgents Framework**: Built on LangChain's DeepAgents with planning, file operations, and sub-agent spawning
- 🔌 **Native MCP Support**: MCPMiddleware for seamless integration with Model Context Protocol servers
- 🐳 **Container Sandboxes**: Run agents in isolated containers for secure code execution (Docker & Apptainer)
- ⚡ **HTCondor Integration**: Submit batch jobs to CERN's HTCondor batch farm system
- 🧠 **Multiple LLM Backends**: Support for OpenAI, Anthropic Claude, and Groq
- 🔧 **Modular Configuration**: YAML-based configuration with environment variable overrides
- 💾 **Conversation Persistence**: Built-in checkpointing for conversation memory
- 🌊 **Streaming Support**: Stream agent responses in real-time
- 📦 **uv Package Management**: Reproducible dependency management with uv

## TODO
### v0.3
- [x] Fix timeout issues with MCP server -- increased timeout client side and provided more pods on the server. Should be able to handle many concurrent requests now and return answers more quickly.
- [ ] Fix known bugs:
  - [ ] Agent seems to get stuck sometimes when using MCP tools in interactive mode. Needs investigation.
  - [ ] Not all tools seem to be available / configured properly with the chatlas agent. Web search tool seems to be missing, for example.
- [ ] Properly set up docker and apptainer sandbox. 
  - [x] Sandboxes set up with new CLI and MCP middleware.
  - [ ] Need to understand how to handle file transfers between host and sandbox. Implement this. 
  - [ ] Set up and test HTCondor submission.
  - [x] Alternative container solution: set up registry with chatlas-deepagents packages pre-installed, mount workdir into sandbox & tell agent to copy files there. -> Docker container has been set up on gitlab (`gitlab-registry.cern.ch/asopio/chatlas-deepagents/chatlas_deepagents`). Can be run with either docker (`docker runn -it`) or apptainer (`apptainer shell --docker-login`).
- [ ] Interface with ATLAS software stack. Create local MCP, tools for ATLAS data sources: AMI, Rucio, Upcoming indico meetings
  - [ ] Simple, preliminary solution: use deepagents skills to wrap command line tools that access ATLAS data sources.
  - [ ] Longer term: create proper MCP server with tools for ATLAS data sources (can interface this with other agent providers eg. Copilot).

### v0.4+
- [ ] Add GitLab remote. Set up CI/CD. Would be cool to have agents running in GitLab runners, eg. to produce automated reviews of paper latex sources.

## Quick Start

### Installation with uv (Recommended)

```bash
# Install uv if not already installed
pip install uv

# Clone the repository
git clone https://github.com/asopio/chatlas-agents.git
cd chatlas-agents

# Sync dependencies with uv
uv sync

# Run commands with uv
uv run python -m chatlas_agents.cli --help
```

### Installation with pip

```bash
# Clone the repository
git clone https://github.com/asopio/chatlas-agents.git
cd chatlas-agents

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your API keys:
```bash
CHATLAS_LLM_PROVIDER=openai
CHATLAS_LLM_API_KEY=your-api-key-here
```

### Running an Agent

The ChATLAS CLI is built on top of DeepAgents CLI, providing an interactive coding assistant with ChATLAS MCP tools.

#### Interactive Mode (Recommended)

```bash
# Run with ChATLAS MCP tools
python -m chatlas_agents.cli run --interactive

# With custom agent name (for separate conversation memory)
python -m chatlas_agents.cli run --interactive --agent my-agent

# With auto-approve (no confirmation for tool execution)
python -m chatlas_agents.cli run --interactive --auto-approve

# With configuration file
python -m chatlas_agents.cli run --interactive --config my-agent.yaml
```

#### Interactive Features

The CLI provides a rich interactive experience:

- **Slash Commands**: Type `/help` to see available commands
  - `/clear` - Clear screen and reset conversation
  - `/tokens` - Show token usage
  - `/help` - Show help
  - `/quit` or `/exit` - Exit

- **Bash Commands**: Execute bash commands directly with `!`
  ```
  !ls -la
  !git status
  ```

- **File Operations**: The agent can read, write, and edit files
- **Code Execution**: Run code in various languages
- **Web Search**: Search the web (requires `TAVILY_API_KEY`)
- **ChATLAS MCP Tools**: Access ATLAS documentation and data

#### Configuration Options

```bash
# Override MCP server URL
python -m chatlas_agents.cli run --interactive --mcp-url https://custom-mcp.cern.ch/mcp

# Set MCP timeout
python -m chatlas_agents.cli run --interactive --mcp-timeout 180

# Enable verbose logging
python -m chatlas_agents.cli run --interactive --verbose
```

#### Using Environment Variables

```bash
# Create a configuration file
python -m chatlas_agents.cli init --output my-agent.yaml

# Edit my-agent.yaml with your settings
# Run the agent
python -m chatlas_agents.cli run --config my-agent.yaml --interactive
```

## Docker Usage

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t chatlas-agents .

# Run the agent
docker run -e CHATLAS_LLM_API_KEY=your-key chatlas-agents run --input "Hello"
```

### Using Docker Compose

```bash
# Set environment variables in .env file
# Start the agent service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## DeepAgents Features

DeepAgents provides advanced capabilities for long-horizon tasks:

### Planning and TODO Management

DeepAgents can break down complex tasks into manageable steps:

```python
from chatlas_agents.agents import create_deep_agent
from chatlas_agents.config import AgentConfig

config = AgentConfig(name="my-agent")
agent = await create_deep_agent(config)

# Agent automatically maintains a TODO list for complex tasks
result = await agent.run("""
    Research quantum computing developments and:
    1. Find top 3 papers from 2024
    2. Summarize key findings
    3. Write a comparative report
""", thread_id="research-session")
```

### MCP Server Integration

ChATLAS Agents provides native MCP (Model Context Protocol) support through the `MCPMiddleware`:

```python
from deepagents import create_deep_agent
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig

# Configure MCP server
mcp_config = MCPServerConfig(
    url="https://chatlas-mcp.app.cern.ch/mcp",
    timeout=60
)

# Create MCP middleware (loads tools from server)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent with MCP support
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
    system_prompt="You are a helpful assistant with access to ChATLAS tools.",
)

# Agent now has access to all MCP server tools
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Search ChATLAS documentation for information about..."}]
})
```

**Key Features:**
- ✅ **No upstream modifications** - Works with standard deepagents and deepagents-cli
- ✅ **Automatic tool loading** - Discovers and loads all tools from MCP server
- ✅ **System prompt injection** - Automatically documents available tools
- ✅ **Composable** - Works alongside other middleware (Skills, Shell, Memory)
- ✅ **Forward compatible** - Uses stable middleware API

See `examples/mcp_middleware_example.py` for a complete working example.
For detailed documentation on MCP integration approaches, see `MCP_INTEGRATION.md` in the repository root.

### Sub-Agent Delegation

Spawn isolated sub-agents for specialized subtasks:

```python
# DeepAgents can automatically spawn sub-agents
# for specific tasks and manage their execution
result = await agent.run("Analyze this dataset and create visualizations")
# Behind the scenes, DeepAgents may spawn sub-agents for:
# - Data analysis
# - Visualization generation
# - Report compilation
```

### File System Operations

Built-in mock file system for agent interactions:

```python
# Agents can create, read, and manage files
result = await agent.run("Create a report.md file with the research findings")
```

### Container Sandbox for Secure Execution

Run code safely in isolated containers. Apptainer is the default (rootless, HPC-friendly).

#### Using Apptainer Sandbox (default)

```python
from chatlas_agents.sandbox import SandboxBackendType

# Create agent with Apptainer sandbox (default)
agent = await create_deep_agent(
  config,
  use_docker_sandbox=True,
  docker_image="docker://python:3.13-slim",  # Apptainer can use Docker images
  sandbox_backend=SandboxBackendType.APPTAINER  # This is the default
)

# Agent can safely execute commands in Apptainer instance
result = await agent.run("""
    Create and execute a data analysis script
""", thread_id="apptainer-session")
```

#### Using Docker Sandbox

```python
from chatlas_agents.sandbox import SandboxBackendType

# Create agent with Docker sandbox
agent = await create_deep_agent(
  config,
  use_docker_sandbox=True,
  docker_image="python:3.13-slim",
  sandbox_backend=SandboxBackendType.DOCKER
)

# Agent can now safely execute shell commands and run code
result = await agent.run("""
    Write a Python script that analyzes data and execute it
""", thread_id="sandbox-session")
```

**Key differences:**
- **Apptainer** (default): Rootless, designed for HPC, no daemon required, can use Docker images with `docker://` prefix
- **Docker**: Requires Docker daemon, root access typically needed
- Both support isolation and secure execution

### Conversation Persistence

Agents automatically persist conversation history:

```python
# All messages in same thread maintain context
await agent.run("My name is Alice", thread_id="user-123")
await agent.run("What's my name?", thread_id="user-123")  # Remembers Alice
```

### Streaming Responses

Stream agent responses in real-time:

```python
async for event in agent.stream("Tell me a story", thread_id="session-1"):
    # Process streaming events
    print(event)
```

### Tool Integration

Tools from the MCP server are automatically integrated:

```python
# Tools are loaded and bound to the LLM automatically
# Agent can call tools as needed during conversation
result = await agent.run("Search for ChATLAS documentation")
```

## Configuration

### LLM Providers

#### OpenAI
```yaml
llm:
  provider: openai
  model: gpt-5-mini
  api_key: sk-...
  temperature: 0.7
```

#### Anthropic Claude
```yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: sk-ant-...
  temperature: 0.7
```

#### Groq
```yaml
llm:
  provider: groq
  model: llama-3.1-70b-versatile
  api_key: gsk_...
  temperature: 0.7
```

### MCP Server Configuration

```yaml
mcp:
  url: https://chatlas-mcp.app.cern.ch/mcp
  timeout: 30
  max_retries: 3
  headers: {}
```

## Project Structure

```
chatlas-agents/
├── chatlas_agents/
│   ├── __init__.py
│   ├── agents/          # Agent implementations
│   ├── config/          # Configuration management
│   ├── llm/             # LLM backend factory
│   ├── mcp/             # MCP client
│   ├── tools/           # Tool wrappers
│   ├── sandbox.py       # Docker sandbox backend
│   ├── htcondor.py      # HTCondor batch job submission
│   ├── graph.py         # DeepAgents graph definition
│   └── cli.py           # Command-line interface
├── configs/             # Example configurations
├── examples/            # Usage examples
├── tests/               # Test suite
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
└── README.md           # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black chatlas_agents/

# Check code style
ruff check chatlas_agents/
```

## CLI Commands

### Run Agent
```bash
python -m chatlas_agents.cli run [OPTIONS]

Options:
  -c, --config PATH          Configuration file path
  -i, --input TEXT           Input text to send to agent
  -I, --interactive          Run in interactive mode
  -t, --thread TEXT          Thread ID for conversation persistence
  --stream                   Enable streaming responses
  --sandbox                  Use container sandbox for secure execution
  --sandbox-image TEXT       Container image to use for sandbox
  --sandbox-backend TEXT     Sandbox backend: docker or apptainer [default: apptainer]
  -v, --verbose              Enable verbose logging
```

### Initialize Configuration
```bash
python -m chatlas_agents.cli init [OPTIONS]

Options:
  -o, --output PATH       Output configuration file path
```

### Version
```bash
python -m chatlas_agents.cli version
```

### HTCondor Batch Job Submission

Submit ChATLAS agent jobs to CERN's HTCondor batch farm system:

```bash
python -m chatlas_agents.cli htcondor-submit [OPTIONS]

Options:
  -n, --job-name TEXT         Name for the HTCondor batch job [required]
  -p, --prompt TEXT          Input prompt to send to the agent [required]
  -c, --config PATH          Configuration file path
  --sandbox-image TEXT       Container image to use for sandbox [default: python:3.13-slim]
  -o, --output-dir PATH      Directory for job output files [default: ./htcondor_jobs]
  -e, --env-file PATH        Path to .env file with environment variables
  --dry-run                  Generate submit file without submitting
  --cpus INTEGER             Number of CPUs to request [default: 1]
  --memory TEXT              Memory to request (e.g., 2GB, 4GB) [default: 2GB]
  --disk TEXT                Disk space to request (e.g., 1GB, 5GB) [default: 1GB]
  -v, --verbose              Enable verbose logging
```

#### HTCondor Usage Examples

```bash
# Submit a basic job
python -m chatlas_agents.cli htcondor-submit \
  --job-name analyze-data \
  --prompt "Analyze the latest CERN experimental data"

# Submit with custom resources and configuration
python -m chatlas_agents.cli htcondor-submit \
  --job-name complex-analysis \
  --prompt "Perform complex statistical analysis" \
  --config my-agent.yaml \
  --cpus 4 \
  --memory 8GB \
  --disk 10GB

# Submit with environment variables (API keys, etc.)
python -m chatlas_agents.cli htcondor-submit \
  --job-name secure-job \
  --prompt "Process sensitive data" \
  --env-file .env \
  --config configs/claude-config.yaml

# Dry run - generate submit file without submitting
python -m chatlas_agents.cli htcondor-submit \
  --job-name test-job \
  --prompt "Test prompt" \
  --dry-run

# Check job status after submission
condor_q <cluster-id>

# View job output
tail -f htcondor_jobs/analyze-data/job.<cluster-id>.*.out
```

For more information on HTCondor at CERN, see: https://batchdocs.web.cern.ch/local/submit.html

## Environment Variables

All configuration options can be set via environment variables with the `CHATLAS_` prefix:

- `CHATLAS_LLM_PROVIDER`: LLM provider (openai, anthropic, groq)
- `CHATLAS_LLM_MODEL`: Model name
- `CHATLAS_LLM_API_KEY`: API key for the provider
- `CHATLAS_LLM_BASE_URL`: Custom API endpoint (optional)
- `CHATLAS_LLM_TEMPERATURE`: Sampling temperature (0.0-2.0)
- `CHATLAS_MCP_URL`: MCP server URL
- `CHATLAS_MCP_TIMEOUT`: Request timeout in seconds
- `CHATLAS_AGENT_NAME`: Agent name
- `CHATLAS_AGENT_VERBOSE`: Enable verbose logging (true/false)
- `CHATLAS_AGENT_MAX_ITERATIONS`: Maximum agent iterations

## Examples

See the `examples/` directory for working examples:
- `deepagent_example.py`: Complete DeepAgent usage with planning and streaming
- `docker_sandbox_example.py`: Secure code execution with Docker sandbox
- `apptainer_sandbox_example.py`: Secure code execution with Apptainer sandbox (HPC environments)

Run an example:

```bash
python examples/deepagent_example.py

# For Apptainer (requires apptainer installed)
python examples/apptainer_sandbox_example.py
```

## Architecture

The ChATLAS agents framework is built on:

1. **DeepAgents**: Official LangChain package for advanced agent capabilities
   - Planning with TODO list management
   - File system operations
   - Sub-agent spawning and delegation
   - Built on LangGraph for flexible orchestration

2. **MCP Client**: Connects to ChATLAS MCP server for:
   - Dynamic tool discovery
   - Tool execution
   - Resource access

3. **LLM Factory**: Creates LLM instances for different providers:
   - OpenAI (GPT-5-mini, etc.)
   - Anthropic (Claude 3.5 Sonnet, etc.)
   - Groq (Llama 3.1, Mixtral, etc.)

4. **Container Sandbox**: Optional isolated execution environment:
   - Apptainer (default): HPC-optimized, rootless containers (ideal for lxplus)
   - Docker: Standard container isolation (requires Docker daemon)
   - Secure code execution
   - Custom container images
   - File operations in containers
   - Shell command execution

5. **Configuration System**: Flexible configuration via:
   - YAML files
   - Environment variables
   - Programmatic API

## License

[Add your license here]

## Contributing

We welcome contributions to the ChATLAS agent project!

### Reporting Bugs

1. Check existing issues first
2. Use the issue template at `.github/ISSUE_TEMPLATE.md`
3. Include:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment info (Python version, OS)
   - Output from `--verbose` flag

### Suggesting Features

1. Open a discussion describing the feature and use case
2. Provide examples of how it would be used
3. Consider alternatives
4. Get feedback before implementation

### Contributing Code

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/chatlas-agents.git
cd chatlas-agents

# Create a feature branch
git checkout -b feature/your-feature-name

# Install dependencies
uv sync

# Make your changes
# Add tests for new functionality
# Update documentation

# Run tests
uv run pytest

# Test manually
uv run python -m chatlas_agents.cli run --input "test" --verbose

# Submit a pull request using .github/PULL_REQUEST_TEMPLATE.md
```

#### Code Style

- Python 3.11+ compatible
- Type hints for public functions
- Docstrings for all classes and functions
- Follow existing code patterns
- Use snake_case for functions/variables, PascalCase for classes

## Support

For issues and questions, please open an issue on GitHub.
