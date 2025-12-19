# AI Agent Instructions for ChATLAS Agent Project

This document provides comprehensive instructions for AI agents (GitHub Copilot, Claude, GPT, etc.) working on the ChATLAS agent codebase.

## Project Overview

The ChATLAS agent is an AI-powered assistant that integrates with ATLAS experiment documentation through the Model Context Protocol (MCP). It combines GPT-4-Turbo with real-time access to ATLAS knowledge bases.

**Key Components:**
- **LLM**: OpenAI GPT-4-Turbo (configurable)
- **Framework**: DeepAgents + LangGraph
- **Protocol**: MCP (JSON-RPC over HTTP)
- **Server**: chatlas-mcp.app.cern.ch/mcp
- **Tools**: RAG-based ATLAS documentation search

## Codebase Structure

```
chatlas-agents/
├── chatlas_agents/          # Main source code
│   ├── agents/              # Agent wrapper and initialization
│   ├── config/              # Configuration management (Settings, AgentConfig)
│   ├── graph.py             # DeepAgent graph creation
│   ├── llm/                 # LLM factory (OpenAI, Anthropic, Groq)
│   ├── mcp/                 # MCP client integration (langchain-mcp-adapters)
│   ├── tools/               # Tool loading from MCP server
│   ├── cli.py               # Command-line interface (typer)
│   ├── sandbox.py           # Docker sandbox backend
│   ├── htcondor.py          # HTCondor batch job submission
│   └── __init__.py
├── .github/                 # GitHub templates and AI instructions
│   ├── copilot-instructions.md  # GitHub Copilot specific
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/                 # Configuration files
├── examples/                # Example usage scripts
├── tests/                   # Test suite
├── AGENTS.md                # This file
└── README.md                # User documentation
```

## Critical Technical Details

### MCP Integration

The agent connects to an MCP server using JSON-RPC over HTTP (NOT REST or SSE).

**Important:**
- MCP server endpoint: `https://chatlas-mcp.app.cern.ch/mcp`
- Tools create sessions on-demand during invocation
- Don't close MCP sessions before tools execute
- All MCP operations are async and must be awaited

**Available Tools:**
- `search_chatlas`: RAG-based ATLAS documentation search
- `dummy_tool`: Echo utility for testing

### Agent Flow

```
User Input → CLI → Config Loading → LLM Init → MCP Tool Loading 
→ DeepAgent Creation → Agent Invocation → Tool Execution → Response
```

### Configuration Hierarchy

1. **Environment Variables** - `CHATLAS_*` prefix
2. **YAML Files** - Configuration files in `configs/`
3. **Code Defaults** - Hardcoded in `config/__init__.py`

### Error Handling

- All MCP errors logged with full traceback
- API key validation on initialization
- Graceful tool invocation failures
- Proper async cleanup on shutdown

**Common Error Patterns:**
- `ClosedResourceError` = session closed prematurely
- `CancelledError` = task was cancelled (timeout?)
- Connection errors = check MCP server connectivity
- Auth errors = check API key

## Development Guidelines

### Before Making Changes

```bash
# Run existing tests
uv run pytest

# Manual test current functionality
uv run python -m chatlas_agents.cli run --input "test" --verbose

# Check related code patterns
grep -r "your_concept" chatlas_agents/
```

### Common Development Tasks

#### Adding a New LLM Provider

**Files to modify:**
1. `chatlas_agents/config/__init__.py` - Add provider enum
2. `chatlas_agents/llm/__init__.py` - Implement provider factory
3. `tests/test_llm.py` - Add provider tests
4. `README.md` - Document new provider

**Test afterward:**
```bash
uv run pytest tests/test_llm.py -v
uv run python -m chatlas_agents.cli run --input "test"
```

#### Adding Configuration Options

**Files to modify:**
1. `chatlas_agents/config/__init__.py` - Add to Settings class
2. Update environment variable documentation in README.md
3. Add to `AgentConfig` dataclass

**Validation:**
```bash
export CHATLAS_YOUR_OPTION="value"
uv run python -m chatlas_agents.cli run --input "test"
```

#### Improving Tool Integration

1. Tools are discovered automatically from MCP server
2. No code changes needed for new MCP tools
3. Update documentation when new tools are available
4. Test tool invocation with `--verbose` flag

#### Fixing Bugs

**Process:**
1. Reproduce with `--verbose` flag
2. Check MCP server connectivity
3. Verify async/await correctness
4. Add test case before fix
5. Verify fix with verbose mode
6. Run full test suite

**Example:**
```bash
# Reproduce
uv run python -m chatlas_agents.cli run --input "query" --verbose

# Check connectivity
curl https://chatlas-mcp.app.cern.ch/mcp

# Fix code, then test
uv run pytest tests/ -v
```

### Code Style Guidelines

**Python Standards:**
- Python 3.11+ compatible
- Type hints for all public functions
- Docstrings for all classes and functions
- Snake_case for functions/variables
- PascalCase for classes
- Follow existing patterns in codebase

**Example Function:**
```python
async def load_mcp_tools(config: MCPServerConfig) -> List[BaseTool]:
    """Load tools from MCP server using LangChain adapters.

    Args:
        config: MCP server configuration

    Returns:
        List of LangChain tools loaded from MCP server

    Raises:
        Exception: If connection to MCP server fails
    """
    try:
        tools = await create_mcp_client_and_load_tools(config)
        logger.info(f"Successfully loaded {len(tools)} tools from MCP server")
        return tools
    except Exception as e:
        logger.error(f"Failed to load tools from MCP server: {e}", exc_info=True)
        raise
```

### Testing Strategy

**Unit Tests:**
```bash
uv run pytest tests/test_llm.py     # LLM factory tests
uv run pytest tests/test_config.py  # Configuration tests
uv run pytest tests/test_mcp.py     # MCP client tests
```

**Integration Tests:**
```bash
# Single query
uv run python -m chatlas_agents.cli run --input "What is ATLAS?"

# Tool invocation
uv run python -m chatlas_agents.cli run --input "Search for photon calibration"

# Interactive mode
uv run python -m chatlas_agents.cli run --interactive
```

**Debugging:**
```bash
# Verbose output
uv run python -m chatlas_agents.cli run --input "query" --verbose

# Check MCP server
curl https://chatlas-mcp.app.cern.ch/mcp

# Check API key
echo $CHATLAS_LLM_API_KEY
```

### Test Checklist

Before submitting changes:
- [ ] Single query mode works
- [ ] Interactive mode works
- [ ] Tools are discovered and loaded
- [ ] MCP server connectivity verified
- [ ] Error messages are clear
- [ ] Logging is appropriate
- [ ] No breaking changes to API

## Key Components

### Configuration Management (`chatlas_agents/config/`)

- `Settings` class: Pydantic settings from environment
- `AgentConfig`: Complete agent configuration dataclass
- `LLMConfig`: LLM provider configuration
- `MCPServerConfig`: MCP server connection settings

### LLM Factory (`chatlas_agents/llm/`)

Creates LLM instances from configuration.

**Supported Providers:**
- OpenAI (GPT-4-Turbo, GPT-3.5-Turbo)
- Anthropic (Claude 3.5 Sonnet)
- Groq (Llama 3.1, Mixtral)

### MCP Integration (`chatlas_agents/mcp/`)

Handles connection to ChATLAS MCP server using `langchain-mcp-adapters`.

**Key Points:**
- Uses HTTP transport (not SSE)
- Creates sessions on-demand for tool invocation
- Connects to `https://chatlas-mcp.app.cern.ch/mcp`

### Agent Graph (`chatlas_agents/graph.py`)

Uses DeepAgents + LangGraph for agent orchestration.

**Built-in Capabilities:**
- Planning and TODO management
- File system operations
- Tool invocation
- Sub-agent spawning

### Tools (`chatlas_agents/tools/`)

Tool loading and management via MCP.

**Available Tools:**
- `search_chatlas`: RAG-based ATLAS documentation search
- `dummy_tool`: Echo utility for testing

## Common Pitfalls

### 1. Session Lifecycle Issues

- Don't close MCP sessions before tools execute
- Tools create sessions on-demand, don't force pre-created sessions
- Let the connection parameter handle session creation

### 2. Async/Await Mistakes

- All MCP operations must be awaited
- Use `asyncio.run()` for CLI entry points
- Properly handle async context managers

### 3. Configuration Loading

- Environment variables override defaults
- Always validate API keys exist
- Support both env vars and YAML files

### 4. Error Messages

- Always log with `exc_info=True` for debugging
- Include context in error messages
- Check for common causes first (connectivity, auth)

### 5. Tool Integration

- Don't modify tool loading for individual tools
- Tools are discovered automatically from MCP server
- Focus on improving the MCP client, not individual tool handling

## Dependencies

### Core Dependencies

- `langchain-core>=0.1.0` - LLM and tool abstractions
- `langchain-mcp-adapters>=0.1.0` - Official MCP integration
- `deepagents` - Agent framework with built-in tools
- `langchain-openai>=0.1.0` - OpenAI LLM provider
- `python-dotenv` - Environment variable management
- `pydantic` - Configuration validation
- `pydantic-settings` - Settings management
- `typer>=0.9.0` - CLI framework
- `rich` - Formatted console output

### Development Dependencies

- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `ruff` - Python linter/formatter

## Pull Request Review Checklist

When reviewing PRs for this project:

- [ ] Code follows style guidelines
- [ ] Type hints present for public APIs
- [ ] Docstrings complete and accurate
- [ ] Tests added for new functionality
- [ ] All existing tests pass
- [ ] No breaking changes to public APIs
- [ ] Error handling is appropriate
- [ ] Logging is useful and not excessive
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

## Performance Considerations

### Expected Timings

| Operation | Time |
|-----------|------|
| MCP server connection | 3-5 seconds |
| Tool discovery | Included in connection |
| LLM inference | 5-15 seconds |
| Tool invocation | 15-30 seconds |
| Total simple query | 10-20 seconds |

### Optimization Strategies

1. **Batch queries** - Use interactive mode for related questions
2. **Thread persistence** - Reuse threads to maintain context
3. **Streaming** - Enable `--stream` for real-time responses
4. **Specific vectorstores** - Use appropriate knowledge bases

## Troubleshooting Guide

### "API key is required"
```bash
export CHATLAS_LLM_API_KEY="sk-..."
```

### "Failed to connect to MCP server"
```bash
# Check connectivity
curl https://chatlas-mcp.app.cern.ch/mcp

# Run with verbose
uv run python -m chatlas_agents.cli run --input "test" --verbose
```

### "Tool invocation timeout"
```bash
export CHATLAS_MCP_TIMEOUT=60
uv run python -m chatlas_agents.cli run --input "query" --verbose
```

### Tests Failing
```bash
# Clean up
rm -rf .venv __pycache__ .pytest_cache

# Reinstall
uv sync

# Run tests
uv run pytest -v
```

## External Resources

- **LangChain Docs**: https://docs.langchain.com/
- **DeepAgents**: https://github.com/langchain-ai/deepagents
- **MCP Protocol**: https://modelcontextprotocol.io/
- **OpenAI API**: https://platform.openai.com/docs/
- **ATLAS Experiment**: https://atlas.cern/

## Summary for AI Agents

**This is a LangChain/DeepAgents-based AI agent that uses MCP tools to answer ATLAS experiment questions.**

**When working on this project:**

1. ✅ Run tests before and after changes
2. ✅ Use `--verbose` for debugging
3. ✅ Check MCP connectivity if tools fail
4. ✅ Verify API key is set
5. ✅ Follow code patterns in existing files
6. ✅ Update documentation with code changes
7. ✅ Handle async/await correctly
8. ✅ Log errors with full context

**Documentation Structure:**
- **README.md** - Complete user documentation
- **AGENTS.md** - This file (AI agent instructions)
- **.github/copilot-instructions.md** - GitHub Copilot specific guidance

---

**Last Updated**: December 15, 2025  
**Project Status**: Production Ready ✅
