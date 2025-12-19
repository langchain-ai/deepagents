# GitHub Copilot Instructions for ChATLAS Agent

This file provides instructions for GitHub Copilot agents working on the ChATLAS agent project.

## Project Overview

The ChATLAS agent is an AI-powered assistant that integrates with ATLAS experiment documentation through the Model Context Protocol (MCP). It combines GPT-4-Turbo with real-time access to ATLAS knowledge bases.

**Key Components:**
- LLM: OpenAI GPT-4-Turbo
- Framework: DeepAgents + LangGraph
- Protocol: MCP (JSON-RPC over HTTP)
- Server: chatlas-mcp.app.cern.ch/mcp

## Codebase Structure

```
chatlas-agents/
├── chatlas_agents/
│   ├── agents/          # Agent wrapper and initialization
│   ├── config/          # Configuration management (Settings, AgentConfig)
│   ├── graph.py         # DeepAgent graph creation
│   ├── llm/             # LLM factory (OpenAI, Anthropic, Groq)
│   ├── mcp/             # MCP client integration (langchain-mcp-adapters)
│   ├── tools/           # Tool loading from MCP server
│   ├── cli.py           # Command-line interface (typer)
│   ├── sandbox.py       # Docker sandbox backend
│   └── __init__.py
├── .github/             # GitHub documentation and templates
│   ├── README.md        # Documentation hub
│   ├── QUICK_START.md   # 5-minute setup guide
│   ├── AGENT_INSTRUCTIONS.md  # Complete user reference
│   ├── CONTRIBUTING.md  # Development guidelines
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/             # Configuration files
├── examples/            # Example usage scripts
├── tests/               # Test suite
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── README.md            # Main project README
```

## Key Concepts for Copilot

### 1. MCP Integration
The agent connects to an MCP server that provides tools:
- **search_chatlas**: RAG-based ATLAS documentation search
- **dummy_tool**: Echo utility for testing

**Important:** The MCP server uses JSON-RPC over HTTP (not REST or SSE). Tools create sessions on-demand during invocation.

### 2. Agent Flow
```
User Input
  ↓
CLI (typer command)
  ↓
Config Loading (env vars or YAML)
  ↓
LLM Initialization (GPT-4-Turbo)
  ↓
MCP Tool Loading (via langchain-mcp-adapters)
  ↓
DeepAgent Creation (LangGraph state graph)
  ↓
Agent Invocation (ainvoke)
  ↓
Tool Execution (if needed)
  ↓
Response Generation
  ↓
User Output
```

### 3. Configuration Hierarchy
1. **Environment Variables** - `CHATLAS_*` prefix
2. **YAML Files** - Configuration files in `configs/`
3. **Code Defaults** - Hardcoded in `config/__init__.py`

### 4. Error Handling
- All MCP errors logged with full traceback
- API key validation on initialization
- Graceful tool invocation failures
- Proper async cleanup on shutdown

## Common Development Tasks

### Adding a New LLM Provider
1. Add enum to `LLMProvider` in `chatlas_agents/config/__init__.py`
2. Implement in `create_llm_from_config()` in `chatlas_agents/llm/__init__.py`
3. Add tests in `tests/test_llm.py`

### Adding Configuration Options
1. Add field to `Settings` class in `chatlas_agents/config/__init__.py`
2. Update environment variable parsing
3. Add to `AgentConfig` dataclass
4. Document in `README.md`

### Improving Tool Integration
1. Tools are discovered automatically from MCP server
2. No code changes needed for new MCP tools
3. Update documentation when new tools are available
4. Test tool invocation with `--verbose` flag

### Bug Fixes
1. Use `--verbose` flag to see detailed logs
2. Check MCP server connectivity first
3. Verify all async/await calls are correct
4. Test with multiple queries before submitting

## Testing Guidelines

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_agents.py -v

# Test with coverage
uv run pytest --cov=chatlas_agents

# Manual testing
uv run python -m chatlas_agents.cli run --input "test query" --verbose
```

### Test Checklist
- [ ] Single query mode works
- [ ] Interactive mode works
- [ ] Tools are discovered and loaded
- [ ] MCP server connectivity verified
- [ ] Error messages are clear
- [ ] Logging is appropriate
- [ ] No breaking changes to API

## Code Style

### Python Standards
- Python 3.11+ compatible
- Type hints for all public functions
- Docstrings for all classes and functions
- Snake_case for functions/variables
- PascalCase for classes
- Follow existing patterns in codebase

### Example Function
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

## Common Pitfalls

1. **Session Lifecycle Issues**
   - Don't close MCP sessions before tools execute
   - Tools create sessions on-demand, don't force pre-created sessions
   - Let the connection parameter handle session creation

2. **Async/Await Mistakes**
   - All MCP operations must be awaited
   - Use `asyncio.run()` for CLI entry points
   - Properly handle async context managers

3. **Configuration Loading**
   - Environment variables override defaults
   - Always validate API keys exist
   - Support both env vars and YAML files

4. **Error Messages**
   - Always log with `exc_info=True` for debugging
   - Include context in error messages
   - Check for common causes first (connectivity, auth)

5. **Tool Integration**
   - Don't modify tool loading for individual tools
   - Tools are discovered automatically from MCP server
   - Focus on improving the MCP client, not individual tool handling

## PR Review Checklist

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

## Documentation Links

For detailed information, refer to:
- **User Documentation**: [README.md](../README.md) - Complete guide for users
- **AI Agent Instructions**: [AGENTS.md](../AGENTS.md) - Instructions for AI agents
- **GitHub Templates**: Issue and PR templates in this directory

## Important Notes for Copilot

1. **MCP Server Endpoint**: `https://chatlas-mcp.app.cern.ch/mcp`
   - Requires HTTP transport (not SSE)
   - Uses JSON-RPC 2.0 protocol
   - Returns 200 OK on initialization

2. **Tool Discovery**: 
   - 2 tools available: `search_chatlas` and `dummy_tool`
   - Both are loaded via MCP server connection
   - No hardcoded tool definitions in codebase

3. **LLM Configuration**:
   - Default: OpenAI GPT-4-Turbo
   - Requires `CHATLAS_LLM_API_KEY` environment variable
   - Validates API key existence on initialization

4. **Async Operations**:
   - All MCP operations are async
   - Agent execution is async-based
   - CLI properly handles async-to-sync boundary

5. **Error Context**:
   - `ClosedResourceError` means session closed before tool execution
   - `CancelledError` means task was cancelled (timeout?)
   - Check MCP server connectivity for connection errors
   - Always check API key for auth errors

## When Making Changes

Before making changes:
1. Run tests: `uv run pytest`
2. Check for similar patterns in codebase
3. Verify async/await correctness
4. Test with verbose flag: `--verbose`
5. Update documentation if changing APIs

After making changes:
1. Ensure all tests pass
2. Manual test: single query, interactive, verbose modes
3. Update docs/docstrings
4. Check for breaking changes
5. Verify error messages are helpful

## Resources

- **LangChain Docs**: https://docs.langchain.com/
- **DeepAgents**: https://github.com/langchain-ai/deepagents
- **MCP Protocol**: https://modelcontextprotocol.io/
- **OpenAI API**: https://platform.openai.com/docs/
- **ATLAS Experiment**: https://atlas.cern/

---

**Last Updated**: December 15, 2025  
**Project Status**: Production Ready ✅
