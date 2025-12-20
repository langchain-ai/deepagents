# GitHub Copilot Instructions for ChATLAS DeepAgents

This file provides GitHub Copilot with project-specific guidance for the ChATLAS DeepAgents repository.

## Project Context

**What is this project?**
A fork of LangChain's DeepAgents library extended to integrate with ChATLAS (CERN ATLAS experiment documentation system). The project uses MCP (Model Context Protocol) to connect AI agents with ATLAS knowledge bases.

**Architecture:**
- **Monorepo** with three modules: deepagents, deepagents-cli, chatlas-agents
- **No circular dependencies**: deepagents ← deepagents-cli ← chatlas-agents
- **Middleware-based extensions**: MCPMiddleware provides MCP server support
- **ChATLAS-specific code**: Lives in `libs/chatlas-agents/`

## Quick Reference for Copilot

### Code Locations

| Component | Location | Modify? |
|-----------|----------|---------|
| Base framework | `libs/deepagents/` | ❌ Avoid (upstream) |
| CLI layer | `libs/deepagents-cli/` | ❌ Avoid (upstream) |
| ChATLAS extensions | `libs/chatlas-agents/` | ✅ Yes (main development) |
| MCP middleware | `libs/chatlas-agents/chatlas_agents/middleware/mcp.py` | ✅ Yes |
| Configuration | `libs/chatlas-agents/chatlas_agents/config/` | ✅ Yes |
| MCP client | `libs/chatlas-agents/chatlas_agents/mcp/` | ✅ Yes |
| Documentation | `.github/*.md` | ✅ Yes (keep updated) |

### Common Patterns

#### MCP Middleware Usage
```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Async initialization
mcp_config = MCPServerConfig(url="...", timeout=60)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
)
```

#### Configuration Loading
```python
from chatlas_agents.config import Settings, AgentConfig

# From environment variables
settings = Settings()

# From YAML file
with open("config.yaml") as f:
    config = AgentConfig.from_yaml(f.read())
```

#### Tool Loading
```python
from chatlas_agents.mcp import create_mcp_client_and_load_tools

# Load MCP tools
tools = await create_mcp_client_and_load_tools(mcp_config)

# Or use middleware (preferred)
mcp_middleware = await MCPMiddleware.create(mcp_config)
agent = create_deep_agent(middleware=[mcp_middleware])
```

### Type Signatures

When suggesting code, use these common types:

```python
from typing import List, Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

# Configuration
class MCPServerConfig(BaseModel):
    url: str = Field(..., description="MCP server URL")
    timeout: int = Field(30, description="Timeout in seconds")

# Agent creation
async def create_agent(
    config: AgentConfig,
    tools: Optional[List[BaseTool]] = None,
) -> CompiledStateGraph:
    ...
```

### Code Style Guidelines

**Python Standards:**
- Python 3.11+ features allowed
- Type hints required for public APIs
- Docstrings required (Google style preferred)
- snake_case for functions/variables
- PascalCase for classes
- UPPER_CASE for constants

**Example Function:**
```python
async def load_mcp_tools(config: MCPServerConfig) -> List[BaseTool]:
    """Load tools from MCP server using LangChain adapters.
    
    Args:
        config: MCP server configuration containing URL and timeout
        
    Returns:
        List of LangChain tools loaded from MCP server
        
    Raises:
        ConnectionError: If connection to MCP server fails
        TimeoutError: If server doesn't respond within timeout
        
    Example:
        >>> config = MCPServerConfig(url="https://mcp.example.com", timeout=30)
        >>> tools = await load_mcp_tools(config)
        >>> print(f"Loaded {len(tools)} tools")
    """
    try:
        tools = await create_mcp_client_and_load_tools(config)
        logger.info(f"Loaded {len(tools)} tools from {config.url}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}", exc_info=True)
        raise
```

### Testing Patterns

When adding new code, suggest tests:

```python
import pytest
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig

@pytest.mark.asyncio
async def test_mcp_middleware_initialization():
    """Test that MCPMiddleware initializes correctly."""
    config = MCPServerConfig(url="http://test-server/mcp", timeout=10)
    middleware = await MCPMiddleware.create(config)
    
    assert middleware.tools is not None
    assert len(middleware.tools) > 0
    
@pytest.mark.asyncio
async def test_mcp_middleware_with_agent():
    """Test MCPMiddleware integration with DeepAgents."""
    from deepagents import create_deep_agent
    
    config = MCPServerConfig(url="http://test-server/mcp", timeout=10)
    middleware = await MCPMiddleware.create(config)
    
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        middleware=[middleware],
    )
    
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "test"}]
    })
    assert result is not None
```

### Common Tasks

#### Adding a New LLM Provider
1. Add to `LLMProvider` enum in `chatlas_agents/config/__init__.py`
2. Implement in `create_llm_from_config()` in `chatlas_agents/llm/__init__.py`
3. Add tests in `tests/test_llm.py`
4. Update README.md

#### Adding Configuration Options
1. Add field to `Settings` class in `chatlas_agents/config/__init__.py`
2. Add to `AgentConfig` if needed
3. Document environment variable in README.md
4. Add validation if needed

#### Adding New Middleware
1. Create in `chatlas_agents/middleware/your_middleware.py`
2. Inherit from `AgentMiddleware`
3. Implement required methods: `before_agent()`, optionally `wrap_model_call()`
4. Export from `chatlas_agents/middleware/__init__.py`
5. Add tests
6. Document usage

### Error Handling Patterns

Always include proper error handling:

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def risky_operation(param: str) -> Optional[str]:
    """Perform operation that might fail."""
    try:
        result = await external_api_call(param)
        logger.info(f"Operation succeeded: {result}")
        return result
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}", exc_info=True)
        return None
    except TimeoutError as e:
        logger.error(f"Operation timed out: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
```

### Async/Await Patterns

This project uses async extensively. Remember:

```python
# Correct: All MCP operations are async
tools = await create_mcp_client_and_load_tools(config)
middleware = await MCPMiddleware.create(config)
result = await agent.ainvoke(input_data)

# Correct: CLI entry point
async def main():
    agent = await create_agent()
    result = await agent.ainvoke(...)
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Incorrect: Forgetting await
tools = create_mcp_client_and_load_tools(config)  # Returns coroutine!
```

### Import Organization

Organize imports in this order:

```python
# Standard library
import asyncio
import logging
from typing import List, Optional, Dict, Any

# Third-party
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field
import typer

# Local/relative
from chatlas_agents.config import Settings, MCPServerConfig
from chatlas_agents.mcp import create_mcp_client_and_load_tools
from chatlas_agents.middleware import MCPMiddleware
```

### Common Pitfalls to Avoid

1. **Don't close MCP sessions early**: Tools need active sessions
2. **Always await async calls**: Especially MCP operations
3. **Validate environment variables**: API keys, URLs must exist
4. **Log errors with context**: Use `exc_info=True` for debugging
5. **Don't modify upstream code**: Keep changes in `chatlas-agents/`

### Environment Variables

Common environment variables in this project:

```bash
# LLM Configuration
CHATLAS_LLM_PROVIDER="openai"        # or "anthropic", "groq"
CHATLAS_LLM_API_KEY="sk-..."
CHATLAS_LLM_MODEL="gpt-4-turbo"

# MCP Configuration
CHATLAS_MCP_URL="https://chatlas-mcp.app.cern.ch/mcp"
CHATLAS_MCP_TIMEOUT="120"

# Optional
TAVILY_API_KEY="..."                  # For web search
ANTHROPIC_API_KEY="..."
OPENAI_API_KEY="..."
```

### Documentation Updates

When making code changes, also update:

- Function/class docstrings
- README.md if changing user-facing features
- `.github/*.md` for architectural changes
- `libs/chatlas-agents/AGENTS.md` for detailed agent instructions
- Type hints for all public APIs

### File Structure Guidelines

```
libs/chatlas-agents/chatlas_agents/
├── __init__.py              # Module exports
├── middleware/
│   ├── __init__.py          # Middleware exports
│   └── mcp.py               # MCPMiddleware implementation
├── config/
│   ├── __init__.py          # Settings, configs
│   └── models.py            # Pydantic models
├── mcp/
│   ├── __init__.py          # MCP client functions
│   └── client.py            # Client implementation
├── llm/
│   ├── __init__.py          # LLM factory
│   └── providers.py         # Provider implementations
└── tools/
    ├── __init__.py          # Tool exports
    └── loaders.py           # Tool loading utilities
```

### Testing Commands

Suggest these for testing:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_mcp_middleware.py -v

# Run with coverage
uv run pytest --cov=chatlas_agents

# Manual testing
uv run python -m chatlas_agents.cli run --input "test" --verbose
```

### Debugging Tips

When suggesting debugging approaches:

```bash
# Check MCP connectivity
curl https://chatlas-mcp.app.cern.ch/mcp

# Verbose logging
uv run python -m chatlas_agents.cli run --input "query" --verbose

# Check environment
echo $CHATLAS_MCP_URL
echo $CHATLAS_LLM_API_KEY

# Inspect tools
uv run python -c "
import asyncio
from chatlas_agents.config import MCPServerConfig
from chatlas_agents.mcp import create_mcp_client_and_load_tools

async def main():
    config = MCPServerConfig(url='https://chatlas-mcp.app.cern.ch/mcp')
    tools = await create_mcp_client_and_load_tools(config)
    for tool in tools:
        print(f'{tool.name}: {tool.description}')

asyncio.run(main())
"
```

## Summary for Copilot

When working on this repository:

1. **Prefer changes in `libs/chatlas-agents/`** - avoid upstream modifications
2. **Use middleware pattern** - follow MCPMiddleware example
3. **All MCP is async** - always await
4. **Type hints required** - for all public APIs
5. **Test thoroughly** - unit tests + manual CLI testing
6. **Document changes** - code, README, and .github/ docs
7. **Follow existing patterns** - consistency is key
8. **Log with context** - helpful error messages

## Resources

- **Detailed agent instructions**: `AGENTS.md` (root), `libs/chatlas-agents/AGENTS.md`
- **MCP integration**: `.github/MCP_INTEGRATION.md`
- **Dependencies**: `.github/DEPENDENCY_ANALYSIS.md`
- **Setup**: `libs/chatlas-agents/SETUP.md`
- **Main docs**: `README.md`

---

**Last Updated**: December 2024  
**Project Status**: Production Ready
