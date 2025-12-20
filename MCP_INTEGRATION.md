# MCP Server Integration for DeepAgents

This document describes the different approaches for integrating Model Context Protocol (MCP) server support into deepagents and deepagents-cli while maintaining forward compatibility.

## Background

DeepAgents v0.3.0 does not provide native MCP server support. However, LangChain provides the `langchain-mcp-adapters` library that enables loading tools from MCP servers. This document explores different integration approaches that:

1. **Minimize changes** to upstream packages (deepagents, deepagents-cli)
2. **Maintain forward compatibility** with future versions
3. **Keep customizations** in `libs/chatlas-agents`
4. **Follow best practices** from the LangChain/DeepAgents ecosystem

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    chatlas-agents                        │
│  (Custom Integration Layer - Our Code)                   │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  MCPMiddleware (NEW)                            │    │
│  │  - Loads tools from MCP server                  │    │
│  │  - Provides tools to deepagents                 │    │
│  │  - Injects tool docs into system prompt        │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                                │
└─────────────────────────┼────────────────────────────────┘
                          ↓
┌─────────────────────────┼────────────────────────────────┐
│              deepagents-cli (Upstream)                    │
│  - CLI interface                                          │
│  - Skills, Memory, Shell middleware                       │
│  - create_cli_agent() factory                             │
└─────────────────────────┼────────────────────────────────┘
                          ↓
┌─────────────────────────┼────────────────────────────────┐
│                deepagents (Upstream)                      │
│  - create_deep_agent() factory                            │
│  - Middleware architecture                                │
│  - Built-in: TodoList, Filesystem, SubAgent middleware    │
└───────────────────────────────────────────────────────────┘
```

## Integration Approaches

### Approach 1: Middleware-based Integration ✅ RECOMMENDED

**Status**: Implemented in `libs/chatlas-agents/chatlas_agents/middleware/mcp.py`

#### Description
Create a custom `MCPMiddleware` in the `chatlas-agents` package that follows the standard deepagents middleware pattern. This approach requires **no changes** to deepagents or deepagents-cli.

#### Implementation

```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Create MCP middleware
mcp_config = MCPServerConfig(
    url="https://chatlas-mcp.app.cern.ch/mcp",
    timeout=30
)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Use with deepagents
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],  # Add MCP middleware
)
```

#### How It Works

1. **MCPMiddleware** inherits from `AgentMiddleware`
2. At initialization, it loads tools from the MCP server using `langchain-mcp-adapters`
3. The middleware exposes tools via its `tools` property (automatically picked up by deepagents)
4. Optionally injects tool descriptions into the system prompt
5. Works seamlessly with other middleware (Skills, Shell, Memory, etc.)

#### Advantages

- ✅ **Zero changes** to deepagents or deepagents-cli
- ✅ **Full control** over MCP integration logic
- ✅ **Composable** with other middleware
- ✅ **Easy to maintain** - all MCP logic in one place
- ✅ **Forward compatible** - uses standard middleware API
- ✅ **Testable** - can be tested independently

#### Disadvantages

- ⚠️ Requires async initialization (`await MCPMiddleware.create()`)
- ⚠️ Users need to explicitly add the middleware

#### File Structure

```
libs/chatlas-agents/
├── chatlas_agents/
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── mcp.py          # MCPMiddleware implementation
│   ├── mcp/
│   │   └── __init__.py     # MCP client utilities (existing)
│   └── config/
│       └── __init__.py     # MCPServerConfig (existing)
```

---

### Approach 2: Optional Dependency with Runtime Detection

**Status**: Alternative approach (not implemented)

#### Description
Add MCP support as an optional feature in deepagents-cli that checks for `langchain-mcp-adapters` availability at runtime.

#### Implementation Concept

```python
# In deepagents-cli/deepagents_cli/integrations/mcp_support.py

def has_mcp_support() -> bool:
    """Check if MCP adapters are available."""
    try:
        import langchain_mcp_adapters
        return True
    except ImportError:
        return False

async def load_mcp_tools_if_available(config: dict) -> list[BaseTool]:
    """Load MCP tools if langchain-mcp-adapters is installed."""
    if not has_mcp_support():
        return []
    
    from langchain_mcp_adapters.client import MultiServerMCPClient
    # ... load tools
    return tools

# In create_cli_agent()
if mcp_config and has_mcp_support():
    mcp_tools = await load_mcp_tools_if_available(mcp_config)
    tools.extend(mcp_tools)
```

#### Advantages

- ✅ Graceful degradation if MCP adapters not installed
- ✅ Could be contributed back to deepagents-cli

#### Disadvantages

- ❌ Requires modifying deepagents-cli
- ❌ Harder to maintain across upstream updates
- ❌ Less control over integration details
- ❌ MCP logic scattered across files

---

### Approach 3: Tools Parameter Pattern

**Status**: Alternative approach (already working in chatlas-agents)

#### Description
Load MCP tools in chatlas-agents and pass them as regular tools to `create_deep_agent()` or `create_cli_agent()`.

#### Implementation

```python
from chatlas_agents.mcp import create_mcp_client_and_load_tools
from deepagents import create_deep_agent

# Load MCP tools
mcp_tools = await create_mcp_client_and_load_tools(mcp_config)

# Pass as regular tools
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=mcp_tools,  # Just regular tools
)
```

#### Advantages

- ✅ Simple and straightforward
- ✅ Already working in chatlas-agents
- ✅ No middleware complexity

#### Disadvantages

- ❌ No system prompt injection for tool documentation
- ❌ Tools not tracked in agent state
- ❌ Less integration with agent lifecycle
- ❌ Can't leverage middleware hooks (before_agent, wrap_model_call, etc.)

---

### Approach 4: Custom Agent Factory

**Status**: Alternative approach (not recommended)

#### Description
Create a custom `create_chatlas_agent()` factory that wraps `create_deep_agent()` with MCP support.

#### Implementation Concept

```python
async def create_chatlas_agent(
    model,
    tools=None,
    mcp_config=None,
    **kwargs
):
    """Create a deep agent with MCP support."""
    all_tools = list(tools) if tools else []
    
    if mcp_config:
        mcp_tools = await create_mcp_client_and_load_tools(mcp_config)
        all_tools.extend(mcp_tools)
    
    return create_deep_agent(
        model=model,
        tools=all_tools,
        **kwargs
    )
```

#### Advantages

- ✅ Simple API
- ✅ Hides MCP complexity

#### Disadvantages

- ❌ Creates another abstraction layer
- ❌ Duplicates create_deep_agent() interface
- ❌ Harder to keep in sync with upstream changes
- ❌ Less flexible than middleware approach

---

## Recommended Implementation: Middleware Pattern

The **Middleware-based Integration (Approach 1)** is the recommended approach because:

1. **No upstream modifications required** - All code stays in `chatlas-agents`
2. **Standard pattern** - Follows the same pattern as Skills, Shell, Memory middleware
3. **Composable** - Works alongside other middleware
4. **Full lifecycle integration** - Hooks into agent initialization and model calls
5. **Forward compatible** - Uses stable middleware API
6. **Easy to maintain** - Single file with clear responsibilities

### Usage Examples

#### With DeepAgents

```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Configure MCP server
mcp_config = MCPServerConfig(
    url="https://chatlas-mcp.app.cern.ch/mcp",
    timeout=30
)

# Create MCP middleware
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent with MCP support
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
    system_prompt="You are a helpful assistant with access to ChATLAS tools.",
)

# Run the agent
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Search ChATLAS documentation"}]
})
```

#### With DeepAgents-CLI (Future Enhancement)

To integrate with deepagents-cli, we could add an optional parameter to `create_cli_agent()`:

```python
# In chatlas-agents wrapper around create_cli_agent
from deepagents_cli.agent import create_cli_agent as upstream_create_cli_agent
from chatlas_agents.middleware import MCPMiddleware

async def create_cli_agent(
    model,
    assistant_id,
    mcp_config=None,
    **kwargs
):
    """Create CLI agent with optional MCP support."""
    middleware = kwargs.get('middleware', [])
    
    if mcp_config:
        mcp_middleware = await MCPMiddleware.create(mcp_config)
        middleware.append(mcp_middleware)
        kwargs['middleware'] = middleware
    
    return upstream_create_cli_agent(
        model=model,
        assistant_id=assistant_id,
        **kwargs
    )
```

Or simply use the middleware directly:

```python
from deepagents_cli.agent import create_cli_agent
from chatlas_agents.middleware import MCPMiddleware

# Create MCP middleware
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Add to middleware list
agent, backend = create_cli_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    assistant_id="my-agent",
    tools=[],
)

# Manually compose with MCP middleware
# (This would require deepagents-cli to support external middleware injection)
```

## Implementation Status

### ✅ Completed

1. **MCPMiddleware** (`libs/chatlas-agents/chatlas_agents/middleware/mcp.py`)
   - Async factory method for initialization
   - Tool loading from MCP server
   - System prompt injection
   - State management
   - Lifecycle hooks

2. **Documentation** (this file)
   - Comprehensive analysis of approaches
   - Usage examples
   - Architecture diagrams

### 🔄 Recommended Next Steps

1. **Testing**
   - Unit tests for MCPMiddleware
   - Integration tests with MCP server
   - Test composition with other middleware

2. **Examples**
   - Example script using MCPMiddleware with deepagents
   - Example integration with deepagents-cli
   - Configuration examples

3. **CLI Integration**
   - Add MCP configuration options to CLI
   - Document CLI usage patterns
   - Add to chatlas-agents CLI wrapper

4. **Documentation Updates**
   - Update README with MCP middleware examples
   - Add migration guide for existing users
   - Document best practices

## Testing

### Unit Tests

```python
import pytest
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig

@pytest.mark.asyncio
async def test_mcp_middleware_initialization():
    config = MCPServerConfig(url="http://test-server/mcp", timeout=10)
    middleware = await MCPMiddleware.create(config)
    
    assert middleware.tools is not None
    assert len(middleware.tools) > 0

@pytest.mark.asyncio
async def test_mcp_middleware_system_prompt():
    config = MCPServerConfig(url="http://test-server/mcp", timeout=10)
    middleware = await MCPMiddleware.create(config)
    
    # Test prompt injection
    tools_desc = middleware._format_tools_description()
    assert "MCP Tools" in tools_desc
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_mcp_middleware_with_deepagents():
    from deepagents import create_deep_agent
    
    mcp_config = MCPServerConfig(url="http://test-server/mcp", timeout=10)
    mcp_middleware = await MCPMiddleware.create(mcp_config)
    
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        middleware=[mcp_middleware],
    )
    
    # Verify agent has MCP tools
    # Run simple task
    result = await agent.ainvoke({"messages": [...]})
    assert result is not None
```

## Conclusion

The **Middleware-based Integration** provides the best balance of:
- **Minimal changes** to upstream packages
- **Forward compatibility** with future versions
- **Full control** over MCP integration
- **Standard patterns** familiar to deepagents users
- **Maintainability** with clear separation of concerns

All MCP-specific code lives in `libs/chatlas-agents`, ensuring we can easily maintain and extend the functionality without being tightly coupled to upstream packages.
