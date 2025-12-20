# MCP Integration Approaches Comparison

This document provides a quick comparison of different approaches to integrate MCP server support into deepagents/deepagents-cli.

## Quick Comparison Table

| Approach | Upstream Changes | Complexity | Maintainability | Flexibility | Recommended |
|----------|------------------|------------|-----------------|-------------|-------------|
| **1. MCPMiddleware** | None | Low | High | High | ✅ **YES** |
| 2. Optional Dependency | Moderate | Medium | Medium | Medium | ⚠️ Maybe |
| 3. Tools Parameter | None | Very Low | High | Low | ✅ For Simple Cases |
| 4. Custom Factory | None | Medium | Low | Medium | ❌ No |

## Detailed Comparison

### Approach 1: MCPMiddleware ✅ RECOMMENDED

**Files**: `libs/chatlas-agents/chatlas_agents/middleware/mcp.py`

```python
# Usage
mcp_middleware = await MCPMiddleware.create(mcp_config)
agent = create_deep_agent(middleware=[mcp_middleware])
```

**Pros:**
- ✅ Zero changes to upstream packages
- ✅ Full lifecycle integration (before_agent, wrap_model_call hooks)
- ✅ System prompt injection for tool documentation
- ✅ Composable with other middleware
- ✅ State management for tools
- ✅ Easy to test independently
- ✅ Forward compatible with future deepagents versions

**Cons:**
- ⚠️ Requires async initialization
- ⚠️ Users need to understand middleware concept

**Best For:**
- Production deployments
- Full-featured integration
- Composing with other middleware (Skills, Memory, Shell)

---

### Approach 2: Optional Dependency Pattern

**Files**: Would require changes to `deepagents-cli`

```python
# Hypothetical usage
agent = create_cli_agent(
    model=model,
    mcp_config=mcp_config,  # Optional parameter
)
```

**Pros:**
- ✅ Graceful degradation if MCP not installed
- ✅ Could be contributed upstream
- ✅ Simple API for users

**Cons:**
- ❌ Requires modifying deepagents-cli
- ❌ Harder to maintain across upstream updates
- ❌ Less control over integration
- ❌ Violates our "minimal changes" requirement

**Best For:**
- If contributing back to upstream
- Not recommended for our use case

---

### Approach 3: Tools Parameter ✅ SIMPLE CASES

**Files**: Uses existing functionality

```python
# Usage
mcp_tools = await create_mcp_client_and_load_tools(mcp_config)
agent = create_deep_agent(tools=mcp_tools)
```

**Pros:**
- ✅ Zero changes to any package
- ✅ Very simple to understand
- ✅ Already works in chatlas-agents
- ✅ No middleware complexity

**Cons:**
- ❌ No system prompt injection
- ❌ Tools not tracked in agent state
- ❌ No lifecycle hooks
- ❌ Can't leverage middleware features

**Best For:**
- Quick prototypes
- Simple integrations
- When middleware features not needed

---

### Approach 4: Custom Factory

**Files**: Would create wrapper in `chatlas-agents`

```python
# Usage
agent = create_chatlas_agent(
    model=model,
    mcp_config=mcp_config,
)
```

**Pros:**
- ✅ Zero upstream changes
- ✅ Simple API

**Cons:**
- ❌ Creates another abstraction layer
- ❌ Duplicates create_deep_agent interface
- ❌ Hard to keep in sync with upstream
- ❌ Less flexible than middleware

**Best For:**
- Not recommended

---

## Implementation Status

### ✅ Implemented

1. **MCPMiddleware (Approach 1)**
   - Full implementation in `chatlas_agents/middleware/mcp.py`
   - Comprehensive test suite
   - Documentation and examples
   - Ready for production use

2. **Tools Parameter (Approach 3)**
   - Already working via `create_mcp_client_and_load_tools()`
   - Used in current chatlas-agents implementation
   - Can be used alongside MCPMiddleware

### ❌ Not Implemented

- Approach 2 (Optional Dependency) - Requires upstream changes
- Approach 4 (Custom Factory) - Not recommended

---

## Usage Recommendations

### For DeepAgents

**Use MCPMiddleware:**

```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

mcp_config = MCPServerConfig(url="...", timeout=60)
mcp_middleware = await MCPMiddleware.create(mcp_config)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
)
```

### For DeepAgents-CLI

**Option A: Tools Parameter (Simple)**

```python
from deepagents_cli.agent import create_cli_agent
from chatlas_agents.mcp import create_mcp_client_and_load_tools

mcp_tools = await create_mcp_client_and_load_tools(mcp_config)

agent, backend = create_cli_agent(
    model=model,
    assistant_id="my-agent",
    tools=mcp_tools,
)
```

**Option B: Helper Wrapper (Recommended)**

```python
# Create a helper in chatlas-agents
async def create_chatlas_cli_agent(model, assistant_id, mcp_config=None, **kwargs):
    tools = kwargs.get('tools', [])
    if mcp_config:
        mcp_middleware = await MCPMiddleware.create(mcp_config)
        tools.extend(mcp_middleware.tools)
    return create_cli_agent(model, assistant_id, tools=tools, **kwargs)

# Use it
agent, backend = await create_chatlas_cli_agent(
    model=model,
    assistant_id="my-agent",
    mcp_config=mcp_config,
)
```

---

## Migration Guide

### From Current chatlas-agents Implementation

**Current (graph.py):**
```python
mcp_tools = await load_mcp_tools(config.mcp)
agent = create_deep_agent(
    model=llm,
    tools=mcp_tools,
)
```

**New (with MCPMiddleware):**
```python
mcp_middleware = await MCPMiddleware.create(config.mcp)
agent = create_deep_agent(
    model=llm,
    middleware=[mcp_middleware],
)
```

**Benefits of Migration:**
- System prompt automatically includes tool documentation
- Tools tracked in agent state
- Composable with other middleware
- More maintainable

---

## Testing Strategy

### Unit Tests

```bash
cd libs/chatlas-agents
pytest tests/test_mcp_middleware.py -v
```

### Integration Tests

```bash
# Test with real MCP server
export CHATLAS_MCP_URL="https://chatlas-mcp.app.cern.ch/mcp"
python examples/mcp_middleware_example.py
```

### CLI Integration Tests

```bash
# Test CLI integration
python examples/mcp_cli_integration_example.py
```

---

## Conclusion

**Recommended Approach: MCPMiddleware (Approach 1)**

Use this for production deployments because it:
1. Requires no upstream changes
2. Provides full integration with agent lifecycle
3. Is composable with other middleware
4. Is forward compatible
5. Has comprehensive tests and documentation

**Fallback: Tools Parameter (Approach 3)**

Use this for:
- Quick prototypes
- Simple use cases
- When middleware features aren't needed

**Avoid: Approaches 2 and 4**
- Approach 2 requires upstream changes
- Approach 4 adds unnecessary abstraction

---

## Further Reading

- [MCP_INTEGRATION.md](../MCP_INTEGRATION.md) - Comprehensive integration guide
- [examples/mcp_middleware_example.py](../libs/chatlas-agents/examples/mcp_middleware_example.py) - Working example with deepagents
- [examples/mcp_cli_integration_example.py](../libs/chatlas-agents/examples/mcp_cli_integration_example.py) - CLI integration patterns
