# DeepAgents MCP Integration

This directory contains the Model Context Protocol (MCP) integration for DeepAgents, enabling seamless connection to MCP servers and tools.

## Overview

The DeepAgents MCP integration allows DeepAgents to:
- Connect to any MCP-compatible server (stdio or HTTP transport)
- Automatically discover and use tools from MCP servers
- Integrate MCP tools alongside native DeepAgents tools
- Support OAuth 2.1 security and consent management

## Features

### Core Capabilities
- **Automatic Tool Discovery**: Dynamically loads tools from configured MCP servers
- **Multiple Transport Support**: Both stdio (subprocess) and HTTP transports
- **Unified Tool Interface**: MCP tools work seamlessly with native tools
- **Configuration Management**: JSON/YAML configuration support

### Security Features
- **OAuth 2.1 Support**: Full OAuth 2.1 resource server implementation
- **Token Validation**: Bearer token authentication for API access
- **Scope-based Authorization**: Fine-grained permissions (tools:read, tools:execute)
- **Consent Framework**: User consent management for tool execution
- **Session Management**: Secure session handling

## Installation

The MCP integration is included with DeepAgents. To use it, ensure you have the required dependencies:

```bash
pip install deepagents[mcp]
```

Or install from source:
```bash
cd deepagents-mcp
pip install -e .
```

## Configuration

### Basic Configuration

Create an `mcp_config.json` file:

```json
{
  "mcp_servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
      "transport": "stdio"
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_TOKEN_HERE"
      },
      "transport": "stdio"
    }
  }
}
```

### Using with DeepAgents

```python
from deepagents import create_deep_agent_async

# Define MCP connections
mcp_connections = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/documents"],
        "transport": "stdio"
    }
}

# Create agent with MCP tools
agent = await create_deep_agent_async(
    tools=[],  # Your native tools
    instructions="You are a helpful assistant with file system access.",
    mcp_connections=mcp_connections
)

# Use the agent
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "List files in the documents folder"}]
})
```

### CLI Usage

Use MCP tools with the DeepAgents CLI:

```bash
python -m deepagents --mcp-config mcp_config.json
```

## Security Configuration

### OAuth 2.1 Setup

For production environments, enable OAuth 2.1 security:

```python
from deepagents_mcp import SecureMCPClient

# Create secure MCP client
client = SecureMCPClient(
    connections=mcp_connections,
    authorization_server_url="https://auth.example.com",
    server_identifier="https://deepagents.local/mcp"
)

# Get tools with authorization
tools = await client.get_tools(authorization_header="Bearer your_token_here")
```

### Consent Management

The consent framework ensures users approve tool executions:

```python
from deepagents_mcp import MCPToolProvider

provider = MCPToolProvider(
    connections=mcp_connections,
    enable_consent=True  # Require user consent for tool execution
)
```

## Available MCP Servers

The integration works with any MCP-compatible server, including:

- **Filesystem**: File system operations
- **GitHub**: Repository management
- **Brave Search**: Web search capabilities
- **Obsidian**: Note-taking integration
- **Supabase**: Database operations
- **LangSmith**: LLM observability
- **And many more...**

## Examples

See the `examples/` directory for complete examples:

- `deepagents_with_mcp.py`: Basic usage example
- `tool_discovery_demo.py`: Tool discovery demonstration
- `math_server.py`: Example MCP server implementation

## Development

### Running Tests

```bash
cd deepagents-mcp
pytest tests/
```

### Project Structure

```
deepagents-mcp/
├── src/
│   └── deepagents_mcp/
│       ├── __init__.py          # Package exports
│       ├── mcp_client.py        # Main MCP client implementation
│       ├── security.py          # OAuth 2.1 implementation
│       ├── consent.py           # Consent framework
│       ├── initialization.py    # MCP initialization
│       └── validation.py        # Input validation
├── examples/                    # Usage examples
├── tests/                       # Test suite
└── docs/                        # Additional documentation
```

## License

This project is part of DeepAgents and follows the same license terms.

## Contributing

Contributions are welcome! Please see the main DeepAgents repository for contribution guidelines.

## Documentation

- [MCP Integration Guide](docs/MCP_INTEGRATION.md)
- [MCP Integration Validation](docs/MCP_INTEGRATION_VALIDATION.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io)