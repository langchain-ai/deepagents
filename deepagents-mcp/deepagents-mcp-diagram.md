# Deep Agents MCP Architecture Diagram

## Overview
This document contains the mermaid diagram representation of the actual Deep Agents architecture with MCP integration, based on code analysis of the deepagents-mcp implementation.

## Actual Architecture Flow

```mermaid
graph TB
    %% Core Components
    CLI[CLI Entry Point<br/>__main__.py] --> Agent[DeepAgent Creation<br/>graph.py]
    Config[MCP Config File<br/>JSON/YAML] --> MCP[MCPToolProvider<br/>mcp_client.py]
    
    %% Agent Creation Flow
    Agent --> State[DeepAgentState<br/>state.py]
    Agent --> Tools[Tool Integration]
    Agent --> Model[Language Model<br/>model.py]
    
    %% Tool Integration
    Tools --> BuiltIn[Built-in Tools<br/>write_todos, files]
    Tools --> User[User Tools<br/>custom functions]
    Tools --> MCPTools[MCP Tools<br/>from servers]
    Tools --> TaskTool[Task Tool<br/>sub-agent delegation]
    
    %% MCP Integration
    MCP --> MultiClient[MultiServerMCPClient<br/>langchain-mcp-adapters]
    MultiClient --> MCPServers[MCP Servers<br/>stdio/HTTP processes]
    MCPServers --> MCPTools
    
    %% Sub-Agent System
    TaskTool --> SubAgents[Sub-Agents<br/>specialized agents]
    SubAgents --> GeneralPurpose[General Purpose Agent]
    SubAgents --> CustomSub[Custom Sub-Agents]
    
    %% State Management
    State --> Todos[Todo Management]
    State --> Files[Mock File System]
    State --> Messages[Message History]
    
    %% Final Assembly
    Agent --> ReactAgent[LangGraph ReAct Agent]
    ReactAgent --> Execution[Agent Execution]
    
    %% Styling
    classDef coreComponent fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef mcpComponent fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef toolComponent fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef stateComponent fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class CLI,Agent,State,Model,ReactAgent,Execution coreComponent
    class Config,MCP,MultiClient,MCPServers,MCPTools mcpComponent
    class Tools,BuiltIn,User,TaskTool,SubAgents,GeneralPurpose,CustomSub toolComponent
    class Todos,Files,Messages stateComponent
```

## Simplified Component View

```mermaid
flowchart TD
    subgraph Core[Deep Agents Core]
        AgentCreation[Agent Creation<br/>create_deep_agent_async]
        State[DeepAgentState<br/>todos, files, messages]
        ReactAgent[LangGraph ReAct Agent]
    end
    
    subgraph MCPLayer[MCP Integration Layer]
        MCPProvider[MCPToolProvider]
        MCPClient[MultiServerMCPClient]
        MCPServers[MCP Servers<br/>stdio/HTTP]
    end
    
    subgraph ToolEcosystem[Tool Ecosystem]
        BuiltInTools[Built-in Tools<br/>todos, file ops]
        UserTools[User Tools<br/>custom functions]
        MCPTools[MCP Tools<br/>dynamic loading]
        TaskTool[Task Tool<br/>sub-agent delegation]
    end
    
    subgraph SubAgentLayer[Sub-Agent Layer]
        GeneralAgent[General Purpose Agent]
        CustomAgents[Custom Sub-Agents]
    end
    
    %% Connections
    AgentCreation --> ToolEcosystem
    MCPProvider --> MCPClient
    MCPClient --> MCPServers
    MCPServers --> MCPTools
    ToolEcosystem --> ReactAgent
    TaskTool --> SubAgentLayer
    ReactAgent --> State
    
    %% Styling
    classDef coreStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    classDef mcpStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef toolStyle fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef subAgentStyle fill:#fff8e1,stroke:#ff9800,stroke-width:2px
    
    class Core coreStyle
    class MCPLayer mcpStyle
    class ToolEcosystem toolStyle
    class SubAgentLayer subAgentStyle
```

## Architecture Description

Based on code analysis, the Deep Agents MCP architecture is a sophisticated multi-layered system:

### Core Architecture Layers

#### **1. Deep Agents Core (Blue)**
- **CLI Entry Point** (`__main__.py`) - Command-line interface with `--mcp-config` support
- **Agent Creation** (`graph.py`) - Main orchestration via `create_deep_agent_async()`
- **DeepAgentState** (`state.py`) - Centralized state management for todos, files, and messages
- **LangGraph ReAct Agent** - The actual execution engine

#### **2. MCP Integration Layer (Purple)**
- **MCPToolProvider** (`mcp_client.py`) - Manages MCP server connections
- **MultiServerMCPClient** (langchain-mcp-adapters) - Handles multiple MCP servers
- **MCP Servers** - External processes via stdio/HTTP transport

#### **3. Tool Ecosystem (Green)**
- **Built-in Tools** - Core functionality (todos, file operations)
- **User Tools** - Custom functions provided by developers
- **MCP Tools** - Dynamically loaded from external MCP servers
- **Task Tool** - Enables sub-agent delegation

#### **4. Sub-Agent Layer (Orange)**
- **General Purpose Agent** - Default sub-agent for broad tasks
- **Custom Sub-Agents** - Specialized agents for specific domains

### Key Architectural Features

1. **Modular Design**: MCP integration is completely optional - core functionality works independently
2. **Async-First**: MCP tool loading requires async agent creation for optimal performance
3. **Tool Unification**: MCP tools seamlessly integrate with native and user tools
4. **State Management**: Centralized state handling through `DeepAgentState` with reducers
5. **Hierarchical Delegation**: Sub-agents can be spawned with specialized toolsets
6. **Transport Flexibility**: Supports both stdio and HTTP MCP server connections
7. **Error Resilience**: Graceful fallback when MCP dependencies are unavailable

### Data Flow

1. **Configuration** → MCP servers defined in JSON/YAML
2. **Tool Discovery** → MCP tools loaded asynchronously 
3. **Tool Integration** → All tools combined into unified interface
4. **Agent Creation** → LangGraph ReAct agent with full toolset
5. **Execution** → State management and sub-agent delegation as needed

### Color Coding
- **Blue**: Core Deep Agents components
- **Purple**: MCP integration layer
- **Green**: Tool ecosystem (built-in, user, MCP, task)
- **Orange**: Sub-agent management and delegation