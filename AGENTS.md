# Instructions for Coding Agents

This document provides essential guidance for AI coding agents (GitHub Copilot, Claude, GPT, etc.) working on the ChATLAS DeepAgents repository.

## Project Overview

This repository is a fork of LangChain's `deepagents` library, extended to integrate with ChATLAS (CERN ATLAS experiment documentation). The project consists of three main modules in a monorepo structure:

```
libs/
├── deepagents/         # Base DeepAgents framework (upstream)
├── deepagents-cli/     # CLI interface with skills and memory
└── chatlas-agents/     # ChATLAS-specific integrations (MCP, ATLAS tools)
```

**Key Features:**
- Native MCP (Model Context Protocol) support via middleware
- ChATLAS MCP server integration for ATLAS documentation search
- ATLAS software compatibility (SetupATLAS on Lxplus)
- HTCondor batch farm integration

## Quick Reference

### Repository Structure

```
chatlas-deepagents/
├── .github/                          # GitHub configuration and documentation
│   ├── copilot-instructions.md       # GitHub Copilot specific guidance
│   ├── DEPENDENCY_ANALYSIS.md        # Module dependency analysis
│   ├── MCP_INTEGRATION.md            # MCP integration guide
│   ├── MCP_APPROACHES_COMPARISON.md  # MCP approach comparison
│   └── IMPLEMENTATION_SUMMARY_MCP.md # MCP implementation summary
├── libs/
│   ├── deepagents/                   # Base framework (minimal changes)
│   ├── deepagents-cli/               # CLI layer (minimal changes)
│   └── chatlas-agents/               # ChATLAS extensions (main development)
│       ├── chatlas_agents/
│       │   ├── middleware/           # MCP middleware implementation
│       │   ├── config/               # Configuration management
│       │   ├── mcp/                  # MCP client utilities
│       │   └── ...
│       ├── .github/                  # Module-specific documentation
│       └── README.md                 # Module documentation
├── AGENTS.md                         # This file (agent instructions)
└── README.md                         # Main project documentation
```

### Module Dependencies

```
deepagents (v0.3.0)
    ↑
    │ (no local dependencies)
    │
    ├─────────────────┐
    ↓                 ↓
deepagents-cli    chatlas-agents (v0.1.0)
(v0.0.10)             ↑
    │                 │
    └─────────────────┘
```

**Important:** No circular dependencies exist. All customizations go in `chatlas-agents`.

### Development Guidelines

1. **Minimal Upstream Changes**: Avoid modifying `deepagents` and `deepagents-cli` unless absolutely necessary
2. **Custom Code Location**: Place ChATLAS-specific code in `libs/chatlas-agents`
3. **Middleware Pattern**: Use middleware for extending agent functionality (see MCPMiddleware example)
4. **Forward Compatibility**: Ensure changes work with future upstream updates
5. **Documentation**: Keep `.github/` documentation updated with architectural decisions

### Setup and Testing

```bash
# Setup from chatlas-agents directory
cd libs/chatlas-agents
uv sync

# Run tests
uv run pytest

# Test CLI
uv run python -m chatlas_agents.cli run --help

# Build and test with dependencies
cd libs/deepagents && uv sync
cd ../deepagents-cli && uv sync
cd ../chatlas-agents && uv sync
```

### Key Implementation: MCP Middleware

The primary ChATLAS extension is MCPMiddleware, which provides MCP server support:

**Location:** `libs/chatlas-agents/chatlas_agents/middleware/mcp.py`

**Usage:**
```python
from chatlas_agents.middleware import MCPMiddleware
from chatlas_agents.config import MCPServerConfig
from deepagents import create_deep_agent

# Create MCP middleware
mcp_config = MCPServerConfig(
    url="https://chatlas-mcp.app.cern.ch/mcp",
    timeout=60
)
mcp_middleware = await MCPMiddleware.create(mcp_config)

# Create agent with MCP support
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    middleware=[mcp_middleware],
)
```

**Key Points:**
- Zero upstream changes required
- Composable with other middleware
- Async initialization required
- Full lifecycle integration

### Common Tasks

#### Adding New Features

1. **New Middleware**: Add to `libs/chatlas-agents/chatlas_agents/middleware/`
2. **New Tools**: Integrate via MCP server or add to middleware
3. **New LLM Provider**: Extend `chatlas_agents/llm/` factory
4. **New Configuration**: Update `chatlas_agents/config/`

#### Debugging

```bash
# Verbose logging
uv run python -m chatlas_agents.cli run --input "test" --verbose

# Check MCP connectivity
curl https://chatlas-mcp.app.cern.ch/mcp

# Run specific tests
uv run pytest tests/test_mcp_middleware.py -v
```

#### Common Pitfalls

1. **Session Lifecycle**: Don't close MCP sessions before tools execute
2. **Async/Await**: All MCP operations must be awaited
3. **Dependencies**: Install in order: deepagents → deepagents-cli → chatlas-agents
4. **Editable Installs**: Use `pip install -e .` or `uv sync` for development

### Code Style

- **Python 3.11+** compatible
- **Type hints** for all public functions
- **Docstrings** for all classes and functions
- **Snake_case** for functions/variables
- **PascalCase** for classes
- Follow **existing patterns** in codebase

### Documentation Structure

- **README.md** (root): Main project documentation for users
- **AGENTS.md** (root): This file - agent instructions
- **.github/copilot-instructions.md**: GitHub Copilot specific guidance
- **.github/*.md**: Technical documentation and architectural decisions
- **libs/chatlas-agents/README.md**: Module-specific documentation
- **libs/chatlas-agents/.github/*.md**: Module-level agent instructions (detailed)

### Important Links

- **Detailed Agent Instructions**: `libs/chatlas-agents/AGENTS.md`
- **MCP Integration Guide**: `.github/MCP_INTEGRATION.md`
- **Dependency Analysis**: `.github/DEPENDENCY_ANALYSIS.md`
- **Setup Instructions**: `libs/chatlas-agents/SETUP.md`

### Testing Checklist

Before submitting changes:
- [ ] All tests pass: `uv run pytest`
- [ ] Linting passes: `uv run ruff check`
- [ ] MCP tools load correctly (if applicable)
- [ ] No breaking changes to public APIs
- [ ] Documentation updated
- [ ] Type hints present
- [ ] Error handling appropriate

### Performance Expectations

| Operation | Expected Time |
|-----------|--------------|
| MCP connection | 3-5 seconds |
| Tool discovery | Included in connection |
| LLM inference | 5-15 seconds |
| Tool invocation | 15-30 seconds |
| Total simple query | 10-20 seconds |

## When to Consult Detailed Documentation

- **New to the codebase**: Read `libs/chatlas-agents/AGENTS.md` for comprehensive guide
- **GitHub Copilot specific**: Check `.github/copilot-instructions.md`
- **MCP integration**: Reference `.github/MCP_INTEGRATION.md` and `.github/MCP_APPROACHES_COMPARISON.md`
- **Dependency issues**: Review `.github/DEPENDENCY_ANALYSIS.md`
- **Module setup**: See `libs/chatlas-agents/SETUP.md`

## Summary for Quick Tasks

**Before making changes:**
1. Run existing tests
2. Check similar patterns in codebase
3. Review relevant documentation in `.github/`

**When making changes:**
1. Keep customizations in `libs/chatlas-agents`
2. Avoid modifying upstream packages
3. Use middleware pattern for extensions
4. Follow existing code style

**After making changes:**
1. Run full test suite
2. Update documentation
3. Verify no breaking changes
4. Check error messages are helpful

---

**For comprehensive guidance, see:**
- Detailed instructions: `libs/chatlas-agents/AGENTS.md`
- GitHub Copilot: `.github/copilot-instructions.md`
- Technical docs: `.github/*.md`
