# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
deepagents is a Python package implementing "Deep Agent" architecture for creating intelligent agents capable of handling complex, multi-step tasks. It uses LangGraph for agent orchestration and Claude Sonnet 4 as the default LLM.

## Common Development Commands

### Setup
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install langgraph>=0.2.6 langchain-anthropic>=0.1.23 langchain>=0.2.14
```

### Running Examples
```bash
# Run the research agent example
cd examples/research
pip install -r requirements.txt
langgraph test  # Run with LangGraph Studio
```

### Building the Package
```bash
# Build distribution
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### Running Tests
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=deepagents

# Run specific test file
pytest tests/test_create_deep_agent.py

# Run specific test class or method
pytest tests/test_create_deep_agent.py::TestCreateDeepAgent::test_create_deep_agent_happy_path
```

## Architecture Overview

### Core Components
- **`src/deepagents/graph.py`**: Main agent creation logic using LangGraph's `create_react_agent`
- **`src/deepagents/state.py`**: State management with `DeepAgentState` extending LangGraph's `AgentState`
- **`src/deepagents/tools.py`**: Built-in tools (file operations, todo management)
- **`src/deepagents/sub_agent.py`**: Sub-agent spawning and task delegation
- **`src/deepagents/prompts.py`**: System prompts and tool descriptions

### Key Design Patterns
1. **Mock Filesystem**: Uses LangGraph state instead of real files for safety. Files are stored in the agent's state with a file reducer for merging.
2. **Sub-Agent Architecture**: Main agent can spawn specialized sub-agents with isolated context and specific tool access.
3. **State Management**: Uses TypedDict for structured state with `todos` list and `files` dictionary.
4. **Tool System**: Tools are LangChain-compatible functions that modify agent state.

### Important Implementation Details
- Default model is Claude Sonnet 4 (`claude-sonnet-4-20250514`) with 64k token limit
- Sub-agents receive quarantined context to prevent information leakage
- File operations validate path safety and perform string replacement validation
- Todo management tracks task status (pending, in_progress, completed)

## Development Notes
- Test suite is in `tests/` directory - run with `pytest`
- No linting/formatting configuration - consider adding black, flake8, mypy
- Package version is in `pyproject.toml` - update when releasing
- Examples in `examples/` directory demonstrate usage patterns