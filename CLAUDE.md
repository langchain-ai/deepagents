# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepAgents is a Python package that implements "deep agents" - LLM-based agents with planning capabilities, file system access, and sub-agent spawning. The architecture is inspired by Claude Code and applications like Deep Research and Manus.

**Monorepo Structure:**
- `libs/deepagents/` - Core deepagents Python package
- `libs/deepagents-cli/` - CLI tool (separate package)
- `examples/` - Example implementations (e.g., research agent)

## Development Commands

### Testing
```bash
# Run unit tests for core package
make test

# Run integration tests
make integration_test

# Run tests for CLI package
cd libs/deepagents-cli && make test
```

### Linting & Formatting
```bash
# Lint all files (ruff + mypy)
make lint

# Lint only core package
make lint_package

# Lint only tests
make lint_tests

# Lint only changed files (git diff against master)
make lint_diff

# Format all files (auto-fix)
make format

# Format only changed files
make format_diff
```

All linting/formatting commands use `uv run --all-groups` to run tools.

### Running Examples
```bash
# Research agent example (requires TAVILY_API_KEY)
cd examples/research
uv run python research_agent.py
```

## Architecture

### Core Factory: `create_deep_agent()`

Located in `libs/deepagents/graph.py`. This is the main entry point that creates a LangGraph CompiledStateGraph with built-in middleware for:

1. **TodoListMiddleware** - Provides `write_todos` tool for task planning
2. **File Middleware** (configurable):
   - **FilesystemMiddleware** (default) - Provides six tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
   - **ClaudeTextEditorMiddleware** (opt-in) - Provides single `str_replace_based_edit_tool` matching Claude's native `text_editor_20250728`
3. **SubAgentMiddleware** - Provides `task` tool for spawning specialized subagents
4. **SummarizationMiddleware** - Manages context length (max 170k tokens before summary)
5. **AnthropicPromptCachingMiddleware** - Optimizes prompt caching for Anthropic models
6. **PatchToolCallsMiddleware** - Internal tool call handling
7. **HumanInTheLoopMiddleware** - Optional, added if `interrupt_on` is provided
8. **ResumableShellToolMiddleware** - Optional, provides `shell`/`bash` tool for command execution

**File Middleware:**
- **Default**: `FilesystemMiddleware` is used automatically if no file middleware is provided
  - Provides 6 tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Claude Native**: Pass `ClaudeTextEditorMiddleware(backend=backend)` in `middleware` parameter
  - User must manually bind the native tool first: `model.bind_tools([{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000}])`
  - Provides exact compatibility with Claude's training but lacks glob/grep search capabilities
  - Example:
    ```python
    agent = create_deep_agent(
        model=model_with_tools,
        middleware=[ClaudeTextEditorMiddleware(backend=backend)]
    )
    ```

**Shell Middleware:**
Shell access is opt-in. Pass `ResumableShellToolMiddleware` in `middleware` parameter.
- **Langchain shell**: Middleware provides the `shell` tool automatically
  - Example:
    ```python
    shell_middleware = ResumableShellToolMiddleware(workspace_root=os.getcwd(), execution_policy=HostExecutionPolicy())
    agent = create_deep_agent(middleware=[shell_middleware])
    ```
- **Claude native bash**: Manually bind the native tool first, then use `add_shell_tool=False`
  - Bind: `model.bind_tools([{"type": "bash_20250124", "name": "bash"}])`
  - Set `add_shell_tool=False` to prevent adding the langchain `shell` tool
  - Middleware automatically detects and handles both langchain and Claude's native bash tool
  - Example:
    ```python
    model = model.bind_tools([{"type": "bash_20250124", "name": "bash"}])
    shell_middleware = ResumableShellToolMiddleware(
        workspace_root=os.getcwd(),
        execution_policy=HostExecutionPolicy(),
        add_shell_tool=False  # Don't add langchain's shell tool
    )
    agent = create_deep_agent(model=model, middleware=[shell_middleware])
    ```

**Default model:** `claude-sonnet-4-5-20250929` with 20k max tokens
**Recursion limit:** 1000

### Middleware System

Middleware is composable and can be used independently. Each middleware adds tools and system prompts to the agent. Key middleware implementations:

- `FilesystemMiddleware` (`libs/deepagents/middleware/filesystem.py`) - Six separate file tools with glob/grep
- `ClaudeTextEditorMiddleware` (`libs/deepagents/middleware/claude_text_editor.py`) - Claude's native text editor tool
- `SubAgentMiddleware` (`libs/deepagents/middleware/subagents.py`) - Subagent spawning
- `ResumableShellToolMiddleware` (`libs/deepagents/middleware/resumable_shell.py`) - Shell access, handles both langchain's shell tool and Claude's native `bash_20250124` tool

### Backend System

Pluggable storage backends for file operations (used by both FilesystemMiddleware and ClaudeTextEditorMiddleware):

- `StateBackend` - Stores files in LangGraph state
- `StoreBackend` - Stores files in LangGraph Store (requires `store` parameter)
- `FilesystemBackend` - Uses actual filesystem
- `CompositeBackend` - Combines multiple backends

Located in `libs/deepagents/backends/`

### Subagents

Two ways to define subagents:

1. **SubAgent** (dict) - Automatically creates agent from config with `name`, `description`, `system_prompt`, `tools`, optional `model`/`middleware`
2. **CompiledSubAgent** (dict) - Uses pre-built LangGraph graph with `name`, `description`, `runnable`

Subagents automatically inherit: TodoListMiddleware, FilesystemMiddleware, SummarizationMiddleware, and share the same backend as parent.

## Key Design Patterns

### Context Management
Deep agents offload large context to the file system to prevent context window overflow. Tool results should be written to files rather than returned directly when large.

### Task Decomposition
Use the `write_todos` tool for multi-step tasks to track progress and adapt plans.

### Subagent Isolation
Spawn subagents for context quarantine - keeps main agent context clean while going deep on subtasks. Each subagent has independent context but shares filesystem access.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Tests run on Python 3.11, 3.12, 3.13 for core package
- Tests run on Python 3.11, 3.13 for CLI package
- Linting is currently disabled (`if: false`)
- Uses `uv` for dependency management with frozen lockfile

## Package Configuration

- Uses `pyproject.toml` with `setuptools` backend
- Source in `libs/` directory with `find_packages` pattern
- Ruff for linting (line length: 150, Google-style docstrings)
- Mypy with strict mode
- Package version: 0.2.4

## Important Notes

- `async_create_deep_agent` has been deprecated - use `create_deep_agent` for both sync and async
- Compatible with MCP tools via `langchain-mcp-adapters`
- The agent is a standard LangGraph graph, so all LangGraph features work (streaming, HITL, memory, Studio)
