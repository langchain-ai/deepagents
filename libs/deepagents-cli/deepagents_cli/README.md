# DeepAgents CLI

Interactive command-line interface for DeepAgents - an AI coding assistant with file operations, web search, and shell command execution.

## Architecture

The CLI is organized into focused modules:

```
cli/
├── __init__.py      # Package exports
├── __main__.py      # Entry point for `python -m deepagents.cli`
├── main.py          # CLI loop, argument parsing, main orchestration
├── config.py        # Configuration, constants, colors, model creation
├── tools.py         # Custom tools (http_request, web_search)
├── ui.py            # Display logic, TokenTracker, help screens
├── input.py         # Input handling, completers, prompt session
├── commands.py      # Slash command and bash command handlers
├── execution.py     # Task execution, streaming, HITL approval
└── agent.py         # Agent creation, management, listing, reset
```

## Module Responsibilities

### `main.py` - Entry Point & Main Loop
- **Purpose**: CLI entry point, argument parsing, main interactive loop
- **Key Functions**:
  - `cli_main()` - Console script entry point (called when you run `deepagents`)
  - `main()` - Async main function that orchestrates agent creation and CLI
  - `simple_cli()` - Main interactive loop handling user input
  - `parse_args()` - Command-line argument parsing
  - `check_cli_dependencies()` - Validates required packages are installed

### `config.py` - Configuration & Constants
- **Purpose**: Centralized configuration, constants, and model creation
- **Key Exports**:
  - `COLORS` - Color scheme for terminal output
  - `DEEP_AGENTS_ASCII` - ASCII art banner
  - `COMMANDS` - Available slash commands
  - `console` - Rich Console instance
  - `create_model()` - Creates OpenAI or Anthropic model based on API keys
  - `get_default_coding_instructions()` - Loads default agent prompt

### `tools.py` - Custom Agent Tools
- **Purpose**: Additional tools for the agent beyond built-in filesystem operations
- **Tools**:
  - `http_request()` - Make HTTP requests to APIs
  - `web_search()` - Search the web using Tavily API
  - `tavily_client` - Initialized Tavily client (if API key available)

### `ui.py` - Display & Rendering
- **Purpose**: All UI rendering and display logic
- **Key Components**:
  - `TokenTracker` - Track and display token usage across the session
  - `render_todo_list()` - Render todo list with checkboxes
  - `show_interactive_help()` - Display available commands during session
  - `show_help()` - Full help screen
  - `format_tool_message_content()` - Format tool messages for display
  - `truncate_value()` - Truncate long values for readable display

### `input.py` - Input Handling
- **Purpose**: User input, completers, and prompt session configuration
- **Key Components**:
  - `FilePathCompleter` - Autocomplete for `@file` mentions
  - `CommandCompleter` - Autocomplete for `/commands`
  - `BashCompleter` - Autocomplete for `!bash` commands
  - `parse_file_mentions()` - Extract `@file` mentions and inject content
  - `create_prompt_session()` - Configure prompt_toolkit session with:
    - Multi-line input (Alt+Enter for newlines, Enter to submit)
    - Command history
    - File/command autocomplete
    - External editor support (Ctrl+E)

### `commands.py` - Command Handlers
- **Purpose**: Handle slash commands (`/help`, `/clear`, etc.) and bash execution
- **Key Functions**:
  - `handle_command()` - Route and execute slash commands
  - `execute_bash_command()` - Execute bash commands prefixed with `!`

### `execution.py` - Task Execution & Streaming
- **Purpose**: Core execution logic, streaming responses, HITL (Human-in-the-Loop)
- **Key Functions**:
  - `execute_task()` - Main execution function that:
    - Parses file mentions
    - Streams agent responses
    - Displays tool calls with icons
    - Renders todo list updates
    - Tracks token usage
    - Handles Ctrl+C interruptions
  - `prompt_for_shell_approval()` - Interactive shell command approval with arrow keys
- **Features**:
  - Dual-stream mode (messages + updates) for HITL support
  - Real-time todo list rendering
  - Spinner with status updates
  - Token tracking integration

### `agent.py` - Agent Management
- **Purpose**: Agent creation, configuration, and management commands
- **Key Functions**:
  - `create_agent_with_config()` - Create agent with:
    - Filesystem backends (working directory + agent directory)
    - Long-term memory middleware
    - Shell execution with HITL approval
    - Custom system prompt
  - `list_agents()` - List all agents in `~/.deepagents/`
  - `reset_agent()` - Reset agent to default or copy from another agent

## Data Flow

```
User Input
    ↓
main.py (simple_cli)
    ↓
input.py (parse_file_mentions, completers)
    ↓
execution.py (execute_task)
    ↓
agent.py (agent created with tools from tools.py)
    ↓
Stream responses → ui.py (render todos, display tool calls)
                 → execution.py (HITL approval if needed)
    ↓
Display output via ui.py (TokenTracker, console)
```

## Key Features

### File Context Injection
Type `@filename` and press Tab to autocomplete and inject file content into your prompt.

### Interactive Commands
- `/help` - Show help
- `/clear` - Clear screen and reset conversation
- `/tokens` - Show token usage
- `/quit` or `/exit` - Exit the CLI

### Bash Commands
Type `!command` to execute bash commands directly (e.g., `!ls`, `!git status`)

### Todo List Tracking
The agent can create and update a visual todo list for multi-step tasks.

### File Operation Summaries & Diff Viewer
- File reads now show a concise summary with the number of lines streamed (e.g., `⏺ Read(example.py)` followed by `⎿  Read 44 lines (lines 1-44)`).
- Writes and edits capture before/after snapshots, reporting lines added or removed plus bytes written.
- A Rich-powered unified diff renders in-line with syntax highlighting so you can review every proposed change before confirming.
- Diff output truncates gracefully for very large edits while still surfacing a summary.
- When Human-in-the-Loop approval is required, the proposed diff is shown *before* you choose Approve/Reject.

### Human-in-the-Loop Shell Approval
Shell commands require user approval with an interactive arrow-key menu.

### Multi-line Input
- **Enter** - Submit (or accept completion if menu is open)
- **Alt+Enter** - Insert newline (Option+Enter on Mac, or ESC then Enter)
- **Ctrl+E** - Open in external editor (nano by default)

## Agent Storage

Each agent stores its state in `~/.deepagents/AGENT_NAME/`:
- `agent.md` - Agent's custom instructions (long-term memory)
- `memories/` - Additional context files
- `history` - Command history

## Development

To modify the CLI:

1. **UI changes** → Edit `ui.py` or `input.py`
2. **Add new tools** → Edit `tools.py`
3. **Change execution flow** → Edit `execution.py`
4. **Add commands** → Edit `commands.py`
5. **Agent configuration** → Edit `agent.py`
6. **Constants/colors** → Edit `config.py`

## Running During Development

```bash
# From project root
uv run python -m deepagents.cli

# Or install in editable mode
uv pip install -e .
deepagents
```

## Entry Point

The CLI is registered in `pyproject.toml` as:
```toml
[project.scripts]
deepagents = "deepagents.cli:cli_main"
```

This means when users install the package, they can run `deepagents` directly.
