# 🚀🧠 Deep Agents CLI

The [deepagents](https://github.com/langchain-ai/deepagents) CLI is an open source coding assistant that runs in your terminal, similar to Claude Code.

**Key Features:**

- **Built-in Tools**: File operations (read, write, edit, glob, grep), shell commands, web search, and subagent delegation
- **Customizable Skills**: Add domain-specific capabilities through a progressive disclosure skill system
- **Persistent Memory**: Agent remembers your preferences, coding style, and project context across sessions
- **Project-Aware**: Automatically detects project roots and loads project-specific configurations

<img src="cli-banner.jpg" alt="deep agent" width="100%"/>

## 🚀 Quickstart

`deepagents-cli` is a Python package that can be installed via pip or uv.

**Install via pip:**

```bash
pip install deepagents-cli
```

**Or using uv (recommended):**

```bash
# Create a virtual environment
uv venv

# Install the package
uv pip install deepagents-cli
```

**Run the agent in your terminal:**

```bash
deepagents
```

**Get help:**

```bash
deepagents help
```

**Common options:**

```bash
# Use a specific agent configuration
deepagents --agent mybot

# Use a specific model (auto-detects provider)
deepagents --model claude-sonnet-4-5-20250929
deepagents --model gpt-4o

# Auto-approve tool usage (skip human-in-the-loop prompts)
deepagents --auto-approve

# Execute code in a remote sandbox
deepagents --sandbox modal        # or runloop, daytona
deepagents --sandbox-id dbx_123   # reuse existing sandbox
```

Type naturally as you would in a chat interface. The agent will use its built-in tools, skills, and memory to help you with tasks.

## Model Configuration

The CLI supports three LLM providers with automatic provider detection based on model name:

**Supported Providers:**

- **OpenAI** - Models like `gpt-4o`, `gpt-5-mini`, `o1-preview`, `o3-mini` (default: `gpt-5-mini`)
- **Anthropic** - Models like `claude-sonnet-4-5-20250929`, `claude-3-opus-20240229` (default: `claude-sonnet-4-5-20250929`)
- **Google** - Models like `gemini-3-pro-preview`, `gemini-2.5-pro` (default: `gemini-3-pro-preview`)

**Specify model at startup:**

```bash
# Auto-detects Anthropic from model name pattern
deepagents --model claude-sonnet-4-5-20250929

# Auto-detects OpenAI from model name pattern
deepagents --model gpt-4o
```

**Or use environment variables:**

```bash
# Set provider-specific model defaults
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
export OPENAI_MODEL="gpt-4o"
export GOOGLE_MODEL="gemini-2.5-pro"

# Set API keys (required)
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

**Model name conventions:**

Model names follow each provider's official naming convention:

- **OpenAI**: See [OpenAI Models Documentation](https://platform.openai.com/docs/models)
- **Anthropic**: See [Anthropic Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models)
- **Google**: See [Google Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini)

The active model is displayed at startup in the CLI interface.

## Built-in Tools

The agent comes with the following built-in tools (always available without configuration):

| Tool | Description |
|------|-------------|
| `ls` | List files and directories |
| `read_file` | Read contents of a file |
| `write_file` | Create or overwrite a file |
| `edit_file` | Make targeted edits to existing files |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) |
| `grep` | Search for text patterns across files |
| `shell` | Execute shell commands (local mode) |
| `execute` | Execute commands in remote sandbox (sandbox mode) |
| `web_search` | Search the web using Tavily API |
| `fetch_url` | Fetch and convert web pages to markdown |
| `task` | Delegate work to subagents for parallel execution |
| `write_todos` | Create and manage task lists for complex work |

> [!WARNING]
> **Human-in-the-Loop (HITL) Approval Required**
>
> Potentially destructive operations require user approval before execution:
>
> - **File operations**: `write_file`, `edit_file`
> - **Command execution**: `shell`, `execute`
> - **External requests**: `web_search`, `fetch_url`
> - **Delegation**: `task` (subagents)
>
> Each operation will prompt for approval showing the action details. Use `--auto-approve` to skip prompts:
>
> ```bash
> deepagents --auto-approve
> ```

## Agent Configuration

Each agent has its own configuration directory at `~/.deepagents/<agent_name>/`, with default `agent`.

```bash
# List all configured agents
deepagents list

# Create a new agent
deepagents create <agent_name>
```

### Environment Variables

#### LangSmith Tracing

The CLI supports separate LangSmith project configuration for agent tracing vs user code tracing:

**Agent Tracing** - Traces deepagents operations (tool calls, agent decisions):

```bash
export DEEPAGENTS_LANGSMITH_PROJECT="my-agent-project"
```

**User Code Tracing** - Traces code executed via shell commands:

```bash
export LANGSMITH_PROJECT="my-user-code-project"
```

**Complete Setup Example:**

```bash
# Enable LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"

# Configure separate projects
export DEEPAGENTS_LANGSMITH_PROJECT="agent-traces"
export LANGSMITH_PROJECT="user-code-traces"

# Run deepagents
deepagents
```

When both are configured, the CLI displays:

```
✓ LangSmith tracing enabled: Deepagents → 'agent-traces'
  User code (shell) → 'user-code-traces'
```

**Why separate projects?**

- Keep agent operations separate from your application code traces
- Easier debugging by isolating agent vs user code behavior
- Different retention policies or access controls per project

**Backwards Compatibility:**
If `DEEPAGENTS_LANGSMITH_PROJECT` is not set, both agent and user code trace to the same project specified by `LANGSMITH_PROJECT`.

## Customization

There are two primary ways to customize any agent: **memory** and **skills**.

Each agent has its own global configuration directory at `~/.deepagents/<agent_name>/`:

```
~/.deepagents/<agent_name>/
  ├── agent.md              # Auto-loaded global personality/style
  └── skills/               # Auto-loaded agent-specific skills
      ├── web-research/
      │   └── SKILL.md
      └── langgraph-docs/
          └── SKILL.md
```

Projects can extend the global configuration with project-specific instructions and skills:

```
my-project/
  ├── .git/
  └── .deepagents/
      ├── agent.md          # Project-specific instructions
      └── skills/           # Project-specific skills
          └── custom-tool/
              └── SKILL.md
```

The CLI automatically detects project roots (via `.git`) and loads:

- Project-specific `agent.md` from `[project-root]/.deepagents/agent.md`
- Project-specific skills from `[project-root]/.deepagents/skills/`

Both global and project configurations are loaded together, allowing you to:

- Keep general coding style/preferences in global agent.md
- Add project-specific context, conventions, or guidelines in project agent.md
- Share project-specific skills with your team (committed to version control)
- Override global skills with project-specific versions (when skill names match)

### agent.md files

`agent.md` files provide persistent memory that is always loaded at session start. Both global and project-level `agent.md` files are loaded together and injected into the system prompt.

**Global `agent.md`** (`~/.deepagents/agent/agent.md`)

- Your personality, style, and universal coding preferences
- General tone and communication style
- Universal coding preferences (formatting, type hints, etc.)
- Tool usage patterns that apply everywhere
- Workflows and methodologies that don't change per-project

**Project `agent.md`** (`.deepagents/agent.md` in project root)

- Project-specific context and conventions
- Project architecture and design patterns
- Coding conventions specific to this codebase
- Testing strategies and deployment processes
- Team guidelines and project structure

**How it works (AgentMemoryMiddleware):**

- Loads both files at startup and injects into system prompt as `<user_memory>` and `<project_memory>`
- Appends [memory management instructions](deepagents_cli/agent_memory.py#L44-L158) on when/how to update memory files

**When the agent updates memory:**

- IMMEDIATELY when you describe how it should behave
- IMMEDIATELY when you give feedback on its work
- When you explicitly ask it to remember something
- When patterns or preferences emerge from your interactions

The agent uses `edit_file` to update memories when learning preferences or receiving feedback.

### Project memory files

Beyond `agent.md`, you can create additional memory files in `.deepagents/` for structured project knowledge. These work similarly to [Anthropic's Memory Tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool). The agent receives [detailed instructions](deepagents_cli/agent_memory.py#L123-L158) on when to read and update these files.

**How it works:**

1. Create markdown files in `[project-root]/.deepagents/` (e.g., `api-design.md`, `architecture.md`, `deployment.md`)
2. The agent checks these files when relevant to a task (not auto-loaded into every prompt)
3. The agent uses `write_file` or `edit_file` to create/update memory files when learning project patterns

**Example workflow:**

```bash
# Agent discovers deployment pattern and saves it
.deepagents/
├── agent.md           # Always loaded (personality + conventions)
├── architecture.md    # Loaded on-demand (system design)
└── deployment.md      # Loaded on-demand (deploy procedures)
```

**When the agent reads memory files:**

- At the start of new sessions (checks what files exist)
- Before answering questions about project-specific topics
- When you reference past work or patterns
- When performing tasks that match saved knowledge domains

**Benefits:**

- **Persistent learning**: Agent remembers project patterns across sessions
- **Team collaboration**: Share project knowledge through version control
- **Contextual retrieval**: Load only relevant memory when needed (reduces token usage)
- **Structured knowledge**: Organize information by domain (APIs, architecture, deployment, etc.)

### Skills

Skills are reusable agent capabilities that provide specialized workflows and domain knowledge. Example skills are provided in the `examples/skills/` directory:

- **web-research** - Structured web research workflow with planning, parallel delegation, and synthesis
- **langgraph-docs** - LangGraph documentation lookup and guidance

To use an example skill globally with the default agent, just copy them to the agent's skills global or project-level skills directory:

```bash
mkdir -p ~/.deepagents/agent/skills
cp -r examples/skills/web-research ~/.deepagents/agent/skills/
```

To manage skills:

```bash
# List all skills (global + project)
deepagents skills list

# List only project skills
deepagents skills list --project

# Create a new global skill from template
deepagents skills create my-skill

# Create a new project skill
deepagents skills create my-tool --project

# View detailed information about a skill
deepagents skills info web-research

# View info for a project skill only
deepagents skills info my-tool --project
```

To use skills (e.g., the langgraph-docs skill), just type a request relevant to a skill and the skill will be used automatically.

```bash
deepagents 
"create a agent.py script that implements a LangGraph agent" 
```

Skills follow Anthropic's [progressive disclosure pattern](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) - the agent knows skills exist but only reads full instructions when needed.

1. **At startup** - SkillsMiddleware scans `~/.deepagents/agent/skills/` and `.deepagents/skills/` directories
2. **Parse metadata** - Extracts YAML frontmatter (name + description) from each `SKILL.md` file
3. **Inject into prompt** - Adds skill list with descriptions to system prompt: "Available Skills: web-research - Use for web research tasks..."
4. **Progressive loading** - Agent reads full `SKILL.md` content with `read_file` only when a task matches the skill's description
5. **Execute workflow** - Agent follows the step-by-step instructions in the skill file

## Development

### Running Tests

To run the test suite:

```bash
uv sync --all-groups

make test
```

### Running During Development

```bash
# From libs/deepagents-cli directory
uv run deepagents

# Or install in editable mode
uv pip install -e .
deepagents
```

### Modifying the CLI

- **UI changes** → Edit `ui.py` or `input.py`
- **Add new tools** → Edit `tools.py`
- **Change execution flow** → Edit `execution.py`
- **Add commands** → Edit `commands.py`
- **Agent configuration** → Edit `agent.py`
- **Skills system** → Edit `skills/` modules
- **Constants/colors** → Edit `config.py`
