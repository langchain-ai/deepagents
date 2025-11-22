# deepagents cli

This is the CLI for deepagents

## Memory & Configuration Structure

The CLI uses a dual-scope memory system with both **global** (per-agent) and **project-specific** configuration:

### Global Configuration

Each agent has its own global configuration directory at `~/.deepagents/<agent_name>/`:

```
~/.deepagents/<agent_name>/
  ├── agent.md              # Auto-loaded global personality/style
  ├── skills/               # Auto-loaded agent-specific skills
  │   ├── web-research/
  │   │   └── SKILL.md
  │   └── langgraph-docs/
  │       └── SKILL.md
```

- **agent.md**: Defines your agent's personality, style, and general instructions (applies to all projects)
- **skills/**: Reusable capabilities that can be invoked across any project

### Project-Specific Configuration

Projects can override or extend the global configuration with project-specific instructions:

```
my-project/
  ├── .git/
  └── .deepagents/
      └── agent.md
```

The CLI automatically detects project roots (via `.git`) and loads project-specific `agent.md` from `[project-root]/.deepagents/agent.md`.

Both global and project agent.md files are loaded together, allowing you to:
- Keep general coding style/preferences in global agent.md
- Add project-specific context, conventions, or guidelines in project agent.md

### How the System Prompt is Constructed

The CLI uses middleware to dynamically construct the system prompt on each model call:

1. **AgentMemoryMiddleware** (runs first):
   - **Prepends** the contents of both agent.md files:
     ```xml
     <user_memory>[~/.deepagents/{agent}/agent.md content]</user_memory>
     <project_memory>[{project}/.deepagents/agent.md content]</project_memory>
     ```
   - **Appends** memory management instructions (how to read/write memory files, decision framework)

2. **SkillsMiddleware** (runs second):
   - **Appends** list of available skills (name + description only, not full SKILL.md content)
   - **Appends** progressive disclosure instructions (how to read full SKILL.md when needed)

3. **Base System Prompt**:
   - Current working directory info
   - Skills directory location
   - Human-in-the-loop guidance

**Final prompt structure:**
```
<user_memory>...</user_memory>
<project_memory>...</project_memory>

[Base system prompt]

[Memory management instructions with project-scoped paths]

[Skills list + progressive disclosure instructions]
```

This approach ensures that agent.md contents are always loaded, while skills use progressive disclosure (metadata shown, full instructions read on-demand).

## Skills

Skills are reusable agent capabilities that can be loaded into the CLI. Each agent has its own skills directory at `~/.deepagents/{AGENT_NAME}/skills/`.

For the default agent (named `agent`), skills are stored in `~/.deepagents/agent/skills/`.

### Example Skills

Example skills are provided in the `examples/skills/` directory:

- **web-research** - Structured web research workflow with planning, parallel delegation, and synthesis
- **langgraph-docs** - LangGraph documentation lookup and guidance

To use an example skill with the default agent, copy it to your agent's skills directory:

```bash
mkdir -p ~/.deepagents/agent/skills
cp -r examples/skills/web-research ~/.deepagents/agent/skills/
```

For a custom agent, replace `agent` with your agent name:

```bash
mkdir -p ~/.deepagents/my-agent/skills
cp -r examples/skills/web-research ~/.deepagents/my-agent/skills/
```

### Managing Skills

```bash
# List available skills
deepagents skills list

# Create a new skill from template
deepagents skills create my-skill

# View detailed information about a skill
deepagents skills info web-research
```

## VS Code Diff Preview

The CLI supports optional VS Code integration to preview code changes as side-by-side diffs before approval.

### Requirements

- VS Code installed with the `code` command in your PATH
- Enable via `--vscode-diff` flag when starting the CLI

### Usage

Start the CLI with VS Code diff preview enabled:

```bash
deepagents --vscode-diff
```

### How It Works

When the agent suggests file changes (write_file or edit_file operations):

1. VS Code automatically opens showing a side-by-side diff
2. You can review the changes in VS Code with full syntax highlighting and editor features
3. Return to the terminal to approve or reject the changes
4. Temporary files are automatically cleaned up

### Benefits

- **Visual Feedback**: See proposed changes with syntax highlighting and editor context
- **Editor Features**: Use VS Code's search, navigation, and inspection tools while reviewing
- **Side-by-Side Comparison**: Clearly see what's being added, removed, or modified
- **Faster Review**: Review changes in your familiar editor environment

### Example

```bash
# Start CLI with VS Code diff preview
deepagents --vscode-diff

# When agent suggests a file change:
# 1. VS Code opens with diff view
# 2. Terminal shows approval prompt
# 3. Review in VS Code, then approve/reject in terminal
```

## Development

### Running Tests

To run the test suite:

```bash
uv sync --all-groups

make test
```
