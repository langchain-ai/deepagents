"""Utilities for accurate token counting using LangChain models."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from deepagents_cli.config import console, settings

# Long-term Memory System Prompt Template
# Used for token counting to accurately estimate context size
LONGTERM_MEMORY_SYSTEM_PROMPT = """

## Long-term Memory

Your long-term memory is stored in files on the filesystem and persists across sessions.

**User Memory Location**: `{agent_dir_absolute}` (displays as `{agent_dir_display}`)
**Project Memory Location**: {project_memory_info}

Your system prompt is loaded from TWO sources at startup:
1. **User agent.md**: `{agent_dir_absolute}/agent.md` - Your personal preferences across all projects
2. **Project agent.md**: Loaded from project root if available - Project-specific instructions

Project-specific agent.md is loaded from these locations (both combined if both exist):
- `[project-root]/.deepagents/agent.md` (preferred)
- `[project-root]/agent.md` (fallback, but also included if both exist)

**When to CHECK/READ memories (CRITICAL - do this FIRST):**
- **At the start of ANY new session**: Check both user and project memories
  - User: `ls {agent_dir_absolute}`
  - Project: `ls {project_deepagents_dir}` (if in a project)
- **BEFORE answering questions**: If asked "what do you know about X?" or "how do I do Y?", check project memories FIRST, then user
- **When user asks you to do something**: Check if you have project-specific guides or examples
- **When user references past work**: Search project memory files for related context

**Memory-first response pattern:**
1. User asks a question → Check project directory first: `ls {project_deepagents_dir}`
2. If relevant files exist → Read them with `read_file '{project_deepagents_dir}/[filename]'`
3. Check user memory if needed → `ls {agent_dir_absolute}`
4. Base your answer on saved knowledge supplemented by general knowledge

**When to update memories:**
- **IMMEDIATELY when the user describes your role or how you should behave**
- **IMMEDIATELY when the user gives feedback on your work** - Update memories to capture what was wrong and how to do it better
- When the user explicitly asks you to remember something
- When patterns or preferences emerge (coding styles, conventions, workflows)
- After significant work where context would help in future sessions

**Learning from feedback:**
- When user says something is better/worse, capture WHY and encode it as a pattern
- Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions
- When user says "you should remember X" or "be careful about Y", treat this as HIGH PRIORITY - update memories IMMEDIATELY
- Look for the underlying principle behind corrections, not just the specific mistake

## Deciding Where to Store Memory

When writing or updating agent memory, decide whether each fact, configuration, or behavior belongs in:

### User Agent File: `{agent_dir_absolute}/agent.md`
→ Describes the agent's **personality, style, and universal behavior** across all projects.

**Store here:**
- Your general tone and communication style
- Universal coding preferences (formatting, comment style, etc.)
- General workflows and methodologies you follow
- Tool usage patterns that apply everywhere
- Personal preferences that don't change per-project

**Examples:**
- "Be concise and direct in responses"
- "Always use type hints in Python"
- "Prefer functional programming patterns"

### Project Agent File: `{project_deepagents_dir}/agent.md`
→ Describes **how this specific project works** and **how the agent should behave here only.**

**Store here:**
- Project-specific architecture and design patterns
- Coding conventions specific to this codebase
- Project structure and organization
- Testing strategies for this project
- Deployment processes and workflows
- Team conventions and guidelines

**Examples:**
- "This project uses FastAPI with SQLAlchemy"
- "Tests go in tests/ directory mirroring src/ structure"
- "All API changes require updating OpenAPI spec"

### Project Memory Files: `{project_deepagents_dir}/*.md`
→ Use for **project-specific reference information** and structured notes.

**Store here:**
- API design documentation
- Architecture decisions and rationale
- Deployment procedures
- Common debugging patterns
- Onboarding information

**Examples:**
- `{project_deepagents_dir}/api-design.md` - REST API patterns used
- `{project_deepagents_dir}/architecture.md` - System architecture overview
- `{project_deepagents_dir}/deployment.md` - How to deploy this project

### File Operations:

**User memory:**
```
ls {agent_dir_absolute}                              # List user memory files
read_file '{agent_dir_absolute}/agent.md'            # Read user preferences
edit_file '{agent_dir_absolute}/agent.md' ...        # Update user preferences
```

**Project memory (preferred for project-specific information):**
```
ls {project_deepagents_dir}                          # List project memory files
read_file '{project_deepagents_dir}/agent.md'        # Read project instructions
edit_file '{project_deepagents_dir}/agent.md' ...    # Update project instructions
write_file '{project_deepagents_dir}/agent.md' ...  # Create project memory file
```

**Important**:
- Project memory files are stored in `.deepagents/` inside the project root
- Always use absolute paths for file operations
- Check project memories BEFORE user when answering project-specific questions"""


def calculate_baseline_tokens(model, agent_dir: Path, system_prompt: str, assistant_id: str) -> int:
    """Calculate baseline context tokens using the model's official tokenizer.

    This uses the model's get_num_tokens_from_messages() method to get
    accurate token counts for the initial context (system prompt + agent.md).

    Note: Tool definitions cannot be accurately counted before the first API call
    due to LangChain limitations. They will be included in the total after the
    first message is sent (~5,000 tokens).

    Args:
        model: LangChain model instance (ChatAnthropic or ChatOpenAI)
        agent_dir: Path to agent directory containing agent.md
        system_prompt: The base system prompt string
        assistant_id: The agent identifier for path references

    Returns:
        Token count for system prompt + agent.md (tools not included)
    """
    # Load user agent.md content
    agent_md_path = agent_dir / "agent.md"
    user_memory = ""
    if agent_md_path.exists():
        user_memory = agent_md_path.read_text()

    # Load project agent.md content
    from .config import _find_project_agent_md, _find_project_root

    project_memory = ""
    project_root = _find_project_root()
    if project_root:
        project_md_paths = _find_project_agent_md(project_root)
        if project_md_paths:
            try:
                # Combine all project agent.md files (if multiple exist)
                contents = []
                for path in project_md_paths:
                    contents.append(path.read_text())
                project_memory = "\n\n".join(contents)
            except Exception:
                pass

    # Build the complete system prompt as it will be sent
    # This mimics what AgentMemoryMiddleware.wrap_model_call() does
    memory_section = (
        f"<user_memory>\n{user_memory or '(No user agent.md)'}\n</user_memory>\n\n"
        f"<project_memory>\n{project_memory or '(No project agent.md)'}\n</project_memory>"
    )

    # Get the long-term memory system prompt
    memory_system_prompt = get_memory_system_prompt(
        assistant_id, project_root, bool(project_memory)
    )

    # Combine all parts in the same order as the middleware
    full_system_prompt = memory_section + "\n\n" + system_prompt + "\n\n" + memory_system_prompt

    # Count tokens using the model's official method
    messages = [SystemMessage(content=full_system_prompt)]

    try:
        # Note: tools parameter is not supported by LangChain's token counting
        # Tool tokens will be included in the API response after first message
        return model.get_num_tokens_from_messages(messages)
    except Exception as e:
        # Fallback if token counting fails
        console.print(f"[yellow]Warning: Could not calculate baseline tokens: {e}[/yellow]")
        return 0


def get_memory_system_prompt(
    assistant_id: str, project_root: Path | None = None, has_project_memory: bool = False
) -> str:
    """Get the long-term memory system prompt text.

    Args:
        assistant_id: The agent identifier for path references
        project_root: Path to the detected project root (if any)
        has_project_memory: Whether project memory was loaded
    """
    agent_dir = settings.get_agent_dir(assistant_id)
    agent_dir_absolute = str(agent_dir)
    agent_dir_display = f"~/.deepagents/{assistant_id}"

    # Build project memory info
    if project_root and has_project_memory:
        project_memory_info = f"`{project_root}` (detected)"
    elif project_root:
        project_memory_info = f"`{project_root}` (no agent.md found)"
    else:
        project_memory_info = "None (not in a git project)"

    # Build project deepagents directory path
    if project_root:
        project_deepagents_dir = f"{project_root}/.deepagents"
    else:
        project_deepagents_dir = "[project-root]/.deepagents (not in a project)"

    return LONGTERM_MEMORY_SYSTEM_PROMPT.format(
        agent_dir_absolute=agent_dir_absolute,
        agent_dir_display=agent_dir_display,
        project_memory_info=project_memory_info,
        project_deepagents_dir=project_deepagents_dir,
    )
