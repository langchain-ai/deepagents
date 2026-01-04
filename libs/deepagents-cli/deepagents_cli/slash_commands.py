"""Custom slash commands support with Claude Code parity.

This module provides user-defined slash commands loaded from markdown files.
Commands can be defined at two levels:
- User-level: ~/.deepagents/{agent_name}/commands/*.md
- Project-level: .deepagents/commands/*.md (overrides user-level)

File format:
```markdown
---
description: Brief description shown in /help
allowed-tools: Bash(git add:*), Read, Grep
argument-hint: [pr-number] [priority]
model: claude-3-5-haiku-20241022
disable-model-invocation: false
---

Your prompt template here with $ARGUMENTS or $1, $2 placeholders.
Shell commands: !`git status`
File inclusion: @src/file.py
```
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SlashCommand:
    """Represents a user-defined slash command."""

    name: str
    template: str
    path: Path
    source: str  # "user" or "project"
    namespace: str | None = None  # subdirectory name
    description: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    argument_hint: str | None = None
    model: str | None = None
    disable_model_invocation: bool = False

    @property
    def display_source(self) -> str:
        """Get display string for command source (e.g., 'project:ci')."""
        if self.namespace:
            return f"{self.source}:{self.namespace}"
        return self.source


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    content = content.strip()

    # Check for frontmatter delimiter
    if not content.startswith("---"):
        return {}, content

    # Find end of frontmatter
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        # No closing delimiter, treat entire content as body
        return {}, content

    frontmatter_str = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        frontmatter = {}

    return frontmatter, body


def parse_allowed_tools(tools_str: str | list | None) -> list[str]:
    """Parse allowed-tools from frontmatter.

    Supports formats:
    - String: "Bash(git add:*), Read, Grep"
    - List: ["Bash(git add:*)", "Read", "Grep"]

    Args:
        tools_str: Raw allowed-tools value from frontmatter

    Returns:
        List of tool specifications
    """
    if not tools_str:
        return []

    if isinstance(tools_str, list):
        return [str(t).strip() for t in tools_str]

    # Parse comma-separated string, being careful with parentheses
    tools = []
    current = ""
    paren_depth = 0

    for char in str(tools_str):
        if char == "(":
            paren_depth += 1
            current += char
        elif char == ")":
            paren_depth -= 1
            current += char
        elif char == "," and paren_depth == 0:
            if current.strip():
                tools.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        tools.append(current.strip())

    return tools


def parse_command_file(path: Path, source: str, namespace: str | None = None) -> SlashCommand:
    """Parse a command markdown file.

    Args:
        path: Path to the .md file
        source: "user" or "project"
        namespace: Subdirectory name if any

    Returns:
        Parsed SlashCommand instance
    """
    content = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    # Extract command name from filename (without .md extension)
    name = path.stem.lower()

    # Handle argument-hint which might be parsed as list if it contains brackets
    argument_hint = frontmatter.get("argument-hint")
    if isinstance(argument_hint, list):
        # Convert list back to string format: ['env', 'version'] -> '[env] [version]'
        argument_hint = " ".join(f"[{item}]" for item in argument_hint)

    return SlashCommand(
        name=name,
        template=body,
        path=path,
        source=source,
        namespace=namespace,
        description=frontmatter.get("description"),
        allowed_tools=parse_allowed_tools(frontmatter.get("allowed-tools")),
        argument_hint=argument_hint,
        model=frontmatter.get("model"),
        disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
    )


def discover_commands_in_directory(
    commands_dir: Path, source: str
) -> dict[str, SlashCommand]:
    """Discover all commands in a directory (including subdirectories).

    Args:
        commands_dir: Path to commands directory
        source: "user" or "project"

    Returns:
        Dict mapping command name to SlashCommand
    """
    commands: dict[str, SlashCommand] = {}

    if not commands_dir.exists():
        return commands

    # Find all .md files in the directory and subdirectories
    for md_file in commands_dir.rglob("*.md"):
        try:
            # Determine namespace from subdirectory
            rel_path = md_file.relative_to(commands_dir)
            namespace = None
            if len(rel_path.parts) > 1:
                # File is in a subdirectory
                namespace = rel_path.parts[0]

            cmd = parse_command_file(md_file, source, namespace)
            commands[cmd.name] = cmd
        except Exception as e:
            logger.warning(f"Failed to parse command file {md_file}: {e}")

    return commands


def discover_commands(
    agent_name: str, project_root: Path | None = None
) -> dict[str, SlashCommand]:
    """Discover all available custom commands.

    Commands are loaded from:
    1. User-level: ~/.deepagents/{agent_name}/commands/
    2. Project-level: {project_root}/.deepagents/commands/

    Project commands override user commands with the same name.

    Args:
        agent_name: Name of the current agent
        project_root: Path to project root (if in a project)

    Returns:
        Dict mapping command name to SlashCommand
    """
    commands: dict[str, SlashCommand] = {}

    # Load user-level commands first
    user_commands_dir = Path.home() / ".deepagents" / agent_name / "commands"
    commands.update(discover_commands_in_directory(user_commands_dir, "user"))

    # Load project-level commands (override user)
    if project_root:
        project_commands_dir = project_root / ".deepagents" / "commands"
        commands.update(discover_commands_in_directory(project_commands_dir, "project"))

    return commands


def substitute_arguments(template: str, args: str) -> str:
    """Substitute argument placeholders in template.

    Supports:
    - $ARGUMENTS: All arguments as a single string
    - $1, $2, $3, ...: Positional arguments

    Args:
        template: Command template with placeholders
        args: Arguments string from user

    Returns:
        Template with placeholders substituted
    """
    result = template

    # Split args into positional arguments (respecting quotes)
    positional = _split_args(args)

    # Replace $ARGUMENTS with full args string
    result = result.replace("$ARGUMENTS", args)

    # Replace positional arguments $1, $2, etc.
    # Sort by number descending to avoid $1 matching $10
    # First, find all positional placeholders used in the template
    max_placeholder = 0
    for match in re.finditer(r"\$(\d+)", result):
        num = int(match.group(1))
        max_placeholder = max(max_placeholder, num)

    # Replace all placeholders from highest to lowest
    for i in range(max_placeholder, 0, -1):
        placeholder = f"${i}"
        if placeholder in result:
            value = positional[i - 1] if i <= len(positional) else ""
            result = result.replace(placeholder, value)

    return result


def _split_args(args: str) -> list[str]:
    """Split arguments string into positional arguments.

    Handles quoted strings as single arguments.

    Args:
        args: Arguments string

    Returns:
        List of positional arguments
    """
    if not args.strip():
        return []

    result = []
    current = ""
    in_quotes = False
    quote_char = None

    for char in args:
        if char in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char.isspace() and not in_quotes:
            if current:
                result.append(current)
                current = ""
        else:
            current += char

    if current:
        result.append(current)

    return result


def execute_shell_injections(template: str, cwd: Path | None = None) -> str:
    """Execute shell command injections in template.

    Replaces !`command` with the command's output.

    Args:
        template: Template with shell injections
        cwd: Working directory for commands

    Returns:
        Template with shell outputs injected
    """
    # Pattern matches !`command` including multiline commands
    pattern = r"!\`([^`]+)\`"

    def replace_shell(match: re.Match) -> str:
        command = match.group(1).strip()
        try:
            result = subprocess.run(
                command,
                check=False, shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd or Path.cwd(),
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"[Command timed out: {command}]"
        except Exception as e:
            return f"[Error executing command: {e}]"

    return re.sub(pattern, replace_shell, template)


def resolve_file_inclusions(
    template: str, cwd: Path | None = None, backend=None
) -> str:
    """Resolve file inclusions in template.

    Replaces @filepath with the file's contents.

    Args:
        template: Template with file references
        cwd: Working directory for relative paths
        backend: Optional backend for file reading

    Returns:
        Template with file contents injected
    """
    # Pattern matches @filepath (stops at whitespace or end of line)
    # Supports paths like @src/file.py, @./relative/path.md, @/absolute/path.txt
    pattern = r"@([\w./\-_]+)"

    working_dir = cwd or Path.cwd()

    def replace_file(match: re.Match) -> str:
        filepath = match.group(1)

        # Resolve path
        path = Path(filepath)
        if not path.is_absolute():
            path = working_dir / path

        try:
            if backend:
                # Use backend if available
                response = backend.read(str(path))
                if hasattr(response, "content"):
                    content = response.content
                    if isinstance(content, bytes):
                        return content.decode("utf-8")
                    if isinstance(content, list):
                        return "\n".join(str(line) for line in content)
                    return str(content)
            else:
                # Direct file read
                if path.exists():
                    return path.read_text(encoding="utf-8")
                return f"[File not found: {filepath}]"
        except Exception as e:
            return f"[Error reading file {filepath}: {e}]"

    return re.sub(pattern, replace_file, template)


def process_template(
    template: str,
    args: str,
    cwd: Path | None = None,
    backend=None,
    execute_shell: bool = True,
    resolve_files: bool = True,
) -> str:
    """Process a command template with all substitutions.

    Processing order:
    1. Substitute $ARGUMENTS and positional args
    2. Execute shell injections (!`command`)
    3. Resolve file inclusions (@filepath)

    Args:
        template: Command template
        args: Arguments from user
        cwd: Working directory
        backend: Optional backend for file operations
        execute_shell: Whether to execute shell injections
        resolve_files: Whether to resolve file inclusions

    Returns:
        Fully processed template
    """
    result = substitute_arguments(template, args)

    if execute_shell:
        result = execute_shell_injections(result, cwd)

    if resolve_files:
        result = resolve_file_inclusions(result, cwd, backend)

    return result


@dataclass
class CommandExecutionResult:
    """Result of executing a custom slash command."""

    prompt: str
    model: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    command: SlashCommand | None = None


def execute_command(
    cmd: SlashCommand,
    args: str,
    cwd: Path | None = None,
    backend=None,
) -> CommandExecutionResult:
    """Execute a custom slash command.

    Args:
        cmd: The command to execute
        args: Arguments from user
        cwd: Working directory
        backend: Optional backend for file operations

    Returns:
        CommandExecutionResult with processed prompt and metadata
    """
    processed = process_template(
        cmd.template,
        args,
        cwd=cwd,
        backend=backend,
    )

    return CommandExecutionResult(
        prompt=processed,
        model=cmd.model,
        allowed_tools=cmd.allowed_tools,
        command=cmd,
    )


def format_commands_for_help(commands: dict[str, SlashCommand]) -> list[tuple[str, str, str]]:
    """Format commands for display in /help.

    Args:
        commands: Dict of available commands

    Returns:
        List of (name, description, source) tuples sorted by name
    """
    result = []
    for name, cmd in sorted(commands.items()):
        desc = cmd.description or "(no description)"
        hint = f" {cmd.argument_hint}" if cmd.argument_hint else ""
        result.append((f"/{name}{hint}", desc, cmd.display_source))
    return result
