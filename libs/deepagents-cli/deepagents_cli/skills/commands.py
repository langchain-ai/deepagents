"""CLI commands for skill management.

These commands are registered with the CLI via cli.py:
- deepagents skills list --agent <agent> [--project]
- deepagents skills create <name>
- deepagents skills info <name>
"""

import argparse
import re
from pathlib import Path
from typing import Any

from deepagents_cli.config import COLORS, Settings, console
from deepagents_cli.skills.load import list_skills

MAX_SKILL_NAME_LENGTH = 64


def _validate_name(name: str) -> tuple[bool, str]:
    """Validate name per Agent Skills spec.

    Requirements (https://agentskills.io/specification):
    - Max 64 characters
    - Lowercase alphanumeric and hyphens only (a-z, 0-9, -)
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - No path traversal sequences

    Args:
        name: The name to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check for empty or whitespace-only names
    if not name or not name.strip():
        return False, "cannot be empty"

    # Check length (spec: max 64 chars)
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "cannot exceed 64 characters"

    # Check for path traversal sequences
    if ".." in name or "/" in name or "\\" in name:
        return False, "cannot contain path components"

    # Spec: lowercase alphanumeric and hyphens only
    # Pattern ensures: no start/end hyphen, no consecutive hyphens
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return (
            False,
            "must be lowercase letters, numbers, and hyphens only "
            "(no uppercase, no underscores, cannot start/end with hyphen)",
        )

    return True, ""


def _validate_skill_path(skill_dir: Path, base_dir: Path) -> tuple[bool, str]:
    """Validate that the resolved skill directory is within the base directory.

    Args:
        skill_dir: The skill directory path to validate
        base_dir: The base skills directory that should contain skill_dir

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        # Resolve both paths to their canonical form
        resolved_skill = skill_dir.resolve()
        resolved_base = base_dir.resolve()

        # Check if skill_dir is within base_dir
        # Use is_relative_to if available (Python 3.9+), otherwise use string comparison
        if hasattr(resolved_skill, "is_relative_to"):
            if not resolved_skill.is_relative_to(resolved_base):
                return False, f"Skill directory must be within {base_dir}"
        else:
            # Fallback for older Python versions
            try:
                resolved_skill.relative_to(resolved_base)
            except ValueError:
                return False, f"Skill directory must be within {base_dir}"

        return True, ""
    except (OSError, RuntimeError) as e:
        return False, f"Invalid path: {e}"


def _list(agent: str, *, project: bool = False) -> None:
    """List all available skills for the specified agent.

    Args:
        agent: Agent identifier for skills (default: agent).
        project: If True, show only project skills.
            If False, show all skills (user + project).
    """
    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    # If --project flag is used, only show project skills
    if project:
        if not project_skills_dir:
            console.print("[yellow]Not in a project directory.[/yellow]")
            console.print(
                "[dim]Project skills require a .git directory in the project root.[/dim]",
                style=COLORS["dim"],
            )
            return

        if not project_skills_dir.exists() or not any(project_skills_dir.iterdir()):
            console.print("[yellow]No project skills found.[/yellow]")
            console.print(
                f"[dim]Project skills will be created in {project_skills_dir}/ when you add them.[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]Create a project skill:\n  deepagents skills create my-skill --project[/dim]",
                style=COLORS["dim"],
            )
            return

        skills = list_skills(user_skills_dir=None, project_skills_dir=project_skills_dir)
        console.print("\n[bold]Project Skills:[/bold]\n", style=COLORS["primary"])
    else:
        # Load both user and project skills
        skills = list_skills(user_skills_dir=user_skills_dir, project_skills_dir=project_skills_dir)

        if not skills:
            console.print("[yellow]No skills found.[/yellow]")
            console.print(
                "[dim]Skills will be created in ~/.deepagents/agent/skills/ when you add them.[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]Create your first skill:\n  deepagents skills create my-skill[/dim]",
                style=COLORS["dim"],
            )
            return

        console.print("\n[bold]Available Skills:[/bold]\n", style=COLORS["primary"])

    # Group skills by source
    user_skills = [s for s in skills if s["source"] == "user"]
    project_skills_list = [s for s in skills if s["source"] == "project"]

    # Show user skills
    if user_skills and not project:
        console.print("[bold cyan]User Skills:[/bold cyan]", style=COLORS["primary"])
        for skill in user_skills:
            skill_path = Path(skill["path"])
            console.print(f"  • [bold]{skill['name']}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print(f"    Location: {skill_path.parent}/", style=COLORS["dim"])
            console.print()

    # Show project skills
    if project_skills_list:
        if not project and user_skills:
            console.print()
        console.print("[bold green]Project Skills:[/bold green]", style=COLORS["primary"])
        for skill in project_skills_list:
            skill_path = Path(skill["path"])
            console.print(f"  • [bold]{skill['name']}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print(f"    Location: {skill_path.parent}/", style=COLORS["dim"])
            console.print()


def _create(skill_name: str, agent: str, project: bool = False) -> None:
    """Create a new skill with a template SKILL.md file.

    Args:
        skill_name: Name of the skill to create.
        agent: Agent identifier for skills
        project: If True, create in project skills directory.
            If False, create in user skills directory.
    """
    # Validate skill name first (per Agent Skills spec)
    is_valid, error_msg = _validate_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] Invalid skill name: {error_msg}")
        console.print(
            "[dim]Per Agent Skills spec: names must be lowercase alphanumeric with hyphens only.\n"
            "Examples: web-research, code-review, data-analysis[/dim]",
            style=COLORS["dim"],
        )
        return

    # Determine target directory
    settings = Settings.from_environment()
    if project:
        if not settings.project_root:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            console.print(
                "[dim]Project skills require a .git directory in the project root.[/dim]",
                style=COLORS["dim"],
            )
            return
        skills_dir = settings.ensure_project_skills_dir()
    else:
        skills_dir = settings.ensure_user_skills_dir(agent)

    skill_dir = skills_dir / skill_name

    # Validate the resolved path is within skills_dir
    is_valid_path, path_error = _validate_skill_path(skill_dir, skills_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        return

    if skill_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] Skill '{skill_name}' already exists at {skill_dir}"
        )
        return

    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Create template SKILL.md (per Agent Skills spec: https://agentskills.io/specification)
    template = f"""---
name: {skill_name}
description: Brief description of what this skill does and when to use it.
# Optional fields per Agent Skills spec:
# license: Apache-2.0
# compatibility: Designed for deepagents CLI
# metadata:
#   author: your-org
#   version: "1.0"
# allowed-tools: Bash(git:*) Read
---

# {skill_name.title().replace("-", " ")} Skill

## Description

[Provide a detailed explanation of what this skill does and when it should be used]

## When to Use

- [Scenario 1: When the user asks...]
- [Scenario 2: When you need to...]
- [Scenario 3: When the task involves...]

## How to Use

### Step 1: [First Action]
[Explain what to do first]

### Step 2: [Second Action]
[Explain what to do next]

### Step 3: [Final Action]
[Explain how to complete the task]

## Best Practices

- [Best practice 1]
- [Best practice 2]
- [Best practice 3]

## Supporting Files

This skill directory can include supporting files referenced in the instructions:
- `helper.py` - Python scripts for automation
- `config.json` - Configuration files
- `reference.md` - Additional reference documentation

## Examples

### Example 1: [Scenario Name]

**User Request:** "[Example user request]"

**Approach:**
1. [Step-by-step breakdown]
2. [Using tools and commands]
3. [Expected outcome]

### Example 2: [Another Scenario]

**User Request:** "[Another example]"

**Approach:**
1. [Different approach]
2. [Relevant commands]
3. [Expected result]

## Notes

- [Additional tips, warnings, or context]
- [Known limitations or edge cases]
- [Links to external resources if helpful]
"""

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(template)

    console.print(f"✓ Skill '{skill_name}' created successfully!", style=COLORS["primary"])
    console.print(f"Location: {skill_dir}\n", style=COLORS["dim"])
    console.print(
        "[dim]Edit the SKILL.md file to customize:\n"
        "  1. Update the description in YAML frontmatter\n"
        "  2. Fill in the instructions and examples\n"
        "  3. Add any supporting files (scripts, configs, etc.)\n"
        "\n"
        f"  nano {skill_md}\n"
        "\n"
        "💡 See examples/skills/ in the deepagents repo for example skills:\n"
        "   - web-research: Structured research workflow\n"
        "   - langgraph-docs: LangGraph documentation lookup\n"
        "\n"
        "   Copy an example: cp -r examples/skills/web-research ~/.deepagents/agent/skills/\n",
        style=COLORS["dim"],
    )


def _info(skill_name: str, *, agent: str = "agent", project: bool = False) -> None:
    """Show detailed information about a specific skill.

    Args:
        skill_name: Name of the skill to show info for.
        agent: Agent identifier for skills (default: agent).
        project: If True, only search in project skills. If False, search in both user and project skills.
    """
    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()

    # Load skills based on --project flag
    if project:
        if not project_skills_dir:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            return
        skills = list_skills(user_skills_dir=None, project_skills_dir=project_skills_dir)
    else:
        skills = list_skills(user_skills_dir=user_skills_dir, project_skills_dir=project_skills_dir)

    # Find the skill
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        console.print("\n[dim]Available skills:[/dim]", style=COLORS["dim"])
        for s in skills:
            console.print(f"  - {s['name']}", style=COLORS["dim"])
        return

    # Read the full SKILL.md file
    skill_path = Path(skill["path"])
    skill_content = skill_path.read_text()

    # Determine source label
    source_label = "Project Skill" if skill["source"] == "project" else "User Skill"
    source_color = "green" if skill["source"] == "project" else "cyan"

    console.print(
        f"\n[bold]Skill: {skill['name']}[/bold] [bold {source_color}]({source_label})[/bold {source_color}]\n",
        style=COLORS["primary"],
    )
    console.print(f"[bold]Description:[/bold] {skill['description']}\n", style=COLORS["dim"])
    console.print(f"[bold]Location:[/bold] {skill_path.parent}/\n", style=COLORS["dim"])

    # List supporting files
    skill_dir = skill_path.parent
    supporting_files = [f for f in skill_dir.iterdir() if f.name != "SKILL.md"]

    if supporting_files:
        console.print("[bold]Supporting Files:[/bold]", style=COLORS["dim"])
        for file in supporting_files:
            console.print(f"  - {file.name}", style=COLORS["dim"])
        console.print()

    # Show the full SKILL.md content
    console.print("[bold]Full SKILL.md Content:[/bold]\n", style=COLORS["primary"])
    console.print(skill_content, style=COLORS["dim"])
    console.print()


def setup_skills_parser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    """Setup the skills subcommand parser with all its subcommands."""
    skills_parser = subparsers.add_parser(
        "skills",
        help="Manage agent skills",
        description="Manage agent skills - create, list, and view skill information",
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_command", help="Skills command")

    # Skills list
    list_parser = skills_subparsers.add_parser(
        "list", help="List all available skills", description="List all available skills"
    )
    list_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    list_parser.add_argument(
        "--project",
        action="store_true",
        help="Show only project-level skills",
    )

    # Skills create
    create_parser = skills_subparsers.add_parser(
        "create",
        help="Create a new skill",
        description="Create a new skill with a template SKILL.md file",
    )
    create_parser.add_argument("name", help="Name of the skill to create (e.g., web-research)")
    create_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    create_parser.add_argument(
        "--project",
        action="store_true",
        help="Create skill in project directory instead of user directory",
    )

    # Skills info
    info_parser = skills_subparsers.add_parser(
        "info",
        help="Show detailed information about a skill",
        description="Show detailed information about a specific skill",
    )
    info_parser.add_argument("name", help="Name of the skill to show info for")
    info_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    info_parser.add_argument(
        "--project",
        action="store_true",
        help="Search only in project skills",
    )
    return skills_parser


def execute_skills_command(args: argparse.Namespace) -> None:
    """Execute skills subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments with skills_command attribute
    """
    # validate agent argument
    if args.agent:
        is_valid, error_msg = _validate_name(args.agent)
        if not is_valid:
            console.print(f"[bold red]Error:[/bold red] Invalid agent name: {error_msg}")
            console.print(
                "[dim]Agent names must only contain letters, numbers, hyphens, and underscores.[/dim]",
                style=COLORS["dim"],
            )
            return

    if args.skills_command == "list":
        _list(agent=args.agent, project=args.project)
    elif args.skills_command == "create":
        _create(args.name, agent=args.agent, project=args.project)
    elif args.skills_command == "info":
        _info(args.name, agent=args.agent, project=args.project)
    else:
        # No subcommand provided, show help
        console.print("[yellow]Please specify a skills subcommand: list, create, or info[/yellow]")
        console.print("\n[bold]Usage:[/bold]", style=COLORS["primary"])
        console.print("  deepagents skills <command> [options]\n")
        console.print("[bold]Available commands:[/bold]", style=COLORS["primary"])
        console.print("  list              List all available skills")
        console.print("  create <name>     Create a new skill")
        console.print("  info <name>       Show detailed information about a skill")
        console.print("\n[bold]Examples:[/bold]", style=COLORS["primary"])
        console.print("  deepagents skills list")
        console.print("  deepagents skills create web-research")
        console.print("  deepagents skills info web-research")
        console.print("\n[dim]For more help on a specific command:[/dim]", style=COLORS["dim"])
        console.print("  deepagents skills <command> --help", style=COLORS["dim"])


__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
