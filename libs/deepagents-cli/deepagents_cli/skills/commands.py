"""CLI commands for skill management.

These commands are registered with the CLI via cli.py:
- deepagents skills list
- deepagents skills create <name>
- deepagents skills info <name>
"""

import argparse
import re
from pathlib import Path
from typing import Any

from deepagents_cli.config import COLORS, console
from deepagents_cli.skills.load import list_skills


def _validate_skill_name(skill_name: str) -> tuple[bool, str]:
    """Validate skill name to prevent path traversal attacks.

    Args:
        skill_name: The skill name to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check for empty or whitespace-only names
    if not skill_name or not skill_name.strip():
        return False, "Skill name cannot be empty"

    # Check for path traversal sequences
    if ".." in skill_name:
        return False, "Skill name cannot contain '..' (path traversal)"

    # Check for absolute paths
    if skill_name.startswith("/") or skill_name.startswith("\\"):
        return False, "Skill name cannot be an absolute path"

    # Check for path separators
    if "/" in skill_name or "\\" in skill_name:
        return False, "Skill name cannot contain path separators"

    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", skill_name):
        return False, "Skill name can only contain letters, numbers, hyphens, and underscores"

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


def _list() -> None:
    """List all available skills for the default agent."""
    # Use default agent's skills directory
    skills_dir = Path.home() / ".deepagents" / "agent" / "skills"

    if not skills_dir.exists() or not any(skills_dir.iterdir()):
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

    # Load skills
    skills = list_skills(skills_dir)

    if not skills:
        console.print("[yellow]No valid skills found.[/yellow]")
        console.print(
            "[dim]Skills must have a SKILL.md file with YAML frontmatter (name, description).[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Skills:[/bold]\n", style=COLORS["primary"])

    for skill in skills:
        skill_path = Path(skill["path"])
        skill_dir_name = skill_path.parent.name

        console.print(f"  • [bold]{skill['name']}[/bold]", style=COLORS["primary"])
        console.print(f"    {skill['description']}", style=COLORS["dim"])
        console.print(
            f"    Location: ~/.deepagents/agent/skills/{skill_dir_name}/", style=COLORS["dim"]
        )
        console.print()


def _create(skill_name: str) -> None:
    """Create a new skill with a template SKILL.md file for the default agent."""
    # Validate skill name first
    is_valid, error_msg = _validate_skill_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] Invalid skill name: {error_msg}")
        console.print(
            "[dim]Skill names must only contain letters, numbers, hyphens, and underscores.[/dim]",
            style=COLORS["dim"],
        )
        return

    # Use default agent's skills directory
    skills_dir = Path.home() / ".deepagents" / "agent" / "skills"
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

    # Create template SKILL.md
    template = f"""---
name: {skill_name}
description: [Brief description of what this skill does]
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


def _info(skill_name: str) -> None:
    """Show detailed information about a specific skill for the default agent."""
    # Use default agent's skills directory
    skills_dir = Path.home() / ".deepagents" / "agent" / "skills"

    # Load skills
    skills = list_skills(skills_dir)

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

    console.print(f"\n[bold]Skill: {skill['name']}[/bold]\n", style=COLORS["primary"])
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
    skills_subparsers.add_parser(
        "list", help="List all available skills", description="List all available skills"
    )

    # Skills create
    create_parser = skills_subparsers.add_parser(
        "create",
        help="Create a new skill",
        description="Create a new skill with a template SKILL.md file",
    )
    create_parser.add_argument("name", help="Name of the skill to create (e.g., web-research)")

    # Skills info
    info_parser = skills_subparsers.add_parser(
        "info",
        help="Show detailed information about a skill",
        description="Show detailed information about a specific skill",
    )
    info_parser.add_argument("name", help="Name of the skill to show info for")
    return skills_parser


def execute_skills_command(args: argparse.Namespace) -> None:
    """Execute skills subcommands based on parsed arguments.

    Args:
        args: Parsed command line arguments with skills_command attribute
    """
    if args.skills_command == "list":
        _list()
    elif args.skills_command == "create":
        _create(args.name)
    elif args.skills_command == "info":
        _info(args.name)
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
