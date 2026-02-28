"""CLI commands for skill management.

These commands are registered with the CLI via main.py:
- deepagents skills list [options]
- deepagents skills create <name> [options]
- deepagents skills info <name> [options]
- deepagents skills delete <name> [options]
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from deepagents.middleware.skills import SkillMetadata

from deepagents_cli.config import COLORS, Settings, console, get_glyphs
from deepagents_cli.ui import (
    build_help_parent,
    show_skills_create_help,
    show_skills_delete_help,
    show_skills_help,
    show_skills_info_help,
    show_skills_list_help,
)

MAX_SKILL_NAME_LENGTH = 64


def _validate_name(name: str) -> tuple[bool, str]:
    """Validate name per Agent Skills spec.

    Requirements (https://agentskills.io/specification):
    - Max 64 characters
    - Unicode lowercase alphanumeric and hyphens only
    - Cannot start or end with hyphen
    - No consecutive hyphens
    - No path traversal sequences

    Unicode lowercase alphanumeric means any character where
    `c.isalpha() and c.islower()` or `c.isdigit()` returns `True`,
    which covers accented Latin characters (e.g., `'cafe'`,
    `'uber-tool'`) and other scripts.  This matches the SDK's
    `_validate_skill_name` implementation.

    Args:
        name: The name to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    # Check for empty or whitespace-only names
    if not name or not name.strip():
        return False, "cannot be empty"

    # Check length (spec: max 64 chars)
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "cannot exceed 64 characters"

    # Check for path traversal sequences (CLI-specific; the SDK validates
    # against the directory name instead, but the CLI accepts user input
    # directly so we need explicit path-safety checks)
    if ".." in name or "/" in name or "\\" in name:
        return False, "cannot contain path components"

    # Structural hyphen checks
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return (
            False,
            "must be lowercase alphanumeric with single hyphens only",
        )

    # Character-by-character check (matches SDK's _validate_skill_name)
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return (
            False,
            "must be lowercase alphanumeric with single hyphens only",
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
        if not resolved_skill.is_relative_to(resolved_base):
            return False, f"Skill directory must be within {base_dir}"
    except (OSError, RuntimeError) as e:
        return False, f"Invalid path: {e}"
    else:
        return True, ""


def _format_info_fields(skill: SkillMetadata) -> list[tuple[str, str]]:
    """Extract non-empty optional metadata fields for display.

    The upstream `_parse_skill_metadata` normalises empty/whitespace license
    and compatibility values to `None`, so the truthy checks below are
    sufficient.

    Args:
        skill: Skill metadata to extract display fields from.

    Returns:
        Ordered list of (label, value) tuples for non-empty fields.
            Fields appear in order: License, Compatibility, Allowed Tools,
            Metadata.
    """
    fields: list[tuple[str, str]] = []
    license_val = skill.get("license")
    if license_val:
        fields.append(("License", license_val))
    compat_val = skill.get("compatibility")
    if compat_val:
        fields.append(("Compatibility", compat_val))
    if skill.get("allowed_tools"):
        fields.append(
            ("Allowed Tools", ", ".join(str(t) for t in skill["allowed_tools"]))
        )
    meta = skill.get("metadata")
    if meta and isinstance(meta, dict):
        formatted = ", ".join(f"{k}={v}" for k, v in meta.items())
        fields.append(("Metadata", formatted))
    return fields


def _list(agent: str, *, project: bool = False) -> None:
    """List all available skills for the specified agent.

    Args:
        agent: Agent identifier for skills (default: agent).
        project: If True, show only project skills.
            If False, show all skills (user + project).
    """
    # Deferred: skills.load imports the deepagents SDK. This module is
    # imported at CLI startup for setup_skills_parser(), so a top-level
    # import here would penalize every command (e.g. `--help`).
    from deepagents_cli.skills.load import list_skills

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # If --project flag is used, only show project skills
    if project:
        if not project_skills_dir:
            console.print("[yellow]Not in a project directory.[/yellow]")
            console.print(
                "[dim]Project skills require a .git directory "
                "in the project root.[/dim]",
                style=COLORS["dim"],
            )
            return

        # Check both project skill directories
        has_deepagents_skills = project_skills_dir.exists() and any(
            project_skills_dir.iterdir()
        )
        has_agent_skills = (
            project_agent_skills_dir
            and project_agent_skills_dir.exists()
            and any(project_agent_skills_dir.iterdir())
        )

        if not has_deepagents_skills and not has_agent_skills:
            console.print("[yellow]No project skills found.[/yellow]")
            console.print(
                f"[dim]Project skills will be created in {project_skills_dir}/ "
                "when you add them.[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]Create a project skill:\n"
                "  deepagents skills create my-skill --project[/dim]",
                style=COLORS["dim"],
            )
            return

        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
        console.print("\n[bold]Project Skills:[/bold]\n", style=COLORS["primary"])
    else:
        # Load skills from all directories (including built-in)
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

        if not skills:
            console.print()
            console.print("[yellow]No skills found.[/yellow]")
            console.print()
            console.print(
                "[dim]Skills are loaded from these directories "
                "(highest precedence first):\n"
                "  1. .agents/skills/                 project skills\n"
                "  2. .deepagents/skills/             project skills (alias)\n"
                "  3. ~/.agents/skills/               user skills\n"
                "  4. ~/.deepagents/<agent>/skills/   user skills (alias)\n"
                "  5. <package>/built_in_skills/      built-in skills[/dim]",
                style=COLORS["dim"],
            )
            console.print(
                "\n[dim]Create your first skill:\n"
                "  deepagents skills create my-skill[/dim]",
                style=COLORS["dim"],
            )
            return

        console.print("\n[bold]Available Skills:[/bold]\n", style=COLORS["primary"])

    # Group skills by source
    user_skills = [s for s in skills if s["source"] == "user"]
    project_skills_list = [s for s in skills if s["source"] == "project"]
    built_in_skills_list = [s for s in skills if s["source"] == "built-in"]

    # Show user skills
    if user_skills and not project:
        console.print("[bold cyan]User Skills:[/bold cyan]", style=COLORS["primary"])
        bullet = get_glyphs().bullet
        for skill in user_skills:
            skill_path = Path(skill["path"])
            name = skill["name"]
            console.print(f"  {bullet} [bold]{name}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill_path.parent}/", style=COLORS["dim"])
            console.print()
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print()

    # Show project skills
    if project_skills_list:
        if not project and user_skills:
            console.print()
        console.print(
            "[bold green]Project Skills:[/bold green]", style=COLORS["primary"]
        )
        bullet = get_glyphs().bullet
        for skill in project_skills_list:
            skill_path = Path(skill["path"])
            name = skill["name"]
            console.print(f"  {bullet} [bold]{name}[/bold]", style=COLORS["primary"])
            console.print(f"    {skill_path.parent}/", style=COLORS["dim"])
            console.print()
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print()

    # Show built-in skills
    if built_in_skills_list and not project:
        if user_skills or project_skills_list:
            console.print()
        console.print(
            "[bold magenta]Built-in Skills:[/bold magenta]", style=COLORS["primary"]
        )
        bullet = get_glyphs().bullet
        for skill in built_in_skills_list:
            name = skill["name"]
            console.print(f"  {bullet} [bold]{name}[/bold]", style=COLORS["primary"])
            console.print()
            console.print(f"    {skill['description']}", style=COLORS["dim"])
            console.print()


def _generate_template(skill_name: str) -> str:
    """Generate a `SKILL.md` template for a new skill.

    The template follows the Agent Skills spec
    (https://agentskills.io/specification) and the skill-creator guidance:

    - Description includes "when to use" trigger information (not the body)
    - Body contains only instructions loaded after the skill triggers

    Args:
        skill_name: Name of the skill (used in frontmatter and heading).

    Returns:
        Complete `SKILL.md` content with YAML frontmatter and markdown body.
    """
    title = skill_name.title().replace("-", " ")
    description = (
        "TODO: Explain what this skill does and when to use it. "
        "Include specific triggers — scenarios, file types, or phrases "
        "that should activate this skill. Example: 'Create and edit PDF "
        "documents. Use when the user asks to merge, split, fill, or "
        "annotate PDF files.'"
    )
    return f"""---
name: {skill_name}
description: "{description}"
# (Warning: SKILL.md files exceeding 10 MB are silently skipped at load time.)
# Optional fields per Agent Skills spec:
# license: Apache-2.0
# compatibility: Designed for Deep Agents CLI
# metadata:
#   author: your-org
#   version: "1.0"
# allowed-tools: Bash(git:*) Read
---

# {title}

## Overview

[TODO: 1-2 sentences explaining what this skill enables]

## Instructions

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

## Examples

### Example 1: [Scenario Name]

**User Request:** "[Example user request]"

**Approach:**
1. [Step-by-step breakdown]
2. [Using tools and commands]
3. [Expected outcome]
"""


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
            "[dim]Per Agent Skills spec: names must be lowercase alphanumeric "
            "with hyphens only.\n"
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
                "[dim]Project skills require a .git directory "
                "in the project root.[/dim]",
                style=COLORS["dim"],
            )
            return
        skills_dir = settings.ensure_project_skills_dir()
        if skills_dir is None:
            console.print(
                "[bold red]Error:[/bold red] Could not create project skills directory."
            )
            return
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
            f"[bold red]Error:[/bold red] Skill '{skill_name}' "
            f"already exists at {skill_dir}"
        )
        return

    # Create skill directory
    skill_dir.mkdir(parents=True, exist_ok=True)

    template = _generate_template(skill_name)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(template)

    checkmark = get_glyphs().checkmark
    console.print(
        f"\n[bold]{checkmark} Skill '{skill_name}' created successfully![/bold]",
        style=COLORS["primary"],
    )
    console.print(f"Location: {skill_dir}\n", style=COLORS["dim"])
    console.print(
        "[dim]Edit the SKILL.md file to customize:\n"
        "  1. Update the description in YAML frontmatter\n"
        "  2. Fill in the instructions and examples\n"
        "  3. Add any supporting files (scripts, configs, etc.)\n"
        "\n"
        f"  nano {skill_md}\n"
        "\n"
        "  See examples/skills/ in the deepagents-cli repo for example skills:\n"
        "   - web-research: Structured research workflow\n"
        "   - langgraph-docs: LangGraph documentation lookup\n"
        "\n"
        "   Copy an example:\n"
        "   cp -r examples/skills/web-research ~/.deepagents/agent/skills/\n",
        style=COLORS["dim"],
    )


def _info(skill_name: str, *, agent: str = "agent", project: bool = False) -> None:
    """Show detailed information about a specific skill.

    Args:
        skill_name: Name of the skill to show info for.
        agent: Agent identifier for skills (default: agent).
        project: If True, only search in project skills.
            If False, search in both user and project skills.
    """
    # Deferred: skills.load imports the deepagents SDK. This module is
    # imported at CLI startup for setup_skills_parser(), so a top-level
    # import here would penalize every command (e.g. `--help`).
    from deepagents_cli.skills.load import list_skills

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # Load skills based on --project flag
    if project:
        if not project_skills_dir:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            return
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    else:
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

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
    skill_content = skill_path.read_text(encoding="utf-8")

    # Determine source label
    source_labels = {
        "project": ("Project Skill", "green"),
        "user": ("User Skill", "cyan"),
        "built-in": ("Built-in Skill", "magenta"),
    }
    source_label, source_color = source_labels.get(skill["source"], ("Skill", "dim"))

    # Check if this project skill shadows a user skill with the same name.
    # This is a cosmetic hint — if the second list_skills() call fails
    # (e.g. permission error reading user dirs) we silently skip the warning
    # rather than crashing the entire `skills info` display.
    shadowed_user_skill = False
    if skill["source"] == "project" and not project:
        try:
            user_only = list_skills(
                user_skills_dir=user_skills_dir,
                project_skills_dir=None,
                user_agent_skills_dir=user_agent_skills_dir,
                project_agent_skills_dir=None,
            )
            shadowed_user_skill = any(s["name"] == skill_name for s in user_only)
        except Exception:  # noqa: BLE001, S110  # Shadow detection is cosmetic, safe to swallow
            pass

    console.print(
        f"\n[bold]Skill: {skill['name']}[/bold] "
        f"[bold {source_color}]({source_label})[/bold {source_color}]\n",
        style=COLORS["primary"],
    )
    if shadowed_user_skill:
        console.print(
            f"[yellow]Note: Overrides user skill '{skill_name}' "
            "of the same name[/yellow]\n"
        )
    console.print(f"[bold]Location:[/bold] {skill_path.parent}/\n", style=COLORS["dim"])
    console.print(
        f"[bold]Description:[/bold] {skill['description']}\n", style=COLORS["dim"]
    )

    # Show optional metadata fields
    for label, value in _format_info_fields(skill):
        console.print(f"[bold]{label}:[/bold] {value}\n", style=COLORS["dim"])

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


def _delete(
    skill_name: str,
    *,
    agent: str = "agent",
    project: bool = False,
    force: bool = False,
) -> None:
    """Delete a skill directory after validation and optional user confirmation.

    Validates the skill name, locates the skill in user or project directories,
    confirms the deletion with the user (unless `force` is `True`), and
    recursively removes the skill directory.

    Args:
        skill_name: Name of the skill to delete.
        agent: Agent identifier for skills.
        project: If `True`, only search in project skills.

            If `False`, search in both user and project skills.
        force: If `True`, skip confirmation prompt.

    Raises:
        SystemExit: If the deletion fails or a safety check is violated.
    """
    # Validate skill name first (per Agent Skills spec)
    is_valid, error_msg = _validate_name(skill_name)
    if not is_valid:
        console.print(f"[bold red]Error:[/bold red] Invalid skill name: {error_msg}")
        return

    # Deferred: skills.load imports the deepagents SDK. This module is
    # imported at CLI startup for setup_skills_parser(), so a top-level
    # import here would penalize every command (e.g. `--help`).
    from deepagents_cli.skills.load import list_skills

    settings = Settings.from_environment()
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    # Load skills based on --project flag
    if project:
        if not project_skills_dir:
            console.print("[bold red]Error:[/bold red] Not in a project directory.")
            return
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    else:
        skills = list_skills(
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )

    # Find the skill
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        console.print("\n[dim]Available skills:[/dim]", style=COLORS["dim"])
        for s in skills:
            source_tag = "[project]" if s["source"] == "project" else "[user]"
            console.print(f"  - {s['name']} {source_tag}", style=COLORS["dim"])
        return

    skill_path = Path(skill["path"])
    skill_dir = skill_path.parent

    # Validate the path is safe to delete
    base_dir = project_skills_dir if skill["source"] == "project" else user_skills_dir
    if not base_dir:
        console.print(
            "[bold red]Error:[/bold red] Cannot determine base skills directory. "
            "Refusing to delete."
        )
        return
    is_valid_path, path_error = _validate_skill_path(skill_dir, base_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        return

    # Determine source label
    source_label = "Project Skill" if skill["source"] == "project" else "User Skill"
    source_color = "green" if skill["source"] == "project" else "cyan"

    # Count files for the confirmation summary (display-only; a permission
    # error in a subdirectory should not abort the entire delete flow).
    try:
        file_count = sum(1 for f in skill_dir.rglob("*") if f.is_file())
    except OSError:
        file_count = -1

    console.print(
        f"\n[bold]Skill:[/bold] {skill_name}"
        f" [bold {source_color}]({source_label})[/bold {source_color}]",
        style=COLORS["primary"],
    )
    console.print(
        f"[bold]Location:[/bold] {skill_dir}/",
        style=COLORS["dim"],
    )
    if file_count >= 0:
        console.print(
            f"[bold]Files:[/bold] {file_count} file(s) will be deleted\n",
            style=COLORS["dim"],
        )
    else:
        console.print(
            "[bold]Files:[/bold] (unable to count files)\n",
            style=COLORS["dim"],
        )

    # Confirmation
    if not force:
        console.print(
            "[yellow]Are you sure you want to delete this skill? (y/N)[/yellow] ",
            end="",
        )
        try:
            response = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return

        if response not in {"y", "yes"}:
            console.print("[dim]Cancelled.[/dim]")
            return

    # Re-validate immediately before deletion to narrow the TOCTOU window
    # (the user may have paused at the confirmation prompt).
    if skill_dir.is_symlink():
        console.print(
            "[bold red]Error:[/bold red] Skill directory is a symlink. "
            "Refusing to delete for safety."
        )
        raise SystemExit(1)

    is_valid_path, path_error = _validate_skill_path(skill_dir, base_dir)
    if not is_valid_path:
        console.print(f"[bold red]Error:[/bold red] {path_error}")
        raise SystemExit(1)

    # Delete the skill directory
    try:
        shutil.rmtree(skill_dir)
        checkmark = get_glyphs().checkmark
        console.print(
            f"{checkmark} Skill '{skill_name}' deleted successfully!",
            style=COLORS["primary"],
        )
    except OSError as e:
        console.print(
            f"[bold red]Error:[/bold red] Failed to fully delete skill: {e}\n"
            f"[yellow]Warning:[/yellow] Some files may have been partially removed.\n"
            f"Please inspect: {skill_dir}/"
        )
        raise SystemExit(1) from e


def setup_skills_parser(
    subparsers: Any,  # noqa: ANN401  # argparse subparsers uses dynamic typing
    *,
    make_help_action: Callable[[Callable[[], None]], type[argparse.Action]],
) -> argparse.ArgumentParser:
    """Setup the skills subcommand parser with all its subcommands.

    Each subcommand gets a dedicated help screen so that
    `deepagents skills -h` shows skills-specific help, not the
    global help.

    Args:
        subparsers: The parent subparsers object to add the skills parser to.
        make_help_action: Factory that accepts a zero-argument help
            callable and returns an argparse Action class wired to it.

    Returns:
        The skills subparser for argument handling.
    """
    help_parent = functools.partial(
        build_help_parent, make_help_action=make_help_action
    )

    skills_parser = subparsers.add_parser(
        "skills",
        help="Manage agent skills",
        description="Manage agent skills - list, create, view, and delete skills.",
        add_help=False,
        parents=help_parent(show_skills_help),
    )
    skills_subparsers = skills_parser.add_subparsers(
        dest="skills_command", help="Skills command"
    )

    # Skills list
    list_parser = skills_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List all available skills",
        description=(
            "List skills from all four skill directories "
            "(user, user alias, project, project alias)."
        ),
        add_help=False,
        parents=help_parent(show_skills_list_help),
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
        description=(
            "Create a new skill with a template SKILL.md file. "
            "By default, skills are created in "
            "~/.deepagents/<agent>/skills/. "
            "Use --project to create in the project's "
            ".deepagents/skills/ directory."
        ),
        add_help=False,
        parents=help_parent(show_skills_create_help),
    )
    create_parser.add_argument(
        "name",
        help="Name of the skill to create (e.g., web-research)",
    )
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
        add_help=False,
        parents=help_parent(show_skills_info_help),
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

    # Skills delete
    delete_parser = skills_subparsers.add_parser(
        "delete",
        help="Delete a skill",
        description="Delete a skill directory and all its contents",
        add_help=False,
        parents=help_parent(show_skills_delete_help),
    )
    delete_parser.add_argument("name", help="Name of the skill to delete")
    delete_parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for skills (default: agent)",
    )
    delete_parser.add_argument(
        "--project",
        action="store_true",
        help="Search only in project skills",
    )
    delete_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
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
            console.print(
                f"[bold red]Error:[/bold red] Invalid agent name: {error_msg}"
            )
            console.print(
                "[dim]Agent names must only contain letters, numbers, "
                "hyphens, and underscores.[/dim]",
                style=COLORS["dim"],
            )
            return

    # "ls" is an argparse alias for "list" — argparse stores the alias
    # as-is in the namespace, so we must match both values.
    if args.skills_command in {"list", "ls"}:
        _list(agent=args.agent, project=args.project)
    elif args.skills_command == "create":
        _create(args.name, agent=args.agent, project=args.project)
    elif args.skills_command == "info":
        _info(args.name, agent=args.agent, project=args.project)
    elif args.skills_command == "delete":
        _delete(args.name, agent=args.agent, project=args.project, force=args.force)
    else:
        # No subcommand provided, show skills help screen
        show_skills_help()


__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
