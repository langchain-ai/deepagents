"""CLI commands for skill management.

These commands are registered with the CLI via cli.py:
- deepagents skills list
- deepagents skills create <name>
- deepagents skills info <name>
"""

from pathlib import Path

from ..config import COLORS, console
from .skill_loader import SkillLoader


def list_skills():
    """List all available skills."""
    skills_dir = Path.home() / ".deepagents" / "skills"

    if not skills_dir.exists() or not any(skills_dir.iterdir()):
        console.print("[yellow]No skills found.[/yellow]")
        console.print(
            "[dim]Skills will be created in ~/.deepagents/skills/ when you add them.[/dim]",
            style=COLORS["dim"],
        )
        console.print(
            f"\n[dim]Create your first skill:\n  deepagents skills create my-skill[/dim]",
            style=COLORS["dim"],
        )
        return

    # Load skills
    loader = SkillLoader(skills_dir=skills_dir)
    skills = loader.list()

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
        console.print(f"    Location: ~/.deepagents/skills/{skill_dir_name}/", style=COLORS["dim"])
        console.print()


def create_skill(skill_name: str):
    """Create a new skill with a template SKILL.md file."""
    skills_dir = Path.home() / ".deepagents" / "skills"
    skill_dir = skills_dir / skill_name

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
        f"  nano {skill_md}\n",
        style=COLORS["dim"],
    )


def show_skill_info(skill_name: str):
    """Show detailed information about a specific skill."""
    skills_dir = Path.home() / ".deepagents" / "skills"

    # Load skills
    loader = SkillLoader(skills_dir=skills_dir)
    skills = loader.list()

    # Find the skill
    skill = next((s for s in skills if s["name"] == skill_name), None)

    if not skill:
        console.print(f"[bold red]Error:[/bold red] Skill '{skill_name}' not found.")
        console.print(f"\n[dim]Available skills:[/dim]", style=COLORS["dim"])
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
