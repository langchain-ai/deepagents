"""Example demonstrating the Skills Middleware.

This example shows how to:
1. Set up skills in a filesystem
2. Configure the SkillsMiddleware
3. Use skills with the deep agent
"""

import asyncio
import tempfile
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware, SkillMetadata


def create_sample_skills(base_path: Path) -> None:
    """Create sample skills for demonstration."""
    # Create user skills directory
    user_skills = base_path / "skills" / "user"
    user_skills.mkdir(parents=True, exist_ok=True)

    # Create web-research skill
    web_research = user_skills / "web-research"
    web_research.mkdir(exist_ok=True)
    (web_research / "SKILL.md").write_text("""---
name: web-research
description: Structured approach to conducting thorough web research on any topic
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
- User needs comprehensive information gathering
- User wants synthesized findings from multiple sources

## Workflow
1. **Understand the query**: Clarify what information the user needs
2. **Search broadly**: Start with general searches to understand the landscape
3. **Deep dive**: Follow promising leads with more specific searches
4. **Organize findings**: Structure information logically
5. **Synthesize**: Provide a coherent summary with key insights

## Best Practices
- Always cite your sources
- Note any conflicting information found
- Highlight areas where information may be outdated
- Suggest follow-up topics if relevant
""")

    # Create code-review skill
    code_review = user_skills / "code-review"
    code_review.mkdir(exist_ok=True)
    (code_review / "SKILL.md").write_text("""---
name: code-review
description: Systematic code review process focusing on quality, security, and best practices
license: MIT
metadata:
  author: Example Team
  version: "1.0"
---

# Code Review Skill

## When to Use
- User asks you to review code
- Before submitting a pull request
- When debugging issues
- During refactoring discussions

## Review Checklist

### 1. Code Quality
- [ ] Clear, descriptive variable/function names
- [ ] Appropriate comments where needed
- [ ] DRY principle followed
- [ ] Single responsibility principle

### 2. Security
- [ ] No hardcoded secrets
- [ ] Input validation in place
- [ ] SQL injection prevention
- [ ] XSS prevention for web code

### 3. Performance
- [ ] No obvious N+1 queries
- [ ] Efficient algorithms used
- [ ] Appropriate caching considered

## Output Format
Provide feedback in this structure:
1. **Critical Issues** (must fix)
2. **Suggestions** (recommended improvements)
3. **Praise** (what's done well)
""")

    # Create project skills directory with a skill
    project_skills = base_path / "skills" / "project"
    project_skills.mkdir(parents=True, exist_ok=True)

    # Create data-analysis skill
    data_analysis = project_skills / "data-analysis"
    data_analysis.mkdir(exist_ok=True)
    (data_analysis / "SKILL.md").write_text("""---
name: data-analysis
description: Data analysis and visualization workflows for exploring datasets
license: Apache-2.0
allowed-tools: read_file execute
---

# Data Analysis Skill

## When to Use
- User provides a dataset to analyze
- User wants statistical insights
- User needs data visualizations

## Workflow
1. **Load Data**: Read the dataset and understand its structure
2. **Explore**: Check data types, missing values, distributions
3. **Clean**: Handle missing data, outliers, type conversions
4. **Analyze**: Perform statistical analysis as needed
5. **Visualize**: Create appropriate charts and graphs
6. **Report**: Summarize findings with actionable insights

## Helper Scripts
This skill includes helper scripts in the skill directory.
""")
    (data_analysis / "helpers.py").write_text("""# Helper functions for data analysis
import json

def summarize_numeric(data: list[float]) -> dict:
    \"\"\"Calculate basic statistics for numeric data.\"\"\"
    if not data:
        return {}
    sorted_data = sorted(data)
    n = len(data)
    return {
        "count": n,
        "mean": sum(data) / n,
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "median": sorted_data[n // 2],
    }
""")


def demonstrate_skills_loading() -> None:
    """Demonstrate how skills are loaded and displayed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)

        # Create sample skills
        create_sample_skills(base_path)

        # Create a filesystem backend
        backend = FilesystemBackend(root_dir=str(base_path), virtual_mode=False)

        # Create the skills middleware
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[
                {"path": str(base_path / "skills" / "user"), "name": "user"},
                {"path": str(base_path / "skills" / "project"), "name": "project"},
            ],
        )

        # Simulate before_agent to load skills
        state_update = middleware.before_agent({}, None)  # type: ignore

        print("=" * 60)
        print("SKILLS LOADED")
        print("=" * 60)

        if state_update and "skills_metadata" in state_update:
            skills: list[SkillMetadata] = state_update["skills_metadata"]
            for skill in skills:
                print(f"\nSkill: {skill['name']}")
                print(f"  Description: {skill['description']}")
                print(f"  Path: {skill['path']}")
                print(f"  Registry: {skill['registry']}")
                if skill.get("allowed_tools"):
                    print(f"  Allowed Tools: {skill['allowed_tools']}")

        # Show what the system prompt would look like
        print("\n" + "=" * 60)
        print("SYSTEM PROMPT INJECTION")
        print("=" * 60)

        skills_locations = middleware._format_skills_locations()
        print("\nSkills Locations:")
        print(skills_locations)

        if state_update:
            skills_list = middleware._format_skills_list(state_update["skills_metadata"])
            print("\nSkills List:")
            print(skills_list)


def show_integration_example() -> None:
    """Show how to integrate SkillsMiddleware with create_deep_agent."""
    print("\n" + "=" * 60)
    print("INTEGRATION EXAMPLE")
    print("=" * 60)
    print("""
To integrate SkillsMiddleware with create_deep_agent:

```python
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware

# Create a backend for skills (persistent storage recommended)
skills_backend = FilesystemBackend(root_dir="/path/to/skills", virtual_mode=False)

# Create skills middleware
skills_middleware = SkillsMiddleware(
    backend=skills_backend,
    registries=[
        {"path": "/path/to/skills/user", "name": "user"},
        {"path": "/path/to/skills/project", "name": "project"},
    ],
)

# Create the agent with skills middleware
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    middleware=[skills_middleware],  # Add to middleware list
    system_prompt="You are a helpful assistant with access to skills.",
)

# The agent will now:
# 1. Load skills metadata before each interaction
# 2. See available skills in its system prompt
# 3. Be able to read full skill instructions using read_file
# 4. Follow skill workflows when appropriate
```
""")


if __name__ == "__main__":
    demonstrate_skills_loading()
    show_integration_example()
