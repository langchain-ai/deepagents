"""Complete demonstration of Skills Middleware - from setup to model view.

This script shows:
1. How to create skills (the file structure)
2. How to configure the middleware
3. What the model actually sees in its system prompt
4. How to verify everything works
"""

import tempfile
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware


def main():
    # =========================================================================
    # STEP 1: Create a skills directory structure
    # =========================================================================
    print("=" * 70)
    print("STEP 1: SKILL FILE STRUCTURE")
    print("=" * 70)
    print("""
A user would create skills in their project like this:

    my_project/
    ├── skills/
    │   ├── user/                    # Personal skills (lower priority)
    │   │   └── my-skill/
    │   │       └── SKILL.md         # Required file
    │   └── project/                 # Project skills (higher priority)
    │       └── another-skill/
    │           ├── SKILL.md         # Required file
    │           └── helper.py        # Optional supporting files
    └── main.py
    """)

    # Create a temporary directory to simulate this
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)

        # =====================================================================
        # STEP 2: Create actual skill files
        # =====================================================================
        print("=" * 70)
        print("STEP 2: WHAT A SKILL.md FILE LOOKS LIKE")
        print("=" * 70)

        # Create user skill directory
        user_skill_dir = base / "skills" / "user" / "web-research"
        user_skill_dir.mkdir(parents=True)

        skill_content = """---
name: web-research
description: Structured approach to researching topics on the web
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
- User needs comprehensive information

## Workflow
1. Clarify the research question
2. Search broadly first
3. Deep dive on promising leads
4. Synthesize findings

## Best Practices
- Always cite sources
- Note conflicting information
"""

        skill_file = user_skill_dir / "SKILL.md"
        skill_file.write_text(skill_content)

        print(f"\nCreated skill at: {skill_file}")
        print("\nSKILL.md contents:")
        print("-" * 40)
        print(skill_content)
        print("-" * 40)

        print("""
KEY POINTS about SKILL.md:
- YAML frontmatter between --- markers (required)
- 'name' must match directory name (e.g., 'web-research' dir = 'web-research' name)
- 'description' is what the model sees in the list
- Everything after frontmatter is the full instructions
""")

        # Create a project skill with helper file
        project_skill_dir = base / "skills" / "project" / "code-review"
        project_skill_dir.mkdir(parents=True)

        project_skill_content = """---
name: code-review
description: Systematic code review focusing on quality and security
allowed-tools: read_file edit_file
---

# Code Review Skill

## Checklist
1. Check for security issues
2. Review code style
3. Look for bugs
"""
        (project_skill_dir / "SKILL.md").write_text(project_skill_content)
        (project_skill_dir / "checklist.txt").write_text("- No hardcoded secrets\n- Input validation\n")

        # =====================================================================
        # STEP 3: Configure the middleware
        # =====================================================================
        print("=" * 70)
        print("STEP 3: CONFIGURING THE MIDDLEWARE")
        print("=" * 70)

        print("""
In your code, you would do:

```python
from deepagents import create_deep_agent, SkillsMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# Point to your skills directory
backend = FilesystemBackend(root_dir="./my_project")

# Configure registries (order matters - later overrides earlier)
middleware = SkillsMiddleware(
    backend=backend,
    registries=[
        {"path": "./my_project/skills/user", "name": "user"},
        {"path": "./my_project/skills/project", "name": "project"},
    ],
)

# Add to your agent
agent = create_deep_agent(middleware=[middleware])
```
""")

        # Actually create the middleware
        backend = FilesystemBackend(root_dir=str(base), virtual_mode=False)
        middleware = SkillsMiddleware(
            backend=backend,
            registries=[
                {"path": str(base / "skills" / "user"), "name": "user"},
                {"path": str(base / "skills" / "project"), "name": "project"},
            ],
        )

        # =====================================================================
        # STEP 4: How skills are loaded (before_agent hook)
        # =====================================================================
        print("=" * 70)
        print("STEP 4: HOW SKILLS ARE LOADED")
        print("=" * 70)

        print("""
When the agent starts, the middleware's `before_agent()` hook runs.
It scans all registries and loads skill metadata into state.
""")

        # Simulate what happens when agent starts
        state_update = middleware.before_agent({}, None)  # type: ignore

        print("Skills loaded into state:")
        print("-" * 40)
        for skill in state_update["skills_metadata"]:
            print(f"  Name: {skill['name']}")
            print(f"  Description: {skill['description']}")
            print(f"  Path: {skill['path']}")
            print(f"  Registry: {skill['registry']}")
            if skill.get("allowed_tools"):
                print(f"  Allowed Tools: {skill['allowed_tools']}")
            print()

        # =====================================================================
        # STEP 5: WHAT THE MODEL ACTUALLY SEES
        # =====================================================================
        print("=" * 70)
        print("STEP 5: WHAT THE MODEL SEES IN ITS SYSTEM PROMPT")
        print("=" * 70)

        print("""
The middleware injects a skills section into the system prompt.
Here's EXACTLY what gets added:
""")
        print("-" * 40)

        # Build the exact system prompt that would be injected
        skills_locations = middleware._format_skills_locations()
        skills_list = middleware._format_skills_list(state_update["skills_metadata"])

        full_skills_prompt = middleware.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        print(full_skills_prompt)
        print("-" * 40)

        # =====================================================================
        # STEP 6: HOW TO VERIFY IT WORKS
        # =====================================================================
        print("=" * 70)
        print("STEP 6: VERIFICATION")
        print("=" * 70)

        print("""
To verify skills work:

1. **Unit tests pass:**
   ```bash
   pytest tests/unit_tests/middleware/test_skills_middleware.py -v
   ```

2. **Integration tests pass:**
   ```bash
   pytest tests/integration_tests/test_skills_integration.py -v
   ```

3. **Manual verification** (what we just did above):
   - Skills are discovered from filesystem ✓
   - Metadata is parsed correctly ✓
   - System prompt is formatted correctly ✓

4. **With a real agent** (requires API key):
   ```python
   agent = create_deep_agent(middleware=[middleware])
   result = agent.invoke({
       "messages": [{"role": "user", "content": "What skills do you have?"}]
   })
   # Agent should list the skills it sees
   ```
""")

        # =====================================================================
        # STEP 7: THE USER EXPERIENCE FLOW
        # =====================================================================
        print("=" * 70)
        print("STEP 7: USER EXPERIENCE FLOW")
        print("=" * 70)

        print("""
1. USER creates skill directories with SKILL.md files

2. USER configures SkillsMiddleware with paths to skill directories

3. USER adds middleware to agent

4. AGENT starts → before_agent() loads all skills

5. AGENT receives message → wrap_model_call() injects skills into prompt

6. MODEL sees skills list in system prompt (names + descriptions + paths)

7. MODEL decides "this task matches the web-research skill"

8. MODEL uses read_file tool to read the full SKILL.md instructions

9. MODEL follows the skill's workflow to complete the task

This is "progressive disclosure" - model only loads full instructions when needed.
""")

        # =====================================================================
        # STEP 8: Show the model's decision flow
        # =====================================================================
        print("=" * 70)
        print("STEP 8: MODEL'S DECISION FLOW (SIMULATED)")
        print("=" * 70)

        print("""
User: "Can you research the latest developments in AI safety?"

Model's reasoning:
1. Sees available skills in system prompt
2. Notices "web-research: Structured approach to researching topics"
3. Decides this skill applies
4. Reads the skill file:
""")
        print(f"   read_file('{skill_file}')")
        print("""
5. Gets the full instructions (workflow, best practices)
6. Follows the skill's workflow:
   - Clarify the research question
   - Search broadly first
   - Deep dive on promising leads
   - Synthesize findings
7. Returns comprehensive research with citations
""")


if __name__ == "__main__":
    main()
