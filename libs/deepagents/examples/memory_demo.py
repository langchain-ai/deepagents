"""Demo showing exactly how MemoryMiddleware works.

This script demonstrates:
1. How memory files are structured
2. How they're loaded
3. What the model actually sees
"""

import tempfile
from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.memory import MemoryMiddleware, MemoryState


def main() -> None:
    print("=" * 70)
    print("MEMORY MIDDLEWARE DEMONSTRATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)

        # =====================================================================
        # STEP 1: Create memory files
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 1: CREATE MEMORY FILES (AGENTS.md)")
        print("=" * 70)

        # User-level memory
        user_dir = base / "user"
        user_dir.mkdir()
        user_memory = """# User Preferences

## Communication Style
- Be concise and direct
- Use technical language appropriately
- Provide code examples when helpful

## Coding Preferences
- Always use type hints in Python
- Prefer functional patterns where appropriate
- Write comprehensive docstrings
"""
        (user_dir / "AGENTS.md").write_text(user_memory)
        print(f"\nCreated user memory: {user_dir / 'AGENTS.md'}")
        print("-" * 40)
        print(user_memory)

        # Project-level memory
        project_dir = base / "project" / ".deepagents"
        project_dir.mkdir(parents=True)
        project_memory = """# Project: MyApp

## Architecture
This is a FastAPI application with:
- SQLAlchemy for ORM
- Alembic for migrations
- pytest for testing

## Code Style
- 4-space indentation
- Google-style docstrings
- All functions must have type hints

## Testing
Run tests with: `pytest tests/ -v`
All PRs require passing tests.

## Deployment
See `docs/deployment.md` for deployment procedures.
"""
        (project_dir / "AGENTS.md").write_text(project_memory)
        print(f"\nCreated project memory: {project_dir / 'AGENTS.md'}")
        print("-" * 40)
        print(project_memory)

        # =====================================================================
        # STEP 2: Configure middleware
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 2: CONFIGURE MIDDLEWARE")
        print("=" * 70)

        backend = FilesystemBackend(root_dir=str(base), virtual_mode=False)

        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(user_dir / "AGENTS.md"), "name": "user"},
                {"path": str(project_dir / "AGENTS.md"), "name": "project"},
            ],
        )

        print("""
Configuration:
```python
middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        {"path": "~/.deepagents/AGENTS.md", "name": "user"},
        {"path": "./.deepagents/AGENTS.md", "name": "project"},
    ],
)
```
""")

        # =====================================================================
        # STEP 3: Load memory (before_agent)
        # =====================================================================
        print("=" * 70)
        print("STEP 3: LOAD MEMORY (before_agent hook)")
        print("=" * 70)

        initial_state: MemoryState = {}
        state_update = middleware.before_agent(initial_state, None)  # type: ignore

        print("\nState update returned:")
        print("-" * 40)
        for name, content in state_update["memory_contents"].items():
            print(f"\n{name}:")
            print(f"  Length: {len(content)} characters")
            print(f"  Preview: {content[:50]}...")

        # =====================================================================
        # STEP 4: What gets injected into system prompt
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 4: WHAT THE MODEL SEES (system prompt injection)")
        print("=" * 70)

        # Simulate the state after before_agent runs
        state_with_memory: MemoryState = {"memory_contents": state_update["memory_contents"]}

        # Format what would be injected
        locations = middleware._format_memory_locations()
        contents = middleware._format_memory_contents(state_with_memory["memory_contents"])

        full_injection = middleware.system_prompt_template.format(
            memory_locations=locations,
            memory_contents=contents,
        )

        print("\nThis is EXACTLY what gets prepended to the system prompt:")
        print("-" * 40)
        print(full_injection)
        print("-" * 40)

        # =====================================================================
        # STEP 5: Complete system prompt example
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: COMPLETE SYSTEM PROMPT (memory + base prompt)")
        print("=" * 70)

        base_prompt = "You are a helpful coding assistant."

        # Simulate modify_request
        class MockRequest:
            def __init__(self):
                self.state = state_with_memory
                self.system_prompt = base_prompt

            def override(self, **kwargs):
                result = MockRequest()
                result.state = self.state
                result.system_prompt = kwargs.get("system_prompt", self.system_prompt)
                return result

        mock_request = MockRequest()
        modified = middleware.modify_request(mock_request)

        print("\nFinal system prompt (truncated for display):")
        print("-" * 40)
        # Show first 2000 chars
        prompt = modified.system_prompt
        if len(prompt) > 2000:
            print(prompt[:2000])
            print(f"\n... [{len(prompt) - 2000} more characters]")
        else:
            print(prompt)

        # =====================================================================
        # STEP 6: Verification checklist
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 6: VERIFICATION CHECKLIST")
        print("=" * 70)

        checks = [
            ("Memory files loaded", len(state_update["memory_contents"]) == 2),
            ("User memory present", "user" in state_update["memory_contents"]),
            ("Project memory present", "project" in state_update["memory_contents"]),
            ("XML tags used", "<user_memory>" in modified.system_prompt),
            ("User content in prompt", "Be concise and direct" in modified.system_prompt),
            ("Project content in prompt", "FastAPI application" in modified.system_prompt),
            ("Base prompt preserved", "helpful coding assistant" in modified.system_prompt),
            ("Memory section header", "Agent Memory" in modified.system_prompt),
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("All checks PASSED - Memory middleware is working correctly!")
        else:
            print("Some checks FAILED - Review the implementation.")


if __name__ == "__main__":
    main()
