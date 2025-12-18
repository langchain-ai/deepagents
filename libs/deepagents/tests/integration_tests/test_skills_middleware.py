#!/usr/bin/env python3
"""Integration tests for SkillsMiddleware.

Tests all backend types:
- StateBackend: In-memory state
- FilesystemBackend: Disk files
- StoreBackend: LangGraph Store (persistent)
- CompositeBackend: Multiple backends combined
"""

import tempfile
from pathlib import Path

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from deepagents.graph import create_agent
from deepagents.middleware.filesystem import FileData, FilesystemMiddleware
from deepagents.middleware.skills import (
    SKILLS_SYSTEM_PROMPT,
    SkillMetadata,
    SkillsMiddleware,
    _list_skills_from_backend,
)


# ========== Realistic Skill Fixtures (based on CLI examples) ==========

WEB_RESEARCH_SKILL = FileData(
    content=[
        "---",
        "name: web-research",
        "description: |",
        "  Use this skill for web research tasks. Provides structured approach",
        "  to conducting comprehensive research with planning and synthesis.",
        "---",
        "",
        "# Web Research Skill",
        "",
        "## When to Use",
        "- Research complex topics requiring multiple sources",
        "- Gather and synthesize current information",
        "- Produce well-sourced reports with citations",
        "",
        "## Process",
        "1. Create research plan",
        "2. Delegate to subagents with `task` tool",
        "3. Synthesize findings into final report",
    ],
    created_at="2024-01-01",
    modified_at="2024-01-01",
)

CODE_REVIEW_SKILL = FileData(
    content=[
        "---",
        "name: code-review",
        "description: Review code for quality, security, and best practices",
        "---",
        "",
        "# Code Review Skill",
        "",
        "## Checklist",
        "- [ ] Code style and formatting",
        "- [ ] Security vulnerabilities",
        "- [ ] Performance considerations",
        "- [ ] Test coverage",
        "",
        "## Output Format",
        "Provide feedback in structured sections.",
    ],
    created_at="2024-01-01",
    modified_at="2024-01-01",
)

LANGGRAPH_DOCS_SKILL = FileData(
    content=[
        "---",
        "name: langgraph-docs",
        "description: Fetch LangGraph documentation for accurate guidance",
        "---",
        "",
        "# LangGraph Docs Skill",
        "",
        "## Instructions",
        "1. Fetch https://docs.langchain.com/llms.txt",
        "2. Select 2-4 relevant documentation URLs",
        "3. Fetch and provide guidance based on docs",
    ],
    created_at="2024-01-01",
    modified_at="2024-01-01",
)

INVALID_SKILL = FileData(
    content=["# Broken Skill", "", "No YAML frontmatter here."],
    created_at="2024-01-01",
    modified_at="2024-01-01",
)


def make_runtime(files: dict | None = None):
    """Create ToolRuntime with files."""
    from langchain.tools import ToolRuntime
    from deepagents.middleware.filesystem import FilesystemState

    state = FilesystemState(messages=[], files=files or {})
    return ToolRuntime(
        state=state, context=None, tool_call_id="test",
        store=None, stream_writer=lambda _: None, config={},
    )


@pytest.mark.requires("langchain_anthropic")
class TestStateBackend:
    """Tests for skill loading from StateBackend (in-memory)."""

    def test_load_skills(self):
        runtime = make_runtime({
            "/skills/web-research/SKILL.md": WEB_RESEARCH_SKILL,
            "/skills/code-review/SKILL.md": CODE_REVIEW_SKILL,
        })
        skills = _list_skills_from_backend(StateBackend(runtime), "/skills")
        assert len(skills) == 2
        assert {s["name"] for s in skills} == {"web-research", "code-review"}

    def test_skip_invalid_skills(self):
        runtime = make_runtime({
            "/skills/valid/SKILL.md": CODE_REVIEW_SKILL,
            "/skills/broken/SKILL.md": INVALID_SKILL,
        })
        skills = _list_skills_from_backend(StateBackend(runtime), "/skills")
        assert len(skills) == 1
        assert skills[0]["name"] == "code-review"

    def test_multiline_description(self):
        runtime = make_runtime({"/skills/research/SKILL.md": WEB_RESEARCH_SKILL})
        skills = _list_skills_from_backend(StateBackend(runtime), "/skills")
        assert len(skills) == 1
        assert "comprehensive research" in skills[0]["description"]

    def test_project_overrides_user(self):
        """Later paths override earlier with same skill name."""
        user_common = FileData(
            content=["---", "name: common", "description: User version", "---"],
            created_at="2024-01-01", modified_at="2024-01-01",
        )
        project_common = FileData(
            content=["---", "name: common", "description: Project version (wins)", "---"],
            created_at="2024-01-01", modified_at="2024-01-01",
        )
        runtime = make_runtime({
            "/user/common/SKILL.md": user_common,
            "/project/common/SKILL.md": project_common,
        })
        backend = StateBackend(runtime)

        all_skills: dict[str, dict] = {}
        for skill in _list_skills_from_backend(backend, "/user"):
            all_skills[skill["name"]] = skill
        for skill in _list_skills_from_backend(backend, "/project"):
            all_skills[skill["name"]] = skill

        assert len(all_skills) == 1
        assert "Project version" in all_skills["common"]["description"]


@pytest.mark.requires("langchain_anthropic")
class TestFilesystemBackend:
    """Tests for skill loading from FilesystemBackend (disk files)."""

    def test_load_skills_from_disk(self):
        """Load skills from real filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: A test skill\n---\n# Test")

            backend = FilesystemBackend(root_dir=tmpdir, virtual_mode=True)
            skills = _list_skills_from_backend(backend, "/")

            assert len(skills) == 1
            assert skills[0]["name"] == "test-skill"

    def test_load_multiple_skills(self):
        """Load multiple skills from filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["skill-a", "skill-b", "skill-c"]:
                skill_dir = Path(tmpdir) / name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Skill {name}\n---\n")

            backend = FilesystemBackend(root_dir=tmpdir, virtual_mode=True)
            skills = _list_skills_from_backend(backend, "/")

            assert len(skills) == 3
            assert {s["name"] for s in skills} == {"skill-a", "skill-b", "skill-c"}


@pytest.mark.requires("langchain_anthropic")
class TestStoreBackend:
    """Tests for skill loading from StoreBackend (LangGraph Store)."""

    def test_load_skills_from_store(self):
        """Load skills from InMemoryStore."""
        from langchain.tools import ToolRuntime
        from deepagents.middleware.filesystem import FilesystemState

        store = InMemoryStore()
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state, context=None, tool_call_id="test",
            store=store, stream_writer=lambda _: None, config={},
        )

        # Put skill files into store
        store.put(
            ("filesystem",),
            "/skills/store-skill/SKILL.md",
            {"content": ["---", "name: store-skill", "description: Skill from store", "---"], "created_at": "2024-01-01", "modified_at": "2024-01-01"},
        )

        backend = StoreBackend(runtime)
        skills = _list_skills_from_backend(backend, "/skills")

        assert len(skills) == 1
        assert skills[0]["name"] == "store-skill"
        assert skills[0]["description"] == "Skill from store"


@pytest.mark.requires("langchain_anthropic")
class TestCompositeBackend:
    """Tests for skill loading from CompositeBackend (multiple sources)."""

    def test_load_skills_from_composite(self):
        """Load skills from composite backend with StateBackend default."""
        runtime = make_runtime({
            "/skills/state-skill/SKILL.md": CODE_REVIEW_SKILL,
        })
        backend = CompositeBackend(default=StateBackend(runtime), routes={})

        skills = _list_skills_from_backend(backend, "/skills")
        assert len(skills) == 1
        assert skills[0]["name"] == "code-review"

    def test_composite_with_filesystem_route(self):
        """Composite backend routing StateBackend + different path."""
        runtime = make_runtime({
            "/default/state-skill/SKILL.md": CODE_REVIEW_SKILL,
            "/override/other-skill/SKILL.md": WEB_RESEARCH_SKILL,
        })

        backend = CompositeBackend(
            default=StateBackend(runtime),
            routes={},  # No special routes, just use default
        )

        # Load from different paths
        default_skills = _list_skills_from_backend(backend, "/default")
        assert len(default_skills) == 1
        assert default_skills[0]["name"] == "code-review"

        override_skills = _list_skills_from_backend(backend, "/override")
        assert len(override_skills) == 1
        assert override_skills[0]["name"] == "web-research"


@pytest.mark.requires("langchain_anthropic")
class TestSkillsFormat:
    """Tests for skills formatting."""

    def test_format_skills_locations(self):
        backend = StateBackend(make_runtime())
        middleware = SkillsMiddleware(
            backend=backend,
            skills_paths=[("/user/skills", "User Skills"), ("/project/skills", "Project Skills")],
        )
        output = middleware._format_skills_locations()
        assert "User Skills" in output
        assert "Project Skills" in output

    def test_format_skills_list(self):
        backend = StateBackend(make_runtime())
        middleware = SkillsMiddleware(backend=backend)
        skills = [
            SkillMetadata(name="web-research", description="Research the web", path="/skills/web-research/SKILL.md", label=None),
            SkillMetadata(name="code-review", description="Review code", path="/skills/code-review/SKILL.md", label=None),
        ]
        output = middleware._format_skills_list(skills)
        assert "web-research" in output
        assert "code-review" in output
        assert "Research the web" in output

    def test_full_system_prompt(self):
        backend = StateBackend(make_runtime())
        middleware = SkillsMiddleware(
            backend=backend,
            skills_paths=[("/user/", "User"), ("/project/", "Project")],
        )
        skills = [
            SkillMetadata(name="web-research", description="Research", path="/user/web-research/SKILL.md", label="User"),
            SkillMetadata(name="code-review", description="Review", path="/project/code-review/SKILL.md", label="Project"),
        ]

        prompt = SKILLS_SYSTEM_PROMPT.format(
            skills_locations=middleware._format_skills_locations(),
            skills_list=middleware._format_skills_list(skills),
        )

        assert "Skills System" in prompt
        assert "Available Skills" in prompt
        assert "Progressive Disclosure" in prompt
        assert "web-research" in prompt


@pytest.mark.requires("langchain_anthropic")
class TestSkillsAgent:
    """Real agent integration tests."""

    def test_system_prompt_injection(self):
        """Verify skills are loaded from state and injected into system prompt."""
        import os
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        if os.environ.get("OPENROUTER_API_KEY"):
            model = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model="anthropic/claude-sonnet-4.5",
            )
        else:
            model = "claude-sonnet-4-20250514"

        captured_prompts = []

        class CapturingMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                captured_prompts.clear()
                if request.system_prompt:
                    captured_prompts.append(request.system_prompt)
                return handler(request)

        agent = create_agent(
            model=model,
            middleware=[
                FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
                SkillsMiddleware(backend=lambda rt: StateBackend(rt), skills_paths=["/skills"]),
                CapturingMiddleware(),
            ],
        )

        # Inject skills via state (programmatic use case)
        agent.invoke({
            "messages": [HumanMessage(content="What skills do you have?")],
            "files": {
                "/skills/web-research/SKILL.md": WEB_RESEARCH_SKILL,
                "/skills/code-review/SKILL.md": CODE_REVIEW_SKILL,
            },
        })

        assert len(captured_prompts) > 0
        prompt = captured_prompts[0]
        assert "Skills System" in prompt
        assert "web-research" in prompt
        assert "code-review" in prompt

    def test_tools_available(self):
        import os
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key available")

        if os.environ.get("OPENROUTER_API_KEY"):
            model = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                model="anthropic/claude-sonnet-4.5",
            )
        else:
            model = "claude-sonnet-4-20250514"

        agent = create_agent(
            model=model,
            middleware=[
                FilesystemMiddleware(backend=lambda rt: StateBackend(rt)),
                SkillsMiddleware(backend=lambda rt: StateBackend(rt)),
            ],
        )

        tools = agent.nodes["tools"].bound._tools_by_name
        assert "read_file" in tools
        assert "write_file" in tools
        assert "ls" in tools


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
