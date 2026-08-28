"""Tests for context-doctor report construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT

from deepagents_code.agent import _MEMORY_READONLY_SYSTEM_PROMPT, get_skill_sources
from deepagents_code.config import ASCII_GLYPHS
from deepagents_code.context_doctor import (
    _bounded,
    build_context_doctor_report,
    estimate_text_tokens,
    format_memory_prompt,
    format_skills_locations,
    format_skills_prompt,
    render_context_doctor_report,
)
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.skills.load import ExtendedSkillMetadata
from deepagents_code.tool_catalog import ToolEntry

if TYPE_CHECKING:
    import pytest


def test_estimate_text_tokens_rounds_up() -> None:
    assert estimate_text_tokens("12345") == 2


def test_bounded_uses_length_aware_ascii_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.context_doctor.get_glyphs", lambda: ASCII_GLYPHS
    )

    result = _bounded("abcdefghij", limit=7)

    assert result == "abcd..."
    assert len(result) == 7


def test_format_memory_prompt_strips_html_comments() -> None:
    prompt = format_memory_prompt(
        [("AGENTS.md", "visible<!-- secret marker -->\ntext")],
        "<agent_memory>\n{agent_memory}\n</agent_memory>",
    )

    assert "secret marker" not in prompt
    assert "AGENTS.md\n\nvisible\ntext" in prompt


def test_report_counts_schemas_and_surfaces_mcp_errors() -> None:
    tools = [
        ToolEntry(
            "read_file",
            "Read a file",
            {"type": "function", "function": {"name": "read_file"}},
        )
    ]
    servers = [
        MCPServerInfo(
            name="docs",
            transport="http",
            tools=(
                MCPToolInfo(
                    name="search",
                    description="Search docs",
                    input_schema={"type": "object", "properties": {}},
                ),
            ),
        ),
        MCPServerInfo(
            name="broken*server",
            transport="http",
            status="error",
            error="connection [failed]",
        ),
    ]

    report = build_context_doctor_report(
        system_prompt="system",
        memory_prompt="memory",
        memory_files=1,
        skills_prompt="skills",
        skills=[],
        built_in_tools=tools,
        mcp_servers=servers,
        conversation_tokens=20,
        provider_tokens=100,
    )
    rendered = render_context_doctor_report(report)

    assert report.injected_tokens > 0
    assert "MCP: docs (1 tool)" in rendered
    assert "MCP: broken*server" in rendered
    assert "connection [failed]" in rendered
    assert "Provider-reported context" in rendered


def test_report_marks_uninspectable_sections_unavailable() -> None:
    report = build_context_doctor_report(
        system_prompt=None,
        memory_prompt=None,
        memory_files=0,
        skills_prompt="",
        skills=[],
        built_in_tools=None,
        mcp_servers=[],
        conversation_tokens=None,
        provider_tokens=None,
    )

    assert render_context_doctor_report(report).count("unavailable") >= 3


def test_format_skills_locations_renders_sources_with_priorities() -> None:
    sources = [
        ("/built_in_skills", "Built-in"),
        ("/user/skills", "User Deepagents"),
        ("/project/skills", "Project Deepagents"),
    ]
    rendered = format_skills_locations(sources)

    assert "**Built-in Skills**: `/built_in_skills`" in rendered
    assert "**User Deepagents Skills**: `/user/skills`" in rendered
    assert (
        "**Project Deepagents Skills**: `/project/skills` (higher priority)" in rendered
    )


def test_format_skills_prompt_includes_locations_and_skills() -> None:
    skills: list[ExtendedSkillMetadata] = [
        ExtendedSkillMetadata(
            name="test-skill",
            description="A test skill",
            path="/path/to/SKILL.md",
            license="MIT",
            compatibility=None,
            metadata={},
            allowed_tools=["read_file"],
            source="user",
        )
    ]
    sources = [
        ("/built_in_skills", "Built-in"),
        ("/user/skills", "User Deepagents"),
    ]
    prompt = format_skills_prompt(skills, sources=sources)

    assert "**Built-in Skills**: `/built_in_skills`" in prompt
    assert "**User Deepagents Skills**: `/user/skills` (higher priority)" in prompt
    assert "- **test-skill**: A test skill (License: MIT)" in prompt
    assert "-> Allowed tools: read_file" in prompt
    assert "-> Read `/path/to/SKILL.md` for full instructions" in prompt


def test_format_memory_prompt_supports_readonly_and_autosave_templates() -> None:
    contents = [("/path/AGENTS.md", "guidelines here")]

    autosave_prompt = format_memory_prompt(contents, MEMORY_SYSTEM_PROMPT)
    readonly_prompt = format_memory_prompt(contents, _MEMORY_READONLY_SYSTEM_PROMPT)

    assert "guidelines here" in autosave_prompt
    assert "Learning from feedback:" in autosave_prompt

    assert "guidelines here" in readonly_prompt
    assert "Automatic memory saving is disabled:" in readonly_prompt
    assert "Learning from feedback:" not in readonly_prompt


def test_get_skill_sources_returns_expected_precedence() -> None:
    sources = get_skill_sources(assistant_id="agent")
    labels = [s[1] for s in sources]

    assert labels[0] == "Built-in"
    assert "User Deepagents" in labels
    assert "User Agents" in labels
