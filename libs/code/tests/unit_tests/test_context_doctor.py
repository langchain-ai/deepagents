"""Tests for context-doctor report construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.config import ASCII_GLYPHS
from deepagents_code.context_doctor import (
    _bounded,
    build_context_doctor_report,
    format_memory_prompt,
    render_context_doctor_report,
)
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tool_catalog import ToolEntry

if TYPE_CHECKING:
    import pytest


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


def test_bounded_uses_length_aware_ascii_ellipsis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "deepagents_code.context_doctor.get_glyphs", lambda: ASCII_GLYPHS
    )

    result = _bounded("abcdefghij", limit=7)

    assert result == "abcd..."
    assert len(result) == 7
