"""Tests for context-doctor report construction."""

from __future__ import annotations

from deepagents_code.context_doctor import (
    build_context_doctor_report,
    estimate_text_tokens,
    format_memory_prompt,
    render_context_doctor_report,
)
from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo
from deepagents_code.tool_catalog import ToolEntry


def test_estimate_text_tokens_rounds_up() -> None:
    assert estimate_text_tokens("12345") == 2


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
