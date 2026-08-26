"""Estimated context audit for the `/context-doctor` command."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.skills.load import ExtendedSkillMetadata
    from deepagents_code.tool_catalog import ToolEntry


@dataclass(frozen=True, slots=True)
class ContextDoctorRow:
    """One component in the context audit."""

    label: str
    tokens: int | None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ContextDoctorReport:
    """Structured context audit ready for presentation."""

    rows: tuple[ContextDoctorRow, ...]
    injected_tokens: int
    conversation_tokens: int | None
    provider_tokens: int | None


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens using the same four-characters-per-token rule as whip.

    Returns:
        Estimated token count.
    """
    return math.ceil(len(text) / 4)


def _schema_tokens(tools: Sequence[ToolEntry]) -> tuple[int, int]:
    schemas = [tool.schema for tool in tools if tool.schema is not None]
    text = "".join(
        json.dumps(schema, separators=(",", ":"), sort_keys=True) for schema in schemas
    )
    return estimate_text_tokens(text), len(schemas)


def _bounded(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _skill_tokens(skill: ExtendedSkillMetadata) -> int:
    line = f"{skill['name']} {skill['description']} {skill['path']}"
    return estimate_text_tokens(line)


def format_memory_prompt(contents: Sequence[tuple[str, str]], template: str) -> str:
    """Approximate the memory fragment after middleware strips HTML comments.

    Returns:
        The formatted memory prompt.
    """
    sections = [
        f"{path}\n\n{re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL).rstrip()}"
        for path, text in contents
    ]
    body = "\n\n".join(section for section in sections if section.strip())
    return template.format(agent_memory=body or "(No memory loaded)")


def format_skills_locations(sources: Sequence[str | tuple[str, ...]]) -> str:
    """Format skills locations matching SkillsMiddleware display.

    Returns:
        The formatted skills locations text.
    """
    if not sources:
        return ""
    from deepagents.middleware.skills import (
        _derive_source_label,  # noqa: PLC2701  # Matches SkillsMiddleware source label derivation
        _source_path,  # noqa: PLC2701  # Matches SkillsMiddleware source path resolution
    )

    normalized_sources: list[tuple[str, str] | str] = [
        (s[0], s[1]) if isinstance(s, tuple) else s for s in sources
    ]
    paths = [_source_path(s) for s in normalized_sources]
    labels = [_derive_source_label(s) for s in normalized_sources]
    last = len(normalized_sources) - 1
    locations = [
        f"**{label} Skills**: `{path}`{' (higher priority)' if i == last else ''}"
        for i, (path, label) in enumerate(zip(paths, labels, strict=True))
    ]
    return "\n".join(locations)


def format_skills_prompt(
    skills: Sequence[ExtendedSkillMetadata],
    sources: Sequence[str | tuple[str, ...]] = (),
) -> str:
    """Approximate the progressive-disclosure skill index sent to the model.

    Returns:
        The formatted skills prompt.
    """
    from deepagents.middleware.skills import SKILLS_SYSTEM_PROMPT

    lines: list[str] = []
    for skill in skills:
        annotations = []
        if skill.get("license"):
            annotations.append(f"License: {skill['license']}")
        if skill.get("compatibility"):
            annotations.append(f"Compatibility: {skill['compatibility']}")
        suffix = f" ({', '.join(annotations)})" if annotations else ""
        lines.append(f"- **{skill['name']}**: {skill['description']}{suffix}")
        if skill["allowed_tools"]:
            lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
        lines.append(f"  -> Read `{skill['path']}` for full instructions")
    skills_list = "\n".join(lines) or "(No skills available yet)"
    skills_locations = format_skills_locations(sources)
    return SKILLS_SYSTEM_PROMPT.format(
        skills_locations=skills_locations,
        skills_load_warnings="",
        skills_list=skills_list,
    )


def _skills_detail(skills: Sequence[ExtendedSkillMetadata]) -> str:
    ranked = sorted(
        ((skill["name"], _skill_tokens(skill)) for skill in skills),
        key=itemgetter(1),
        reverse=True,
    )[:5]
    if not ranked:
        return ""
    return "largest: " + ", ".join(f"{name} ~{tokens}" for name, tokens in ranked)


def _mcp_row(server: MCPServerInfo) -> ContextDoctorRow:
    if server.status != "ok":
        detail = _bounded(server.error or server.status.replace("_", " "))
        return ContextDoctorRow(_bounded(f"MCP: {server.name}"), 0, detail)
    definitions = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in server.tools
    ]
    text = "".join(
        json.dumps(definition, separators=(",", ":"), sort_keys=True)
        for definition in definitions
    )
    count = len(server.tools)
    noun = "tool" if count == 1 else "tools"
    return ContextDoctorRow(
        _bounded(f"MCP: {server.name} ({count} {noun})"), estimate_text_tokens(text)
    )


def build_context_doctor_report(
    *,
    system_prompt: str | None,
    memory_prompt: str | None,
    memory_files: int,
    skills_prompt: str | None,
    skills: Sequence[ExtendedSkillMetadata],
    built_in_tools: Sequence[ToolEntry] | None,
    mcp_servers: Sequence[MCPServerInfo],
    conversation_tokens: int | None,
    provider_tokens: int | None,
) -> ContextDoctorReport:
    """Build a fresh-session audit and reconcile it with live usage when available.

    Returns:
        The structured context audit.
    """
    rows = [
        ContextDoctorRow(
            "System prompt (base)",
            None if system_prompt is None else estimate_text_tokens(system_prompt),
            "unavailable for custom or remote agent" if system_prompt is None else "",
        ),
        ContextDoctorRow(
            f"AGENTS.md memory ({memory_files} files)",
            None if memory_prompt is None else estimate_text_tokens(memory_prompt),
            "unavailable for custom or remote agent" if memory_prompt is None else "",
        ),
        ContextDoctorRow(
            f"Skills index ({len(skills)} loaded)",
            None if skills_prompt is None else estimate_text_tokens(skills_prompt),
            _skills_detail(skills),
        ),
    ]
    if built_in_tools is None:
        rows.append(
            ContextDoctorRow(
                "Built-in tool schemas",
                None,
                "unavailable for custom or remote agent",
            )
        )
    else:
        tokens, schemas = _schema_tokens(built_in_tools)
        detail = "sent with every request"
        if schemas != len(built_in_tools):
            detail = f"{schemas} of {len(built_in_tools)} schemas available"
        rows.append(
            ContextDoctorRow(
                f"Built-in tool schemas ({len(built_in_tools)} tools)",
                tokens,
                detail,
            )
        )
    rows.extend(_mcp_row(server) for server in mcp_servers)
    injected = sum(row.tokens or 0 for row in rows)
    return ContextDoctorReport(
        rows=tuple(rows),
        injected_tokens=injected,
        conversation_tokens=conversation_tokens,
        provider_tokens=provider_tokens,
    )


def render_context_doctor_report(report: ContextDoctorReport) -> str:
    """Render a plain-text report.

    Returns:
        Text safe for display without markup parsing.
    """
    lines = ["Fresh-session context audit (estimated tokens)", ""]
    width = max(len(row.label) for row in report.rows)
    for row in report.rows:
        value = "unavailable" if row.tokens is None else f"~{row.tokens:,}"
        detail = f"  {row.detail}" if row.detail else ""
        lines.append(f"{row.label:<{width}}  {value:>11}{detail}")
    total_label = "TOTAL injected before conversation"
    lines.append(f"{total_label:<{width}}  ~{report.injected_tokens:>10,}")
    if report.conversation_tokens is not None:
        lines.append(
            f"{'Conversation history':<{width}}  ~{report.conversation_tokens:>10,}"
        )
    if report.provider_tokens is not None:
        explained = report.injected_tokens + (report.conversation_tokens or 0)
        remainder = report.provider_tokens - explained
        provider_label = "Provider-reported context"
        delta_label = "Unattributed / estimation delta"
        lines.extend(
            [
                f"{provider_label:<{width}}  {report.provider_tokens:>11,}",
                f"{delta_label:<{width}}  {remainder:>+11,}",
            ]
        )
    lines.extend(
        [
            "",
            "Approximate: section counts use about four characters per token.",
            "Trim skills or disable an MCP server, then run /context-doctor again.",
        ]
    )
    return "\n".join(lines)
