"""Unit tests for per-subagent tool/skill scoping resolution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import tool

from deepagents_code.agent import (
    _resolve_subagent_skill_sources,
    _resolve_subagent_tool_allowlist,
)


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return query


@tool
def github_create_issue(title: str) -> str:
    """Create an issue."""
    return title


class TestToolAllowlistResolution:
    """Test _resolve_subagent_tool_allowlist."""

    def test_resolves_named_tools_in_allowlist_order(self) -> None:
        """The allowlist picks session tools by name, order preserved."""
        session = [web_search, github_create_issue]

        resolved = _resolve_subagent_tool_allowlist(
            "researcher", ["github_create_issue", "web_search"], session
        )

        assert resolved == [github_create_issue, web_search]

    def test_duck_typed_tools_without_name_are_skipped(self) -> None:
        """Entries without a usable .name never enter the pool."""
        resolved = _resolve_subagent_tool_allowlist(
            "researcher", ["web_search"], [SimpleNamespace(), web_search]
        )

        assert resolved == [web_search]

    def test_unknown_tool_raises_with_available_names(self) -> None:
        """A typo aborts assembly and names what IS available."""
        with pytest.raises(ValueError, match=r"unknown tool.*web-serch") as excinfo:
            _resolve_subagent_tool_allowlist("researcher", ["web-serch"], [web_search])

        assert "web_search" in str(excinfo.value)

    def test_empty_allowlist_yields_no_tools(self) -> None:
        """tools: [] resolves to an empty list, not to inheritance."""
        assert _resolve_subagent_tool_allowlist("analyst", [], [web_search]) == []


class TestSkillSourceResolution:
    """Test _resolve_subagent_skill_sources."""

    def test_resolves_named_skills_to_their_directories(self, tmp_path: Path) -> None:
        """Only the named skill directories are returned, in name order."""
        root = tmp_path / "skills"
        (root / "source-citation").mkdir(parents=True)
        (root / "browser-flows").mkdir()
        (root / "not-a-skill.txt").write_text("stray file")

        sources = _resolve_subagent_skill_sources(
            "researcher", ["browser-flows"], [root]
        )

        assert sources == [str(root / "browser-flows")]

    def test_later_roots_override_earlier_ones(self, tmp_path: Path) -> None:
        """Precedence mirrors the main agent's source order."""
        user = tmp_path / "user"
        project = tmp_path / "project"
        (user / "shared").mkdir(parents=True)
        (project / "shared").mkdir(parents=True)

        sources = _resolve_subagent_skill_sources("coder", ["shared"], [user, project])

        assert sources == [str(project / "shared")]

    def test_none_and_missing_roots_are_skipped(self, tmp_path: Path) -> None:
        """None roots (skills disabled dirs) and absent dirs are ignored."""
        root = tmp_path / "skills"
        (root / "alpha").mkdir(parents=True)

        sources = _resolve_subagent_skill_sources(
            "coder", ["alpha"], [None, tmp_path / "does-not-exist", root]
        )

        assert sources == [str(root / "alpha")]

    def test_unknown_skill_raises_with_available_names(self, tmp_path: Path) -> None:
        """A typo aborts assembly and lists resolvable skills."""
        root = tmp_path / "skills"
        (root / "alpha").mkdir(parents=True)

        with pytest.raises(ValueError, match=r"unknown skill.*alph") as excinfo:
            _resolve_subagent_skill_sources("coder", ["alph"], [root])

        assert "alpha" in str(excinfo.value)
