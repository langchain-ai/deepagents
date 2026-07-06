"""Tests for the `dcode tools` command group."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from deepagents_code import managed_tools
from deepagents_code._env_vars import OFFLINE, RIPGREP_INSTALLER
from deepagents_code.tool_catalog import ToolEntry, ToolGroup
from deepagents_code.tools_commands import run_tools_command


def _run_text(args: argparse.Namespace) -> tuple[int, str]:
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False, width=200)
    with patch("deepagents_code.config.console", test_console):
        code = run_tools_command(args)
    return code, buf.getvalue()


class TestToolsInstall:
    """Tests for `dcode tools install` dispatch."""

    def test_install_success_text(self, tmp_path: Path) -> None:
        installed = tmp_path / "[/green]" / "rg"
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=installed),
            patch.object(managed_tools, "prepend_managed_bin_to_path"),
            patch.object(managed_tools, "managed_rg_path", return_value=installed),
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "Managed ripgrep" in output
        assert str(installed) in output

    def test_install_reuses_system_rg(self, tmp_path: Path) -> None:
        system_rg = Path("/usr/bin/rg")
        managed = tmp_path / "rg"
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=system_rg),
            patch.object(managed_tools, "prepend_managed_bin_to_path") as prepend,
            patch.object(managed_tools, "managed_rg_path", return_value=managed),
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "already on PATH" in output
        prepend.assert_not_called()

    def test_install_json_success(self, tmp_path: Path, capsys) -> None:
        installed = tmp_path / "rg"
        args = argparse.Namespace(tools_command="install", output_format="json")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=installed),
            patch.object(managed_tools, "prepend_managed_bin_to_path"),
            patch.object(managed_tools, "managed_rg_path", return_value=installed),
        ):
            code = run_tools_command(args)
        assert code == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["command"] == "tools install"
        assert envelope["data"]["status"] == "ok"
        assert envelope["data"]["path"] == str(installed)

    def test_install_skipped_system_installer(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RIPGREP_INSTALLER, "system")
        monkeypatch.delenv(OFFLINE, raising=False)
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=None),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "system" in output

    def test_install_skipped_offline(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv(OFFLINE, "1")
        monkeypatch.delenv(RIPGREP_INSTALLER, raising=False)
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=None),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "OFFLINE" in output

    def test_install_failure_returns_nonzero(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv(OFFLINE, raising=False)
        monkeypatch.delenv(RIPGREP_INSTALLER, raising=False)
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(managed_tools, "ensure_ripgrep", return_value=None),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 1
        assert "Could not install" in output

    def test_install_checksum_mismatch_returns_nonzero(self, tmp_path: Path) -> None:
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(
                managed_tools,
                "ensure_ripgrep",
                side_effect=managed_tools.ChecksumMismatchError("bad"),
            ),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 1
        assert "SHA-256" in output

    def test_install_unexpected_error_returns_nonzero(self, tmp_path: Path) -> None:
        """An unexpected exception degrades to a clean error, not a traceback."""
        args = argparse.Namespace(tools_command="install", output_format="text")
        with (
            patch.object(
                managed_tools,
                "ensure_ripgrep",
                side_effect=OSError("boom"),
            ),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 1
        assert "unexpectedly" in output
        assert "boom" not in output  # internals stay in the logs, not stdout

    def test_no_subcommand_shows_help(self) -> None:
        args = argparse.Namespace(tools_command=None)
        with patch("deepagents_code.ui.show_tools_help") as show_help:
            code = run_tools_command(args)
        assert code == 0
        show_help.assert_called_once()


_SAMPLE_GROUPS = [
    ToolGroup(
        label="Built-in",
        source="built-in",
        tools=(
            ToolEntry(name="read_file", description="Read a file"),
            ToolEntry(name="execute", description="Run a shell command"),
        ),
    ),
    ToolGroup(
        label="docs",
        source="mcp",
        tools=(ToolEntry(name="search_docs", description="Search the docs"),),
    ),
]


class TestToolsList:
    """Tests for `dcode tools list` dispatch."""

    def test_list_text_output(self) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        with patch(
            "deepagents_code.tool_catalog.collect_tool_groups",
            return_value=_SAMPLE_GROUPS,
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "3 tools available" in output
        assert "Built-in" in output
        assert "read_file" in output
        assert "Run a shell command" in output
        # MCP tools grouped under their server name.
        assert "docs" in output
        assert "search_docs" in output

    def test_list_json_output(self, capsys) -> None:
        args = argparse.Namespace(tools_command="list", output_format="json")
        with patch(
            "deepagents_code.tool_catalog.collect_tool_groups",
            return_value=_SAMPLE_GROUPS,
        ):
            code = run_tools_command(args)
        assert code == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["command"] == "tools list"
        data = envelope["data"]
        assert data["count"] == 3
        assert len(data["tools"]) == 3
        first = data["tools"][0]
        assert first == {
            "name": "read_file",
            "description": "Read a file",
            "group": "Built-in",
            "source": "built-in",
        }
        assert data["tools"][-1]["source"] == "mcp"
        assert data["tools"][-1]["group"] == "docs"

    def test_list_forwards_runtime_options(self) -> None:
        """`--no-mcp`, `--mcp-config`, and interpreter resolution reach the catalog."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            interpreter=True,
            sandbox="none",
            no_mcp=True,
            mcp_config="/tmp/mcp.json",
            trust_project_mcp=True,
        )
        with patch(
            "deepagents_code.tool_catalog.collect_tool_groups",
            return_value=[],
        ) as collect:
            code = run_tools_command(args)
        assert code == 0
        collect.assert_called_once_with(
            enable_interpreter=True,
            include_mcp=False,
            mcp_config_path="/tmp/mcp.json",
            trust_project_mcp=True,
        )

    def test_list_consults_persisted_project_mcp_trust_by_default(self) -> None:
        """Absent `--trust-project-mcp` lets MCP discovery use stored trust."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            interpreter=False,
            sandbox="none",
            no_mcp=False,
            mcp_config=None,
        )
        with patch(
            "deepagents_code.tool_catalog.collect_tool_groups",
            return_value=[],
        ) as collect:
            code = run_tools_command(args)
        assert code == 0
        collect.assert_called_once_with(
            enable_interpreter=False,
            include_mcp=True,
            mcp_config_path=None,
            trust_project_mcp=None,
        )

    def test_list_interpreter_defaults_to_settings(self) -> None:
        """With no explicit flag, the resolved local default drives `js_eval`."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            interpreter=None,
            sandbox="none",
            no_mcp=False,
            mcp_config=None,
            trust_project_mcp=False,
        )
        with (
            patch(
                "deepagents_code._server_config._resolve_enable_interpreter",
                return_value=True,
            ),
            patch(
                "deepagents_code.tool_catalog.collect_tool_groups",
                return_value=[],
            ) as collect,
        ):
            code = run_tools_command(args)
        assert code == 0
        assert collect.call_args.kwargs["enable_interpreter"] is True
        assert collect.call_args.kwargs["include_mcp"] is True

    def test_list_singular_noun_for_one_tool(self) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        one_group = [
            ToolGroup(
                label="Built-in",
                source="built-in",
                tools=(ToolEntry(name="ls", description="List files"),),
            )
        ]
        with patch(
            "deepagents_code.tool_catalog.collect_tool_groups",
            return_value=one_group,
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "1 tool available" in output
