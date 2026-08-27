"""Tests for the `dcode tools` command group."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from deepagents_code import managed_tools
from deepagents_code._env_vars import OFFLINE, RIPGREP_INSTALLER
from deepagents_code.client.commands.tools import _truncate, run_tools_command
from deepagents_code.tool_catalog import (
    ToolCatalog,
    ToolEntry,
    ToolGroup,
    UnavailableServer,
)


def _run_text(args: argparse.Namespace, *, width: int = 200) -> tuple[int, str]:
    buf = io.StringIO()
    test_console = Console(file=buf, highlight=False, width=width)
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

    def test_install_unavailable_returns_specific_message(self, tmp_path: Path) -> None:
        args = argparse.Namespace(tools_command="install", output_format="text")
        error = managed_tools.ManagedToolUnavailableError(
            tool="ripgrep",
            reason="artifact_not_found",
            message="Managed ripgrep artifact for linux/x86_64 was not found.",
        )
        with (
            patch.object(managed_tools, "ensure_ripgrep", side_effect=error),
            patch.object(managed_tools, "managed_rg_path", return_value=tmp_path / "x"),
        ):
            code, output = _run_text(args)
        assert code == 1
        assert "linux/x86_64" in output
        assert "unexpectedly" not in output

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


_SAMPLE_GROUPS = (
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
)
_SAMPLE_CATALOG = ToolCatalog(groups=_SAMPLE_GROUPS, unavailable=(), mcp_error=None)


class TestToolsList:
    """Tests for `dcode tools list` dispatch."""

    def test_list_json_includes_unavailable_and_mcp_error(self, capsys) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        # `output_format` defaults to text on the namespace; force json below.
        args.output_format = "json"
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            ),
            unavailable=(
                UnavailableServer(
                    name="needslogin", status="unauthenticated", detail="run login"
                ),
            ),
            mcp_error="MCP discovery failed; showing built-in tools only.",
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code = run_tools_command(args)
        # No explicit --mcp-config on this namespace, so degradation is exit 0.
        assert code == 0
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["unavailable"] == [
            {"name": "needslogin", "status": "unauthenticated", "detail": "run login"}
        ]
        assert data["mcp_error"] == "MCP discovery failed; showing built-in tools only."

    def test_list_invalid_allow_fs_tools_exits(self) -> None:
        """A malformed `--allow-fs-tools` fails fast with exit 2, before catalog.

        `_parse_allow_fs_tools_flag` is unit-tested exhaustively in isolation;
        this pins the command-level contract that the bad value aborts the
        `tools list` request rather than degrading to an unrestricted listing.
        """
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            interpreter=False,
            sandbox="none",
            allow_fs_tools="bogus",
            no_mcp=True,
            mcp_config=None,
            trust_project_mcp=False,
        )
        with (
            patch(
                "deepagents_code.tool_catalog.collect_catalog",
                return_value=ToolCatalog(groups=()),
            ) as collect,
            pytest.raises(SystemExit) as exc_info,
        ):
            run_tools_command(args)
        assert exc_info.value.code == 2
        collect.assert_not_called()

    def test_list_end_to_end_offline_renders_real_built_ins(self) -> None:
        """Real `collect_catalog` compiles the agent offline and renders it."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="text",
            interpreter=False,
            sandbox="none",
            no_mcp=True,
            mcp_config=None,
            trust_project_mcp=False,
        )
        code, output = _run_text(args)
        assert code == 0
        assert "tools available" in output
        assert "Built-in" in output
        # Representative built-in tools the default agent always binds.
        assert "read_file" in output
        assert "execute" in output


class TestTruncate:
    """Tests for `_truncate` description clipping."""
