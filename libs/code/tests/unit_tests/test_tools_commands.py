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
