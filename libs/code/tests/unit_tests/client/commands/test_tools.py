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

    def test_list_text_output(self) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        with patch(
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=_SAMPLE_CATALOG,
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
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=_SAMPLE_CATALOG,
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
        # No discovery problems for this catalog.
        assert data["unavailable"] == []
        assert data["mcp_error"] is None

    def test_list_reports_unavailable_servers_text(self) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            ),
            unavailable=(
                UnavailableServer(name="broken", status="error", detail="boom"),
            ),
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code, output = _run_text(args)
        assert code == 0
        # The broken server is shown with its status and reason, not hidden.
        assert "Unavailable MCP servers" in output
        assert "broken" in output
        assert "error" in output
        assert "boom" in output

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

    def test_list_explicit_mcp_config_failure_exits_nonzero(self, capsys) -> None:
        """A failed explicit --mcp-config is a failed user request → exit 1."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            mcp_config="/tmp/mcp.json",
        )
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            ),
            mcp_error="MCP discovery failed; showing built-in tools only.",
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code = run_tools_command(args)
        assert code == 1
        data = json.loads(capsys.readouterr().out)["data"]
        assert data["mcp_error"]

    def test_list_discovery_failure_without_explicit_config_exits_zero(self) -> None:
        """Best-effort auto-discovery failure stays exit 0 (built-ins render)."""
        args = argparse.Namespace(
            tools_command="list", output_format="text", mcp_config=None
        )
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            ),
            mcp_error="MCP discovery failed; showing built-in tools only.",
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "showing built-in tools only" in output

    def test_list_forwards_runtime_options(self) -> None:
        """Agent tool options reach the catalog."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            interpreter=True,
            sandbox="none",
            allow_fs_tools="ls,read_file",
            no_mcp=True,
            mcp_config="/tmp/mcp.json",
            trust_project_mcp=True,
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=ToolCatalog(groups=()),
        ) as collect:
            code = run_tools_command(args)
        # --no-mcp means no discovery, so mcp_error is None → exit 0 even with
        # an explicit --mcp-config on the namespace.
        assert code == 0
        collect.assert_called_once_with(
            assistant_id="agent",
            enable_interpreter=True,
            fs_tools=["ls", "read_file"],
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
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=ToolCatalog(groups=()),
        ) as collect:
            code = run_tools_command(args)
        assert code == 0
        collect.assert_called_once_with(
            assistant_id="agent",
            enable_interpreter=False,
            fs_tools=None,
            include_mcp=True,
            mcp_config_path=None,
            trust_project_mcp=None,
        )

    def test_list_forwards_selected_agent(self) -> None:
        """`--agent` selects the agent whose built-in subagents are cataloged."""
        args = argparse.Namespace(
            tools_command="list",
            output_format="json",
            agent="custom-agent",
            resume_thread=None,
            interpreter=False,
            sandbox="none",
            no_mcp=False,
            mcp_config=None,
            trust_project_mcp=False,
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=ToolCatalog(groups=()),
        ) as collect:
            code = run_tools_command(args)
        assert code == 0
        assert collect.call_args.kwargs["assistant_id"] == "custom-agent"

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
                "deepagents_code.tool_catalog.collect_catalog",
                return_value=ToolCatalog(groups=()),
            ) as collect,
        ):
            code = run_tools_command(args)
        assert code == 0
        assert collect.call_args.kwargs["enable_interpreter"] is True
        assert collect.call_args.kwargs["include_mcp"] is True

    def test_list_singular_noun_for_one_tool(self) -> None:
        args = argparse.Namespace(tools_command="list", output_format="text")
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            )
        )
        with patch(
            "deepagents_code.tool_catalog.collect_catalog",
            return_value=catalog,
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "1 tool available" in output

    def test_long_description_clipped_to_terminal_width(self) -> None:
        """On a narrow terminal, descriptions clip and no row overflows width."""
        long_desc = "abcdefg " * 60  # ~480 chars, far wider than the terminal
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="read_file", description=long_desc),),
                ),
            )
        )
        args = argparse.Namespace(tools_command="list", output_format="text")
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code, output = _run_text(args, width=40)
        assert code == 0
        assert "read_file" in output
        # The full description cannot fit; it must have been clipped.
        assert long_desc.rstrip() not in output
        # Every rendered row stays within the terminal width (no_wrap + crop).
        assert all(len(line) <= 40 for line in output.splitlines())

    def test_disabled_server_renders_without_detail_suffix(self) -> None:
        """An unavailable server with no detail prints status alone, no `: `."""
        catalog = ToolCatalog(
            groups=(
                ToolGroup(
                    label="Built-in",
                    source="built-in",
                    tools=(ToolEntry(name="ls", description="List files"),),
                ),
            ),
            unavailable=(
                UnavailableServer(name="offsvc", status="disabled", detail=""),
            ),
        )
        args = argparse.Namespace(tools_command="list", output_format="text")
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code, output = _run_text(args)
        assert code == 0
        assert "Unavailable MCP servers" in output
        assert "offsvc" in output
        assert "disabled by user" in output
        # Empty detail → status stands alone, no trailing `: `.
        assert "disabled:" not in output

    def test_pending_reenable_renders_reconnect_guidance(self) -> None:
        catalog = ToolCatalog(
            groups=(),
            unavailable=(
                UnavailableServer(
                    name="notion",
                    status="disabled",
                    detail="Re-enabled — press Ctrl+R to load.",
                ),
            ),
        )
        args = argparse.Namespace(tools_command="list", output_format="text")
        with patch(
            "deepagents_code.tool_catalog.collect_catalog", return_value=catalog
        ):
            code, output = _run_text(args)

        assert code == 0
        assert "Re-enabled — press Ctrl+R to load." in output
        assert "disabled by user" not in output

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

    @pytest.mark.parametrize(
        ("text", "width", "ellipsis", "expected"),
        [
            ("hello", 10, "...", "hello"),  # fits comfortably
            ("hello", 5, "...", "hello"),  # exact fit, no clip
            ("hello world", 9, "...", "hello..."),  # clip + rstrip before ellipsis
            ("hello world", 8, "...", "hello..."),  # trailing space dropped
            ("abcdef", 3, "...", "abc"),  # width == len(ellipsis)
            ("abcdef", 2, "...", "ab"),  # width < len(ellipsis)
            ("abcdef", 0, "...", "abcdef"),  # zero width → unchanged
            ("abcdef", -5, "...", "abcdef"),  # negative width → unchanged
            ("hello world", 8, "…", "hello w…"),  # single-char ellipsis
        ],
    )
    def test_truncate(
        self, text: str, width: int, ellipsis: str, expected: str
    ) -> None:
        assert _truncate(text, width, ellipsis) == expected
