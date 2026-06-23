"""Unit tests for the `dcode doctor` command."""

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from deepagents_code.doctor import (
    DiagnosticItem,
    DiagnosticSection,
    _commit_hash,
    collect_sections,
    run_doctor_command,
)
from deepagents_code.main import parse_args


class TestDoctorArgs:
    """Tests for `doctor` argument parsing."""

    def test_command_parsed(self) -> None:
        """`dcode doctor` selects the doctor command."""
        with patch.object(sys, "argv", ["deepagents", "doctor"]):
            args = parse_args()
        assert args.command == "doctor"

    def test_json_flag(self) -> None:
        """`dcode doctor --json` selects JSON output."""
        with patch.object(sys, "argv", ["deepagents", "doctor", "--json"]):
            args = parse_args()
        assert args.command == "doctor"
        assert args.output_format == "json"


class TestDiagnosticSection:
    """Tests for the section dataclass health aggregation."""

    def test_ok_when_all_items_ok(self) -> None:
        """A section is healthy when every item is healthy."""
        section = DiagnosticSection(
            title="X",
            items=[DiagnosticItem("a", "1"), DiagnosticItem("b", "2")],
        )
        assert section.ok is True

    def test_not_ok_when_any_item_fails(self) -> None:
        """A single failing item makes the section unhealthy."""
        section = DiagnosticSection(
            title="X",
            items=[DiagnosticItem("a", "1"), DiagnosticItem("b", "2", ok=False)],
        )
        assert section.ok is False


class TestCollectSections:
    """Tests for the diagnostic data collection."""

    def test_section_titles(self) -> None:
        """All sections are collected in display order."""
        sections = collect_sections()
        assert [s.title for s in sections] == [
            "Diagnostics",
            "Updates",
            "Tracing",
            "Configuration",
        ]

    def test_diagnostics_reports_version(self) -> None:
        """The Diagnostics section reports the running CLI version."""
        from deepagents_code._version import __version__

        diagnostics = collect_sections()[0]
        labels = {item.label: item.value for item in diagnostics.items}
        assert labels["deepagents-code"] == __version__
        assert "Commit hash" in labels
        assert labels["Commit hash"]
        assert "Platform" in labels
        assert "Install method" in labels


class TestCollectTracing:
    """Tests for the Tracing diagnostic section."""

    def _section(self, **kwargs: object) -> DiagnosticSection:
        from deepagents_code.config import TracingStatus
        from deepagents_code.doctor import _collect_tracing

        defaults: dict[str, object] = {
            "enabled": False,
            "has_credentials": False,
            "endpoint": None,
            "project": None,
            "replica_project": None,
        }
        defaults.update(kwargs)
        status = TracingStatus(**defaults)  # type: ignore[arg-type]
        with patch("deepagents_code.config.get_tracing_status", return_value=status):
            return _collect_tracing()

    def test_disabled_is_healthy(self) -> None:
        """A disabled, keyless setup is informational, not a failure."""
        section = self._section(enabled=False, project="deepagents-code")
        assert section.title == "Tracing"
        assert section.ok is True
        labels = {item.label: item.value for item in section.items}
        assert labels["Tracing"] == "disabled"
        assert labels["Credentials"] == "not configured"
        assert labels["Project"] == "deepagents-code"

    def test_enabled_without_credentials_is_unhealthy(self) -> None:
        """Tracing on with no key and no endpoint is a genuine problem."""
        section = self._section(enabled=True, has_credentials=False)
        assert section.ok is False
        creds = next(i for i in section.items if i.label == "Credentials")
        assert creds.ok is False

    def test_enabled_with_credentials_is_healthy(self) -> None:
        """A configured key keeps the section healthy and reports the project."""
        section = self._section(enabled=True, has_credentials=True, project="my-proj")
        assert section.ok is True
        labels = {item.label: item.value for item in section.items}
        assert labels["Tracing"] == "enabled"
        assert labels["Credentials"] == "configured"
        assert labels["Project"] == "my-proj"

    def test_keyless_custom_endpoint_is_healthy(self) -> None:
        """A custom endpoint is a valid keyless setup, so it stays healthy."""
        section = self._section(
            enabled=True,
            has_credentials=False,
            endpoint="http://localhost:1984",
        )
        assert section.ok is True
        labels = {item.label: item.value for item in section.items}
        assert labels["Endpoint"] == "http://localhost:1984"

    def test_endpoint_is_sanitized(self) -> None:
        """Endpoint diagnostics redact userinfo, path, query, and fragment."""
        section = self._section(
            enabled=True,
            has_credentials=False,
            endpoint=(
                "https://user:secret@example.com:8443/trace?api_key=secret-token#frag"
            ),
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Endpoint"] == "https://example.com:8443"
        assert "secret" not in labels["Endpoint"]
        assert "api_key" not in labels["Endpoint"]

    def test_replica_project_listed_when_set(self) -> None:
        """A configured replica project is surfaced as its own item."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            replica_project="replica",
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Replica project"] == "replica"


class TestCommitHash:
    """Tests for git commit hash detection."""

    def test_uses_absolute_git_path(self, tmp_path) -> None:
        """Git metadata probing must not rely on subprocess PATH lookup."""
        git = tmp_path / "git"
        git.write_text("", encoding="utf-8")
        git.chmod(0o755)

        with (
            patch("shutil.which", return_value=str(git)),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="abc123\n"),
            ) as run,
        ):
            assert _commit_hash(str(tmp_path)) == "abc123"

        argv = run.call_args.args[0]
        assert Path(argv[0]).is_absolute()
        assert argv[1:] == ["rev-parse", "--short", "HEAD"]

    def test_missing_git_returns_unknown(self) -> None:
        """Missing Git should degrade to `unknown` without spawning a process."""
        with (
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert _commit_hash("/tmp") == "unknown"

        run.assert_not_called()


class TestRunDoctorCommand:
    """Tests for the text and JSON rendering paths."""

    def _run_text(self) -> tuple[int, str]:
        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        args = argparse.Namespace(output_format="text")
        with patch("deepagents_code.config.console", test_console):
            code = run_doctor_command(args)
        return code, buf.getvalue()

    def test_text_output_contains_sections(self) -> None:
        """Text output renders each section title and key facts."""
        code, output = self._run_text()
        assert code == 0
        assert "Diagnostics" in output
        assert "Updates" in output
        assert "Tracing" in output
        assert "Configuration" in output
        assert "deepagents-code" in output

    def test_json_output_envelope(self, capsys) -> None:
        """JSON output is a stable envelope with section data."""
        args = argparse.Namespace(output_format="json")
        code = run_doctor_command(args)
        assert code == 0

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["command"] == "doctor"
        assert envelope["schema_version"] == 1
        data = envelope["data"]
        assert data["healthy"] is True
        titles = [section["title"] for section in data["sections"]]
        assert titles == ["Diagnostics", "Updates", "Tracing", "Configuration"]

    def test_unhealthy_returns_nonzero(self) -> None:
        """An unhealthy section yields a non-zero exit code."""
        unhealthy = [
            DiagnosticSection(
                title="Diagnostics",
                items=[DiagnosticItem("deepagents (SDK)", "not installed", ok=False)],
            )
        ]
        args = argparse.Namespace(output_format="text")
        buf = io.StringIO()
        with (
            patch("deepagents_code.doctor.collect_sections", return_value=unhealthy),
            patch(
                "deepagents_code.config.console",
                Console(file=buf, highlight=False, width=200),
            ),
        ):
            code = run_doctor_command(args)
        assert code == 1


class TestPathStatus:
    """Tests for the path-existence diagnostic item."""

    def test_existing_path_is_healthy(self, tmp_path) -> None:
        """An existing path reports `exists` and stays healthy."""
        from deepagents_code.doctor import _path_status

        item = _path_status("Data directory", tmp_path)
        assert item.ok is True
        assert "exists" in item.value

    def test_missing_path_is_healthy(self, tmp_path) -> None:
        """A not-yet-created path is informational, not a failure."""
        from deepagents_code.doctor import _path_status

        item = _path_status("Data directory", tmp_path / "absent")
        assert item.ok is True
        assert "not created" in item.value

    def test_unreadable_path_is_unhealthy(self, monkeypatch) -> None:
        """An unreadable path is flagged as a genuine problem (`ok=False`)."""
        from pathlib import Path

        from deepagents_code.doctor import _path_status

        def _raise(self: Path) -> object:  # noqa: ARG001  # must match Path.stat signature
            msg = "permission denied"
            raise PermissionError(msg)

        monkeypatch.setattr(Path, "stat", _raise)
        item = _path_status("Config file", "/some/protected/path")
        assert item.ok is False
        assert "unreadable" in item.value


class TestDoctorHelp:
    """Tests for the doctor help screen."""

    def test_help_renders(self) -> None:
        """`show_doctor_help` prints usage and examples."""
        from deepagents_code.ui import show_doctor_help

        buf = io.StringIO()
        test_console = Console(file=buf, highlight=False, width=200)
        with patch("deepagents_code.ui.console", test_console):
            show_doctor_help()
        output = buf.getvalue()
        assert "dcode doctor [options]" in output
        assert "Usage:" in output
