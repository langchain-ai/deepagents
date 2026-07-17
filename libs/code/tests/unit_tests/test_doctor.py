"""Unit tests for the `dcode doctor` command."""

import argparse
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from deepagents_code.doctor import (
    DiagnosticItem,
    DiagnosticSection,
    _build_commit,
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
            "explicitly_disabled": False,
            "has_credentials": False,
            "endpoint": None,
            "project": None,
            "project_is_default": False,
            "replica_project": None,
        }
        defaults.update(kwargs)
        # `defaults` is intentionally heterogeneous (`dict[str, object]`), so the
        # unpack can't be statically matched to each field's type.
        status = TracingStatus(**defaults)  # ty: ignore[invalid-argument-type]
        with patch("deepagents_code.config.get_tracing_status", return_value=status):
            return _collect_tracing()

    def test_not_configured_is_healthy(self) -> None:
        """An unconfigured, keyless setup is informational, not a failure."""
        section = self._section(enabled=False, project="deepagents-code")
        assert section.title == "Tracing"
        assert section.ok is True
        labels = {item.label: item.value for item in section.items}
        assert labels["Tracing"] == "not configured"
        assert labels["Credentials"] == "not set"
        assert labels["Project"] == "deepagents-code"

    def test_explicitly_disabled_reads_disabled(self) -> None:
        """An explicit opt-out reads `disabled`, not `not configured`."""
        section = self._section(enabled=False, explicitly_disabled=True)
        assert section.ok is True
        labels = {item.label: item.value for item in section.items}
        assert labels["Tracing"] == "disabled"
        assert labels["Credentials"] == "not set"

    def test_default_project_is_marked(self) -> None:
        """An unconfigured project shows the default marker."""
        section = self._section(project="deepagents-code", project_is_default=True)
        labels = {item.label: item.value for item in section.items}
        assert labels["Project"] == "deepagents-code (default)"

    def test_explicit_project_has_no_default_marker(self) -> None:
        """An explicitly set project name is reported verbatim."""
        section = self._section(project="deepagents-code", project_is_default=False)
        labels = {item.label: item.value for item in section.items}
        assert labels["Project"] == "deepagents-code"

    def test_unset_project_renders_unset(self) -> None:
        """A missing project renders the `(unset)` placeholder."""
        section = self._section(project=None)
        labels = {item.label: item.value for item in section.items}
        assert labels["Project"] == "(unset)"

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

    def test_gateway_yes_for_default_endpoint(self) -> None:
        """No custom endpoint means the SDK default (managed gateway) is used."""
        section = self._section(enabled=True, has_credentials=True, endpoint=None)
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "yes"

    def test_gateway_yes_for_langsmith_host(self) -> None:
        """A `smith.langchain.com` endpoint routes through the managed gateway."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            endpoint="https://api.smith.langchain.com",
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "yes"
        # A configured endpoint surfaces both items; they must not interfere.
        assert "Endpoint" in labels

    def test_gateway_no_for_custom_endpoint(self) -> None:
        """A self-hosted/dev endpoint is not the LangSmith managed gateway."""
        section = self._section(
            enabled=True,
            has_credentials=False,
            endpoint="http://localhost:1984",
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "no"

    def test_gateway_unknown_for_unparseable_endpoint(self) -> None:
        """An endpoint with no parseable host reports `unknown`, never `no`."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            endpoint="not a url",
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "unknown"

    def test_gateway_no_when_replica_endpoint_is_self_hosted(self) -> None:
        """A self-hosted replica target counts even with no primary endpoint."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            endpoint=None,
            runs_endpoints=("http://localhost:1984",),
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "no"

    def test_gateway_yes_when_replica_endpoint_is_gateway(self) -> None:
        """Replica targets on the managed gateway keep the report `yes`."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            endpoint=None,
            runs_endpoints=("https://eu.api.smith.langchain.com",),
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Gateway"] == "yes"

    def test_gateway_absent_when_not_enabled(self) -> None:
        """The Gateway line only appears when tracing is enabled."""
        section = self._section(enabled=False)
        labels = {item.label: item.value for item in section.items}
        assert "Gateway" not in labels

    def test_replica_project_listed_when_set(self) -> None:
        """A configured replica project is surfaced as its own item."""
        section = self._section(
            enabled=True,
            has_credentials=True,
            replica_project="replica",
        )
        labels = {item.label: item.value for item in section.items}
        assert labels["Replica project"] == "replica"


class TestEndpointGatewayState:
    """Tests for the single-endpoint tracing gateway-host classifier."""

    def test_exact_host_is_gateway(self) -> None:
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("https://smith.langchain.com") == "yes"

    def test_regional_subdomain_is_gateway(self) -> None:
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("https://eu.api.smith.langchain.com") == "yes"

    def test_whitespace_padded_endpoint_is_gateway(self) -> None:
        """A padded env value (a classic misconfiguration) still classifies."""
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("  https://smith.langchain.com  ") == "yes"

    def test_self_hosted_is_not_gateway(self) -> None:
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("http://localhost:1984") == "no"

    def test_suffix_lookalike_host_is_not_gateway(self) -> None:
        """A host containing the domain as a substring is not the gateway."""
        from deepagents_code.doctor import _endpoint_gateway_state

        assert (
            _endpoint_gateway_state("https://smith.langchain.com.evil.example") == "no"
        )

    def test_prefix_lookalike_host_is_not_gateway(self) -> None:
        """The leading-dot boundary rejects a host that only shares a suffix."""
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("https://notsmith.langchain.com") == "no"

    def test_unparseable_endpoint_is_unknown(self) -> None:
        """A string with no parseable host is `unknown`, not a false `no`."""
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("not a url") == "unknown"
        assert _endpoint_gateway_state("") == "unknown"

    def test_bracket_malformed_ipv6_is_unknown(self) -> None:
        """A `urlsplit` `ValueError` degrades to `unknown` rather than crashing."""
        from deepagents_code.doctor import _endpoint_gateway_state

        assert _endpoint_gateway_state("http://[::1") == "unknown"


class TestCollectUpdates:
    """Tests for the Updates diagnostic section."""

    def _labels(self, cache_file: Path) -> dict[str, str]:
        """Collect the Updates labels, reading `checked_at` from `cache_file`.

        Patches `CACHE_FILE` rather than `get_last_update_check_time` so the
        section flows through the genuine reader and the epoch -> ISO ->
        relative-time conversion, not a stub.
        """
        from deepagents_code.doctor import _collect_updates

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.update_check.is_update_check_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.is_auto_update_enabled",
                return_value=True,
            ),
            patch(
                "deepagents_code.update_check.get_cached_update_available",
                return_value=(False, "1.0.0"),
            ),
            patch("deepagents_code.update_check.CACHE_FILE", cache_file),
        ):
            section = _collect_updates()
        return {item.label: item.value for item in section.items}

    def test_last_checked_shows_relative_time(self, tmp_path: Path) -> None:
        """A check stamped an hour ago renders as `1h ago` via the real read."""
        cache = tmp_path / "latest_version.json"
        cache.write_text(
            json.dumps({"checked_at": time.time() - 3600}), encoding="utf-8"
        )
        assert self._labels(cache)["Last checked"] == "1h ago"

    def test_last_checked_just_now_on_future_stamp(self, tmp_path: Path) -> None:
        """A future stamp (clock skew) renders as `just now`, not a crash."""
        cache = tmp_path / "latest_version.json"
        cache.write_text(
            json.dumps({"checked_at": time.time() + 3600}), encoding="utf-8"
        )
        assert self._labels(cache)["Last checked"] == "just now"

    def test_last_checked_never_without_cache(self, tmp_path: Path) -> None:
        """An absent cache reports `never` rather than crashing."""
        assert self._labels(tmp_path / "latest_version.json")["Last checked"] == "never"

    def test_last_checked_never_on_corrupt_stamp(self, tmp_path: Path) -> None:
        """A non-finite stamp fails soft to `never` instead of crashing doctor."""
        cache = tmp_path / "latest_version.json"
        cache.write_text(json.dumps({"checked_at": float("nan")}), encoding="utf-8")
        assert self._labels(cache)["Last checked"] == "never"


class TestCommitHash:
    """Tests for git commit hash detection."""

    def test_uses_absolute_git_path(self, tmp_path) -> None:
        """Git metadata probing must not rely on subprocess PATH lookup."""
        git = tmp_path / "git"
        git.write_text("", encoding="utf-8")
        git.chmod(0o755)

        with (
            patch("deepagents_code.doctor._build_commit", return_value=None),
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
            patch("deepagents_code.doctor._build_commit", return_value=None),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert _commit_hash("/tmp") == "unknown"

        run.assert_not_called()

    def test_baked_commit_preferred_over_git(self) -> None:
        """A build-stamped commit wins for a wheel and skips the live git probe."""
        with (
            patch("deepagents_code.doctor._build_commit", return_value="deadbee"),
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("shutil.which") as which,
            patch("subprocess.run") as run,
        ):
            assert _commit_hash("/tmp") == "deadbee"

        which.assert_not_called()
        run.assert_not_called()

    def test_editable_install_ignores_baked_commit(self) -> None:
        """An editable install ignores a (possibly stale) stamp and probes git."""
        with (
            patch("deepagents_code.doctor._build_commit", return_value="deadbee"),
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch("shutil.which", return_value=None),
            patch("subprocess.run") as run,
        ):
            assert _commit_hash("/tmp") == "unknown"

        run.assert_not_called()

    def test_build_commit_missing_module(self) -> None:
        """No generated module (editable/dev install) yields `None`."""
        with patch.dict(sys.modules, {"deepagents_code._build_info": None}):
            assert _build_commit() is None

    def test_build_commit_reads_stamped_value(self) -> None:
        """A generated module exposes its stamped commit."""
        stub = SimpleNamespace(BUILD_COMMIT="abc1234")
        with patch.dict(sys.modules, {"deepagents_code._build_info": stub}):
            assert _build_commit() == "abc1234"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_build_commit_blank_value_is_none(self, value: str | None) -> None:
        """A present module with a blank stamp yields `None` (falls back to git)."""
        stub = SimpleNamespace(BUILD_COMMIT=value)
        with patch.dict(sys.modules, {"deepagents_code._build_info": stub}):
            assert _build_commit() is None

    def test_build_commit_corrupt_module_returns_none(self) -> None:
        """A present-but-corrupt stamp degrades to `None` instead of crashing."""

        class _Corrupt:
            def __getattr__(self, name: str) -> str:
                msg = "corrupt stamp"
                raise ValueError(msg)

        with patch.dict(sys.modules, {"deepagents_code._build_info": _Corrupt()}):
            assert _build_commit() is None


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
        assert "dcode config" in output
        assert "dcode config get <key>" in output
        assert "dcode --version" in output
        assert "dcode -v" in output

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
        assert "dcode config" in output
        assert "dcode config get <key>" in output
        assert "dcode --version" in output
        assert "dcode -v" in output
