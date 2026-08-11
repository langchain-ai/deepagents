"""Unit tests for the editable-install dependency floor check."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

import pytest

import deepagents_code._dep_floor_check as dep_floor_check
from deepagents_code._dep_floor_check import (
    _find_floor_violations,
    _load_cli_requirements,
    _quote_arg,
    consume_dep_floor_notice,
    warn_if_editable_deps_stale,
)


@pytest.fixture(autouse=True)
def _editable_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to an editable install.

    Tests that exercise the non-editable gate re-patch the same symbol.
    """
    monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: True)


@pytest.fixture(autouse=True)
def _clear_dep_floor_notice() -> None:
    """Reset the stashed TUI notice so one test's warning cannot leak."""
    dep_floor_check._dep_floor_notice = None


def _patch_versions(monkeypatch: pytest.MonkeyPatch, versions: dict[str, str]) -> None:
    """Resolve `importlib.metadata.version` lookups from `versions` only."""

    def fake_version(name: str) -> str:
        try:
            return versions[name]
        except KeyError:
            raise dep_floor_check.importlib.metadata.PackageNotFoundError(
                name
            ) from None

    monkeypatch.setattr(dep_floor_check.importlib.metadata, "version", fake_version)


class TestEditableGate:
    """The PEP 610 editable gate decides whether the check runs at all."""

    def test_non_editable_install_skips_floor_logic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Released installs must not parse requirements or read versions."""
        monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: False)

        def _fail(*_args: object, **_kwargs: object) -> None:
            msg = "floor logic ran for a non-editable install"
            raise AssertionError(msg)

        monkeypatch.setattr(dep_floor_check, "_load_cli_requirements", _fail)
        monkeypatch.setattr(dep_floor_check, "_find_floor_violations", _fail)
        monkeypatch.setattr(dep_floor_check.importlib.metadata, "version", _fail)

        warn_if_editable_deps_stale()  # must not raise or call any of the above


class TestWarnAndContinue:
    """Editable installs with stale deps warn and return normally."""

    def test_below_floor_dep_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A dep below its floor names the dist, versions, and remediation."""
        monkeypatch.setattr(
            dep_floor_check,
            "_load_cli_requirements",
            lambda: ["quickjs-rs>=0.2.5", "packaging>=26.2"],
        )
        _patch_versions(monkeypatch, {"quickjs-rs": "0.2.4", "packaging": "26.2"})

        result = warn_if_editable_deps_stale()

        assert result is None  # warn-and-continue: returns, never exits
        captured = capsys.readouterr()
        assert captured.out == ""
        text = captured.err
        assert "Warning" in text
        assert "quickjs-rs" in text
        assert "0.2.4" in text
        assert "0.2.5" in text
        assert "Refresh the active environment:" in text
        assert "uv pip install --python" in text
        assert "--upgrade" in text
        # The satisfied floor is not listed as a violation.
        assert "packaging 26.2" not in text

    def test_all_deps_satisfied_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An editable install at or above every floor prints nothing."""
        monkeypatch.setattr(
            dep_floor_check,
            "_load_cli_requirements",
            lambda: ["quickjs-rs>=0.2.5", "packaging>=26.2,<27"],
        )
        _patch_versions(monkeypatch, {"quickjs-rs": "0.2.6", "packaging": "26.2"})

        warn_if_editable_deps_stale()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestExactPins:
    """`==` pins participate in the floor comparison; wildcards do not."""

    def test_older_than_exact_pin_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An install older than a hard pin (e.g. the SDK's) is stale."""
        monkeypatch.setattr(
            dep_floor_check, "_load_cli_requirements", lambda: ["deepagents==0.7.4"]
        )
        _patch_versions(monkeypatch, {"deepagents": "0.7.3"})

        warn_if_editable_deps_stale()

        captured = capsys.readouterr()
        assert captured.out == ""
        text = captured.err
        assert "deepagents" in text
        assert "0.7.3" in text
        assert "0.7.4" in text

    def test_exact_pin_satisfied_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An install at the pinned version reports no violation."""
        _patch_versions(monkeypatch, {"deepagents": "0.7.4"})
        assert _find_floor_violations(["deepagents==0.7.4"]) == []

    def test_wildcard_equality_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`==X.*` has no single parseable version, so it cannot be a floor."""
        _patch_versions(monkeypatch, {"some-dep": "0.9"})
        assert _find_floor_violations(["some-dep==1.0.*"]) == []


class TestQuoteArg:
    """The refresh command is quoted for the platform's shell conventions."""

    def test_posix_single_quotes_spaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POSIX shells get `shlex.quote`-style single-quoted arguments."""
        monkeypatch.setattr(dep_floor_check.sys, "platform", "darwin")
        assert _quote_arg("/path with spaces/bin/python") == (
            "'/path with spaces/bin/python'"
        )
        assert _quote_arg("/plain/path") == "/plain/path"

    def test_windows_uses_cmdline_quoting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows gets argv quoting `cmd.exe` understands, not single quotes."""
        monkeypatch.setattr(dep_floor_check.sys, "platform", "win32")
        # cmd.exe does not treat single quotes as quoting, so a POSIX-quoted
        # path would arrive with the quote characters included.
        assert _quote_arg("C:\\Python\\python.exe") == "C:\\Python\\python.exe"
        assert _quote_arg("C:\\Program Files\\Python\\python.exe") == (
            '"C:\\Program Files\\Python\\python.exe"'
        )


class TestBestEffort:
    """Malformed or missing metadata degrades to a debug log, never a crash."""

    def test_checkout_pyproject_reads(self) -> None:
        """The live checkout's pyproject parses into requirement strings."""
        entries = _load_cli_requirements()
        assert entries is not None
        assert entries
        assert all(isinstance(entry, str) for entry in entries)

    def test_missing_distribution_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A requirement whose dist is not installed never crashes."""
        _patch_versions(monkeypatch, {})  # nothing "installed"
        assert _find_floor_violations(["no-such-dist>=1.0"]) == []

    def test_unparseable_requirement_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Garbage requirement strings never crash the check."""
        _patch_versions(monkeypatch, {"packaging": "26.2"})
        assert _find_floor_violations(["!!! not a requirement !!!"]) == []

    def test_marker_false_requirement_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A requirement gated to another platform reports no violation."""
        _patch_versions(monkeypatch, {"some-dep": "0.1"})
        assert (
            _find_floor_violations(
                ["some-dep>=9.9; sys_platform == 'nonexistent_platform'"]
            )
            == []
        )

    def test_unparseable_installed_version_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed installed version string reports no violation."""
        _patch_versions(monkeypatch, {"some-dep": "not-a-version"})
        assert _find_floor_violations(["some-dep>=1.0"]) == []

    def test_check_swallows_unexpected_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any unexpected failure inside the check is swallowed."""

        def _boom() -> list[str]:
            msg = "simulated metadata failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(dep_floor_check, "_load_cli_requirements", _boom)
        warn_if_editable_deps_stale()  # must not raise

    def test_missing_packaging_does_not_break_the_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken optional check dependency never prevents CLI startup."""
        monkeypatch.setattr(
            dep_floor_check, "_load_cli_requirements", lambda: ["some-dep>=1.0"]
        )
        original_import = builtins.__import__

        def _missing_packaging(
            name: str,
            globals_: Mapping[str, object] | None = None,
            locals_: Mapping[str, object] | None = None,
            fromlist: Sequence[str] | None = (),
            level: int = 0,
        ) -> ModuleType:
            if name.startswith("packaging"):
                msg = "simulated missing packaging"
                raise ModuleNotFoundError(msg)
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _missing_packaging)

        warn_if_editable_deps_stale()  # must not raise


class TestAnnounceVsStash:
    """`announce` chooses stderr print (non-TUI) vs. stashed toast (TUI)."""

    def _stale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dep_floor_check, "_load_cli_requirements", lambda: ["quickjs-rs>=0.2.5"]
        )
        _patch_versions(monkeypatch, {"quickjs-rs": "0.2.4"})

    def test_announce_prints_and_does_not_stash(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Non-interactive launches print to stderr and leave nothing stashed."""
        self._stale(monkeypatch)

        warn_if_editable_deps_stale(announce=True)

        assert "Warning" in capsys.readouterr().err
        assert consume_dep_floor_notice() is None

    def test_no_announce_stashes_and_does_not_print(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Interactive launches stash a plain-text notice and print nothing."""
        self._stale(monkeypatch)

        warn_if_editable_deps_stale(announce=False)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        notice = consume_dep_floor_notice()
        assert notice is not None
        assert "Warning" in notice
        assert "quickjs-rs" in notice
        assert "0.2.4" in notice
        assert "0.2.5" in notice
        assert "uv pip install --python" in notice
        # The TUI toast renders markup=False, so no Rich tags survive.
        assert "[bold yellow]" not in notice

    def test_consume_clears_the_notice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The stashed notice is returned once, then cleared."""
        self._stale(monkeypatch)
        warn_if_editable_deps_stale(announce=False)

        assert consume_dep_floor_notice() is not None
        assert consume_dep_floor_notice() is None

    def test_no_violations_stashes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A satisfied editable install stashes no notice for the TUI."""
        monkeypatch.setattr(
            dep_floor_check, "_load_cli_requirements", lambda: ["quickjs-rs>=0.2.5"]
        )
        _patch_versions(monkeypatch, {"quickjs-rs": "0.2.6"})

        warn_if_editable_deps_stale(announce=False)

        assert consume_dep_floor_notice() is None
