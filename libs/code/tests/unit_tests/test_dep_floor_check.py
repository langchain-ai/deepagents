"""Unit tests for the editable-install dependency floor check."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest
from rich.console import Console

import deepagents_code._dep_floor_check as dep_floor_check
from deepagents_code import config
from deepagents_code._dep_floor_check import (
    _find_floor_violations,
    _load_cli_requirements,
    warn_if_editable_deps_stale,
)


@contextmanager
def _patched_config_console(console: Console) -> Iterator[None]:
    """Point `deepagents_code.config`'s cached console at `console`.

    `_get_console()` resolves through `globals()`, so assigning the
    module-level `console` attribute is enough to capture output.
    """
    sentinel = object()
    previous = config.__dict__.get("console", sentinel)
    config.__dict__["console"] = console
    try:
        yield
    finally:
        if previous is sentinel:
            config.__dict__.pop("console", None)
        else:
            config.__dict__["console"] = previous


@pytest.fixture
def console_output(tmp_path: Path) -> Iterator[Path]:
    """Route the config console to a file and yield its path for assertions."""
    out = tmp_path / "console.txt"
    with out.open("w", encoding="utf-8") as stream:
        console = Console(file=stream, force_terminal=False, width=200)
        with _patched_config_console(console):
            yield out


@pytest.fixture(autouse=True)
def _editable_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to an editable install.

    Tests that exercise the non-editable gate re-patch the same symbol.
    """
    monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: True)


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
        self, monkeypatch: pytest.MonkeyPatch, console_output: Path
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
        text = console_output.read_text(encoding="utf-8")
        assert "Warning" in text
        assert "quickjs-rs" in text
        assert "0.2.4" in text
        assert "0.2.5" in text
        assert (
            "uv pip install --python ~/.local/share/dcode-dev/bin/python "
            "-e <repo>/libs/code --upgrade" in text
        )
        # The satisfied floor is not listed as a violation.
        assert "packaging 26.2" not in text

    def test_all_deps_satisfied_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch, console_output: Path
    ) -> None:
        """An editable install at or above every floor prints nothing."""
        monkeypatch.setattr(
            dep_floor_check,
            "_load_cli_requirements",
            lambda: ["quickjs-rs>=0.2.5", "packaging>=26.2,<27"],
        )
        _patch_versions(monkeypatch, {"quickjs-rs": "0.2.6", "packaging": "26.2"})

        warn_if_editable_deps_stale()

        assert console_output.read_text(encoding="utf-8") == ""


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
