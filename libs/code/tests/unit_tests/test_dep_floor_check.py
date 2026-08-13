"""Unit tests for the editable-install dependency floor check."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

import pytest

import deepagents_code._dep_floor_check as dep_floor_check
import deepagents_code.main as main_module
from deepagents_code._dep_floor_check import (
    _DEBUG_VIOLATION,
    _find_floor_violations,
    _FloorViolation,
    _load_cli_requirements,
    _mismatch_fingerprint,
    _quote_arg,
    format_dep_floor_warning,
    is_dep_floor_mismatch_muted,
    mute_dep_floor_mismatch,
    prompt_if_editable_deps_stale,
    warn_if_editable_deps_stale,
)
from deepagents_code.main import _TrustAction, _TrustPromptOutcome

_VIOLATIONS = [
    _FloorViolation(
        dist_name="quickjs-rs",
        installed="0.2.4",
        floor="0.2.5",
        required="quickjs-rs>=0.2.5",
    )
]


@pytest.fixture(autouse=True)
def _editable_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to an editable install.

    Tests that exercise the non-editable gate re-patch the same symbol.
    """
    monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: True)


@pytest.fixture(autouse=True)
def dismissal_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the dismissal store: every test gets a fresh, writable file."""
    path = tmp_path / "dep_floor_dismissed.json"
    monkeypatch.setattr(dep_floor_check, "_default_dismissal_path", lambda: path)
    return path


class TestEditableGate:
    """The PEP 610 editable gate decides whether the check runs at all."""

    def test_non_editable_install_skips_floor_logic(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
        assert capsys.readouterr().err == ""


class TestWarnAndContinue:
    """Editable installs with stale deps warn on stderr and return normally."""

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

    def test_compatible_release_floor_is_honored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`~=` declares a floor, so an older install is a violation."""
        _patch_versions(monkeypatch, {"some-dep": "1.4.1"})
        violations = _find_floor_violations(["some-dep~=1.4.2"])
        assert [(v.dist_name, v.installed, v.floor) for v in violations] == [
            ("some-dep", "1.4.1", "1.4.2")
        ]

    def test_unreadable_dist_metadata_skips_only_that_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corrupt `.dist-info` must not abandon the rest of the scan.

        Letting the failure escape would discard violations already found and
        leave later entries unexamined — a false all-clear for every dep.
        """

        def fake_version(name: str) -> str:
            if name == "broken-dep":
                msg = "corrupt .dist-info"
                raise OSError(msg)
            return "0.9"

        monkeypatch.setattr(dep_floor_check.importlib.metadata, "version", fake_version)

        violations = _find_floor_violations(["broken-dep>=1.0", "later-dep>=1.0"])

        assert [v.dist_name for v in violations] == ["later-dep"]


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
        # `_load_cli_requirements` reads the `pyproject.toml` two directories
        # above the installed module. Wheel installs (e.g. the release
        # pipeline's built-wheel test run) have no checkout `pyproject.toml`
        # there, so the function takes its designed `None` fallback and this
        # test has nothing to assert against.
        pyproject = (
            Path(dep_floor_check.__file__).resolve().parent.parent / "pyproject.toml"
        )
        if not pyproject.is_file():
            pytest.skip("no checkout pyproject.toml adjacent to the installed package")
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


class TestDebugDepFloor:
    """`DEEPAGENTS_CODE_DEBUG_DEP_FLOOR` synthesizes a stale-deps warning."""

    def test_debug_var_synthesizes_warning_on_non_editable_install(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The synthetic warning bypasses the editable gate and real versions."""
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_DEP_FLOOR", "1")
        monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: False)

        def _fail(*_args: object, **_kwargs: object) -> None:
            msg = "real floor logic ran under the debug var"
            raise AssertionError(msg)

        monkeypatch.setattr(dep_floor_check, "_load_cli_requirements", _fail)
        monkeypatch.setattr(dep_floor_check, "_find_floor_violations", _fail)
        monkeypatch.setattr(dep_floor_check.importlib.metadata, "version", _fail)

        warn_if_editable_deps_stale()

        captured = capsys.readouterr()
        assert captured.out == ""
        text = captured.err
        assert "Warning" in text
        assert "packaging" in text
        assert "0.0.1" in text
        assert "uv pip install --python" in text

    def test_debug_var_drives_the_interactive_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The synthetic violation reaches the interactive prompt path."""
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_DEP_FLOOR", "1")
        monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: False)
        seen: list[list[_FloorViolation]] = []

        def _prompt(_console: object, violations: list[_FloorViolation]) -> object:
            seen.append(violations)
            return _TrustAction.ALLOW_ONCE

        monkeypatch.setattr(main_module, "prompt_for_dep_floor_mismatch", _prompt)

        assert prompt_if_editable_deps_stale() is None
        assert seen == [[_DEBUG_VIOLATION]]

    def test_debug_var_falsy_falls_back_to_real_check(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`DEEPAGENTS_CODE_DEBUG_DEP_FLOOR=0` means no synthetic warning."""
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_DEP_FLOOR", "0")
        monkeypatch.setattr(dep_floor_check, "_is_editable_install", lambda: False)

        warn_if_editable_deps_stale()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestMuteStore:
    """The per-checkout dismissal store silences only the muted mismatch."""

    def test_muted_mismatch_suppresses_the_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            dep_floor_check, "_collect_violations", lambda: list(_VIOLATIONS)
        )
        assert mute_dep_floor_mismatch(_mismatch_fingerprint(_VIOLATIONS))

        warn_if_editable_deps_stale()

        assert capsys.readouterr().err == ""

    def test_changed_mismatch_re_arms_the_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A different violation set must not match the muted fingerprint."""
        monkeypatch.setattr(
            dep_floor_check, "_collect_violations", lambda: list(_VIOLATIONS)
        )
        assert mute_dep_floor_mismatch(_mismatch_fingerprint(_VIOLATIONS))

        changed = [
            _FloorViolation(
                dist_name="quickjs-rs",
                installed="0.2.4",
                floor="0.3.0",
                required="quickjs-rs>=0.3.0",
            )
        ]
        monkeypatch.setattr(
            dep_floor_check, "_collect_violations", lambda: list(changed)
        )
        warn_if_editable_deps_stale()

        assert "Warning" in capsys.readouterr().err

    def test_mute_is_keyed_per_checkout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        dismissal_store: Path,
    ) -> None:
        """A dismissal recorded under one checkout never mutes another."""
        monkeypatch.setattr(
            dep_floor_check, "_collect_violations", lambda: list(_VIOLATIONS)
        )
        assert mute_dep_floor_mismatch(_mismatch_fingerprint(_VIOLATIONS))
        stored = json.loads(dismissal_store.read_text(encoding="utf-8"))
        assert len(stored["dismissed"]) == 1

        monkeypatch.setattr(dep_floor_check, "_checkout_key", lambda: "/other/checkout")
        warn_if_editable_deps_stale()

        assert "Warning" in capsys.readouterr().err

    def test_corrupt_store_re_arms(self, dismissal_store: Path) -> None:
        """An unreadable store degrades to 'not muted', never a crash."""
        dismissal_store.write_text("not json", encoding="utf-8")
        assert not is_dep_floor_mismatch_muted("anything")
        # Muting over the corrupt file recovers cleanly.
        assert mute_dep_floor_mismatch("fresh")
        assert is_dep_floor_mismatch_muted("fresh")

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("[]", id="root-not-a-mapping"),
            pytest.param('{"dismissed": []}', id="dismissed-not-a-mapping"),
            pytest.param('{"dismissed": {"/checkout": 123}}', id="fingerprint-not-str"),
        ],
    )
    def test_malformed_store_shapes_re_arm(
        self, dismissal_store: Path, payload: str
    ) -> None:
        """Every shape that is not `{"dismissed": {str: str}}` re-arms."""
        dismissal_store.write_text(payload, encoding="utf-8")
        assert not is_dep_floor_mismatch_muted("anything")

    def test_one_bad_entry_does_not_unmute_the_others(
        self, dismissal_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-string fingerprint drops its own row, not the whole map."""
        dismissal_store.write_text(
            json.dumps({"dismissed": {"/other": 123, "/mine": "fp"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(dep_floor_check, "_checkout_key", lambda: "/mine")

        assert is_dep_floor_mismatch_muted("fp")

    def test_mute_write_failure_reports_false_and_leaves_no_temp_file(
        self, dismissal_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed atomic replace is reported, not swallowed as success."""

        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "simulated replace failure"
            raise OSError(msg)

        monkeypatch.setattr(dep_floor_check.Path, "replace", _boom)

        assert not mute_dep_floor_mismatch("fp")
        assert not list(dismissal_store.parent.glob("*.tmp"))


class TestInteractivePrompt:
    """`prompt_if_editable_deps_stale` maps picker outcomes to launch control."""

    @pytest.fixture(autouse=True)
    def _violations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dep_floor_check, "_collect_violations", lambda: list(_VIOLATIONS)
        )

    def _stub_prompt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        action: _TrustAction | _TrustPromptOutcome,
    ) -> None:
        monkeypatch.setattr(
            main_module, "prompt_for_dep_floor_mismatch", lambda _c, _v: action
        )

    def test_no_violations_continues_without_prompting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dep_floor_check, "_collect_violations", list)

        def _fail(*_args: object) -> None:
            msg = "prompt shown without violations"
            raise AssertionError(msg)

        monkeypatch.setattr(main_module, "prompt_for_dep_floor_mismatch", _fail)
        assert prompt_if_editable_deps_stale() is None

    def test_muted_mismatch_continues_without_prompting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert mute_dep_floor_mismatch(_mismatch_fingerprint(_VIOLATIONS))

        def _fail(*_args: object) -> None:
            msg = "prompt shown for a muted mismatch"
            raise AssertionError(msg)

        monkeypatch.setattr(main_module, "prompt_for_dep_floor_mismatch", _fail)
        assert prompt_if_editable_deps_stale() is None

    def test_allow_once_continues_without_muting(
        self, monkeypatch: pytest.MonkeyPatch, dismissal_store: Path
    ) -> None:
        self._stub_prompt(monkeypatch, _TrustAction.ALLOW_ONCE)

        assert prompt_if_editable_deps_stale() is None
        assert not dismissal_store.exists()

    def test_remember_mutes_the_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_prompt(monkeypatch, _TrustAction.REMEMBER)

        assert prompt_if_editable_deps_stale() is None
        assert is_dep_floor_mismatch_muted(_mismatch_fingerprint(_VIOLATIONS))

    def test_remember_warns_when_the_mute_cannot_be_saved(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._stub_prompt(monkeypatch, _TrustAction.REMEMBER)
        monkeypatch.setattr(
            dep_floor_check, "mute_dep_floor_mismatch", lambda _fp: False
        )

        assert prompt_if_editable_deps_stale() is None
        assert "could not be saved" in capsys.readouterr().err

    def test_cancelled_aborts_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_prompt(monkeypatch, _TrustPromptOutcome.CANCELLED)
        assert prompt_if_editable_deps_stale() is _TrustPromptOutcome.CANCELLED

    def test_picking_abort_launch_aborts_the_launch(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Selecting "Abort launch" in the picker must not start the TUI.

        Exercises the real `prompt_for_dep_floor_mismatch`, stubbing only the
        picker, because the regression this guards lived in the translation
        between the picker's `DENY` and the launch-control outcome.
        """
        monkeypatch.setattr(main_module, "_trust_picker_has_terminal", lambda: True)
        monkeypatch.setattr(
            main_module,
            "_run_trust_action_picker",
            lambda *_args, **_kwargs: _TrustAction.DENY,
        )

        assert prompt_if_editable_deps_stale() is _TrustPromptOutcome.CANCELLED
        assert "Continuing this session" not in capsys.readouterr().err

    def test_unexpected_outcome_aborts_rather_than_continuing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unmapped outcome must refuse, not fall through to "continue"."""
        self._stub_prompt(monkeypatch, _TrustAction.DENY)
        assert prompt_if_editable_deps_stale() is _TrustPromptOutcome.CANCELLED

    def test_interrupt_aborts_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_prompt(monkeypatch, _TrustPromptOutcome.INTERRUPTED)
        assert prompt_if_editable_deps_stale() is _TrustPromptOutcome.INTERRUPTED

    def test_check_failure_continues_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unexpected metadata failures remain non-fatal before the TUI."""

        def _fail() -> list[_FloorViolation]:
            msg = "simulated metadata failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(dep_floor_check, "_collect_violations", _fail)

        assert prompt_if_editable_deps_stale() is None


class TestCliMainChannel:
    """`cli_main` picks the prompt or the warning from the launch mode."""

    @pytest.fixture
    def channels(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Count which channel `cli_main` reaches, without running either."""
        calls = {"prompt": 0, "warn": 0}

        def _prompt() -> None:
            """Stand in for the prompt, always choosing to continue."""
            calls["prompt"] += 1

        def _warn() -> None:
            calls["warn"] += 1

        monkeypatch.setattr(dep_floor_check, "prompt_if_editable_deps_stale", _prompt)
        monkeypatch.setattr(dep_floor_check, "warn_if_editable_deps_stale", _warn)
        monkeypatch.setattr(main_module, "check_cli_dependencies", lambda: None)
        monkeypatch.setattr(main_module, "apply_stdin_pipe", lambda _args: None)
        return calls

    def _run_cli_main(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str], *, tty: bool
    ) -> None:
        """Invoke `cli_main` far enough to pass the dep-floor gate, then stop.

        `--no-mcp` with `--mcp-config` is rejected on the statement right
        after the gate, so it stops the launch without stubbing session setup.
        """
        monkeypatch.setattr(main_module, "_trust_picker_has_terminal", lambda: tty)
        monkeypatch.setattr(
            main_module.sys,
            "argv",
            ["dcode", *argv, "--no-mcp", "--mcp-config", "/some/path"],
        )

        with pytest.raises(SystemExit) as exc:
            main_module.cli_main()

        assert exc.value.code == 2

    def test_terminal_interactive_launch_prompts(
        self, monkeypatch: pytest.MonkeyPatch, channels: dict[str, int]
    ) -> None:
        self._run_cli_main(monkeypatch, [], tty=True)
        assert channels == {"prompt": 1, "warn": 0}

    def test_piped_stdin_interactive_launch_warns_instead_of_prompting(
        self, monkeypatch: pytest.MonkeyPatch, channels: dict[str, int]
    ) -> None:
        """`cat x | dcode -m ...` still mounts the TUI but cannot be prompted.

        `apply_stdin_pipe` routes piped text into `initial_prompt` there, not
        `non_interactive_message`, so the launch looks interactive while stdin
        is a pipe. Prompting would read EOF and abort the launch outright.
        """
        self._run_cli_main(monkeypatch, ["-m", "explain this"], tty=False)
        assert channels == {"prompt": 0, "warn": 1}

    def test_headless_launch_warns(
        self, monkeypatch: pytest.MonkeyPatch, channels: dict[str, int]
    ) -> None:
        self._run_cli_main(monkeypatch, ["-n", "summarize"], tty=True)
        assert channels == {"prompt": 0, "warn": 1}


def test_dep_floor_prompt_escapes_dynamic_rich_markup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Requirement extras and bracketed checkout paths render literally."""
    from rich.console import Console

    violation = _FloorViolation(
        dist_name="langgraph-cli[inmem]",
        installed="0.4.0[local]",
        floor="0.5.0",
        required="langgraph-cli[inmem]>=0.5.0",
    )
    monkeypatch.setattr(
        dep_floor_check,
        "_checkout_root",
        lambda: dep_floor_check.Path("/tmp/[bold]"),
    )
    monkeypatch.setattr(
        main_module,
        "_select_trust_action",
        lambda *_args, **_kwargs: _TrustAction.ALLOW_ONCE,
    )

    main_module.prompt_for_dep_floor_mismatch(Console(stderr=True), [violation])

    text = capsys.readouterr().err
    assert "langgraph-cli[inmem]" in text
    assert "0.4.0[local]" in text
    assert "/tmp/[bold]" in text


def test_dep_floor_prompt_separates_guidance_and_actions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The remediation, risk warning, and chooser each start after a blank line."""
    from rich.console import Console

    picker: dict[str, object] = {}

    def _select(*_args: object, **kwargs: object) -> _TrustAction:
        picker.update(kwargs)
        return _TrustAction.ALLOW_ONCE

    monkeypatch.setattr(dep_floor_check, "refresh_command", lambda: "uv sync")
    monkeypatch.setattr(main_module, "_select_trust_action", _select)

    main_module.prompt_for_dep_floor_mismatch(Console(stderr=True), _VIOLATIONS)

    text = capsys.readouterr().err
    assert "\n\nRefresh the active environment:" in text
    assert "  uv sync\n\nRunning stale source" in text
    assert "hard-to-diagnose ways.\n\n" in text
    assert picker["remember_label"] == "Continue and hide until versions change"


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
