"""Tests for the `dcode auth` CLI subcommands."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import IO

import pytest

from deepagents_code import auth_store
from deepagents_code.auth_commands import run_auth_command


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` and `DEFAULT_STATE_DIR` into a temp directory."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


@pytest.mark.usefixtures("fake_home")
class TestSet:
    """`auth set` reads keys from stdin or an env var, never argv."""

    def test_set_from_stdin(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A piped key is stored and a confirmation (not the key) is printed."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-ant-secret\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 0
        assert auth_store.get_stored_key("anthropic") == "sk-ant-secret"
        out = capsys.readouterr().out
        assert "Stored credential for anthropic." in out
        assert "sk-ant-secret" not in out

    def test_set_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--from-env` copies the key from a process env var."""
        monkeypatch.setenv("MY_KEY", "sk-openai-abc")
        code = run_auth_command(
            _ns(auth_command="set", provider="openai", from_env="MY_KEY")
        )
        assert code == 0
        assert auth_store.get_stored_key("openai") == "sk-openai-abc"

    def test_set_from_unset_env_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--from-env` on an unset variable exits non-zero and stores nothing."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        code = run_auth_command(
            _ns(auth_command="set", provider="groq", from_env="MISSING_VAR")
        )
        assert code == 1
        assert auth_store.get_stored_key("groq") is None
        assert "MISSING_VAR is not set or is empty" in capsys.readouterr().err

    def test_set_rejects_tty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An interactive terminal is rejected so the command never hangs."""

        class _TTY(io.StringIO):
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(sys, "stdin", _TTY())
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 1
        assert auth_store.get_stored_key("anthropic") is None
        assert "interactive terminal" in capsys.readouterr().err

    def test_set_empty_stdin_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty piped key is rejected rather than stored as a blank."""
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 1
        assert auth_store.get_stored_key("anthropic") is None


@pytest.mark.usefixtures("fake_home")
class TestRemove:
    """`auth remove` deletes a stored credential and is idempotent."""

    def test_remove_existing(self, capsys: pytest.CaptureFixture[str]) -> None:
        auth_store.set_stored_key("anthropic", "sk-ant")
        code = run_auth_command(_ns(auth_command="remove", provider="anthropic"))
        assert code == 0
        assert auth_store.get_stored_key("anthropic") is None
        assert "Removed stored credential for anthropic." in capsys.readouterr().out

    def test_remove_absent_is_noop(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = run_auth_command(_ns(auth_command="delete", provider="anthropic"))
        assert code == 0
        assert "No stored credential for anthropic." in capsys.readouterr().out


@pytest.mark.usefixtures("fake_home")
class TestStatus:
    """`auth status` reports the resolution source the TUI shows."""

    def test_status_stored(self, capsys: pytest.CaptureFixture[str]) -> None:
        auth_store.set_stored_key("anthropic", "sk-ant")
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "stored" in out

    def test_status_env(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "env: ANTHROPIC_API_KEY" in capsys.readouterr().out

    def test_status_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "missing" in capsys.readouterr().out


@pytest.mark.usefixtures("fake_home")
def test_path_prints_resolved_location(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`auth path` prints the resolved `auth.json` location."""
    code = run_auth_command(_ns(auth_command="path"))
    assert code == 0
    expected = fake_home / ".deepagents" / ".state" / "auth.json"
    assert capsys.readouterr().out.strip() == str(expected)


@pytest.mark.usefixtures("fake_home")
def test_no_subcommand_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare `auth` invocation renders the help screen."""
    code = run_auth_command(_ns(auth_command=None))
    assert code == 0
    assert "dcode auth <command>" in capsys.readouterr().out


# --- Subprocess round-trip (per issue coverage requirements) ----------------


def _run_cli(
    argv: list[str],
    *,
    home: Path,
    stdin: int | IO[bytes] | None = subprocess.DEVNULL,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke `cli_main` in a subprocess with an isolated `HOME`."""
    code = """
        import json
        import sys
        from unittest.mock import patch

        from deepagents_code.main import cli_main

        argv = ["deepagents", *json.loads(sys.argv[1])]
        with (
            patch.object(sys, "argv", argv),
            patch("deepagents_code.main.check_cli_dependencies"),
        ):
            cli_main()
    """
    import os

    env = dict(os.environ)
    env["HOME"] = str(home)
    # Drop provider env vars so subprocess status is deterministic.
    for key in list(env):
        if key.endswith("_API_KEY"):
            del env[key]
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), json.dumps(argv)],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=stdin,
        env=env,
        check=False,
    )


def test_subprocess_set_from_file_then_status(tmp_path: Path) -> None:
    """End-to-end: a key piped from a file is stored and reported as `stored`."""
    home = tmp_path / "home"
    home.mkdir()
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-ant-from-file\n", encoding="utf-8")

    with key_file.open("rb") as fh:
        set_result = _run_cli(["auth", "set", "anthropic"], home=home, stdin=fh)
    assert set_result.returncode == 0, set_result.stderr
    assert "Stored credential for anthropic." in set_result.stdout
    assert "sk-ant-from-file" not in set_result.stdout

    status_result = _run_cli(["auth", "status", "anthropic"], home=home)
    assert status_result.returncode == 0, status_result.stderr
    assert "stored" in status_result.stdout


def test_subprocess_from_env_unset_fails(tmp_path: Path) -> None:
    """`--from-env` on an unset variable exits non-zero with a clear error."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_cli(
        ["auth", "set", "anthropic", "--from-env", "NOPE_NOT_SET"], home=home
    )
    assert result.returncode == 1
    assert "NOPE_NOT_SET is not set or is empty" in result.stderr
