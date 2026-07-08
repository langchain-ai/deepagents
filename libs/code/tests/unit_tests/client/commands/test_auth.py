"""Tests for the `dcode auth` CLI subcommands."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from deepagents_code import auth_store
from deepagents_code.client.commands.auth import (
    _known_providers,
    _resolution_label,
    run_auth_command,
)


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


@pytest.fixture
def clean_model_caches() -> Generator[None]:
    """Clear `model_config`'s module-level caches around a test.

    `ModelConfig.load()` memoizes the default-path result in a module global,
    so tests that redirect `DEFAULT_CONFIG_PATH` must clear it before reading
    and after, lest a warm or freshly-populated cache leak across tests.
    """
    from deepagents_code import model_config

    model_config.clear_caches()
    yield
    model_config.clear_caches()


def _write_corrupt_store() -> None:
    """Write an unparseable `auth.json` at the resolved store path."""
    path = auth_store.auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")


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

    def test_set_from_env_takes_precedence_over_stdin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--from-env` wins over a piped key and never consumes stdin."""
        monkeypatch.setenv("MY_KEY", "sk-from-env")
        stdin = io.StringIO("sk-from-stdin\n")
        monkeypatch.setattr(sys, "stdin", stdin)
        code = run_auth_command(
            _ns(auth_command="set", provider="openai", from_env="MY_KEY")
        )
        assert code == 0
        assert auth_store.get_stored_key("openai") == "sk-from-env"
        # stdin was not read: an unread StringIO is still at position 0.
        assert stdin.tell() == 0

    def test_set_from_env_strips_surrounding_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key copied from an env var is stored stripped of surrounding space."""
        monkeypatch.setenv("MY_KEY", "  sk-padded-key  \n")
        code = run_auth_command(
            _ns(auth_command="set", provider="openai", from_env="MY_KEY")
        )
        assert code == 0
        assert auth_store.get_stored_key("openai") == "sk-padded-key"

    def test_set_from_whitespace_only_env_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A whitespace-only env var is rejected, exercising the `strip()` arm."""
        monkeypatch.setenv("MY_KEY", "   \n")
        code = run_auth_command(
            _ns(auth_command="set", provider="groq", from_env="MY_KEY")
        )
        assert code == 1
        assert auth_store.get_stored_key("groq") is None
        assert "MY_KEY is not set or is empty" in capsys.readouterr().err

    def test_set_from_stdin_preserves_existing_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rotating a key from stdin keeps the stored endpoint."""
        auth_store.set_stored_key(
            "openai", "sk-old", base_url="https://gateway.example/v1"
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-new\n"))

        code = run_auth_command(
            _ns(auth_command="set", provider="openai", from_env=None)
        )

        assert code == 0
        assert auth_store.get_stored_key("openai") == "sk-new"
        assert auth_store.get_stored_base_url("openai") == "https://gateway.example/v1"

    def test_set_from_env_preserves_existing_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rotating a key from an env var keeps the stored endpoint."""
        auth_store.set_stored_key(
            "openai", "sk-old", base_url="https://gateway.example/v1"
        )
        monkeypatch.setenv("MY_KEY", "sk-new")

        code = run_auth_command(
            _ns(auth_command="set", provider="openai", from_env="MY_KEY")
        )

        assert code == 0
        assert auth_store.get_stored_key("openai") == "sk-new"
        assert auth_store.get_stored_base_url("openai") == "https://gateway.example/v1"

    def test_set_langsmith_with_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--project` stores a custom LangSmith project alongside the key."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("lsv2_test\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project="my-app",
            )
        )
        assert code == 0
        assert auth_store.get_stored_key("langsmith") == "lsv2_test"
        assert auth_store.get_stored_project("langsmith") == "my-app"

    def test_set_langsmith_preserves_existing_project(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rotating the key without `--project` keeps the stored project."""
        auth_store.set_stored_key("langsmith", "old", project="my-app")
        monkeypatch.setattr(sys, "stdin", io.StringIO("new\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="langsmith", from_env=None, project=None)
        )
        assert code == 0
        assert auth_store.get_stored_key("langsmith") == "new"
        assert auth_store.get_stored_project("langsmith") == "my-app"

    def test_set_langsmith_empty_project_clears_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit empty `--project` clears a previously stored project."""
        auth_store.set_stored_key("langsmith", "old", project="my-app")
        monkeypatch.setattr(sys, "stdin", io.StringIO("new\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="langsmith", from_env=None, project="")
        )
        assert code == 0
        assert auth_store.get_stored_key("langsmith") == "new"
        assert auth_store.get_stored_project("langsmith") is None

    def test_set_project_rejected_for_non_langsmith(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--project` is only valid for the langsmith service."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-ant\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="anthropic",
                from_env=None,
                project="my-app",
            )
        )
        assert code == 1
        assert auth_store.get_stored_key("anthropic") is None
        assert "--project is only valid for langsmith" in capsys.readouterr().err

    def test_set_langsmith_base_url_eu_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--base-url eu` resolves the shorthand to the canonical EU URL."""
        from deepagents_code.config import LANGSMITH_EU_ENDPOINT

        monkeypatch.setattr(sys, "stdin", io.StringIO("lsv2_test\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url="eu",
            )
        )
        assert code == 0
        assert auth_store.get_stored_base_url("langsmith") == LANGSMITH_EU_ENDPOINT

    def test_set_langsmith_base_url_full_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full custom endpoint URL is stored verbatim."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("lsv2_test\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url="https://langsmith.internal.example.com",
            )
        )
        assert code == 0
        assert (
            auth_store.get_stored_base_url("langsmith")
            == "https://langsmith.internal.example.com"
        )

    def test_set_base_url_invalid_scheme_rejected(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-http(s) `--base-url` exits non-zero and stores nothing."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("lsv2_test\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url="ftp://nope.example.com",
            )
        )
        assert code == 1
        assert auth_store.get_stored_key("langsmith") is None
        assert "--base-url must be an http(s) URL" in capsys.readouterr().err

    def test_set_base_url_malformed_ipv6_rejected_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A malformed IPv6 `--base-url` is a clean input error, not a traceback."""
        # `urlparse("http://[::1")` raises ValueError; the guard must catch it and
        # exit 1 with the standard message rather than crash the CLI.
        monkeypatch.setattr(sys, "stdin", io.StringIO("lsv2_test\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url="http://[::1",
            )
        )
        assert code == 1
        assert auth_store.get_stored_key("langsmith") is None
        assert "--base-url must be an http(s) URL" in capsys.readouterr().err

    def test_set_empty_base_url_clears_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit empty `--base-url` clears a previously stored endpoint."""
        from deepagents_code.config import LANGSMITH_EU_ENDPOINT

        auth_store.set_stored_key("langsmith", "old", base_url=LANGSMITH_EU_ENDPOINT)
        monkeypatch.setattr(sys, "stdin", io.StringIO("new\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url="",
            )
        )
        assert code == 0
        assert auth_store.get_stored_key("langsmith") == "new"
        assert auth_store.get_stored_base_url("langsmith") is None

    def test_set_base_url_none_preserves_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rotating the key without `--base-url` keeps the stored endpoint."""
        from deepagents_code.config import LANGSMITH_EU_ENDPOINT

        auth_store.set_stored_key("langsmith", "old", base_url=LANGSMITH_EU_ENDPOINT)
        monkeypatch.setattr(sys, "stdin", io.StringIO("new\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="langsmith",
                from_env=None,
                project=None,
                base_url=None,
            )
        )
        assert code == 0
        assert auth_store.get_stored_base_url("langsmith") == LANGSMITH_EU_ENDPOINT

    def test_set_non_langsmith_base_url_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid `--base-url` is stored for a non-LangSmith provider."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-openai-abc\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="openai",
                from_env=None,
                project=None,
                base_url="https://proxy.internal.example.com/v1",
            )
        )
        assert code == 0
        assert (
            auth_store.get_stored_base_url("openai")
            == "https://proxy.internal.example.com/v1"
        )

    def test_set_non_langsmith_base_url_rejects_region_alias(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Region aliases are LangSmith-only: `eu` stays a literal (invalid) URL."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-openai-abc\n"))
        code = run_auth_command(
            _ns(
                auth_command="set",
                provider="openai",
                from_env=None,
                project=None,
                base_url="eu",
            )
        )
        assert code == 1
        assert auth_store.get_stored_key("openai") is None
        assert "--base-url must be an http(s) URL" in capsys.readouterr().err

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

    def test_set_whitespace_only_stdin_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A whitespace-only piped key is stripped and rejected, not stored."""
        monkeypatch.setattr(sys, "stdin", io.StringIO("   \n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 1
        assert auth_store.get_stored_key("anthropic") is None

    def test_set_corrupt_store_errors(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A corrupt store fails loudly without echoing the key being stored."""
        _write_corrupt_store()
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-ant-secret\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "Error:" in err
        assert "sk-ant-secret" not in err

    def test_set_write_failure_is_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An `OSError` on write surfaces as a clean error, not a traceback."""

        def _raise(_data: dict) -> tuple[str, ...]:
            msg = "No space left on device"
            raise OSError(msg)

        monkeypatch.setattr(auth_store, "_write_raw", _raise)
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-ant-secret\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "Failed to write credential file" in err
        assert "disk space" in err
        assert "sk-ant-secret" not in err
        assert auth_store.get_stored_key("anthropic") is None

    def test_set_surfaces_chmod_warning(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A chmod that cannot lock the file down is surfaced on stderr."""
        original_chmod = Path.chmod

        def _deny_chmod(self: Path, mode: int) -> None:
            if self.name == "auth.json":
                msg = "simulated chmod denial"
                raise OSError(msg)
            original_chmod(self, mode)

        monkeypatch.setattr(Path, "chmod", _deny_chmod)
        monkeypatch.setattr(sys, "stdin", io.StringIO("sk-ant-secret\n"))
        code = run_auth_command(
            _ns(auth_command="set", provider="anthropic", from_env=None)
        )
        assert code == 0
        captured = capsys.readouterr()
        assert auth_store.get_stored_key("anthropic") == "sk-ant-secret"
        assert "Warning:" in captured.err
        assert "world-readable" in captured.err
        assert "sk-ant-secret" not in captured.err

    def test_set_rejects_openai_codex_without_reading_key(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`openai_codex` is OAuth-only, so CLI API-key storage is rejected."""
        from deepagents_code.model_config import CODEX_PROVIDER

        stdin = io.StringIO("sk-ignored\n")
        monkeypatch.setattr(sys, "stdin", stdin)

        code = run_auth_command(
            _ns(auth_command="set", provider=CODEX_PROVIDER, from_env=None)
        )

        assert code == 1
        assert stdin.tell() == 0
        assert auth_store.get_stored_key(CODEX_PROVIDER) is None
        err = capsys.readouterr().err
        assert "ChatGPT OAuth" in err
        assert "openai_codex" in err
        assert "sk-ignored" not in err

    def test_set_from_env_rejects_openai_codex(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--from-env` cannot create ignored API keys for `openai_codex`."""
        from deepagents_code.model_config import CODEX_PROVIDER

        monkeypatch.setenv("MY_KEY", "sk-ignored")

        code = run_auth_command(
            _ns(auth_command="set", provider=CODEX_PROVIDER, from_env="MY_KEY")
        )

        assert code == 1
        assert auth_store.get_stored_key(CODEX_PROVIDER) is None
        err = capsys.readouterr().err
        assert "ChatGPT OAuth" in err
        assert "sk-ignored" not in err


@pytest.mark.usefixtures("fake_home")
class TestRemove:
    """`auth remove` deletes a stored credential and is idempotent."""

    def test_remove_existing(self, capsys: pytest.CaptureFixture[str]) -> None:
        auth_store.set_stored_key("anthropic", "sk-ant")
        code = run_auth_command(_ns(auth_command="remove", provider="anthropic"))
        assert code == 0
        assert auth_store.get_stored_key("anthropic") is None
        assert "Removed stored credential for anthropic." in capsys.readouterr().out

    def test_remove_rm_alias(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The `rm` alias removes a stored credential like `remove`/`delete`."""
        auth_store.set_stored_key("anthropic", "sk-ant")
        code = run_auth_command(_ns(auth_command="rm", provider="anthropic"))
        assert code == 0
        assert auth_store.get_stored_key("anthropic") is None
        assert "Removed stored credential for anthropic." in capsys.readouterr().out

    def test_remove_absent_is_noop(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = run_auth_command(_ns(auth_command="delete", provider="anthropic"))
        assert code == 0
        assert "No stored credential for anthropic." in capsys.readouterr().out

    def test_remove_corrupt_store_errors(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A corrupt store makes `remove` exit non-zero with a clear error."""
        _write_corrupt_store()
        code = run_auth_command(_ns(auth_command="remove", provider="anthropic"))
        assert code == 1
        assert "Error:" in capsys.readouterr().err

    def test_remove_openai_codex_deletes_oauth_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`openai_codex` removal targets the ChatGPT OAuth token store."""
        from deepagents_code.integrations import openai_codex
        from deepagents_code.model_config import CODEX_PROVIDER

        token = tmp_path / "chatgpt-auth.json"
        token.write_text("token", encoding="utf-8")
        monkeypatch.setattr(openai_codex, "default_store_path", lambda: token)

        code = run_auth_command(_ns(auth_command="remove", provider=CODEX_PROVIDER))

        assert code == 0
        assert not token.exists()
        assert auth_store.get_stored_key(CODEX_PROVIDER) is None
        assert "Removed stored credential for openai_codex." in capsys.readouterr().out

    def test_remove_openai_codex_absent_is_noop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Missing ChatGPT OAuth tokens are treated like absent API keys."""
        from deepagents_code.integrations import openai_codex
        from deepagents_code.model_config import CODEX_PROVIDER

        token = tmp_path / "missing-chatgpt-auth.json"
        monkeypatch.setattr(openai_codex, "default_store_path", lambda: token)

        code = run_auth_command(_ns(auth_command="remove", provider=CODEX_PROVIDER))

        assert code == 0
        assert "No stored credential for openai_codex." in capsys.readouterr().out

    def test_remove_openai_codex_reports_delete_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """OAuth token deletion failures return a clean CLI error."""
        from deepagents_code.integrations import openai_codex
        from deepagents_code.model_config import CODEX_PROVIDER

        def _raise() -> bool:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(openai_codex, "logout", _raise)

        code = run_auth_command(_ns(auth_command="remove", provider=CODEX_PROVIDER))

        assert code == 1
        err = capsys.readouterr().err
        assert "failed to remove stored credential for openai_codex" in err
        assert "permission denied" in err


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
        monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "env: ANTHROPIC_API_KEY" in capsys.readouterr().out

    def test_status_service_stored(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A stored service (langsmith) resolves via the service status path."""
        auth_store.set_stored_key("langsmith", "lsv2_test")
        code = run_auth_command(_ns(auth_command="status", provider="langsmith"))
        assert code == 0
        out = capsys.readouterr().out
        assert "langsmith" in out
        assert "stored" in out

    def test_status_env_uses_prefixed_override(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-env")
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "env: DEEPAGENTS_CODE_ANTHROPIC_API_KEY" in capsys.readouterr().out

    def test_status_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "missing" in capsys.readouterr().out

    def test_status_requires_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`status` without a provider points users at `list` instead."""
        code = run_auth_command(_ns(auth_command="status", provider=None))
        assert code == 1
        err = capsys.readouterr().err
        assert "requires a provider" in err
        assert "dcode auth list" in err

    def test_status_unknown_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        """An unrecognized provider is not an error: defer auth to the SDK.

        `status` accepts an explicit provider without validating it against the
        known set, so a name that is neither stored, installed, nor declared in
        `config.toml` reports `credentials unknown` and exits `0` rather than
        failing.
        """
        code = run_auth_command(
            _ns(auth_command="status", provider="zzz-nonexistent-provider")
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "zzz-nonexistent-provider" in out
        assert "credentials unknown" in out

    def test_status_single_provider_warns_when_store_corrupt(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A corrupt store is surfaced on stderr instead of a silent `missing`.

        `get_provider_auth_status` swallows the corrupt-store error and would
        report `missing`; the explicit warning keeps the row from being read as
        authoritative.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)
        _write_corrupt_store()
        code = run_auth_command(_ns(auth_command="status", provider="anthropic"))
        assert code == 0
        assert "Warning:" in capsys.readouterr().err


@pytest.mark.usefixtures("fake_home")
class TestList:
    """`auth list` prints a row per known provider, unioning every source."""

    def test_list_shows_stored_providers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Stored providers appear, including one with no installed package.

        A stored well-known provider resolves to `stored`; a stored provider
        the SDK doesn't recognize still appears (proving the `| stored` arm of
        the union), even though its resolution label is `credentials unknown`.
        """
        auth_store.set_stored_key("anthropic", "sk-ant")
        auth_store.set_stored_key("zzz-custom-provider", "sk-custom")
        code = run_auth_command(_ns(auth_command="list"))
        assert code == 0
        out = capsys.readouterr().out
        assert "zzz-custom-provider" in out
        anthropic_row = next(
            line for line in out.splitlines() if line.startswith("anthropic")
        )
        assert "stored" in anthropic_row

    def test_list_ls_alias(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The `ls` alias renders the same listing as `list`."""
        auth_store.set_stored_key("zzz-custom-provider", "sk-custom")
        code = run_auth_command(_ns(auth_command="ls"))
        assert code == 0
        assert "zzz-custom-provider" in capsys.readouterr().out

    @pytest.mark.usefixtures("clean_model_caches")
    def test_list_includes_config_declared_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A provider declared in `config.toml` with `api_key_env` is listed.

        Exercises the `config_providers` arm of the `_known_providers` union;
        `DEFAULT_CONFIG_PATH` is not redirected by `fake_home`, so point it at a
        temp config here.
        """
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[models.providers.zzz-config-provider]\n"
            'api_key_env = "ZZZ_CONFIG_API_KEY"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_CONFIG_PATH", cfg)
        code = run_auth_command(_ns(auth_command="list"))
        assert code == 0
        assert "zzz-config-provider" in capsys.readouterr().out

    @pytest.mark.usefixtures("clean_model_caches")
    def test_list_includes_codex_when_openai_is_installed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`openai_codex` appears with `langchain-openai` despite no API key env."""
        monkeypatch.setattr(
            "deepagents_code.model_config.get_available_models",
            lambda: {"openai": ["gpt-5.3-codex"]},
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "missing.toml",
        )

        from deepagents_code.model_config import SERVICE_API_KEY_ENV

        expected = sorted({"openai", "openai_codex", *SERVICE_API_KEY_ENV})
        assert _known_providers() == (expected, None)
        code = run_auth_command(_ns(auth_command="list"))

        assert code == 0
        assert "openai_codex" in capsys.readouterr().out

    @pytest.mark.usefixtures("clean_model_caches")
    def test_list_shows_services_when_no_model_providers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Services (e.g. LangSmith tracing) are always listed, even with no models.

        Mirrors the TUI `/auth` manager, where services are configurable
        regardless of whether any model-provider package is installed.
        """
        monkeypatch.setattr("deepagents_code.model_config.get_available_models", dict)
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "missing.toml",
        )

        from deepagents_code.model_config import SERVICE_API_KEY_ENV

        assert _known_providers() == (sorted(SERVICE_API_KEY_ENV), None)
        code = run_auth_command(_ns(auth_command="list"))
        assert code == 0
        assert "langsmith" in capsys.readouterr().out

    def test_list_warns_when_store_corrupt(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A corrupt store is surfaced on stderr while listing continues."""
        _write_corrupt_store()
        code = run_auth_command(_ns(auth_command="list"))
        assert code == 0
        assert "Warning:" in capsys.readouterr().err

    def test_known_providers_returns_corruption_message(self) -> None:
        """A corrupt store is reported as the second tuple element, not raised.

        Guards the data contract that lets `list` surface the warning from a
        single store read instead of a sibling `_warn_if_store_unreadable`
        re-read.
        """
        _write_corrupt_store()
        _providers, warning = _known_providers()
        assert warning is not None


class TestResolutionLabel:
    """`_resolution_label` maps each `ProviderAuthState` to a plain-text source."""

    def test_not_required_uses_detail_then_default(self) -> None:
        from deepagents_code.model_config import ProviderAuthState, ProviderAuthStatus

        with_detail = ProviderAuthStatus(
            state=ProviderAuthState.NOT_REQUIRED,
            provider="ollama",
            detail="local endpoint",
        )
        without_detail = ProviderAuthStatus(
            state=ProviderAuthState.NOT_REQUIRED, provider="ollama"
        )
        assert _resolution_label(with_detail) == "local endpoint"
        assert _resolution_label(without_detail) == "no API key required"

    def test_implicit_default(self) -> None:
        from deepagents_code.model_config import ProviderAuthState, ProviderAuthStatus

        status = ProviderAuthStatus(
            state=ProviderAuthState.IMPLICIT, provider="google_vertexai"
        )
        assert _resolution_label(status) == "implicit auth"

    def test_managed_default(self) -> None:
        from deepagents_code.model_config import ProviderAuthState, ProviderAuthStatus

        status = ProviderAuthStatus(state=ProviderAuthState.MANAGED, provider="custom")
        assert _resolution_label(status) == "custom auth"

    def test_unknown_falls_through_to_default(self) -> None:
        from deepagents_code.model_config import ProviderAuthState, ProviderAuthStatus

        status = ProviderAuthStatus(state=ProviderAuthState.UNKNOWN, provider="mystery")
        assert _resolution_label(status) == "credentials unknown"

    def test_configured_without_env_var_falls_back(self) -> None:
        from deepagents_code.model_config import (
            ProviderAuthSource,
            ProviderAuthState,
            ProviderAuthStatus,
        )

        status = ProviderAuthStatus(
            state=ProviderAuthState.CONFIGURED,
            provider="anthropic",
            source=ProviderAuthSource.ENV,
            env_var=None,
        )
        assert _resolution_label(status) == "configured"


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
