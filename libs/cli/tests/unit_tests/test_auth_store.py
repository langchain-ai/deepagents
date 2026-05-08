"""Tests for the user-level provider credential store."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from deepagents_cli import auth_store


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` and `DEFAULT_STATE_DIR` into a temp directory."""
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_cli.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


def _auth_file(home: Path) -> Path:
    return home / ".deepagents" / ".state" / "auth.json"


@pytest.mark.usefixtures("fake_home")
class TestRoundTrip:
    """Persist/read happy paths."""

    def test_missing_file_returns_empty(self) -> None:
        """No file on disk yields an empty mapping and no key."""
        assert auth_store.load_credentials() == {}
        assert auth_store.get_stored_key("anthropic") is None
        assert auth_store.list_configured_providers() == []

    def test_set_then_get(self) -> None:
        """A set key round-trips through disk."""
        auth_store.set_stored_key("anthropic", "sk-ant-secret")
        assert auth_store.get_stored_key("anthropic") == "sk-ant-secret"

    def test_set_strips_whitespace(self) -> None:
        """Surrounding whitespace is stripped before persistence."""
        auth_store.set_stored_key("openai", "  sk-openai  \n")
        assert auth_store.get_stored_key("openai") == "sk-openai"

    def test_multiple_providers_isolated(self) -> None:
        """Distinct providers don't overwrite each other."""
        auth_store.set_stored_key("anthropic", "k1")
        auth_store.set_stored_key("openai", "k2")
        assert auth_store.get_stored_key("anthropic") == "k1"
        assert auth_store.get_stored_key("openai") == "k2"
        assert auth_store.list_configured_providers() == ["anthropic", "openai"]

    def test_overwrites_existing(self) -> None:
        """Setting an existing provider replaces the prior value."""
        auth_store.set_stored_key("anthropic", "old")
        auth_store.set_stored_key("anthropic", "new")
        assert auth_store.get_stored_key("anthropic") == "new"

    def test_delete_returns_true_when_removed(self) -> None:
        """Deleting an existing entry returns `True` and clears the value."""
        auth_store.set_stored_key("openai", "k")
        assert auth_store.delete_stored_key("openai") is True
        assert auth_store.get_stored_key("openai") is None

    def test_delete_missing_returns_false(self) -> None:
        """Deleting an unknown provider is a no-op."""
        assert auth_store.delete_stored_key("anthropic") is False


@pytest.mark.usefixtures("fake_home")
class TestValidation:
    """Input validation rejects shapes we never want on disk."""

    def test_empty_key_rejected(self) -> None:
        """Empty/whitespace keys are rejected so they don't mask env vars."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            auth_store.set_stored_key("anthropic", "   ")

    def test_empty_provider_rejected(self) -> None:
        """Empty provider names are rejected."""
        with pytest.raises(ValueError, match="Provider name cannot be empty"):
            auth_store.set_stored_key("", "k")


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX mode bits don't apply on Windows"
)
class TestPermissions:
    """The credential file and its parent must be private to the user."""

    def test_file_mode_is_0600(self, fake_home: Path) -> None:
        """Stored credential file is readable/writable only by the owner."""
        auth_store.set_stored_key("anthropic", "k")
        path = _auth_file(fake_home)
        assert path.stat().st_mode & 0o777 == 0o600

    def test_parent_dir_mode_is_0700(self, fake_home: Path) -> None:
        """Parent state directory is locked to the owner."""
        auth_store.set_stored_key("anthropic", "k")
        path = _auth_file(fake_home)
        assert path.parent.stat().st_mode & 0o777 == 0o700

    def test_chmod_failure_returned_as_warning(
        self,
        fake_home: Path,  # noqa: ARG002 - fixture activates the temp state dir
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chmod that can't lock down the file shows up in WriteOutcome."""
        from pathlib import Path as _Path

        original_chmod = _Path.chmod

        def deny_file_chmod(self: _Path, mode: int) -> None:
            if self.name == "auth.json":
                msg = "simulated chmod denial"
                raise OSError(msg)
            original_chmod(self, mode)

        monkeypatch.setattr(_Path, "chmod", deny_file_chmod)
        outcome = auth_store.set_stored_key("anthropic", "k")
        assert any("0600" in w for w in outcome.warnings)
        assert any("simulated chmod denial" in w for w in outcome.warnings)

    def test_clean_save_returns_no_warnings(self, fake_home: Path) -> None:  # noqa: ARG002 - fixture activates the temp state dir
        """A successful save reports an empty warnings tuple."""
        outcome = auth_store.set_stored_key("anthropic", "k")
        assert outcome.warnings == ()


@pytest.mark.usefixtures("fake_home")
class TestCorruption:
    """Bad payloads surface a deletion hint instead of crashing later."""

    def test_invalid_json_raises(self, fake_home: Path) -> None:
        """Invalid JSON raises with a remediation hint."""
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        with pytest.raises(RuntimeError, match="Delete the file"):
            auth_store.load_credentials()

    def test_unknown_version_raises(self, fake_home: Path) -> None:
        """A future schema version is rejected."""
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 999, "credentials": {}}))
        with pytest.raises(RuntimeError, match="unsupported version"):
            auth_store.load_credentials()

    def test_non_object_payload_rejected(self, fake_home: Path) -> None:
        """A non-object payload (e.g., a list) is rejected."""
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(RuntimeError, match="not a JSON object"):
            auth_store.load_credentials()

    def test_malformed_credential_entries_skipped(self, fake_home: Path) -> None:
        """Entries missing required fields are silently dropped on read."""
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        payload = {
            "version": 1,
            "credentials": {
                "good": {"type": "api_key", "key": "k", "added_at": "2026-01-01"},
                "no_key": {"type": "api_key", "added_at": "2026-01-01"},
                "unknown_type": {"type": "future_kind", "value": "x"},
                "not_a_dict": "scalar",
            },
        }
        path.write_text(json.dumps(payload))
        creds = auth_store.load_credentials()
        assert set(creds.keys()) == {"good"}

    def test_oauth_entries_silently_skipped_until_implemented(
        self, fake_home: Path
    ) -> None:
        """OAuth type stub is reserved; unimplemented producers don't show up.

        Forward-compat invariant: when OAuth lands, the producer side is
        added; until then a hand-written `oauth` entry is skipped (rather
        than crashed on or surfaced as an api_key).
        """
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        payload = {
            "version": 1,
            "credentials": {
                "anthropic": {
                    "type": "oauth",
                    "access_token": "at",
                    "refresh_token": "rt",
                    "expires_at": "2026-01-01",
                },
            },
        }
        path.write_text(json.dumps(payload))
        creds = auth_store.load_credentials()
        assert creds == {}
        assert auth_store.get_stored_key("anthropic") is None


@pytest.mark.usefixtures("fake_home")
class TestSafety:
    """Make sure secret values don't leak to logs or repr."""

    def test_logs_never_contain_key(self, caplog: pytest.LogCaptureFixture) -> None:
        """Storing/deleting a secret does not include the secret in log output."""
        secret = "sk-do-not-log-9b8a7c"
        with caplog.at_level("DEBUG", logger="deepagents_cli.auth_store"):
            auth_store.set_stored_key("anthropic", secret)
            auth_store.delete_stored_key("anthropic")
        for record in caplog.records:
            assert secret not in record.getMessage()

    def test_stale_tmp_file_is_logged_on_pre_unlink(
        self,
        fake_home: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A leftover `.tmp` from a prior crash is logged when removed."""
        path = _auth_file(fake_home)
        path.parent.mkdir(parents=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text("stale")
        with caplog.at_level("WARNING", logger="deepagents_cli.auth_store"):
            auth_store.set_stored_key("anthropic", "k")
        assert any(
            "stale credential temp file" in r.getMessage() for r in caplog.records
        )

    def test_atomic_write_keeps_old_file_on_failure(
        self,
        fake_home: Path,  # noqa: ARG002 - fixture activates the temp state dir
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the temp-write step fails, the prior file stays intact."""
        auth_store.set_stored_key("anthropic", "original")
        original_open = os.open

        def boom(
            path: str | bytes | os.PathLike,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if str(path).endswith(".tmp"):
                msg = "simulated failure"
                raise OSError(msg)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", boom)
        with pytest.raises(OSError, match="simulated failure"):
            auth_store.set_stored_key("openai", "second")
        # Restore so load_credentials can read.
        monkeypatch.setattr(os, "open", original_open)
        assert auth_store.get_stored_key("anthropic") == "original"
        assert auth_store.get_stored_key("openai") is None
