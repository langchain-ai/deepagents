"""Unit tests for topic wiki runner setup and preflight helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import topic_wiki_helpers as helpers


@pytest.fixture(autouse=True)
def clear_hub_binary_cache() -> None:
    """Reset cached hub-compatible binary state between tests."""
    helpers._HUB_COMPATIBLE_BINARIES.clear()


def test_resolve_langsmith_binary_prefers_langsmith(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pick `langsmith` first when both command names are available."""

    def fake_which(cmd: str) -> str | None:
        mapping = {
            "langsmith": "/usr/local/bin/langsmith",
            "langsmith-cli": "/usr/local/bin/langsmith-cli",
        }
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    assert helpers._resolve_langsmith_binary() == "/usr/local/bin/langsmith"


def test_resolve_langsmith_binary_falls_back_to_langsmith_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use `langsmith-cli` when `langsmith` is unavailable."""

    def fake_which(cmd: str) -> str | None:
        mapping = {
            "langsmith": None,
            "langsmith-cli": "/usr/local/bin/langsmith-cli",
        }
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    assert helpers._resolve_langsmith_binary() == "/usr/local/bin/langsmith-cli"


def test_resolve_langsmith_binary_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a clear error when no supported CLI binary is found."""
    monkeypatch.setattr(helpers.shutil, "which", lambda _cmd: None)

    with pytest.raises(helpers.TopicWikiError, match="LangSmith CLI was not found on PATH"):
        helpers._resolve_langsmith_binary()


def test_ensure_hub_command_support_accepts_supported_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mark binaries as compatible when `hub --help` exits successfully."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="help", stderr="")

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    helpers._ensure_hub_command_support("/usr/local/bin/langsmith")

    assert "/usr/local/bin/langsmith" in helpers._HUB_COMPATIBLE_BINARIES


def test_ensure_hub_command_support_raises_for_incompatible_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise actionable guidance when the CLI lacks `hub` support."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="",
            stderr="Error: No such command 'hub'.",
        )

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    with pytest.raises(helpers.TopicWikiError, match="does not support `hub` commands"):
        helpers._ensure_hub_command_support("/usr/local/bin/langsmith-cli")


def test_ensure_mode_prerequisites_requires_api_key_for_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require `LANGSMITH_API_KEY` for sandbox-backed modes."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(helpers.TopicWikiError, match="LANGSMITH_API_KEY is required"):
        helpers._ensure_mode_prerequisites("ingest")


def test_ensure_mode_prerequisites_allows_init_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not require `LANGSMITH_API_KEY` for init mode."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    helpers._ensure_mode_prerequisites("init")


def test_run_langsmith_cli_uses_binary_name_in_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Include the resolved binary name in auth failure command output."""
    monkeypatch.setattr(helpers, "_resolve_langsmith_binary", lambda: "/usr/local/bin/langsmith-cli")
    monkeypatch.setattr(helpers, "_ensure_hub_command_support", lambda _binary: None)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="LANGSMITH_API_KEY is missing",
        )

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    with pytest.raises(helpers.TopicWikiError, match="Command: langsmith-cli hub push repo"):
        helpers._run_langsmith_cli(["hub", "push", "repo"])
