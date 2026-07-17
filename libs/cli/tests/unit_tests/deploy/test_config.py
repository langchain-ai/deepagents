"""Tests for deploy CLI dotenv loading."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import deepagents_cli.config as config_module
from deepagents_cli.config import _load_dotenv

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_project_dotenv_loads_api_key_but_ignores_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "LANGSMITH_API_KEY=project-key\nLANGSMITH_ENDPOINT=https://attacker.example\n"
    )
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
    monkeypatch.setattr(config_module, "_GLOBAL_DOTENV_PATH", tmp_path / "missing")

    assert _load_dotenv(start_path=project)

    assert os.environ["LANGSMITH_API_KEY"] == "project-key"
    assert "LANGSMITH_ENDPOINT" not in os.environ


def test_project_dotenv_does_not_override_shell_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("LANGSMITH_ENDPOINT=https://attacker.example\n")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://trusted.example")
    monkeypatch.setattr(config_module, "_GLOBAL_DOTENV_PATH", tmp_path / "missing")

    _load_dotenv(start_path=project)

    assert os.environ["LANGSMITH_ENDPOINT"] == "https://trusted.example"


def test_project_dotenv_ignores_proxy_tls_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        "HTTPS_PROXY=https://proxy.example.invalid\n"
        "SSL_CERT_FILE=/tmp/attacker-ca.pem\n"
        "NO_PROXY=api.smith.langchain.com\n"
    )
    for key in ("HTTPS_PROXY", "SSL_CERT_FILE", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "_GLOBAL_DOTENV_PATH", tmp_path / "missing")

    assert _load_dotenv(start_path=project)

    assert "HTTPS_PROXY" not in os.environ
    assert "SSL_CERT_FILE" not in os.environ
    assert "NO_PROXY" not in os.environ


def test_global_dotenv_can_set_endpoint_after_project_endpoint_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("LANGSMITH_ENDPOINT=https://attacker.example\n")
    global_dotenv = tmp_path / "global.env"
    global_dotenv.write_text("LANGSMITH_ENDPOINT=https://global.example\n")
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
    monkeypatch.setattr(config_module, "_GLOBAL_DOTENV_PATH", global_dotenv)

    _load_dotenv(start_path=project)

    assert os.environ["LANGSMITH_ENDPOINT"] == "https://global.example"
