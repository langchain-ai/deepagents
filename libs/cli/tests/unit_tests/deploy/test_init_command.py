"""Tests for `deepagents init`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.commands import execute_init_command


def _ns(name: str | None, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(name=name, force=force)


def test_init_scaffolds_new_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    execute_init_command(_ns("my-agent"))
    project = tmp_path / "my-agent"
    assert (project / "agent.json").is_file()
    agent = json.loads((project / "agent.json").read_text())
    assert agent["name"] == "my-agent"
    assert (project / "AGENTS.md").is_file()
    assert (project / ".gitignore").is_file()
    assert ".deepagents/" in (project / ".gitignore").read_text()
    assert (project / "skills").is_dir()


def test_init_refuses_existing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        execute_init_command(_ns("x"))


def test_init_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "agent.json").write_text("{}")
    execute_init_command(_ns("x", force=True))
    agent = json.loads((tmp_path / "x" / "agent.json").read_text())
    assert agent["name"] == "x"
