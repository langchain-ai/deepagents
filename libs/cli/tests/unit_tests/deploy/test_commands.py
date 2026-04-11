"""Tests for deploy CLI commands (scaffolding only — no subprocess calls)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from deepagents_cli.deploy.config import (
    AGENTS_MD_FILENAME,
    DEFAULT_CONFIG_FILENAME,
    MCP_FILENAME,
    SKILLS_DIRNAME,
    STARTER_SKILL_NAME,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestInitProject:
    """Test the `_init_project` scaffolding function."""

    def test_creates_expected_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        from deepagents_cli.deploy.commands import _init_project

        _init_project(name="my-agent")

        root = tmp_path / "my-agent"
        assert (root / DEFAULT_CONFIG_FILENAME).is_file()
        assert (root / AGENTS_MD_FILENAME).is_file()
        assert (root / ".env").is_file()
        assert (root / MCP_FILENAME).is_file()
        assert (root / SKILLS_DIRNAME).is_dir()
        assert (root / SKILLS_DIRNAME / STARTER_SKILL_NAME / "SKILL.md").is_file()

    def test_files_are_utf8(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        from deepagents_cli.deploy.commands import _init_project

        _init_project(name="enc-test")

        root = tmp_path / "enc-test"
        for f in root.rglob("*"):
            if f.is_file():
                f.read_text(encoding="utf-8")

    def test_refuses_existing_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "exists").mkdir()

        from deepagents_cli.deploy.commands import _init_project

        with pytest.raises(SystemExit):
            _init_project(name="exists")

    def test_force_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "proj").mkdir()

        from deepagents_cli.deploy.commands import _init_project

        _init_project(name="proj", force=True)
        assert (tmp_path / "proj" / DEFAULT_CONFIG_FILENAME).is_file()
