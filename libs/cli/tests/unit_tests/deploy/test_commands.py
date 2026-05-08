"""Tests for deploy CLI commands (scaffolding only — no subprocess calls)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

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

    from deepagents_cli.deploy.commands import DeployConfig


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


class TestAutoWireIssuesBoard:
    def test_skips_when_memories_not_hub(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_cli.deploy.commands import _auto_wire_issues_board_if_hub

        called = {"upsert": False}

        def _upsert(**_: object) -> None:
            called["upsert"] = True

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._upsert_issues_board_config",
            _upsert,
        )
        cfg = SimpleNamespace(
            agent=SimpleNamespace(name="my-agent"),
            memories=SimpleNamespace(backend="store", identifier=""),
        )

        _auto_wire_issues_board_if_hub(cast("DeployConfig", cfg))
        assert called["upsert"] is False

    def test_skips_when_api_key_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_cli.deploy.commands import _auto_wire_issues_board_if_hub

        called = {"lookup": False}

        def _lookup(**_: object) -> str | None:
            called["lookup"] = True
            return None

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_langsmith_api_key",
            lambda: None,
        )
        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_tracer_session_id_by_project_name",
            _lookup,
        )
        cfg = SimpleNamespace(
            agent=SimpleNamespace(name="my-agent"),
            memories=SimpleNamespace(backend="hub", identifier=""),
        )

        _auto_wire_issues_board_if_hub(cast("DeployConfig", cfg))
        assert called["lookup"] is False

    def test_hub_wires_board_with_default_identifier(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_cli.deploy.commands import _auto_wire_issues_board_if_hub

        observed: dict[str, str] = {}

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_langsmith_api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_tracer_session_id_by_project_name",
            lambda **_: "session-123",
        )

        def _upsert(**kwargs: str) -> None:
            observed.update(kwargs)

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._upsert_issues_board_config",
            _upsert,
        )
        cfg = SimpleNamespace(
            agent=SimpleNamespace(name="my-agent"),
            memories=SimpleNamespace(backend="hub", identifier=""),
        )

        _auto_wire_issues_board_if_hub(cast("DeployConfig", cfg))

        assert observed["session_id"] == "session-123"
        assert observed["api_key"] == "test-key"
        assert observed["context_hub_repo_handle"] == "my-agent"

    def test_hub_wires_board_with_explicit_identifier(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_cli.deploy.commands import _auto_wire_issues_board_if_hub

        observed: dict[str, str] = {}

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_langsmith_api_key",
            lambda: "test-key",
        )
        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._resolve_tracer_session_id_by_project_name",
            lambda **_: "session-123",
        )

        def _upsert(**kwargs: str) -> None:
            observed.update(kwargs)

        monkeypatch.setattr(
            "deepagents_cli.deploy.commands._upsert_issues_board_config",
            _upsert,
        )
        cfg = SimpleNamespace(
            agent=SimpleNamespace(name="my-agent"),
            memories=SimpleNamespace(backend="hub", identifier="acme/custom-agent"),
        )

        _auto_wire_issues_board_if_hub(cast("DeployConfig", cfg))

        assert observed["context_hub_repo_handle"] == "custom-agent"
