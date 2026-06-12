"""Tests for prebuilt LangChain/LangSmith skill installation."""

from __future__ import annotations

import io
import tarfile
import time
from typing import TYPE_CHECKING

import httpx
import pytest

from deepagents_code.skills.prebuilt import (
    PREBUILT_COLLECTIONS,
    PrebuiltSkillsError,
    install_collection,
    resolve_collections,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_tarball(
    files: dict[str, str], *, root: str = "langchain-skills-main"
) -> bytes:
    """Build an in-memory `.tar.gz` mirroring a GitHub source tarball.

    Args:
        files: Mapping of archive-relative path to file contents.
        root: Top-level directory GitHub wraps the source in.

    Returns:
        Gzipped tar bytes.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name=f"{root}/{path}")
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _patch_download(monkeypatch: pytest.MonkeyPatch, content: bytes) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_args, **_kwargs: _FakeResponse(content))


class TestResolveCollections:
    def test_default_returns_all(self) -> None:
        assert resolve_collections([]) == list(PREBUILT_COLLECTIONS.values())

    def test_all_keyword_returns_all(self) -> None:
        assert resolve_collections(["all"]) == list(PREBUILT_COLLECTIONS.values())

    def test_specific_key(self) -> None:
        resolved = resolve_collections(["langchain"])
        assert [c.key for c in resolved] == ["langchain"]

    def test_case_insensitive_and_deduped(self) -> None:
        resolved = resolve_collections(["LangChain", "langchain"])
        assert [c.key for c in resolved] == ["langchain"]

    def test_unknown_raises(self) -> None:
        with pytest.raises(PrebuiltSkillsError, match="Unknown skill collection"):
            resolve_collections(["nope"])


class TestInstallCollection:
    def test_installs_skills(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = _make_tarball(
            {
                "config/skills/skill-a/SKILL.md": "A",
                "config/skills/skill-b/SKILL.md": "B",
                "config/skills/skill-b/scripts/run.py": "print('x')",
                "README.md": "ignored",
            }
        )
        _patch_download(monkeypatch, content)

        result = install_collection(
            PREBUILT_COLLECTIONS["langchain"], tmp_path, force=False
        )

        assert result.installed == ["skill-a", "skill-b"]
        assert result.skipped == []
        assert (tmp_path / "skill-a" / "SKILL.md").read_text() == "A"
        assert (tmp_path / "skill-b" / "scripts" / "run.py").exists()

    def test_skips_existing_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "skill-a").mkdir()
        (tmp_path / "skill-a" / "SKILL.md").write_text("OLD")
        content = _make_tarball({"config/skills/skill-a/SKILL.md": "NEW"})
        _patch_download(monkeypatch, content)

        result = install_collection(
            PREBUILT_COLLECTIONS["langchain"], tmp_path, force=False
        )

        assert result.installed == []
        assert result.skipped == ["skill-a"]
        assert (tmp_path / "skill-a" / "SKILL.md").read_text() == "OLD"

    def test_force_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "skill-a").mkdir()
        (tmp_path / "skill-a" / "SKILL.md").write_text("OLD")
        content = _make_tarball({"config/skills/skill-a/SKILL.md": "NEW"})
        _patch_download(monkeypatch, content)

        result = install_collection(
            PREBUILT_COLLECTIONS["langchain"], tmp_path, force=True
        )

        assert result.installed == ["skill-a"]
        assert (tmp_path / "skill-a" / "SKILL.md").read_text() == "NEW"

    def test_no_skills_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = _make_tarball({"README.md": "nothing here"})
        _patch_download(monkeypatch, content)

        with pytest.raises(PrebuiltSkillsError, match="No skills found"):
            install_collection(PREBUILT_COLLECTIONS["langchain"], tmp_path)

    def test_unsafe_path_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        content = _make_tarball({"config/skills/../evil/SKILL.md": "bad"})
        _patch_download(monkeypatch, content)

        with pytest.raises(PrebuiltSkillsError, match="Unsafe path"):
            install_collection(PREBUILT_COLLECTIONS["langchain"], tmp_path)

    def test_download_error_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "nope"
            raise httpx.ConnectError(msg)

        monkeypatch.setattr(httpx, "get", _boom)

        with pytest.raises(PrebuiltSkillsError, match="Failed to download"):
            install_collection(PREBUILT_COLLECTIONS["langchain"], tmp_path)
