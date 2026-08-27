"""Tests for `.deepagentsignore` rules and backend filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import PERMISSION_DENIED, BackendProtocol, GrepResult

from deepagents_code.deepagentsignore import (
    DeepagentsIgnore,
    IgnoringBackend,
    IgnoringSandboxBackend,
)


def _ignore(root: Path, profile: Path) -> DeepagentsIgnore:
    return DeepagentsIgnore.from_project(root, project_root=root, profile_root=profile)


def test_loads_defaults_profile_and_project_in_order(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    project = tmp_path / "project"
    profile.mkdir()
    project.mkdir()
    (profile / ".deepagentsignore").write_text("*.key\n")
    (project / ".deepagentsignore").write_text("!public.key\nsecret/\n")

    ignore = _ignore(project, profile)

    assert ignore.is_ignored_relative("node_modules/pkg/index.js")
    assert ignore.is_ignored_relative("private.key")
    assert not ignore.is_ignored_relative("public.key")
    assert ignore.is_ignored_relative("secret/token.txt")


def test_matching_supports_anchoring_globstar_classes_and_case(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text(
        "/root.txt\nlogs/**/*.log\nfile[0-9].txt\nSecrets/\n"
    )
    ignore = _ignore(tmp_path, profile)

    assert ignore.is_ignored_relative("root.txt")
    assert not ignore.is_ignored_relative("nested/root.txt")
    assert ignore.is_ignored_relative("logs/app.log")
    assert ignore.is_ignored_relative("logs/old/app.log")
    assert ignore.is_ignored_relative("file7.txt")
    assert ignore.is_ignored_relative("Secrets/token.txt")
    assert not ignore.is_ignored_relative("secrets/token.txt")


def test_directory_only_rule_does_not_match_a_regular_file(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("secret/\n")
    ignore = _ignore(tmp_path, profile)

    assert not ignore.is_ignored_relative("nested/secret")
    assert ignore.is_ignored_relative("nested/secret", is_dir=True)
    assert ignore.is_ignored_relative("nested/secret/value.txt")


def test_invalid_character_class_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("[z-a]\nsecret.txt\n")

    ignore = _ignore(tmp_path, profile)

    assert not ignore.is_ignored_relative("z")
    assert ignore.is_ignored_relative("secret.txt")
    assert "Skipping invalid .deepagentsignore pattern" in caplog.text


def test_escaped_comment_and_negation_are_literals(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("\\#notes\n\\!important\n")
    ignore = _ignore(tmp_path, profile)

    assert ignore.is_ignored_relative("#notes")
    assert ignore.is_ignored_relative("!important")


def test_unreadable_ignore_file_is_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    ignore_file = tmp_path / ".deepagentsignore"
    ignore_file.write_text("secret\n")
    original = Path.read_text

    def fail_read(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == ignore_file:
            msg = "denied"
            raise PermissionError(msg)
        return original(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(PermissionError, match="denied"):
        _ignore(tmp_path, profile)


def test_backend_blocks_ignored_symlink_name_and_target(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    visible = tmp_path / "visible.txt"
    visible.write_text("visible\n")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret\n")
    (tmp_path / "secret-link.txt").symlink_to(visible)
    (tmp_path / "visible-link.txt").symlink_to(secret)
    (tmp_path / ".deepagentsignore").write_text("secret.txt\nsecret-link.txt\n")
    raw = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    backend = IgnoringBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=raw.cwd,
        virtual_mode=raw.virtual_mode,
    )

    assert backend.read(str(tmp_path / "secret-link.txt")).error is not None
    assert backend.read(str(tmp_path / "visible-link.txt")).error is not None


def test_backend_filters_all_file_operations(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("secret.txt\nhidden/\n")
    (tmp_path / "visible.txt").write_text("needle\n")
    (tmp_path / "secret.txt").write_text("needle\n")
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "value.txt").write_text("needle\n")
    raw = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = IgnoringBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=raw.cwd,
        virtual_mode=raw.virtual_mode,
    )

    assert backend.read("/secret.txt").error is not None
    assert backend.write("/secret.txt", "new").error is not None
    assert backend.edit("/secret.txt", "needle", "new").error is not None
    assert backend.delete("/hidden").error is not None
    assert {entry["path"] for entry in backend.ls("/").entries or []} == {
        "/.deepagentsignore",
        "/profile/",
        "/visible.txt",
    }
    assert [item["path"] for item in backend.glob("*.txt", "/").matches or []] == [
        "/visible.txt"
    ]
    assert [item["path"] for item in backend.grep("needle", "/").matches or []] == [
        "/visible.txt"
    ]
    assert backend.upload_files([("/secret.txt", b"x")])[0].error == PERMISSION_DENIED
    assert backend.download_files(["/secret.txt"])[0].error == PERMISSION_DENIED


def test_grep_applies_max_count_after_filtering(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("secret.txt\n")
    raw = MagicMock(spec=BackendProtocol)
    raw.grep.side_effect = [
        GrepResult(
            matches=[{"path": "/secret.txt", "line": 1, "text": "needle"}],
            truncated=True,
        ),
        GrepResult(
            matches=[
                {"path": "/secret.txt", "line": 1, "text": "needle"},
                {"path": "/visible.txt", "line": 1, "text": "needle"},
            ],
            truncated=True,
        ),
    ]
    backend = IgnoringBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=tmp_path,
        virtual_mode=True,
    )

    result = backend.grep("needle", "/", max_count=1)

    assert [match["path"] for match in result.matches or []] == ["/visible.txt"]
    assert result.truncated
    assert [call.kwargs["max_count"] for call in raw.grep.call_args_list] == [1, 2]


async def test_async_grep_overfetch_is_bounded(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("secret.txt\n")
    raw = MagicMock(spec=BackendProtocol)
    raw.agrep = AsyncMock(
        return_value=GrepResult(
            matches=[{"path": "/secret.txt", "line": 1, "text": "needle"}],
            truncated=True,
        )
    )
    backend = IgnoringBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=tmp_path,
        virtual_mode=True,
    )

    result = await backend.agrep("needle", "/", max_count=1)

    assert result.matches == []
    assert result.truncated
    counts = [call.kwargs["max_count"] for call in raw.agrep.call_args_list]
    assert counts[-1] == 10_000
    assert counts == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        10_000,
    ]


async def test_backend_async_methods_filter_and_preserve_metadata(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / ".deepagentsignore").write_text("secret.txt\n")
    (tmp_path / "visible.txt").write_text("needle\n")
    (tmp_path / "secret.txt").write_text("needle\n")
    raw = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = IgnoringBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=raw.cwd,
        virtual_mode=raw.virtual_mode,
    )

    assert (await backend.aread("/secret.txt")).error is not None
    grep = await backend.agrep("needle", "/", max_count=2)
    assert [item["path"] for item in grep.matches or []] == ["/visible.txt"]
    glob = await backend.aglob("*.txt", "/")
    assert [item["path"] for item in glob.matches or []] == ["/visible.txt"]
    assert (await backend.awrite("/secret.txt", "new")).error is not None
    assert (await backend.aedit("/secret.txt", "needle", "new")).error is not None
    assert (await backend.adelete("/secret.txt")).error is not None
    uploads = await backend.aupload_files([("/secret.txt", b"x")])
    downloads = await backend.adownload_files(["/secret.txt"])
    assert uploads[0].error == PERMISSION_DENIED
    assert downloads[0].error == PERMISSION_DENIED


def test_shell_execution_remains_an_explicit_bypass(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("shell-visible\n")
    (tmp_path / ".deepagentsignore").write_text("secret.txt\n")
    raw = LocalShellBackend(
        root_dir=tmp_path,
        virtual_mode=False,
        inherit_env=True,
    )
    backend = IgnoringSandboxBackend(
        raw,
        _ignore(tmp_path, profile),
        backend_root=raw.cwd,
        virtual_mode=raw.virtual_mode,
    )

    assert backend.read(str(secret)).error is not None
    result = backend.execute("cat secret.txt", timeout=5)
    assert result.exit_code == 0
    assert result.output.strip() == "shell-visible"
    assert backend.id == raw.id
