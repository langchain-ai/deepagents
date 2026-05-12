"""Unit tests for topic wiki runner setup and preflight helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import topic_wiki_helpers as helpers


@pytest.fixture(autouse=True)
def clear_hub_binary_cache() -> None:
    """Reset cached hub-compatible binary state between tests."""
    helpers._HUB_COMPATIBLE_BINARIES.clear()


def test_resolve_langsmith_binary_prefers_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pick `langsmith` first when both command names are available."""

    def fake_which(cmd: str) -> str | None:
        mapping = {
            "langsmith": "/usr/local/bin/langsmith",
            "langsmith-cli": "/usr/local/bin/langsmith-cli",
        }
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    assert helpers._resolve_langsmith_binary() == "/usr/local/bin/langsmith"


def test_resolve_langsmith_binary_falls_back_to_langsmith_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use `langsmith-cli` when `langsmith` is unavailable."""

    def fake_which(cmd: str) -> str | None:
        mapping = {
            "langsmith": None,
            "langsmith-cli": "/usr/local/bin/langsmith-cli",
        }
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    assert helpers._resolve_langsmith_binary() == "/usr/local/bin/langsmith-cli"


def test_resolve_langsmith_binary_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when no supported CLI binary is found."""
    monkeypatch.setattr(helpers.shutil, "which", lambda _cmd: None)

    with pytest.raises(
        helpers.TopicWikiError, match="LangSmith CLI was not found on PATH"
    ):
        helpers._resolve_langsmith_binary()


def test_ensure_hub_command_support_accepts_supported_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark binaries as compatible when `hub --help` exits successfully."""

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="help", stderr=""
        )

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    helpers._ensure_hub_command_support("/usr/local/bin/langsmith")

    assert "/usr/local/bin/langsmith" in helpers._HUB_COMPATIBLE_BINARIES


def test_ensure_hub_command_support_raises_for_incompatible_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_ensure_mode_prerequisites_requires_api_key_for_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require `LANGSMITH_API_KEY` for sandbox-backed modes."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(helpers.TopicWikiError, match="LANGSMITH_API_KEY is required"):
        helpers._ensure_mode_prerequisites("ingest")


def test_ensure_mode_prerequisites_allows_init_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not require `LANGSMITH_API_KEY` for init mode."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    helpers._ensure_mode_prerequisites("init")


def test_run_langsmith_cli_uses_binary_name_in_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Include the resolved binary name in auth failure command output."""
    monkeypatch.setattr(
        helpers, "_resolve_langsmith_binary", lambda: "/usr/local/bin/langsmith-cli"
    )
    monkeypatch.setattr(helpers, "_ensure_hub_command_support", lambda _binary: None)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="LANGSMITH_API_KEY is missing",
        )

    monkeypatch.setattr(helpers.subprocess, "run", fake_run)

    with pytest.raises(
        helpers.TopicWikiError, match="Command: langsmith-cli hub push repo"
    ):
        helpers._run_langsmith_cli(["hub", "push", "repo"])


def test_ensure_internal_repo_source_creates_internal_repo() -> None:
    """Create repos with `source=internal` before first push."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    deps = helpers.CliDeps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: "ok",
        tempdir_factory=lambda: tempfile.TemporaryDirectory(),
    )
    helpers._ensure_internal_repo_source("-/go-programming-language-demo", deps)

    assert calls == [
        [
            "api",
            "repos",
            "-X",
            "POST",
            "-F",
            "repo_handle=go-programming-language-demo",
            "-F",
            "repo_type=agent",
            "-F",
            "is_public=false",
            "-F",
            "source=internal",
        ]
    ]


def test_ensure_internal_repo_source_ignores_conflict() -> None:
    """Treat create conflicts as success because the repo already exists."""

    def fake_run_langsmith_cli(_args: list[str]) -> subprocess.CompletedProcess[str]:
        msg = "langsmith api repos failed with exit code 1: 409 conflict"
        raise helpers.TopicWikiError(msg)

    deps = helpers.CliDeps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: "ok",
        tempdir_factory=lambda: tempfile.TemporaryDirectory(),
    )

    helpers._ensure_internal_repo_source("-/go-programming-language-demo", deps)


def test_ensure_internal_repo_source_skips_explicit_owner() -> None:
    """Skip pre-create when hub ids include explicit owners."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    deps = helpers.CliDeps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: "ok",
        tempdir_factory=lambda: tempfile.TemporaryDirectory(),
    )
    helpers._ensure_internal_repo_source("acme/go-programming-language-demo", deps)

    assert calls == []


def _make_symlink_or_skip(link_path: Path, target_path: Path) -> None:
    """Create a symlink or skip test when unsupported by the platform."""
    try:
        link_path.symlink_to(target_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is not supported in this environment: {exc}")


def test_ensure_no_symlinks_rejects_workspace_symlink(tmp_path: Path) -> None:
    """Reject workspaces that contain symlink entries."""
    workspace_dir = tmp_path / "workspace"
    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True)
    (workspace_dir / "wiki").mkdir()
    (workspace_dir / "wiki" / "index.md").write_text("# Index\n", encoding="utf-8")

    target_file = tmp_path / "outside.md"
    target_file.write_text("outside\n", encoding="utf-8")
    _make_symlink_or_skip(raw_dir / "seed.md", target_file)

    with pytest.raises(helpers.TopicWikiError, match="Symlinks are not supported"):
        helpers._ensure_no_symlinks(workspace_dir)


def test_write_if_missing_rejects_symlink_target(tmp_path: Path) -> None:
    """Refuse writes when the destination path is a symlink."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir(parents=True)
    link_path = wiki_dir / "index.md"
    outside = tmp_path / "outside.md"
    _make_symlink_or_skip(link_path, outside)

    with pytest.raises(
        helpers.TopicWikiError, match="Refusing to write to symlink path"
    ):
        helpers._write_if_missing(link_path, "# Safe\n")

    assert not outside.exists()


def test_stage_sources_avoids_symlink_destination(tmp_path: Path) -> None:
    """Avoid writing staged files to a symlinked destination path."""
    source_file = tmp_path / "note.md"
    source_file.write_text("hello\n", encoding="utf-8")
    workspace_dir = tmp_path / "workspace"
    raw_dir = workspace_dir / "raw"
    raw_dir.mkdir(parents=True)

    external_target = tmp_path / "external.md"
    external_target.write_text("keep\n", encoding="utf-8")
    _make_symlink_or_skip(raw_dir / "note.md", external_target)

    staged = helpers._stage_sources([source_file], workspace_dir)

    assert [path.name for path in staged] == ["note-2.md"]
    assert external_target.read_text(encoding="utf-8") == "keep\n"


def test_run_pull_mode_rejects_workspace_symlink_before_mode(tmp_path: Path) -> None:
    """Fail pull-mode execution when pulled workspace contains symlinks."""
    calls: list[tuple[str, ...]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ["hub", "pull"]:
            workspace_dir = Path(args[args.index("--dir") + 1])
            (workspace_dir / "raw").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "wiki").mkdir(parents=True, exist_ok=True)
            target_file = tmp_path / "outside.md"
            target_file.write_text("outside\n", encoding="utf-8")
            _make_symlink_or_skip(workspace_dir / "wiki" / "index.md", target_file)
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )
        if args[:2] == ["hub", "push"]:
            pytest.fail("push should not run when workspace contains symlinks")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    deps = helpers.CliDeps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: "ok",
        tempdir_factory=lambda: tempfile.TemporaryDirectory(),
    )
    config = helpers.RunnerConfig(
        mode="lint",
        topic="Go",
        hub_id="-/go-programming-language-demo",
        topic_dir=tmp_path / "unused",
        sources=(),
        note=None,
        question=None,
        model=None,
    )

    with pytest.raises(helpers.TopicWikiError, match="Symlinks are not supported"):
        helpers._run_pull_mode(config, deps)

    assert any(call[:2] == ("hub", "pull") for call in calls)
