"""Unit tests for wiki runner setup and ingest helpers."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import wiki_helpers as helpers


@pytest.fixture(autouse=True)
def clear_hub_binary_cache() -> None:
    """Reset cached hub-compatible binary state between tests."""
    helpers._HUB_COMPATIBLE_BINARIES.clear()


def _make_deps(
    run_langsmith_cli: Callable[[list[str]], subprocess.CompletedProcess[str]],
    run_agent_mode: Callable[..., str] | None = None,
    run_agent_review_mode: Callable[..., str] | None = None,
    ask_user: Callable[[str], str] | None = None,
) -> helpers.CliDeps:
    """Build injectable dependencies for helper tests."""

    return helpers.CliDeps(
        run_langsmith_cli=run_langsmith_cli,
        run_agent_mode=run_agent_mode or (lambda *_args: "apply"),
        run_agent_review_mode=run_agent_review_mode or (lambda *_args: "review"),
        ask_user=ask_user or (lambda _prompt: "y"),
        tempdir_factory=lambda: tempfile.TemporaryDirectory(),
    )


def test_resolve_langsmith_binary_prefers_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve to `langsmith` when available."""

    def fake_which(cmd: str) -> str | None:
        mapping = {"langsmith": "/usr/local/bin/langsmith"}
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    assert helpers._resolve_langsmith_binary() == "/usr/local/bin/langsmith"


def test_resolve_langsmith_binary_raises_when_langsmith_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise when `langsmith` is unavailable."""

    def fake_which(cmd: str) -> str | None:
        mapping = {"langsmith": None}
        return mapping.get(cmd)

    monkeypatch.setattr(helpers.shutil, "which", fake_which)

    with pytest.raises(helpers.WikiError, match="LangSmith CLI was not found on PATH"):
        helpers._resolve_langsmith_binary()


def test_resolve_langsmith_binary_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a clear error when no supported CLI binary is found."""
    monkeypatch.setattr(helpers.shutil, "which", lambda _cmd: None)

    with pytest.raises(helpers.WikiError, match="LangSmith CLI was not found on PATH"):
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

    with pytest.raises(helpers.WikiError, match="does not support `hub` commands"):
        helpers._ensure_hub_command_support("/usr/local/bin/langsmith")


def test_ensure_mode_prerequisites_requires_api_key_for_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require `LANGSMITH_API_KEY` for sandbox-backed modes."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(helpers.WikiError, match="LANGSMITH_API_KEY is required"):
        helpers._ensure_mode_prerequisites("ingest")


def test_ensure_mode_prerequisites_allows_init_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not require `LANGSMITH_API_KEY` for init mode."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    helpers._ensure_mode_prerequisites("init")


def test_parse_config_accepts_repo_owner_inputs() -> None:
    """Parse direct repo/owner inputs into normalized fields."""
    config = helpers.parse_config(
        [
            "--mode",
            "init",
            "--topic",
            "Ada",
            "--repo",
            "acme/ada-wiki",
            "--owner",
            "acme",
        ]
    )

    assert config.repo == "ada-wiki"
    assert config.owner == "acme"


def test_parse_config_defaults_topic_from_repo() -> None:
    """Default topic name from repo when --topic is omitted."""
    config = helpers.parse_config(
        [
            "--mode",
            "init",
            "--repo",
            "ada-lovelace-wiki",
        ]
    )

    assert config.topic == "Ada Lovelace Wiki"
    assert config.review is False


def test_parse_config_sets_review_when_requested() -> None:
    """Enable review mode when --review is passed."""
    config = helpers.parse_config(
        [
            "--mode",
            "ingest",
            "--repo",
            "ada-lovelace-wiki",
            "--source",
            __file__,
            "--review",
        ]
    )

    assert config.review is True


def test_parse_config_rejects_owner_repo_conflict() -> None:
    """Reject conflicting owner information across flags."""
    with pytest.raises(SystemExit):
        helpers.parse_config(
            [
                "--mode",
                "init",
                "--topic",
                "Ada",
                "--repo",
                "acme/ada-wiki",
                "--owner",
                "other",
            ]
        )


def test_parse_config_requires_source_for_ingest() -> None:
    """Require at least one source path in ingest mode."""
    with pytest.raises(SystemExit):
        helpers.parse_config(
            [
                "--mode",
                "ingest",
                "--topic",
                "Ada",
                "--repo",
                "ada-wiki",
            ]
        )


def test_parse_config_requires_question_for_query() -> None:
    """Require a question in query mode."""
    with pytest.raises(SystemExit):
        helpers.parse_config(
            [
                "--mode",
                "query",
                "--topic",
                "Ada",
                "--repo",
                "ada-wiki",
            ]
        )


def test_resolve_internal_source_flag_prefers_repo_source() -> None:
    """Prefer `--repo-source internal` when hub help exposes it."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args == ["hub", "init", "--help"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="--repo-source [internal|external]",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    assert helpers._resolve_internal_source_flag(deps) == (
        "--repo-source",
        "internal",
    )


def test_resolve_internal_source_flag_raises_when_missing() -> None:
    """Fail when the CLI has no internal-source init option."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args == ["hub", "init", "--help"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="--name TEXT",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    with pytest.raises(helpers.WikiError, match="cannot guarantee `source=internal`"):
        helpers._resolve_internal_source_flag(deps)


def test_verify_internal_repo_source_passes() -> None:
    """Accept repos with source explicitly marked internal."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args[:2] == ["hub", "get"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"source": "internal"}',
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    helpers._verify_internal_repo_source("-/ada", deps)


def test_verify_internal_repo_source_fails_when_not_internal() -> None:
    """Reject repos with non-internal source value."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args[:2] == ["hub", "get"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"source": "external"}',
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    with pytest.raises(helpers.WikiError, match="must use `source=internal`"):
        helpers._verify_internal_repo_source("-/ada", deps)


def test_run_init_uses_hub_only_flow(tmp_path: Path) -> None:
    """Run init via hub commands only, then verify internal source."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args == ["hub", "init", "--help"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="--repo-source [internal|external]",
                stderr="",
            )
        if args[:2] == ["hub", "get"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"source": "internal"}',
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)
    config = helpers.RunnerConfig(
        mode="init",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "workspace",
        sources=(),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    result = helpers._run_init(config, deps)

    assert result.hub_url and result.hub_url.endswith("/hub/ada-wiki")
    assert not any(call and call[0] == "api" for call in calls)
    init_calls = [call for call in calls if call[:2] == ["hub", "init"]]
    assert any("--repo-source" in call for call in init_calls)


def test_run_init_passes_description_when_supported(tmp_path: Path) -> None:
    """Pass description on init when hub help advertises description flags."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args == ["hub", "init", "--help"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="--repo-source [internal|external]\\n--description TEXT",
                stderr="",
            )
        if args[:2] == ["hub", "get"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"source": "internal"}',
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)
    config = helpers.RunnerConfig(
        mode="init",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "workspace",
        sources=(),
        note=None,
        question=None,
        model=None,
        description="Ada research wiki",
        review=False,
    )

    helpers._run_init(config, deps)

    init_calls = [call for call in calls if call[:2] == ["hub", "init"]]
    assert any("--description" in call for call in init_calls)


def test_run_init_fails_if_internal_source_cannot_be_verified(tmp_path: Path) -> None:
    """Fail init when hub metadata reports a non-internal repo source."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args == ["hub", "init", "--help"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="--repo-source [internal|external]",
                stderr="",
            )
        if args[:2] == ["hub", "get"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"source": "external"}',
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)
    config = helpers.RunnerConfig(
        mode="init",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "workspace",
        sources=(),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    with pytest.raises(helpers.WikiError, match="must use `source=internal`"):
        helpers._run_init(config, deps)


def _make_symlink_or_skip(link_path: Path, target_path: Path) -> None:
    """Create a symlink or skip test when unsupported by the platform."""
    try:
        link_path.symlink_to(target_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation is not supported in this environment: {exc}")


def test_expand_sources_supports_recursive_directories(tmp_path: Path) -> None:
    """Expand source directories recursively and deterministically."""
    src_dir = tmp_path / "raw-src"
    nested = src_dir / "nested"
    nested.mkdir(parents=True)

    file_a = src_dir / "a.md"
    file_b = nested / "b.txt"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")

    expanded = helpers._expand_sources([src_dir])

    assert expanded == [file_a.resolve(), file_b.resolve()]


def test_expand_sources_rejects_empty_directory(tmp_path: Path) -> None:
    """Reject empty source directories."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(helpers.WikiError, match="Source directory is empty"):
        helpers._expand_sources([empty])


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

    with pytest.raises(helpers.WikiError, match="Symlinks are not supported"):
        helpers._ensure_no_symlinks(workspace_dir)


def test_write_if_missing_rejects_symlink_target(tmp_path: Path) -> None:
    """Refuse writes when the destination path is a symlink."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir(parents=True)
    link_path = wiki_dir / "index.md"
    outside = tmp_path / "outside.md"
    _make_symlink_or_skip(link_path, outside)

    with pytest.raises(helpers.WikiError, match="Refusing to write to symlink path"):
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


def test_run_ingest_workspace_runs_review_then_apply(tmp_path: Path) -> None:
    """Apply ingest only after review approval."""
    source = tmp_path / "source.md"
    source.write_text("hello\n", encoding="utf-8")

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "raw").mkdir(parents=True)
    (workspace_dir / "wiki").mkdir(parents=True)
    (workspace_dir / "wiki" / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
    (workspace_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_review(*_args: object) -> str:
        calls.append("review")
        return "review summary"

    def fake_apply(*_args: object) -> str:
        calls.append("apply")
        return "apply summary"

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
        run_agent_review_mode=fake_review,
        ask_user=lambda _prompt: "y",
    )

    config = helpers.RunnerConfig(
        mode="ingest",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(source,),
        note=None,
        question=None,
        model=None,
        description=None,
        review=True,
    )

    result = helpers._run_ingest_workspace(config, workspace_dir, deps)

    assert result.should_push is True
    assert result.answer == "apply summary"
    assert calls == ["review", "apply"]


def test_run_ingest_workspace_cancelled_skips_apply(tmp_path: Path) -> None:
    """Skip apply and push when review is not approved."""
    source = tmp_path / "source.md"
    source.write_text("hello\n", encoding="utf-8")

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "raw").mkdir(parents=True)
    (workspace_dir / "wiki").mkdir(parents=True)
    (workspace_dir / "wiki" / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
    (workspace_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_review(*_args: object) -> str:
        calls.append("review")
        return "review summary"

    def fake_apply(*_args: object) -> str:
        calls.append("apply")
        return "apply summary"

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
        run_agent_review_mode=fake_review,
        ask_user=lambda _prompt: "n",
    )

    config = helpers.RunnerConfig(
        mode="ingest",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(source,),
        note=None,
        question=None,
        model=None,
        description=None,
        review=True,
    )

    result = helpers._run_ingest_workspace(config, workspace_dir, deps)

    assert result.should_push is False
    assert result.answer and "canceled" in result.answer.lower()
    assert calls == ["review"]


def test_run_ingest_workspace_default_skips_review(tmp_path: Path) -> None:
    """Apply ingest directly when review mode is not enabled."""
    source = tmp_path / "source.md"
    source.write_text("hello\n", encoding="utf-8")

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "raw").mkdir(parents=True)
    (workspace_dir / "wiki").mkdir(parents=True)
    (workspace_dir / "wiki" / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
    (workspace_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_review(*_args: object) -> str:
        calls.append("review")
        return "review summary"

    def fake_apply(*_args: object) -> str:
        calls.append("apply")
        return "apply summary"

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
        run_agent_review_mode=fake_review,
        ask_user=lambda _prompt: "n",
    )

    config = helpers.RunnerConfig(
        mode="ingest",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(source,),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    result = helpers._run_ingest_workspace(config, workspace_dir, deps)

    assert result.should_push is True
    assert result.answer == "apply summary"
    assert calls == ["apply"]


def test_run_pull_mode_skips_push_when_ingest_cancelled(tmp_path: Path) -> None:
    """Avoid hub push when ingest review is declined."""
    source = tmp_path / "source.md"
    source.write_text("hello\n", encoding="utf-8")

    calls: list[tuple[str, ...]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ["hub", "pull"]:
            workspace = Path(args[args.index("--dir") + 1])
            (workspace / "raw").mkdir(parents=True, exist_ok=True)
            (workspace / "wiki").mkdir(parents=True, exist_ok=True)
            (workspace / "wiki" / "index.md").write_text(
                "# Ada Wiki\n", encoding="utf-8"
            )
            (workspace / "log.md").write_text(
                "# Change Log\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    deps = _make_deps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: "apply",
        run_agent_review_mode=lambda *_args: "review",
        ask_user=lambda _prompt: "n",
    )

    config = helpers.RunnerConfig(
        mode="ingest",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(source,),
        note=None,
        question=None,
        model=None,
        description=None,
        review=True,
    )

    result = helpers._run_pull_mode(config, deps)

    assert result.answer and "canceled" in result.answer.lower()
    assert not any(call[:2] == ("hub", "push") for call in calls)
