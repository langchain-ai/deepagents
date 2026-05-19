"""Unit tests for LLM wiki setup and ingest helpers."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lint as lint_helpers
import helpers
import query as query_helpers


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


def _log_headings(log_text: str) -> list[str]:
    """Return parseable timeline heading lines from log markdown."""
    return [line for line in log_text.splitlines() if line.startswith("## [")]


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
            "--repo",
            "acme/ada-wiki",
            "--owner",
            "acme",
        ]
    )

    assert config.repo == "ada-wiki"
    assert config.owner == "acme"


def test_parse_config_defaults_topic_from_repo() -> None:
    """Default topic name from repo."""
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


def test_resolve_internal_source_flag_allows_missing_flag() -> None:
    """Allow init when the CLI omits explicit internal-source flags."""

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args == ["hub", "init", "--help"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="--name TEXT",
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    assert helpers._resolve_internal_source_flag(deps) == ()


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


def test_verify_internal_repo_source_fails_on_missing_source_field() -> None:
    """Reject init flows when both hub and API metadata omit source information."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["hub", "get"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"full_name": "langchain-ai/ada"}',
                stderr="",
            )
        assert args[:2] == ["api", "/api/v1/repos/-/ada"]
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout='{"full_name": "langchain-ai/ada"}',
            stderr="",
        )

    deps = _make_deps(fake_run_langsmith_cli)

    with pytest.raises(helpers.WikiError, match="also lacked source metadata"):
        helpers._verify_internal_repo_source("-/ada", deps)

    assert calls[0][:2] == ["hub", "get"]
    assert calls[1][:2] == ["api", "/api/v1/repos/-/ada"]


def test_run_init_ensures_internal_source_via_api(tmp_path: Path) -> None:
    """Run init and verify source defaults are enforced through repos API checks."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
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
    assert any(call[:2] == ["api", "/api/v1/repos/-/ada-wiki"] for call in calls)
    init_calls = [call for call in calls if call[:2] == ["hub", "init"]]
    assert any("--repo-source" in call for call in init_calls)


def test_run_init_without_source_flag_support(tmp_path: Path) -> None:
    """Run init when hub help omits source flags and verify source via metadata."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
        if args == ["hub", "init", "--help"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="--name TEXT\n--description TEXT",
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
    init_calls = [call for call in calls if call[:2] == ["hub", "init"]]
    assert init_calls
    assert all("--repo-source" not in call for call in init_calls)
    assert all("--source" not in call for call in init_calls)
    assert all("--internal" not in call for call in init_calls)


def test_run_init_creates_internal_repo_when_missing(tmp_path: Path) -> None:
    """Create missing repos through `/api/v1/repos` with `source=internal`."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            raise helpers.WikiError("HTTP 404")
        if args[:2] == ["api", "/api/v1/repos"]:
            body = json.loads(args[5])
            assert body["repo_handle"] == "ada-wiki"
            assert body["repo_type"] == "agent"
            assert body["is_public"] is False
            assert body["source"] == "internal"
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
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
    assert any(call[:2] == ["api", "/api/v1/repos"] for call in calls)


def test_run_init_creates_agents_md_when_missing(tmp_path: Path) -> None:
    """Create the default `AGENTS.md` scaffold when it is missing."""
    workspace_dir = tmp_path / "workspace"
    assert not (workspace_dir / "AGENTS.md").exists()

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
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
        topic_dir=workspace_dir,
        sources=(),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    helpers._run_init(config, deps)

    agents_text = (workspace_dir / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_text.startswith("# Ada Wiki\n\n")
    assert "wiki schema/config for agent behavior" in agents_text


def test_run_init_preserves_existing_agents_md(tmp_path: Path) -> None:
    """Preserve existing `AGENTS.md` content on repeated init runs."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True)
    agents_path = workspace_dir / "AGENTS.md"
    original_content = "# Custom\n\nKeep this file unchanged.\n"
    agents_path.write_text(original_content, encoding="utf-8")

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
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
        topic_dir=workspace_dir,
        sources=(),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    helpers._run_init(config, deps)

    assert agents_path.read_text(encoding="utf-8") == original_content


def test_run_init_passes_description_when_supported(tmp_path: Path) -> None:
    """Pass description on init when hub help advertises description flags."""
    calls: list[list[str]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "internal"}}',
                stderr="",
            )
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
        if args[:2] == ["api", "/api/v1/repos/-/ada-wiki"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout='{"repo": {"source": "external"}}',
                stderr="",
            )
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


def test_refresh_index_builds_categorized_content_catalog(tmp_path: Path) -> None:
    """Build a content-oriented index with sections, summaries, and metadata."""
    workspace_dir = tmp_path / "workspace"
    wiki_dir = workspace_dir / "wiki"
    (wiki_dir / "entities").mkdir(parents=True)
    (wiki_dir / "concepts").mkdir(parents=True)
    (wiki_dir / "sources").mkdir(parents=True)
    (wiki_dir / "timeline").mkdir(parents=True)
    (wiki_dir / "query").mkdir(parents=True)

    (wiki_dir / "entities" / "ada-lovelace.md").write_text(
        (
            "# Ada Lovelace\n\n"
            "Pioneering analyst of the Analytical Engine.\n\n"
            "- Evidence from `/raw/letters.md`.\n"
            "- Last major revision: 1843-07-01.\n"
        ),
        encoding="utf-8",
    )
    (wiki_dir / "concepts" / "analytical-engine.md").write_text(
        (
            "# Analytical Engine\n\n"
            "A [mechanical](https://example.com/mechanical) design for general-purpose computation.\n"
        ),
        encoding="utf-8",
    )
    (wiki_dir / "sources" / "letters.md").write_text(
        (
            "# Letters Source Notes\n\n"
            "Primary-source extracts from Ada correspondence.\n\n"
            "Referenced files: `/raw/letters.md`, `/raw/notes.txt`.\n"
        ),
        encoding="utf-8",
    )
    (wiki_dir / "query" / "what-did-ada-do.md").write_text(
        "# What Did Ada Do?\n\nDurable query answer with canonical citations.\n",
        encoding="utf-8",
    )
    (wiki_dir / "timeline" / "milestones.md").write_text(
        "# Milestones\n\n1835-01-01: early collaboration notes.\n",
        encoding="utf-8",
    )
    (wiki_dir / "history.md").write_text(
        "# History\n\nChronological milestones page.\n",
        encoding="utf-8",
    )

    helpers._refresh_index("Ada", workspace_dir)
    index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")

    assert "Content catalog for wiki navigation and retrieval." in index_text
    assert "Read this page first during query workflows." in index_text
    assert "## Entities" in index_text
    assert "## Concepts" in index_text
    assert "## Sources" in index_text
    assert "## Queries" in index_text
    assert "## Timelines" in index_text
    assert "## Other Pages" in index_text
    assert "(entities/ada-lovelace.md)" in index_text
    assert "Pioneering analyst of the Analytical Engine." in index_text
    assert "date: 1843-07-01" in index_text
    assert "(sources/letters.md)" in index_text
    assert "sources: 2" in index_text
    assert "A mechanical design for general-purpose computation." in index_text
    assert "(query/what-did-ada-do.md)" in index_text
    assert "(history.md)" in index_text


def test_refresh_index_writes_empty_catalog_when_no_pages(tmp_path: Path) -> None:
    """Write the default content-catalog scaffold when no wiki pages exist."""
    workspace_dir = tmp_path / "workspace"
    wiki_dir = workspace_dir / "wiki"
    wiki_dir.mkdir(parents=True)

    helpers._refresh_index("Ada", workspace_dir)
    index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")

    assert "# Ada Wiki" in index_text
    assert "Content catalog for wiki navigation and retrieval." in index_text
    assert "## Other Pages" in index_text
    assert "_No pages yet._" in index_text


def test_append_log_entry_normalizes_and_truncates_fields(tmp_path: Path) -> None:
    """Keep structured log headers one-line and grep-friendly."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True)

    noisy_summary = " ".join(["summary"] * 100)
    helpers._append_log_entry(
        workspace_dir,
        "query.review",
        "skip",
        metadata={
            "question": "What changed\nthis week?",
            "note": "alpha   beta\tgamma",
        },
        summary=noisy_summary,
    )

    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    headings = _log_headings(log_text)

    assert headings
    assert re.match(r"^## \[\d{4}-\d{2}-\d{2}\] query\.review \| ", headings[0])
    assert "outcome=skip" in headings[0]
    assert 'question="What changed this week?"' in headings[0]
    assert "note=alpha_beta_gamma" not in headings[0]
    assert 'note="alpha beta gamma"' in headings[0]
    assert "\n" not in headings[0]
    assert "- summary:" in log_text
    assert "..." in log_text


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
    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    assert "ingest.review | outcome=completed" in log_text
    assert "ingest.apply | outcome=applied" in log_text


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

    assert result.should_push is True
    assert result.answer and "canceled" in result.answer.lower()
    assert calls == ["review"]
    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    assert "ingest.review | outcome=completed" in log_text
    assert "ingest.apply | outcome=canceled" in log_text


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
    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    assert "ingest.apply | outcome=applied" in log_text
    assert "ingest.review |" not in log_text


def test_run_pull_mode_pushes_when_ingest_cancelled(tmp_path: Path) -> None:
    """Still push when ingest review is declined so timeline logs persist."""
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
    assert any(call[:2] == ("hub", "push") for call in calls)


def test_build_lint_prompt_covers_health_checks_and_web_policy() -> None:
    """Lint prompt should include required checks and web-fallback rules."""
    prompt = lint_helpers.build_lint_prompt("Ada", "Focus on source freshness.")

    assert "single-pass lint reconciliation" in prompt
    assert "Reconcile contradictions across wiki pages" in prompt
    assert "stale claims superseded by newer evidence" in prompt
    assert "orphan pages with no inbound links" in prompt
    assert "missing cross-references" in prompt
    assert "important concept lacks a dedicated page" in prompt
    assert "Read recent `/log.md` entries first" in prompt
    assert "Use model-native web browsing/search only if available" in prompt
    assert "If web access is unavailable" in prompt
    assert "Do not edit `/log.md`" in prompt
    assert "do not create a separate lint report directory" in prompt
    assert "## Reconciled Changes" in prompt
    assert "## Remaining Gaps" in prompt
    assert "## Suggested Next Questions and Sources" in prompt


def test_run_lint_workspace_returns_summary_and_updates_index_log(tmp_path: Path) -> None:
    """Lint should run as apply-only, refresh index, and append a rich log detail."""
    workspace_dir = tmp_path / "workspace"
    wiki_dir = workspace_dir / "wiki"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "index.md").write_text("# Ada Wiki\n\n## Pages\n", encoding="utf-8")
    (wiki_dir / "history.md").write_text("# History\n", encoding="utf-8")
    (workspace_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_apply(*_args: object) -> str:
        calls.append("apply")
        return (
            "## Reconciled Changes\n- Fixed contradiction between timeline pages.\n\n"
            "## Remaining Gaps\n- Need external confirmation for publication year.\n\n"
            "## Suggested Next Questions and Sources\n- Look for primary publication records."
        )

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
    )

    config = helpers.RunnerConfig(
        mode="lint",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(),
        note="Focus on contradictions.",
        question=None,
        model=None,
        description=None,
        review=False,
    )

    summary = lint_helpers.run_lint_workspace(config, workspace_dir, deps)

    assert calls == ["apply"]
    assert "## Reconciled Changes" in summary
    assert "publication year" in summary

    index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")
    assert "(history.md)" in index_text

    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    assert "lint.apply | outcome=applied" in log_text
    assert "Fixed contradiction between timeline pages." in log_text

    assert not (wiki_dir / "lint").exists()


def test_run_pull_mode_lint_returns_summary_and_pushes(tmp_path: Path) -> None:
    """Lint mode should return summary text and still push updates."""
    calls: list[tuple[str, ...]] = []

    def fake_run_langsmith_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(args))
        if args[:2] == ["hub", "pull"]:
            workspace = Path(args[args.index("--dir") + 1])
            (workspace / "raw").mkdir(parents=True, exist_ok=True)
            wiki_dir = workspace / "wiki"
            wiki_dir.mkdir(parents=True, exist_ok=True)
            (wiki_dir / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
            (wiki_dir / "history.md").write_text("# History\n", encoding="utf-8")
            (workspace / "log.md").write_text("# Change Log\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    deps = _make_deps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=lambda *_args: (
            "## Reconciled Changes\n- Added missing backlinks.\n\n"
            "## Remaining Gaps\n- Need stronger citation for milestone date.\n\n"
            "## Suggested Next Questions and Sources\n- Check primary archives."
        ),
    )

    config = helpers.RunnerConfig(
        mode="lint",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(),
        note=None,
        question=None,
        model=None,
        description=None,
        review=False,
    )

    result = helpers._run_pull_mode(config, deps)

    assert "## Reconciled Changes" in (result.answer or "")
    assert any(call[:2] == ("hub", "push") for call in calls)


def test_build_query_prompt_prefers_query_pages_for_discovery() -> None:
    """Guide query analysis to use prior query pages as a discovery pre-pass."""
    prompt = query_helpers.build_query_prompt("Ada", "What did Ada do?")

    assert (
        "Read `/wiki/index.md` first and use its categorized summaries/metadata to choose "
        "candidate pages." in prompt
    )
    assert "Read recent `/log.md` entries (latest ~10 `## [` headings)" in prompt
    assert (
        "Prefer checking relevant prior `/wiki/query/*.md` pages first as a discovery step."
        in prompt
    )
    assert (
        "Use those query pages to identify likely canonical `/wiki/*.md` pages and topics."
        in prompt
    )
    assert "Read the canonical wiki pages before final synthesis." in prompt


def test_build_query_prompt_sets_query_pages_as_routing_hints() -> None:
    """Require canonical evidence and uncertainty when only query pages support claims."""
    prompt = query_helpers.build_query_prompt("Ada", "What did Ada do?")

    assert "Treat `/log.md` as operational recency context" in prompt
    assert (
        "Treat `/wiki/query/*.md` pages as routing hints, not primary evidence."
        in prompt
    )
    assert "Cite canonical wiki pages for final claims whenever possible." in prompt
    assert (
        "If a claim is only supported by query pages, explicitly note uncertainty and "
        "missing canonical grounding." in prompt
    )


def test_parse_query_decision_extracts_answer_and_file_signal() -> None:
    """Parse structured query decision output with file recommendation."""
    raw = (
        "ANSWER:\nAda introduced a programmable workflow. See `wiki/overview.md`.\n\n"
        "FILING_DECISION: file\n"
        "FILING_REASON: durable synthesis"
    )

    decision = query_helpers.parse_query_decision(raw)

    assert decision.should_file is True
    assert "programmable workflow" in decision.answer
    assert decision.reason == "durable synthesis"


def test_parse_query_decision_defaults_to_skip_when_marker_missing() -> None:
    """Default to skip when decision markers are not present."""
    raw = "ANSWER:\nInsufficient evidence in the current wiki."

    decision = query_helpers.parse_query_decision(raw)

    assert decision.should_file is False
    assert decision.answer == "Insufficient evidence in the current wiki."
    assert "defaulted to skip" in decision.reason.lower()


def test_run_query_workspace_skip_keeps_query_read_only(tmp_path: Path) -> None:
    """Skip filing writes no wiki pages but still records the query interaction."""
    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "wiki").mkdir(parents=True)
    (workspace_dir / "wiki" / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
    log_path = workspace_dir / "log.md"
    log_path.write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_review(*_args: object) -> str:
        calls.append("review")
        return (
            "ANSWER:\nThis answer is ad-hoc and not reusable.\n\n"
            "FILING_DECISION: skip\n"
            "FILING_REASON: low reuse value"
        )

    def fake_apply(*_args: object) -> str:
        calls.append("apply")
        return "should not run"

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
        run_agent_review_mode=fake_review,
    )

    config = helpers.RunnerConfig(
        mode="query",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(),
        note=None,
        question="What changed this week?",
        model=None,
        description=None,
        review=False,
    )

    result = query_helpers.run_query_workspace(config, workspace_dir, deps)

    assert result.should_push is True
    assert result.filed_path is None
    assert "ad-hoc" in result.answer
    assert calls == ["review"]
    log_text = log_path.read_text(encoding="utf-8")
    assert "query.review | outcome=skip" in log_text
    assert "query.apply |" not in log_text


def test_run_query_workspace_files_durable_answer(tmp_path: Path) -> None:
    """File durable query answers into `wiki/query/<slug>.md` and log the run."""
    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "wiki").mkdir(parents=True)
    (workspace_dir / "wiki" / "index.md").write_text("# Ada Wiki\n", encoding="utf-8")
    (workspace_dir / "log.md").write_text("# Change Log\n", encoding="utf-8")

    calls: list[str] = []

    def fake_review(*_args: object) -> str:
        calls.append("review")
        return (
            "ANSWER:\nAda introduced analytical workflow patterns. See `wiki/history.md`.\n\n"
            "FILING_DECISION: file\n"
            "FILING_REASON: reusable synthesis"
        )

    def fake_apply(
        workspace_dir_arg: Path, _topic: str, prompt: str, _model: str | None
    ) -> str:
        calls.append("apply")
        assert "/wiki/query/what-did-ada-do.md" in prompt
        target = workspace_dir_arg / "wiki" / "query" / "what-did-ada-do.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# What Did Ada Do\n\n## Answer\n\nDurable synthesis.\n",
            encoding="utf-8",
        )
        return "filed"

    deps = _make_deps(
        run_langsmith_cli=lambda args: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
        run_agent_mode=fake_apply,
        run_agent_review_mode=fake_review,
    )

    config = helpers.RunnerConfig(
        mode="query",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(),
        note=None,
        question="What did Ada do?",
        model=None,
        description=None,
        review=False,
    )

    result = query_helpers.run_query_workspace(config, workspace_dir, deps)

    assert result.should_push is True
    assert result.filed_path == "/wiki/query/what-did-ada-do.md"
    assert "analytical workflow" in result.answer
    assert calls == ["review", "apply"]

    index_text = (workspace_dir / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "(query/what-did-ada-do.md)" in index_text

    log_text = (workspace_dir / "log.md").read_text(encoding="utf-8")
    assert "query.review | outcome=file" in log_text
    assert "query.apply | outcome=filed" in log_text
    assert 'path=/wiki/query/what-did-ada-do.md' in log_text


def test_run_pull_mode_pushes_when_query_is_not_filed(tmp_path: Path) -> None:
    """Push even when query is skipped so timeline entries persist."""
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
            (workspace / "log.md").write_text("# Change Log\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    def fake_apply(*_args: object) -> str:
        msg = "write phase should not run for skipped query filing"
        raise AssertionError(msg)

    deps = _make_deps(
        run_langsmith_cli=fake_run_langsmith_cli,
        run_agent_mode=fake_apply,
        run_agent_review_mode=lambda *_args: (
            "ANSWER:\nAd-hoc answer.\n\n"
            "FILING_DECISION: skip\n"
            "FILING_REASON: low reuse"
        ),
    )

    config = helpers.RunnerConfig(
        mode="query",
        topic="Ada",
        repo="ada-wiki",
        owner=None,
        topic_dir=tmp_path / "unused",
        sources=(),
        note=None,
        question="What happened today?",
        model=None,
        description=None,
        review=False,
    )

    result = helpers._run_pull_mode(config, deps)

    assert "Ad-hoc answer" in (result.answer or "")
    assert any(call[:2] == ("hub", "push") for call in calls)
