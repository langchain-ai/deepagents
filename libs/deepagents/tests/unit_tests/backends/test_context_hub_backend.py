"""Tests for ContextHubBackend with mocked langsmith.Client."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from langsmith.schemas import AgentEntry, FileEntry, SkillEntry
from langsmith.utils import LangSmithAPIError, LangSmithConflictError, LangSmithNotFoundError

from deepagents.backends import CompositeBackend, ContextHubBackend, FilesystemBackend

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT_HASH = "abcd1234" * 8  # 64-char hex
_COMMIT_URL = "https://host/context/test-agent/ef567890?organizationId=org-id"
_LEGACY_COMMIT_URL = "https://host/hub/-/test-agent:ef567890"
_HASHLESS_COMMIT_URL = "https://host/context/test-agent?organizationId=org-id"


def _make_backend(
    **files: Any,
) -> tuple[ContextHubBackend, MagicMock]:
    """Build a ContextHubBackend with a mocked langsmith.Client.

    `files` pre-populates the `pull_agent` return as `AgentContext.files`.
    Entry values can be `FileEntry`/`AgentEntry`/`SkillEntry` instances.
    """
    mock_client = MagicMock()
    context = SimpleNamespace(
        commit_id="00000000-0000-0000-0000-000000000000",
        commit_hash=_COMMIT_HASH,
        files=files,
    )
    mock_client.pull_agent.return_value = context
    mock_client.push_agent.return_value = _COMMIT_URL
    backend = ContextHubBackend("-/test-agent", client=mock_client)
    return backend, mock_client


def _wait_for_content(backend: ContextHubBackend, path: str, expected: str, *, timeout: float = 1.0) -> None:
    """Wait until an accepted local mutation is visible through `read`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = backend.read(path)
        if result.file_data is not None and result.file_data["content"] == expected:
            return
        time.sleep(0.001)
    msg = f"{path} never became visible with expected content"
    raise AssertionError(msg)


def _flush_pending(backend: ContextHubBackend, count: int) -> None:
    """Release a batch after exactly `count` mutations have been accepted."""
    queue = backend._mutations
    with queue.condition:
        assert queue.condition.wait_for(lambda: len(queue.pending) == count, timeout=1)
        queue.deadline = time.monotonic()
        queue.condition.notify_all()


def test_read_returns_content() -> None:
    backend, _ = _make_backend(**{"AGENTS.md": FileEntry(type="file", content="# hi\nworld")})
    result = backend.read("/AGENTS.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "# hi\nworld"


def test_read_missing_returns_not_found() -> None:
    backend, _ = _make_backend()
    result = backend.read("/missing.md")
    assert result.error == "File '/missing.md' not found"
    assert result.file_data is None


def test_read_slices_by_offset_limit() -> None:
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="1\n2\n3\n4\n5")})
    result = backend.read("/a.md", offset=1, limit=2)
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "2\n3\n"


def test_read_surfaces_pagination_metadata() -> None:
    """`ContextHubBackend.read` propagates the pagination metadata from `slice_read_response`."""
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="1\n2\n3\n4\n5")})
    result = backend.read("/a.md", offset=1, limit=2)
    assert result.total_lines == 5
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.next_offset == 3


def test_pull_runs_only_once_for_multiple_reads() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    backend.read("/a.md")
    backend.read("/a.md")
    backend.ls("/")
    assert mock_client.pull_agent.call_count == 1


def test_pull_404_treated_as_empty_repo() -> None:
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithNotFoundError("not found")
    backend = ContextHubBackend("-/new-agent", client=mock_client)

    result = backend.read("/any.md")
    assert result.error == "File '/any.md' not found"  # empty cache, not a hub error


def test_pull_non_404_failure_surfaces_as_error() -> None:
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithAPIError("hub 5xx")
    backend = ContextHubBackend("-/x", client=mock_client)

    result = backend.read("/anything")
    assert result.error is not None
    assert "Hub unavailable" in result.error
    assert "hub 5xx" in result.error


def test_pull_non_langsmith_failure_propagates() -> None:
    """Unexpected runtime errors should bubble up instead of being masked."""
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = RuntimeError("boom")
    backend = ContextHubBackend("-/x", client=mock_client)

    with pytest.raises(RuntimeError, match="boom"):
        backend.read("/anything")


def test_has_prior_commits_false_for_missing_repo() -> None:
    """A repo that 404s on pull has never been committed to."""
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithNotFoundError("not found")
    backend = ContextHubBackend("-/fresh", client=mock_client)

    assert backend.has_prior_commits() is False


def test_has_prior_commits_true_for_existing_repo() -> None:
    """An existing repo with a commit hash should report prior commits."""
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    assert backend.has_prior_commits() is True


def test_has_prior_commits_flips_after_first_write() -> None:
    """A fresh repo flips from no-prior-commits to has-prior-commits after a write."""
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithNotFoundError("not found")
    mock_client.push_agent.return_value = "https://host/context/fresh/ef567890?organizationId=org-id"
    backend = ContextHubBackend("-/fresh", client=mock_client)

    assert backend.has_prior_commits() is False
    backend.write("/seed.md", "hello")
    assert backend.has_prior_commits() is True


def test_write_commits_file() -> None:
    backend, mock_client = _make_backend()
    result = backend.write("/notes.md", "# hi")

    assert result.error is None
    assert result.path == "/notes.md"
    mock_client.push_agent.assert_called_once()
    call = mock_client.push_agent.call_args
    assert call.args[0] == "-/test-agent"
    files_arg = call.kwargs["files"]
    assert "notes.md" in files_arg
    assert files_arg["notes.md"].content == "# hi"
    assert files_arg["notes.md"].type == "file"


def test_write_sends_parent_commit_from_pull() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    backend.read("/a.md")  # prime cache to populate commit_hash
    backend.write("/b.md", "b")

    call = mock_client.push_agent.call_args
    assert call.kwargs["parent_commit"] == _COMMIT_HASH


@pytest.mark.parametrize(
    "commit_url",
    [
        pytest.param(_COMMIT_URL, id="current"),
        pytest.param(_LEGACY_COMMIT_URL, id="legacy"),
    ],
)
def test_write_updates_commit_hash_from_url(commit_url: str) -> None:
    backend, mock_client = _make_backend()
    mock_client.push_agent.side_effect = [commit_url, _COMMIT_URL]
    backend.write("/a.md", "a")
    backend.write("/b.md", "b")

    second_call = mock_client.push_agent.call_args_list[1]
    assert second_call.kwargs["parent_commit"] == "ef567890"


def test_hashless_context_url_reloads_authoritative_tree_and_next_parent() -> None:
    remote_hash = "feedface" * 8
    initial = SimpleNamespace(
        commit_id="initial",
        commit_hash=_COMMIT_HASH,
        files={"shared.md": FileEntry(type="file", content="first\nedit me\n")},
    )
    reloaded = SimpleNamespace(
        commit_id="reloaded",
        commit_hash=remote_hash,
        files={
            "batch-a.md": FileEntry(type="file", content="durable"),
            "shared.md": FileEntry(type="file", content="first\nremote line\nedit me\n"),
        },
    )
    mock_client = MagicMock()
    authoritative_pull_started = threading.Event()
    release_authoritative_pull = threading.Event()
    pull_calls = 0

    def pull_agent(_identifier: str):
        nonlocal pull_calls
        pull_calls += 1
        if pull_calls == 1:
            return initial
        authoritative_pull_started.set()
        if not release_authoritative_pull.wait(timeout=2):
            msg = "timed out waiting to release authoritative pull"
            raise TimeoutError(msg)
        return reloaded

    mock_client.pull_agent.side_effect = pull_agent
    mock_client.push_agent.side_effect = [_HASHLESS_COMMIT_URL, _COMMIT_URL]
    backend = ContextHubBackend("-/test-agent", client=mock_client)

    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_a = pool.submit(backend.write, "/batch-a.md", "durable")
        assert authoritative_pull_started.wait(timeout=1)
        try:
            batch_b = pool.submit(backend.edit, "/shared.md", "edit me", "edited")
            with backend._mutations.condition:
                assert backend._mutations.condition.wait_for(lambda: len(backend._mutations.pending) == 1, timeout=1)
        finally:
            release_authoritative_pull.set()
        assert batch_a.result(timeout=2).error is None
        edit_result = batch_b.result(timeout=2)

    assert edit_result.error is None
    assert edit_result.occurrences == 1
    assert mock_client.pull_agent.call_count == 2
    second_push = mock_client.push_agent.call_args_list[1]
    assert second_push.kwargs["parent_commit"] == remote_hash
    assert second_push.kwargs["files"]["shared.md"].content == "first\nremote line\nedited\n"
    shared = backend.read("/shared.md")
    assert shared.file_data is not None
    assert shared.file_data["content"] == "first\nremote line\nedited\n"


def test_hashless_snapshot_fails_only_pending_edit_with_stale_precondition() -> None:
    backend, mock_client = _make_backend(**{"shared.md": FileEntry(type="file", content="edit me")})
    assert backend.read("/shared.md").error is None
    remote_hash = "feedface" * 8
    authoritative = SimpleNamespace(
        commit_id="authoritative",
        commit_hash=remote_hash,
        files={
            "batch-a.md": FileEntry(type="file", content="durable"),
            "shared.md": FileEntry(type="file", content="changed remotely"),
        },
    )
    authoritative_pull_started = threading.Event()
    release_authoritative_pull = threading.Event()

    def pull_agent(_identifier: str):
        authoritative_pull_started.set()
        if not release_authoritative_pull.wait(timeout=2):
            msg = "timed out waiting to release authoritative pull"
            raise TimeoutError(msg)
        return authoritative

    mock_client.pull_agent.side_effect = pull_agent
    mock_client.push_agent.side_effect = [_HASHLESS_COMMIT_URL, _COMMIT_URL]

    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_a = pool.submit(backend.write, "/batch-a.md", "durable")
        assert authoritative_pull_started.wait(timeout=1)
        try:
            batch_b = pool.submit(backend.edit, "/shared.md", "edit me", "edited")
            with backend._mutations.condition:
                assert backend._mutations.condition.wait_for(lambda: len(backend._mutations.pending) == 1, timeout=1)
                pending = backend._mutations.pending[0]
        finally:
            release_authoritative_pull.set()
        assert batch_a.result(timeout=2).error is None
        edit_result = batch_b.result(timeout=2)

    assert "Hub unavailable" in (edit_result.error or "")
    assert isinstance(pending.error, LangSmithConflictError)
    mock_client.push_agent.assert_called_once()
    shared = backend.read("/shared.md")
    assert shared.file_data is not None
    assert shared.file_data["content"] == "changed remotely"

    assert backend.write("/next.md", "next").error is None
    second_push = mock_client.push_agent.call_args_list[1]
    assert second_push.kwargs["parent_commit"] == remote_hash


def test_hashless_commit_without_reloaded_hash_fails() -> None:
    initial = SimpleNamespace(commit_id="initial", commit_hash=_COMMIT_HASH, files={})
    unresolved = SimpleNamespace(
        commit_id="unresolved",
        commit_hash=None,
        files={"local.md": FileEntry(type="file", content="local")},
    )
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = [initial, unresolved]
    mock_client.push_agent.return_value = _HASHLESS_COMMIT_URL
    backend = ContextHubBackend("-/test-agent", client=mock_client)

    with pytest.raises(RuntimeError, match="commit succeeded but its hash could not be resolved"):
        backend.write("/local.md", "local")

    assert mock_client.pull_agent.call_count == 2
    assert backend._worker is None
    assert backend._cache is None
    assert backend._mutations.in_flight == []
    assert backend._mutations.pending == []


def test_worker_start_failure_cleans_mutation_state() -> None:
    backend, mock_client = _make_backend()
    failure = RuntimeError("thread start failed")
    worker = MagicMock()
    worker.start.side_effect = failure

    with (
        patch("deepagents.backends.context_hub.threading.Thread", return_value=worker),
        pytest.raises(RuntimeError, match="thread start failed"),
    ):
        backend.write("/failed.md", "failed")

    with backend._mutations.condition:
        assert backend._mutations.pending == []
        assert backend._mutations.in_flight == []
        assert backend._mutations.deadline is None
        assert backend._worker is None

    result = backend.write("/recovered.md", "recovered")
    assert result.error is None
    mock_client.push_agent.assert_called_once()


def test_write_updates_cache_after_commit() -> None:
    backend, _ = _make_backend()
    backend.write("/a.md", "hello")

    result = backend.read("/a.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "hello"


def test_write_sibling_of_linked_entry_allowed() -> None:
    backend, mock_client = _make_backend(**{"skills/code-reviewer": SkillEntry(type="skill", repo_handle="code-reviewer")})
    result = backend.write("/skills/code-reviewer.md", "sibling")
    assert result.error is None
    mock_client.push_agent.assert_called_once()


def test_commit_failure_invalidates_cache() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    mock_client.push_agent.side_effect = LangSmithAPIError("500")

    result = backend.write("/b.md", "b")
    assert result.error is not None
    assert "Hub unavailable" in result.error

    # Next read should trigger a re-pull because cache was invalidated.
    backend.read("/a.md")
    assert mock_client.pull_agent.call_count == 2


def test_edit_replaces_single_occurrence() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="hello world")})
    result = backend.edit("/a.md", "world", "earth")

    assert result.error is None
    assert result.occurrences == 1
    call = mock_client.push_agent.call_args
    assert call.kwargs["files"]["a.md"].content == "hello earth"


def test_edit_file_not_found() -> None:
    backend, _ = _make_backend()
    result = backend.edit("/missing.md", "x", "y")
    assert result.error is not None
    assert "not found" in result.error


def test_edit_ambiguous_match_without_replace_all() -> None:
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="x x x")})
    result = backend.edit("/a.md", "x", "y")
    assert result.error is not None
    assert "appears" in result.error.lower() or "3 times" in result.error


def test_edit_replace_all() -> None:
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="x x x")})
    result = backend.edit("/a.md", "x", "y", replace_all=True)
    assert result.error is None
    assert result.occurrences == 3


def test_ls_flat_repo() -> None:
    backend, _ = _make_backend(
        **{
            "AGENTS.md": FileEntry(type="file", content="a"),
            "tools.json": FileEntry(type="file", content="{}"),
        }
    )
    result = backend.ls("/")
    assert result.entries is not None
    paths = sorted(e["path"] for e in result.entries)
    assert paths == ["/AGENTS.md", "/tools.json"]


def test_ls_surfaces_virtual_directories() -> None:
    backend, _ = _make_backend(
        **{
            "AGENTS.md": FileEntry(type="file", content="a"),
            "memories/day1.md": FileEntry(type="file", content="m1"),
            "memories/day2.md": FileEntry(type="file", content="m2"),
        }
    )
    result = backend.ls("/")
    assert result.entries is not None

    paths = {e["path"]: e for e in result.entries}
    assert "/AGENTS.md" in paths
    assert paths["/AGENTS.md"]["is_dir"] is False
    assert "/memories" in paths
    assert paths["/memories"]["is_dir"] is True


def test_ls_nested_path() -> None:
    backend, _ = _make_backend(
        **{
            "memories/a.md": FileEntry(type="file", content="x"),
            "memories/b.md": FileEntry(type="file", content="y"),
        }
    )
    result = backend.ls("/memories")
    assert result.entries is not None
    paths = sorted(e["path"] for e in result.entries)
    assert paths == ["/memories/a.md", "/memories/b.md"]


def test_ls_surfaces_pull_error() -> None:
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithAPIError("5xx")
    backend = ContextHubBackend("-/x", client=mock_client)

    result = backend.ls("/")
    assert result.error is not None
    assert "Hub unavailable" in result.error


def test_grep_finds_matches() -> None:
    backend, _ = _make_backend(
        **{
            "a.md": FileEntry(type="file", content="hello\nworld\n"),
            "b.md": FileEntry(type="file", content="nothing"),
        }
    )
    result = backend.grep("hello")
    assert result.error is None
    assert result.matches is not None
    assert len(result.matches) == 1
    assert result.matches[0]["path"] == "/a.md"
    assert result.matches[0]["line"] == 1


def test_grep_with_path_prefix() -> None:
    backend, _ = _make_backend(
        **{
            "memories/a.md": FileEntry(type="file", content="hello"),
            "AGENTS.md": FileEntry(type="file", content="hello"),
        }
    )
    result = backend.grep("hello", path="/memories")
    assert result.matches is not None
    paths = {m["path"] for m in result.matches}
    assert paths == {"/memories/a.md"}


def test_grep_with_glob_filter() -> None:
    """`grep` honors the glob filter via the precompiled, hoisted matcher."""
    backend, _ = _make_backend(
        **{
            "src/main.py": FileEntry(type="file", content="import os"),
            "src/notes.md": FileEntry(type="file", content="import os"),
        }
    )
    result = backend.grep("import", glob="*.py")
    assert result.matches is not None
    paths = {m["path"] for m in result.matches}
    assert paths == {"/src/main.py"}


def test_grep_glob_matches_full_path_not_basename() -> None:
    """A directory-qualified glob matches against the full path, not the basename.

    Context Hub matches the glob against the whole file path, so `src/*.py`
    selects the nested file but excludes a same-named top-level file. A
    basename-only matcher would select neither, so this pins the intended
    full-path semantics.
    """
    backend, _ = _make_backend(
        **{
            "src/main.py": FileEntry(type="file", content="import os"),
            "main.py": FileEntry(type="file", content="import os"),
        }
    )
    result = backend.grep("import", glob="src/*.py")
    assert result.matches is not None
    paths = {m["path"] for m in result.matches}
    assert paths == {"/src/main.py"}


def test_grep_glob_does_not_brace_expand() -> None:
    """Context Hub uses `fnmatch`, which treats braces literally.

    Unlike the wcmatch-backed in-memory grep helpers (which enable brace
    expansion), `*.{py,md}` here matches a file literally named `x.{py,md}`
    rather than expanding to `*.py` or `*.md`. This locks in the intended
    asymmetry between the two grep paths.
    """
    backend, _ = _make_backend(
        **{
            "a.py": FileEntry(type="file", content="hit"),
            "b.md": FileEntry(type="file", content="hit"),
            "literal.{py,md}": FileEntry(type="file", content="hit"),
        }
    )
    result = backend.grep("hit", glob="*.{py,md}")
    assert result.matches is not None
    paths = {m["path"] for m in result.matches}
    assert paths == {"/literal.{py,md}"}


def test_grep_with_special_characters() -> None:
    backend, _ = _make_backend(**{"notes.md": FileEntry(type="file", content="items[0]\n")})
    result = backend.grep("[")
    assert result.error is None
    assert result.matches == [{"path": "/notes.md", "line": 1, "text": "items[0]"}]


def _grep_cap_backend() -> ContextHubBackend:
    """Backend with three total matches across two files for cap tests."""
    backend, _ = _make_backend(
        **{
            "a.md": FileEntry(type="file", content="hit\nhit\n"),
            "b.md": FileEntry(type="file", content="hit\n"),
        }
    )
    return backend


def test_grep_max_count_over_cap_truncates() -> None:
    """More matches than `max_count` returns the cap flagged truncated."""
    result = _grep_cap_backend().grep("hit", max_count=2)
    assert result.matches is not None
    assert len(result.matches) == 2
    assert result.truncated is True


def test_grep_max_count_exact_cap_not_truncated() -> None:
    """Exactly `max_count` matches with none dropped is reported complete."""
    result = _grep_cap_backend().grep("hit", max_count=3)
    assert result.matches is not None
    assert len(result.matches) == 3
    assert result.truncated is False


def test_grep_max_count_none_returns_all() -> None:
    """`max_count=None` returns every match, untruncated."""
    result = _grep_cap_backend().grep("hit")
    assert result.matches is not None
    assert len(result.matches) == 3
    assert result.truncated is False


def test_glob_matches_pattern() -> None:
    backend, _ = _make_backend(
        **{
            "a.md": FileEntry(type="file", content="x"),
            "b.txt": FileEntry(type="file", content="y"),
            "c.md": FileEntry(type="file", content="z"),
        }
    )
    result = backend.glob("*.md")
    assert result.matches is not None
    paths = sorted(m["path"] for m in result.matches)
    assert paths == ["/a.md", "/c.md"]


async def test_glob_refused_pattern_returns_error() -> None:
    """Sync and inherited async glob calls report patterns refused by `wcmatch`."""
    backend, _ = _make_backend()
    pattern = "{a,b}" * 40 + "*.py"

    for result in (backend.glob(pattern), await backend.aglob(pattern)):
        assert result.error is not None
        assert "Invalid glob pattern" in result.error
        assert result.matches is None


def test_upload_text_file_succeeds() -> None:
    backend, mock_client = _make_backend()
    responses = backend.upload_files([("/note.md", b"hello")])
    assert len(responses) == 1
    assert responses[0].error is None
    mock_client.push_agent.assert_called_once()


def test_upload_binary_rejected() -> None:
    backend, _ = _make_backend()
    responses = backend.upload_files([("/x.bin", b"\x80\x81\xff")])
    assert responses[0].error == "invalid_path"


def test_upload_partial_success() -> None:
    backend, _ = _make_backend()
    responses = backend.upload_files([("/ok.md", b"hello"), ("/bad.bin", b"\x80")])
    assert len(responses) == 2
    assert responses[0].error is None
    assert responses[1].error == "invalid_path"


def test_upload_multiple_files_uses_single_commit() -> None:
    """Batch upload of N text files must produce one push_agent call, not N."""
    backend, mock_client = _make_backend()
    responses = backend.upload_files(
        [
            ("/a.md", b"alpha"),
            ("/b.md", b"beta"),
            ("/nested/c.md", b"gamma"),
        ]
    )
    assert all(r.error is None for r in responses)

    mock_client.push_agent.assert_called_once()
    call = mock_client.push_agent.call_args
    files_payload = call.kwargs["files"]
    assert set(files_payload.keys()) == {"a.md", "b.md", "nested/c.md"}
    assert files_payload["a.md"].content == "alpha"
    assert files_payload["b.md"].content == "beta"
    assert files_payload["nested/c.md"].content == "gamma"


def test_upload_partial_success_only_commits_valid_files() -> None:
    """Mixed valid/invalid input still flushes valid files in a single commit."""
    backend, mock_client = _make_backend()
    responses = backend.upload_files(
        [
            ("/ok.md", b"hello"),
            ("/bad.bin", b"\x80"),
            ("/also-ok.md", b"world"),
        ]
    )
    assert responses[0].error is None
    assert responses[1].error == "invalid_path"
    assert responses[2].error is None

    mock_client.push_agent.assert_called_once()
    files_payload = mock_client.push_agent.call_args.kwargs["files"]
    assert set(files_payload.keys()) == {"ok.md", "also-ok.md"}


def test_upload_failure_propagates_to_all_pending_paths() -> None:
    """If the batch commit fails, every otherwise-valid file gets the hub error."""
    backend, mock_client = _make_backend()
    mock_client.push_agent.side_effect = LangSmithAPIError("503")

    responses = backend.upload_files([("/a.md", b"alpha"), ("/b.md", b"beta"), ("/bad.bin", b"\x80")])
    assert "Hub unavailable" in (responses[0].error or "")
    assert "Hub unavailable" in (responses[1].error or "")
    assert responses[2].error == "invalid_path"
    # Single batch call attempted, not three.
    mock_client.push_agent.assert_called_once()


def test_upload_duplicate_path_keeps_last_write() -> None:
    """If the same path appears twice in a batch, the later content wins."""
    backend, mock_client = _make_backend()
    responses = backend.upload_files([("/dup.md", b"first"), ("/dup.md", b"second")])
    assert all(r.error is None for r in responses)

    mock_client.push_agent.assert_called_once()
    files_payload = mock_client.push_agent.call_args.kwargs["files"]
    assert files_payload["dup.md"].content == "second"


def test_single_file_write_still_uses_one_commit() -> None:
    """Refactor regression: per-file write() must still issue exactly one commit."""
    backend, mock_client = _make_backend()
    backend.write("/note.md", "hi")
    mock_client.push_agent.assert_called_once()
    files_payload = mock_client.push_agent.call_args.kwargs["files"]
    assert set(files_payload.keys()) == {"note.md"}


def test_download_existing_file() -> None:
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="hi")})
    responses = backend.download_files(["/a.md"])
    assert len(responses) == 1
    assert responses[0].content == b"hi"
    assert responses[0].error is None


def test_download_missing_file() -> None:
    backend, _ = _make_backend()
    responses = backend.download_files(["/nope.md"])
    assert responses[0].error == "file_not_found"


def test_download_propagates_pull_failure() -> None:
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = LangSmithAPIError("5xx")
    backend = ContextHubBackend("-/x", client=mock_client)

    responses = backend.download_files(["/a.md"])
    assert len(responses) == 1
    assert responses[0].error is not None
    assert "Hub unavailable" in responses[0].error


def test_get_linked_entries_returns_repo_handles() -> None:
    backend, _ = _make_backend(
        **{
            "skills/reviewer": SkillEntry(type="skill", repo_handle="reviewer"),
            "subagents/planner": AgentEntry(type="agent", repo_handle="planner"),
            "AGENTS.md": FileEntry(type="file", content="a"),
        }
    )
    entries = backend.get_linked_entries()
    assert entries == {"skills/reviewer": "reviewer", "subagents/planner": "planner"}


def test_server_expanded_linked_files_are_readable() -> None:
    """Files expanded by the server under a linked skill path should be readable."""
    backend, _ = _make_backend(
        **{
            "skills/s": SkillEntry(type="skill", repo_handle="s"),
            "skills/s/skill.md": FileEntry(type="file", content="expanded"),
        }
    )
    result = backend.read("/skills/s/skill.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "expanded"


def test_default_client_constructed_when_not_provided() -> None:
    """Verify the default Client() path is exercised (even though we don't call it)."""
    with patch("deepagents.backends.context_hub.Client") as mock_client_cls:
        ContextHubBackend("-/x")
        mock_client_cls.assert_called_once_with()


def test_composite_backend_routes_prefix_correctly(tmp_path: Path) -> None:
    """Paths through a CompositeBackend route reach the hub with the prefix stripped.

    Wires a ContextHubBackend under `/memories/` and verifies write/read/ls/
    grep/glob round-trip: writes to `/memories/x` land at hub key `x`,
    reads come back with the route prefix restored, and writes outside the
    route go to the default backend.
    """
    hub_backend, mock_client = _make_backend()
    default_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    composite = CompositeBackend(
        default=default_backend,
        routes={"/memories/": hub_backend},
    )

    # Write under the routed prefix — should reach hub as "notes.md" (stripped).
    composite.write("/memories/notes.md", "hello hub")
    call = mock_client.push_agent.call_args
    assert "notes.md" in call.kwargs["files"]
    assert call.kwargs["files"]["notes.md"].content == "hello hub"

    # Read through the composite returns the hub value.
    read = composite.read("/memories/notes.md")
    assert read.error is None
    assert read.file_data is not None
    assert read.file_data["content"] == "hello hub"

    # Write outside the route goes to the default backend, hub is untouched.
    composite.write("/fs-only.txt", "default side")
    assert mock_client.push_agent.call_count == 1
    assert (tmp_path / "fs-only.txt").read_text() == "default side"

    # ls on the route root yields paths WITH the /memories/ prefix restored.
    ls_mem = composite.ls("/memories/")
    assert ls_mem.entries is not None
    assert any(e["path"] == "/memories/notes.md" for e in ls_mem.entries)

    # grep scoped to the route finds the hub-backed file.
    grep_mem = composite.grep("hello", path="/memories")
    assert grep_mem.matches is not None
    assert any(m["path"] == "/memories/notes.md" for m in grep_mem.matches)

    # glob scoped to the route finds the hub-backed file.
    glob_mem = composite.glob("*.md", path="/memories")
    assert glob_mem.matches is not None
    assert any(m["path"] == "/memories/notes.md" for m in glob_mem.matches)


def test_delete_commits_none_entry() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="bye")})
    result = backend.delete("/a.md")

    assert result.error is None
    assert result.path == "/a.md"
    mock_client.push_agent.assert_called_once()
    call = mock_client.push_agent.call_args
    files_arg = call.kwargs["files"]
    # A None entry is the deletion marker.
    assert "a.md" in files_arg
    assert files_arg["a.md"] is None
    assert call.kwargs["parent_commit"] == _COMMIT_HASH


def test_delete_directory_commits_all_nested_entries() -> None:
    backend, mock_client = _make_backend(
        **{
            "work/a.md": FileEntry(type="file", content="a"),
            "work/sub/b.md": FileEntry(type="file", content="b"),
            "keep.md": FileEntry(type="file", content="k"),
        }
    )
    result = backend.delete("/work")

    assert result.error is None
    assert result.path == "/work"
    files_arg = mock_client.push_agent.call_args.kwargs["files"]
    # Every nested entry under /work is marked for deletion...
    assert files_arg["work/a.md"] is None
    assert files_arg["work/sub/b.md"] is None
    # ...and the sibling outside the subtree is left alone.
    assert "keep.md" not in files_arg


def test_delete_missing_returns_not_found() -> None:
    backend, mock_client = _make_backend()
    result = backend.delete("/ghost.md")

    assert result.path is None
    assert result.error is not None
    assert "not found" in result.error
    mock_client.push_agent.assert_not_called()


def test_delete_updates_cache_after_commit() -> None:
    backend, _ = _make_backend(**{"a.md": FileEntry(type="file", content="bye")})
    assert backend.read("/a.md").error is None

    backend.delete("/a.md")

    # File is gone from the cache; no re-pull needed.
    assert backend.read("/a.md").error is not None


def test_delete_failure_invalidates_cache() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    mock_client.push_agent.side_effect = LangSmithAPIError("500")

    result = backend.delete("/a.md")
    assert result.error is not None
    assert "Hub unavailable" in result.error

    backend.read("/a.md")
    assert mock_client.pull_agent.call_count == 2


async def test_adelete_commits_none_entry() -> None:
    """Async delete (protocol default `asyncio.to_thread`) behaves like sync."""
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="bye")})
    result = await backend.adelete("/a.md")

    assert result.error is None
    assert result.path == "/a.md"
    mock_client.push_agent.assert_called_once()
    files_arg = mock_client.push_agent.call_args.kwargs["files"]
    # A None entry is the deletion marker, same as the sync path.
    assert files_arg["a.md"] is None
    # Cache was updated, so a follow-up read finds the file gone without re-pulling.
    assert backend.read("/a.md").error is not None


async def test_adelete_missing_returns_not_found() -> None:
    backend, mock_client = _make_backend()
    result = await backend.adelete("/ghost.md")

    assert result.path is None
    assert result.error is not None
    assert "not found" in result.error
    mock_client.push_agent.assert_not_called()


def test_batch_window_is_anchored_to_first_mutation() -> None:
    backend, mock_client = _make_backend()
    backend.read("/missing.md")
    queue = backend._mutations

    with patch("deepagents.backends.context_hub._BATCH_WINDOW_SECONDS", 10.0):
        with queue.condition:
            first = backend._queue_changes_locked({"first.md": "first"})
            first_deadline = queue.deadline
            second = backend._queue_changes_locked({"second.md": "second"})

            assert queue.deadline == first_deadline
            queue.deadline = time.monotonic()
            queue.condition.notify_all()

        backend._wait_for_mutation(first)
        backend._wait_for_mutation(second)

    mock_client.push_agent.assert_called_once()


def test_eight_concurrent_writes_coalesce_into_one_commit() -> None:
    backend, mock_client = _make_backend()
    barrier = threading.Barrier(8)

    def write(index: int):
        barrier.wait(timeout=2)
        return backend.write(f"/file-{index}.md", f"value-{index}")

    with patch("deepagents.backends.context_hub._BATCH_WINDOW_SECONDS", 10.0), ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write, index) for index in range(8)]
        _flush_pending(backend, 8)
        results = [future.result(timeout=2) for future in futures]

    assert all(result.error is None for result in results)
    mock_client.pull_agent.assert_called_once_with("-/test-agent")
    mock_client.push_agent.assert_called_once()
    payload = mock_client.push_agent.call_args.kwargs["files"]
    assert set(payload) == {f"file-{index}.md" for index in range(8)}
    for index in range(8):
        result = backend.read(f"/file-{index}.md")
        assert result.file_data is not None
        assert result.file_data["content"] == f"value-{index}"
    assert backend._worker is None


def test_mixed_mutations_and_upload_share_one_batch() -> None:
    backend, mock_client = _make_backend(
        **{
            "edit.md": FileEntry(type="file", content="before"),
            "delete.md": FileEntry(type="file", content="remove"),
        }
    )
    backend.read("/edit.md")
    barrier = threading.Barrier(4)

    def write():
        barrier.wait(timeout=2)
        return backend.write("/write.md", "written")

    def edit():
        barrier.wait(timeout=2)
        return backend.edit("/edit.md", "before", "after")

    def delete():
        barrier.wait(timeout=2)
        return backend.delete("/delete.md")

    def upload():
        barrier.wait(timeout=2)
        return backend.upload_files([("/upload-a.md", b"a"), ("/upload-b.md", b"b")])

    with patch("deepagents.backends.context_hub._BATCH_WINDOW_SECONDS", 10.0), ThreadPoolExecutor(max_workers=4) as pool:
        write_future = pool.submit(write)
        edit_future = pool.submit(edit)
        delete_future = pool.submit(delete)
        upload_future = pool.submit(upload)
        _flush_pending(backend, 4)
        write_result = write_future.result(timeout=2)
        edit_result = edit_future.result(timeout=2)
        delete_result = delete_future.result(timeout=2)
        upload_results = upload_future.result(timeout=2)

    assert write_result.error is None
    assert edit_result.error is None
    assert delete_result.error is None
    assert all(result.error is None for result in upload_results)
    mock_client.push_agent.assert_called_once()
    payload = mock_client.push_agent.call_args.kwargs["files"]
    assert set(payload) == {"write.md", "edit.md", "delete.md", "upload-a.md", "upload-b.md"}
    assert payload["write.md"].content == "written"
    assert payload["edit.md"].content == "after"
    assert payload["delete.md"] is None
    assert payload["upload-a.md"].content == "a"
    assert payload["upload-b.md"].content == "b"


def test_follow_on_edit_uses_in_flight_write_and_waits_for_commit() -> None:
    backend, mock_client = _make_backend()
    backend.read("/missing.md")
    first_started = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    calls: list[dict[str, Any]] = []
    active = 0
    max_active = 0

    def push_agent(identifier: str, **kwargs: Any) -> str:
        nonlocal active, max_active
        with calls_lock:
            index = len(calls)
            calls.append(kwargs)
            active += 1
            max_active = max(max_active, active)
        try:
            if index == 0:
                first_started.set()
                if not release_first.wait(timeout=2):
                    msg = "timed out waiting to release first commit"
                    raise TimeoutError(msg)
            return f"https://host/hub/{identifier}:{index + 1:08x}"
        finally:
            with calls_lock:
                active -= 1

    mock_client.push_agent.side_effect = push_agent
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.write, "/chain.md", "hello")
        assert first_started.wait(timeout=1)
        second = pool.submit(backend.edit, "/chain.md", "hello", "goodbye")
        try:
            _wait_for_content(backend, "/chain.md", "goodbye")
            with calls_lock:
                assert len(calls) == 1
        finally:
            release_first.set()
        assert first.result(timeout=2).error is None
        edit_result = second.result(timeout=2)

    assert edit_result.error is None
    assert edit_result.occurrences == 1
    assert len(calls) == 2
    assert max_active == 1
    assert calls[1]["parent_commit"] == "00000001"
    assert calls[1]["files"]["chain.md"].content == "goodbye"


def test_pending_mutation_is_visible_to_cached_read_operations() -> None:
    backend, mock_client = _make_backend(**{"deleted.md": FileEntry(type="file", content="delete me")})
    backend.read("/deleted.md")
    push_started = threading.Event()
    release_push = threading.Event()

    def push_agent(_identifier: str, **_kwargs: Any) -> str:
        push_started.set()
        if not release_push.wait(timeout=2):
            msg = "timed out waiting to release commit"
            raise TimeoutError(msg)
        return _COMMIT_URL

    mock_client.push_agent.side_effect = push_agent
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(backend.write, "/pending.md", "visible pending text")
        assert push_started.wait(timeout=1)
        try:
            delete_future = pool.submit(backend.delete, "/deleted.md")
            with backend._mutations.condition:
                assert backend._mutations.condition.wait_for(lambda: len(backend._mutations.pending) == 1, timeout=1)

            read = backend.read("/pending.md")
            assert read.file_data is not None
            assert read.file_data["content"] == "visible pending text"
            assert backend.read("/deleted.md").error == "File '/deleted.md' not found"
            ls_result = backend.ls("/")
            assert ls_result.entries is not None
            assert "/pending.md" in {entry["path"] for entry in ls_result.entries}
            assert "/deleted.md" not in {entry["path"] for entry in ls_result.entries}
            grep_result = backend.grep("pending")
            assert grep_result.matches is not None
            assert [match["path"] for match in grep_result.matches] == ["/pending.md"]
            glob_result = backend.glob("*.md")
            assert glob_result.matches is not None
            assert "/pending.md" in {match["path"] for match in glob_result.matches}
            download = backend.download_files(["/pending.md"])
            assert download[0].content == b"visible pending text"
        finally:
            release_push.set()
        assert future.result(timeout=2).error is None
        assert delete_future.result(timeout=2).error is None


def test_in_flight_failure_rejects_queued_waiter_and_recovers_next_burst() -> None:
    backend, mock_client = _make_backend()
    backend.read("/missing.md")
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    def push_agent(_identifier: str, **_kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            if not release_first.wait(timeout=2):
                msg = "timed out waiting to fail first commit"
                raise TimeoutError(msg)
            msg = "503"
            raise LangSmithAPIError(msg)
        return _COMMIT_URL

    mock_client.push_agent.side_effect = push_agent
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.write, "/first.md", "first")
        assert first_started.wait(timeout=1)
        second = pool.submit(backend.write, "/second.md", "second")
        try:
            _wait_for_content(backend, "/second.md", "second", timeout=0.5)
        finally:
            release_first.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert "Hub unavailable" in (first_result.error or "")
    assert "Hub unavailable" in (second_result.error or "")
    mock_client.push_agent.assert_called_once()
    assert backend._worker is None
    assert backend.read("/second.md").error == "File '/second.md' not found"
    assert mock_client.pull_agent.call_count == 2

    recovered = backend.write("/recovered.md", "recovered")
    assert recovered.error is None
    assert mock_client.push_agent.call_count == 2
    content = backend.read("/recovered.md")
    assert content.file_data is not None
    assert content.file_data["content"] == "recovered"


def test_conflict_replays_in_flight_batch_and_rejects_invalid_pending_edit() -> None:
    initial = SimpleNamespace(
        commit_id="initial",
        commit_hash=_COMMIT_HASH,
        files={
            "base.md": FileEntry(type="file", content="before\nkeep"),
            "folder/old.md": FileEntry(type="file", content="old"),
            "pending.md": FileEntry(type="file", content="edit me"),
        },
    )
    remote_hash = "feedface" * 8
    reloaded = SimpleNamespace(
        commit_id="remote",
        commit_hash=remote_hash,
        files={
            "base.md": FileEntry(type="file", content="remote header\nbefore\nkeep"),
            "folder/old.md": FileEntry(type="file", content="old"),
            "folder/remote.md": FileEntry(type="file", content="remote"),
            "pending.md": FileEntry(type="file", content="changed remotely"),
        },
    )
    reload_started = threading.Event()
    release_reload = threading.Event()
    mock_client = MagicMock()

    def pull_agent(_identifier: str) -> SimpleNamespace:
        if mock_client.pull_agent.call_count == 1:
            return initial
        reload_started.set()
        if not release_reload.wait(timeout=2):
            msg = "timed out waiting to release conflict reload"
            raise TimeoutError(msg)
        return reloaded

    mock_client.pull_agent.side_effect = pull_agent
    mock_client.push_agent.side_effect = [LangSmithConflictError("409"), _COMMIT_URL]
    backend = ContextHubBackend("-/test-agent", client=mock_client)

    with patch("deepagents.backends.context_hub._BATCH_WINDOW_SECONDS", 10.0), ThreadPoolExecutor(max_workers=3) as pool:
        edit = pool.submit(backend.edit, "/base.md", "before", "after")
        delete = pool.submit(backend.delete, "/folder")
        _flush_pending(backend, 2)
        assert reload_started.wait(timeout=1)
        try:
            pending = pool.submit(backend.edit, "/pending.md", "edit me", "edited")
            with backend._mutations.condition:
                assert backend._mutations.condition.wait_for(lambda: len(backend._mutations.pending) == 1, timeout=1)
        finally:
            release_reload.set()
        edit_result = edit.result(timeout=2)
        delete_result = delete.result(timeout=2)
        pending_result = pending.result(timeout=2)

    assert edit_result.error is None
    assert edit_result.occurrences == 1
    assert delete_result.error is None
    assert "Hub unavailable" in (pending_result.error or "")
    assert mock_client.push_agent.call_count == 2
    first, second = mock_client.push_agent.call_args_list
    assert first.kwargs["parent_commit"] == _COMMIT_HASH
    assert second.kwargs["parent_commit"] == remote_hash
    assert set(second.kwargs["files"]) == {
        "base.md",
        "folder/old.md",
        "folder/remote.md",
    }
    assert second.kwargs["files"]["base.md"].content == "remote header\nafter\nkeep"
    assert second.kwargs["files"]["folder/old.md"] is None
    assert second.kwargs["files"]["folder/remote.md"] is None
    pending_read = backend.read("/pending.md")
    assert pending_read.file_data is not None
    assert pending_read.file_data["content"] == "changed remotely"


def test_conflict_stops_after_three_retries() -> None:
    contexts = [SimpleNamespace(commit_id=str(index), commit_hash=f"{index + 1:064x}", files={}) for index in range(4)]
    mock_client = MagicMock()
    mock_client.pull_agent.side_effect = contexts
    mock_client.push_agent.side_effect = LangSmithConflictError("409")
    backend = ContextHubBackend("-/test-agent", client=mock_client)

    result = backend.write("/local.md", "local")

    assert "Hub unavailable" in (result.error or "")
    assert mock_client.push_agent.call_count == 4
    assert mock_client.pull_agent.call_count == 4
    assert backend._worker is None


def test_later_same_path_enqueue_wins_coalesced_batch() -> None:
    backend, mock_client = _make_backend()
    backend.read("/missing.md")

    with patch("deepagents.backends.context_hub._BATCH_WINDOW_SECONDS", 10.0), ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.write, "/same.md", "first")
        _wait_for_content(backend, "/same.md", "first")
        second = pool.submit(backend.write, "/same.md", "second")
        _wait_for_content(backend, "/same.md", "second")
        _flush_pending(backend, 2)
        assert first.result(timeout=2).error is None
        assert second.result(timeout=2).error is None

    mock_client.push_agent.assert_called_once()
    payload = mock_client.push_agent.call_args.kwargs["files"]
    assert payload["same.md"].content == "second"
