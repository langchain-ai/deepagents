"""Tests for ContextHubBackend with mocked langsmith.Client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from langsmith.schemas import AgentEntry, FileEntry, SkillEntry
from langsmith.utils import LangSmithNotFoundError

from deepagents.backends import (
    CompositeBackend,
    ContextHubBackend,
    FilesystemBackend,
)

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT_HASH = "abcd1234" * 8  # 64-char hex
_COMMIT_URL = "https://host/hub/-/test-agent:ef567890"


def _make_backend(
    **files: Any,
) -> tuple[ContextHubBackend, MagicMock]:
    """Build a ContextHubBackend with a mocked langsmith.Client.

    ``files`` pre-populates the ``pull_agent`` return as ``AgentContext.files``.
    Entry values can be ``FileEntry``/``AgentEntry``/``SkillEntry`` instances.
    """
    mock_client = MagicMock()
    context = SimpleNamespace(
        owner="-",
        repo="test-agent",
        commit_id="00000000-0000-0000-0000-000000000000",
        commit_hash=_COMMIT_HASH,
        files=files,
    )
    mock_client.pull_agent.return_value = context
    mock_client.push_agent.return_value = _COMMIT_URL
    backend = ContextHubBackend("-/test-agent", client=mock_client)
    return backend, mock_client


def test_read_returns_content() -> None:
    backend, _ = _make_backend(
        **{"AGENTS.md": FileEntry(type="file", content="# hi\nworld")}
    )
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
    assert result.file_data["content"] == "2\n3"


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
    mock_client.pull_agent.side_effect = RuntimeError("hub 5xx")
    backend = ContextHubBackend("-/x", client=mock_client)

    result = backend.read("/anything")
    assert result.error is not None
    assert "Hub unavailable" in result.error
    assert "hub 5xx" in result.error


def test_write_commits_file() -> None:
    backend, mock_client = _make_backend()
    result = backend.write("/AGENTS.md", "# hi")

    assert result.error is None
    assert result.path == "/AGENTS.md"
    mock_client.push_agent.assert_called_once()
    call = mock_client.push_agent.call_args
    assert call.args[0] == "-/test-agent"
    files_arg = call.kwargs["files"]
    assert "AGENTS.md" in files_arg
    assert files_arg["AGENTS.md"].content == "# hi"
    assert files_arg["AGENTS.md"].type == "file"


def test_write_sends_parent_commit_from_pull() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    backend.read("/a.md")  # prime cache to populate commit_hash
    backend.write("/b.md", "b")

    call = mock_client.push_agent.call_args
    assert call.kwargs["parent_commit"] == _COMMIT_HASH


def test_write_updates_commit_hash_from_url() -> None:
    backend, mock_client = _make_backend()
    backend.write("/a.md", "a")
    backend.write("/b.md", "b")

    second_call = mock_client.push_agent.call_args_list[1]
    assert second_call.kwargs["parent_commit"] == "ef567890"


def test_write_updates_cache_after_commit() -> None:
    backend, _ = _make_backend()
    backend.write("/a.md", "hello")

    result = backend.read("/a.md")
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "hello"


def test_write_under_linked_entry_rejected() -> None:
    backend, mock_client = _make_backend(
        **{"skills/code-reviewer": SkillEntry(type="skill", repo_handle="code-reviewer")}
    )
    result = backend.write("/skills/code-reviewer/prompt.md", "malicious")

    assert result.error is not None
    assert "Cannot write to a linked entry" in result.error
    mock_client.push_agent.assert_not_called()


def test_write_at_linked_root_rejected() -> None:
    backend, _ = _make_backend(
        **{"skills/code-reviewer": SkillEntry(type="skill", repo_handle="code-reviewer")}
    )
    result = backend.write("/skills/code-reviewer", "override")
    assert result.error is not None
    assert "Cannot write to a linked entry" in result.error


def test_write_sibling_of_linked_entry_allowed() -> None:
    backend, mock_client = _make_backend(
        **{"skills/code-reviewer": SkillEntry(type="skill", repo_handle="code-reviewer")}
    )
    result = backend.write("/skills/code-reviewer.md", "sibling")
    assert result.error is None
    mock_client.push_agent.assert_called_once()


def test_write_under_linked_agent_rejected() -> None:
    backend, _ = _make_backend(
        **{"subagents/reviewer": AgentEntry(type="agent", repo_handle="reviewer")}
    )
    result = backend.write("/subagents/reviewer/config.json", "x")
    assert result.error is not None
    assert "Cannot write to a linked entry" in result.error


def test_commit_failure_invalidates_cache() -> None:
    backend, mock_client = _make_backend(**{"a.md": FileEntry(type="file", content="a")})
    mock_client.push_agent.side_effect = RuntimeError("500")

    result = backend.write("/b.md", "b")
    assert result.error is not None
    assert "Hub unavailable" in result.error

    # Next read should trigger a re-pull because cache was invalidated.
    backend.read("/a.md")
    assert mock_client.pull_agent.call_count == 2


def test_edit_replaces_single_occurrence() -> None:
    backend, mock_client = _make_backend(
        **{"a.md": FileEntry(type="file", content="hello world")}
    )
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


def test_edit_under_linked_entry_rejected() -> None:
    backend, _ = _make_backend(
        **{
            "skills/s": SkillEntry(type="skill", repo_handle="s"),
            "skills/s/f.md": FileEntry(type="file", content="hello"),
        }
    )
    result = backend.edit("/skills/s/f.md", "hello", "bye")
    assert result.error is not None
    assert "Cannot write to a linked entry" in result.error


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
    mock_client.pull_agent.side_effect = RuntimeError("5xx")
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


def test_grep_invalid_regex() -> None:
    backend, _ = _make_backend()
    result = backend.grep("[unclosed")
    assert result.error is not None
    assert "Invalid regex" in result.error


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
    responses = backend.upload_files(
        [("/ok.md", b"hello"), ("/bad.bin", b"\x80")]
    )
    assert len(responses) == 2
    assert responses[0].error is None
    assert responses[1].error == "invalid_path"


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
    mock_client.pull_agent.side_effect = RuntimeError("5xx")
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
    with patch("langsmith.Client") as mock_client_cls:
        ContextHubBackend("-/x")
        mock_client_cls.assert_called_once_with()


def test_composite_backend_routes_prefix_correctly(tmp_path: Path) -> None:
    """Paths through a CompositeBackend route reach the hub with the prefix stripped.

    Wires a ContextHubBackend under ``/memories/`` and verifies write/read/ls/
    grep/glob round-trip: writes to ``/memories/x`` land at hub key ``x``,
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


