"""Tests for autocomplete fuzzy search functionality."""

import locale
import logging
import os
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from deepagents_code.command_registry import CommandEntry, get_slash_commands
from deepagents_code.tui.widgets import autocomplete as autocomplete_module
from deepagents_code.tui.widgets.autocomplete import (
    CompletionController,
    FuzzyFileController,
    MultiCompletionManager,
    SlashCommandController,
    ThreadCompletionController,
    _fuzzy_score,
    _fuzzy_search,
    _get_git_executable,
    _get_project_files,
    _run_git_ls_files,
    _scope_files_to_cwd,
)

_RANKING_CORPUS = [
    "README.md",
    "Makefile",
    "pyproject.toml",
    "setup.py",
    "src/main.py",
    "src/utils.py",
    "src/chat_models.py",
    "src/chat-model-base.py",
    "src/helpers/string_utils.py",
    "src/helpers/__init__.py",
    "src/models/chat/openai.py",
    "src/models/llms/base.py",
    "src/Runnables/RunnableConfig.py",
    "libs/core/langchain_core/messages/ai.py",
    "libs/core/langchain_core/runnables/base.py",
    "libs/core/tests/unit_tests/test_messages.py",
    "tests/test_main.py",
    "tests/test_utils.py",
    "tests/integration/test_chat.py",
    "docs/api.md",
    "docs/how_to/streaming.md",
    "docs/Ça_va.md",
    "web/src/components/ChatInput.tsx",
    "web/src/components/chat-input.css",
    "a/b/c/d/e/f/g/deeply_nested_file_name.txt",
    "x.py",
    "xx",
    "win\\Src\\Path_Utils.py",
    ".github/workflows/ci.yml",
    ".gitignore",
    ".env.example",
]
"""Fixed file list for comparing `_fuzzy_search` with `_reference_fuzzy_search`."""

_RANKING_QUERIES = [
    "",
    *"acmpstx._-/",
    "chat",
    "chatmod",
    "cht",
    "tst_cht",
    "mdoel",
    "bse",
    "runnablez",
    "readme",
    "READ",
    "Init",
    "libs/core",
    "src/m",
    "py",
    "a.b",
    "ça",
    "xyzq",
    "deepnest",
    "nested_file",
    ".git",
    "ci",
    "streamin",
    "toml",
    "x",
    "xx",
]
"""Queries that cover substring, word-boundary, fuzzy, and no-match paths."""


def _reference_score(query: str, candidate: str) -> float:
    """Score a file the plain way, with `ratio()` on every fuzzy candidate.

    Returns:
        The score; callers drop values below 15.
    """
    query_lower = query.lower()
    normalized = candidate.replace("\\", "/")
    candidate_lower = normalized.lower()
    filename = normalized.rsplit("/", 1)[-1].lower()
    filename_start = candidate_lower.rfind("/") + 1
    bonus = 1 / len(candidate)
    if query_lower in filename:
        idx = filename.find(query_lower)
        if idx == 0:
            return 150 + bonus
        return (120 if filename[idx - 1] in "_-." else 100) + bonus
    if query_lower in candidate_lower:
        idx = candidate_lower.find(query_lower)
        if idx == filename_start:
            return 80 + bonus
        boundary = idx == 0 or candidate[idx - 1] in "/_-."
        return (60 if boundary else 40) + bonus
    filename_ratio = SequenceMatcher(None, query_lower, filename).ratio()
    if filename_ratio > 0.4:  # reference threshold
        return filename_ratio * 30
    return SequenceMatcher(None, query_lower, candidate_lower).ratio() * 15


def _reference_fuzzy_search(
    query: str, candidates: list[str], *, include_dotfiles: bool
) -> list[str]:
    """Rank files with `_reference_score`.

    Returns:
        Up to 50 paths, best match first.
    """
    if not include_dotfiles:
        candidates = [
            c for c in candidates if not any(p.startswith(".") for p in c.split("/"))
        ]
    if not query:
        return sorted(candidates, key=lambda p: (p.count("/"), p.lower()))[:50]
    scored = [(_reference_score(query, c), c) for c in candidates]
    scored = [item for item in scored if item[0] >= 15]  # reference threshold
    scored.sort(key=lambda item: -item[0])
    return [c for _, c in scored[:50]]


class TestFuzzyScore:
    """Tests for the _fuzzy_score function."""


class TestFuzzySearch:
    """Tests for the _fuzzy_search function."""

    @pytest.fixture
    def sample_files(self):
        """Sample file list for testing."""
        return [
            "README.md",
            "setup.py",
            "src/main.py",
            "src/utils.py",
            "src/helpers/string_utils.py",
            "tests/test_main.py",
            "tests/test_utils.py",
            ".github/workflows/ci.yml",
            ".gitignore",
            "docs/api.md",
        ]

    @pytest.mark.parametrize("include_dotfiles", [False, True])
    def test_matches_reference_ranking(self, include_dotfiles: bool) -> None:
        """Ranked results match a plain scorer that runs `ratio()` on every file.

        `_fuzzy_search` skips `SequenceMatcher.ratio()` when a cheap upper bound
        shows a file cannot reach `_MIN_FUZZY_SCORE`. The skip must never change
        which files are returned or their order.
        """
        for query in _RANKING_QUERIES:
            expected = _reference_fuzzy_search(
                query, _RANKING_CORPUS, include_dotfiles=include_dotfiles
            )
            actual = _fuzzy_search(
                query, _RANKING_CORPUS, limit=50, include_dotfiles=include_dotfiles
            )
            assert actual == expected, query


class TestHelperFunctions:
    """Tests for helper functions."""


class TestSlashCommandController:
    """Tests for SlashCommandController."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view):
        """Create a SlashCommandController with mock view."""
        return SlashCommandController(get_slash_commands(), mock_view)

    @pytest.mark.parametrize("query", ["/res", "/resume"])
    def test_resume_ranks_threads_first(self, controller, mock_view, query):
        controller.on_text_changed(query, len(query))

        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert suggestions[0][0] == "/threads"
        assert sum(label == "/threads" for label, _ in suggestions) == 1
        assert controller._suggestion_names[0] == "/threads"

    def test_substring_description_match_fresh(self, controller, mock_view):
        """Typing 'fresh' surfaces /clear via substring on 'Start a fresh thread'."""
        controller.on_text_changed("/fresh", 6)

        mock_view.render_completion_suggestions.assert_called()
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        assert any("/clear" in s[0] for s in suggestions)


class TestScoreCommand:
    """Direct unit tests for SlashCommandController._score_command."""

    @staticmethod
    def score(search: str, cmd: str, desc: str, keywords: str = "") -> float:
        """Proxy score helper with explicit type signature for static analysis."""
        return SlashCommandController._score_command(search, cmd, desc, keywords)


class TestFuzzyFileControllerCanHandle:
    """Tests for FuzzyFileController.can_handle method."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view, tmp_path):
        """Create a FuzzyFileController."""
        return FuzzyFileController(mock_view, cwd=tmp_path)

    def test_invalid_cursor_positions(self, controller):
        """Handles invalid cursor positions gracefully."""
        assert controller.can_handle("@file", 0) is False
        assert controller.can_handle("@file", -1) is False
        assert controller.can_handle("@file", 100) is False


class TestThreadCompletionController:
    """Tests for durable `@@` thread references."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view: MagicMock) -> ThreadCompletionController:
        controller = ThreadCompletionController(mock_view)
        controller.update_threads(
            [
                {
                    "thread_id": "11111111-2222-3333-4444-555555555555",
                    "agent_name": "coder",
                    "updated_at": "2026-04-14T12:00:00Z",
                    "initial_prompt": "Fix the parser",
                    "git_branch": "feature/parser",
                    "cwd": "/workspace/project",
                },
                {
                    "thread_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "agent_name": "researcher",
                    "updated_at": "2026-04-13T12:00:00Z",
                    "initial_prompt": "Research caching",
                    "git_branch": "main",
                    "cwd": "/workspace/docs",
                },
            ]
        )
        return controller

    def test_handles_double_at_but_file_controller_does_not(
        self,
        controller: ThreadCompletionController,
        mock_view: MagicMock,
        tmp_path: Path,
    ) -> None:
        assert controller.can_handle("see @@pars", 10)
        assert controller.can_handle("(@@pars", 7)
        assert not controller.can_handle("email@@pars", 11)
        assert not controller.can_handle("compare @@(thread:abc)", 21)
        assert not FuzzyFileController(mock_view, cwd=tmp_path).can_handle("@@pars", 6)

    def test_searches_thread_metadata_and_inserts_token(
        self, controller: ThreadCompletionController, mock_view: MagicMock
    ) -> None:
        controller.on_text_changed("compare @@feature", 17)
        suggestions = mock_view.render_completion_suggestions.call_args.args[0]
        assert suggestions[0][0] == "Fix the parser"
        assert suggestions[0][1].endswith(" · 11111111")

        assert controller.apply_selection(0, "compare @@feature", 17)
        mock_view.replace_completion_range.assert_called_once_with(
            8,
            17,
            "@@(thread:11111111-2222-3333-4444-555555555555)",
        )

    def test_multiword_query_matches_initial_prompt(
        self, controller: ThreadCompletionController, mock_view: MagicMock
    ) -> None:
        controller.on_text_changed("compare @@fix parser", 20)
        suggestions = mock_view.render_completion_suggestions.call_args.args[0]
        assert suggestions[0][0] == "Fix the parser"

    @pytest.mark.parametrize("query", ["release AUDIT", "fix parser", "feature/parser"])
    def test_saved_name_search_and_label(
        self,
        controller: ThreadCompletionController,
        mock_view: MagicMock,
        query: str,
    ) -> None:
        thread_id = "11111111-2222-3333-4444-555555555555"
        controller.update_threads(
            [
                {
                    "thread_id": thread_id,
                    "agent_name": "coder",
                    "updated_at": None,
                    "thread_name": "Release audit",
                    "initial_prompt": "Fix the parser",
                    "git_branch": "feature/parser",
                }
            ]
        )
        text = f"compare @@{query}"
        controller.on_text_changed(text, len(text))
        mock_view.render_completion_suggestions.assert_called_once()
        suggestions = mock_view.render_completion_suggestions.call_args.args[0]
        assert suggestions[0][0] == "Release audit"
        assert controller.apply_selection(0, text, len(text))
        mock_view.replace_completion_range.assert_called_once_with(
            8, len(text), f"@@(thread:{thread_id})"
        )

    @pytest.mark.parametrize(
        ("name", "prompt", "label"),
        [
            (None, "Fix the parser", "Fix the parser"),
            ("", "Fix the parser", "Fix the parser"),
            (None, None, "11111111"),
            ("Release\n\taudit\x00", "Fix the parser", "Release audit"),
            ("a" * 100, "Fix the parser", "a" * 77 + "..."),
        ],
    )
    def test_label_sanitization_and_fallback(
        self,
        controller: ThreadCompletionController,
        mock_view: MagicMock,
        name: str | None,
        prompt: str | None,
        label: str,
    ) -> None:
        controller.update_threads(
            [
                {
                    "thread_id": "11111111-2222-3333-4444-555555555555",
                    "agent_name": "coder",
                    "updated_at": None,
                    "thread_name": name,
                    "initial_prompt": prompt,
                }
            ]
        )
        controller.on_text_changed("@@", 2)
        suggestions = mock_view.render_completion_suggestions.call_args.args[0]
        assert suggestions[0][0] == label


class TestMultiCompletionManager:
    """Tests for MultiCompletionManager."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_view, tmp_path):
        """Create a MultiCompletionManager with both controllers."""
        slash_ctrl = SlashCommandController(get_slash_commands(), mock_view)
        file_ctrl = FuzzyFileController(mock_view, cwd=tmp_path)
        # Cast needed: lists are invariant, so the inferred type
        # list[SlashCommandController | FuzzyFileController] won't match
        # list[CompletionController] even though both satisfy the protocol.
        controllers = cast("list[CompletionController]", [slash_ctrl, file_ctrl])
        return MultiCompletionManager(controllers)


class TestSlashCommandControllerUpdateCommands:
    """Tests for SlashCommandController.update_commands()."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()


class TestSlashCommandControllerDisplaySeparation:
    """Popup shows the label but completion inserts the machine name."""

    @pytest.fixture
    def mock_view(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def controller(self, mock_view: MagicMock) -> SlashCommandController:
        commands = [
            CommandEntry(
                name="/skill:my-plugin:review",
                description="(my-plugin) Review code",
                hidden_keywords="my-plugin review",
                argument_hint="",
                display_name="/skill:review",
            ),
        ]
        return SlashCommandController(commands, mock_view)


@pytest.mark.parametrize(
    "control",
    ["\n", "\t", "\x1b", "\x85", "\u2028", "\u2029", "\u202e", "\u200b", "\u034f"],
)
@pytest.mark.parametrize("warm", [False, True])
async def test_file_completion_rejects_unsafe_names(
    control: str, warm: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe repository filenames never enter the popup or chat input."""
    view = MagicMock()
    controller = FuzzyFileController(view, cwd=tmp_path)
    monkeypatch.setattr(
        autocomplete_module,
        "_get_project_files",
        lambda _root: [f"unsafe{control}@secret.txt", "re\u0301sume\u0301.md"],
    )

    if warm:
        await controller.warm_cache(force=True)
    controller.on_text_changed("@", 1)
    view.render_completion_suggestions.assert_called_once_with(
        [("@re\u0301sume\u0301.md", "md")], 0
    )
    assert controller.apply_selection(0, "@", 1)
    view.replace_completion_range.assert_called_once_with(
        0, 1, "@re\u0301sume\u0301.md"
    )


async def test_file_completion_uses_refreshed_file_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suggestions follow the file list after a forced background refresh."""
    view = MagicMock()
    controller = FuzzyFileController(view, cwd=tmp_path)
    files = ["old_module.py"]
    monkeypatch.setattr(autocomplete_module, "_get_project_files", lambda _root: files)

    controller.on_text_changed("@module", 7)
    assert view.render_completion_suggestions.call_args[0][0] == [
        ("@old_module.py", "py")
    ]

    files.append("new_module.py")
    await controller.warm_cache(force=True)
    controller.on_text_changed("@module", 7)
    labels = {label for label, _ in view.render_completion_suggestions.call_args[0][0]}
    assert labels == {"@old_module.py", "@new_module.py"}


class TestGetProjectFiles:
    """Tests for _get_project_files."""

    @staticmethod
    def _init_repo(root: Path) -> None:
        """Initialize a throwaway git repo with a test identity.

        Commit signing is disabled locally so commits succeed even when the
        host has `commit.gpgsign=true` set globally (no signing key is
        available in the throwaway repo).
        """
        for args in (
            ["init"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
            ["config", "commit.gpgsign", "false"],
        ):
            subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    def test_includes_tracked_and_untracked_files(self, tmp_path: Path) -> None:
        """Both committed and untracked-but-not-ignored files are returned.

        Tracked files are listed before untracked ones so they rank ahead in
        completion results.
        """
        self._init_repo(tmp_path)
        (tmp_path / "tracked.py").write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "tracked.py"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        (tmp_path / "untracked.py").write_text("y = 2\n")

        files = _get_project_files(tmp_path)

        assert "tracked.py" in files
        assert "untracked.py" in files
        assert files.index("tracked.py") < files.index("untracked.py")

    def test_excludes_ignored_files(self, tmp_path: Path) -> None:
        """Files matched by .gitignore are not returned."""
        self._init_repo(tmp_path)
        (tmp_path / ".gitignore").write_text("ignored.py\n")
        (tmp_path / "ignored.py").write_text("z = 3\n")
        (tmp_path / "visible.py").write_text("a = 4\n")

        files = _get_project_files(tmp_path)

        assert "visible.py" in files
        assert "ignored.py" not in files

    def test_empty_git_listing_does_not_fall_back_to_glob(self, tmp_path: Path) -> None:
        """Successful empty git output is authoritative."""
        self._init_repo(tmp_path)
        (tmp_path / ".git" / "info" / "exclude").write_text("ignored.py\n")
        (tmp_path / "ignored.py").write_text("z = 3\n")

        files = _get_project_files(tmp_path)

        assert files == []

    def test_non_ascii_names_are_returned_verbatim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for #6965: non-ASCII names come back as real paths.

        Under `core.quotePath`, plain `git ls-files` prints such names C-quoted
        with octal escapes. Unquoted output must still be decoded as UTF-8: a
        legacy Windows code page is simulated through `locale.getencoding`,
        which `subprocess` consults when no encoding is given.
        """
        self._init_repo(tmp_path)
        # Pin git's default so a global `core.quotePath=false` can't mask it.
        subprocess.run(
            ["git", "config", "core.quotePath", "true"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "résumé.md").write_text("tracked\n")
        subprocess.run(
            ["git", "add", "résumé.md"], cwd=tmp_path, check=True, capture_output=True
        )
        (tmp_path / "naïve.py").write_text("untracked\n")
        monkeypatch.setattr(locale, "getencoding", lambda: "cp1252")

        files = _get_project_files(tmp_path)

        assert files == ["résumé.md", "naïve.py"]

    @pytest.mark.skipif(
        os.name == "nt" or sys.platform == "darwin",
        reason="Windows and macOS (APFS) require Unicode filenames",
    )
    def test_non_utf8_names_do_not_alias_valid_files(self, tmp_path: Path) -> None:
        """Invalid UTF-8 names must not become a different selectable filename."""
        self._init_repo(tmp_path)
        invalid_name = os.fsdecode(b"bad-\xff.md")
        valid_name = "bad-\ufffd.md"
        (tmp_path / invalid_name).write_text("invalid name")
        (tmp_path / valid_name).write_text("valid name")

        files = _get_project_files(tmp_path)
        assert set(files) == {invalid_name, valid_name}

        view = MagicMock()
        controller = FuzzyFileController(view, cwd=tmp_path)
        controller.on_text_changed("@", 1)
        view.render_completion_suggestions.assert_called_once_with(
            [(f"@{valid_name}", "md")], 0
        )
        assert controller.apply_selection(0, "@", 1)
        view.replace_completion_range.assert_called_once_with(0, 1, f"@{valid_name}")

    def test_deduplicates_repeated_paths(self, tmp_path: Path) -> None:
        """A path emitted more than once by git ls-files appears only once.

        An unmerged (conflicted) file is reported once per merge stage by
        `git ls-files`, which is the real source of the duplicate entries the
        de-duplication guards against.
        """
        self._init_repo(tmp_path)
        conflict = tmp_path / "conflict.py"
        conflict.write_text("base\n")
        subprocess.run(
            ["git", "add", "conflict.py"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        base_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "checkout", "-b", "other"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        conflict.write_text("theirs\n")
        subprocess.run(
            ["git", "commit", "-am", "theirs"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", base_branch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        conflict.write_text("ours\n")
        subprocess.run(
            ["git", "commit", "-am", "ours"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        # The merge fails with a conflict; the conflicted state is the point.
        subprocess.run(
            ["git", "merge", "other"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
        )

        git_path = _get_git_executable()
        assert git_path is not None
        _, raw = _run_git_ls_files(git_path, tmp_path, [])
        # Premise: the conflicted path is reported more than once.
        assert raw.count("conflict.py") > 1

        files = _get_project_files(tmp_path)

        assert files.count("conflict.py") == 1

    def test_untracked_failure_keeps_tracked_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed untracked scan must not discard the tracked list.

        When the optional `--others --exclude-standard` call fails or times
        out, the already-successful tracked listing stays authoritative instead
        of falling back to the shallow glob walk.
        """
        self._init_repo(tmp_path)
        nested = tmp_path / "a" / "b" / "c" / "d" / "e"
        nested.mkdir(parents=True)
        deep = nested / "deep.py"
        deep.write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "a"], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        real_run = autocomplete_module._run_git_ls_files

        def fake_run(
            git_path: str, root: Path, extra_args: list[str]
        ) -> tuple[bool, list[str]]:
            if "--others" in extra_args:
                return False, []
            return real_run(git_path, root, extra_args)

        monkeypatch.setattr(autocomplete_module, "_run_git_ls_files", fake_run)

        files = _get_project_files(tmp_path)

        assert "a/b/c/d/e/deep.py" in files

    def test_git_stderr_uses_stable_locale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Git diagnostics must use the English locale expected by log filtering."""

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        captured: dict[str, object] = {}

        def fake_run(*_args: object, **kwargs: object) -> _Result:
            captured.update(kwargs)
            return _Result()

        monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
        monkeypatch.setenv("AUTOCOMPLETE_TEST_ENV", "preserved")
        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        _run_git_ls_files("git", tmp_path, [])

        env = cast("dict[str, str]", captured["env"])
        assert env["LC_ALL"] == "C"
        assert env["AUTOCOMPLETE_TEST_ENV"] == "preserved"

    def test_non_repo_directory_is_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-Git directory (exit 128) must not emit a debug log.

        Running `git ls-files` outside a work tree is the expected trigger for
        the glob fallback, so the failure path stays silent.
        """
        git_path = _get_git_executable()
        assert git_path is not None

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            ok, files = _run_git_ls_files(git_path, tmp_path, [])

        assert ok is False
        assert files == []
        assert caplog.records == []

    def test_genuine_failure_logs_details(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A real Git failure logs root/cwd, args, exit code, and stderr."""

        class _Result:
            returncode = 129
            stdout = ""
            stderr = "fatal: unknown option `--bogus'\n"

        def fake_run(*_args: object, **_kwargs: object) -> _Result:
            return _Result()

        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            ok, files = _run_git_ls_files("git", tmp_path, ["--bogus"])

        assert ok is False
        assert files == []
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert str(tmp_path) in message
        assert "--bogus" in message
        assert "exit=129" in message
        assert "unknown option" in message

    def test_failure_stderr_is_sanitized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Control characters in git stderr are neutralized before logging."""

        class _Result:
            returncode = 1
            stdout = ""
            stderr = "fatal: broken\x1b[31mred\r\nsecond line\n"

        def fake_run(*_args: object, **_kwargs: object) -> _Result:
            return _Result()

        monkeypatch.setattr(autocomplete_module.subprocess, "run", fake_run)

        with caplog.at_level(logging.DEBUG, logger="deepagents_code"):
            _run_git_ls_files("git", tmp_path, [])

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "\x1b" not in message
        assert "\r" not in message

    def test_glob_fallback_when_git_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without git, files are discovered via glob, excluding dotpaths."""
        monkeypatch.setattr(autocomplete_module, "_get_git_executable", lambda: None)
        (tmp_path / "visible.py").write_text("a = 1\n")
        (tmp_path / ".hidden.py").write_text("secret = 1\n")
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text("b = 2\n")

        files = _get_project_files(tmp_path)

        assert "visible.py" in files
        assert "pkg/mod.py" in files
        assert ".hidden.py" not in files


class TestFuzzyFileControllerScope:
    """Tests for cwd-scoped file completion behavior."""

    @pytest.fixture
    def mock_view(self):
        """Create a mock CompletionView."""
        return MagicMock()

    def test_scopes_git_file_list_to_cwd(self, mock_view, monkeypatch, tmp_path):
        """When cwd is nested, suggestions are scoped to that subtree."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        mock_files = [
            "README.md",
            "apps/cli/main.py",
            "apps/cli/utils/helpers.py",
            "apps/web/index.ts",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py", "utils/helpers.py"]

        controller.on_text_changed("@", 1)
        suggestions = mock_view.render_completion_suggestions.call_args[0][0]
        labels = [label for label, _ in suggestions]
        assert "@main.py" in labels
        assert "@utils/helpers.py" in labels
        assert not any("apps/web" in label for label in labels)

    def test_keeps_project_root_scope_when_cwd_is_root(
        self, mock_view, monkeypatch, tmp_path
    ):
        """When cwd is project root, file list remains repo-relative."""
        (tmp_path / ".git").mkdir()
        mock_files = ["README.md", "apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=tmp_path)
        assert controller._get_files() == mock_files

    def test_scopes_git_file_list_with_symlinked_cwd(
        self, mock_view, monkeypatch, tmp_path
    ):
        """Symlinked cwd should still scope suggestions to the resolved subtree."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        real_cwd = project_root / "apps" / "cli"
        real_cwd.mkdir(parents=True)
        symlink_cwd = project_root / "APPS_CLI_LINK"
        try:
            symlink_cwd.symlink_to(real_cwd, target_is_directory=True)
        except OSError:  # pragma: no cover - platform/permission dependent
            return

        mock_files = [
            "README.md",
            "apps/cli/main.py",
            "apps/cli/utils/helpers.py",
            "apps/web/index.ts",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=symlink_cwd)
        assert controller._get_files() == ["main.py", "utils/helpers.py"]

    def test_excludes_sibling_with_shared_prefix(
        self, mock_view, monkeypatch, tmp_path
    ):
        """A sibling sharing a name prefix (apps/cli vs apps/cli-tools) is excluded.

        Guards the trailing slash in the scope prefix: without it, `apps/cli`
        would also match `apps/cli-tools/...`.
        """
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        mock_files = [
            "apps/cli/main.py",
            "apps/cli-tools/runner.py",
        ]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py"]

    def test_empty_when_cwd_subtree_has_no_files(
        self, mock_view, monkeypatch, tmp_path
    ):
        """A nested cwd with no files under it yields an empty suggestion list."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "empty"
        cwd.mkdir(parents=True)

        mock_files = ["README.md", "apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: mock_files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == []

    def test_refresh_cache_rescopes_to_cwd(self, mock_view, monkeypatch, tmp_path):
        """refresh_cache re-runs scoping against the latest file list."""
        project_root = tmp_path
        (project_root / ".git").mkdir()
        cwd = project_root / "apps" / "cli"
        cwd.mkdir(parents=True)

        files = ["apps/cli/main.py"]
        monkeypatch.setattr(
            autocomplete_module, "_get_project_files", lambda _root: files
        )

        controller = FuzzyFileController(mock_view, cwd=cwd)
        assert controller._get_files() == ["main.py"]

        files.append("apps/cli/added.py")
        controller.refresh_cache()
        assert controller._get_files() == ["main.py", "added.py"]

    def test_scope_helper_fails_closed_when_cwd_outside_root(self):
        """A cwd outside project_root returns [] rather than wrong-base paths.

        The input paths are project-root-relative; if cwd is not under the root
        they would resolve to the wrong base, so the helper fails closed.
        """
        files = ["src/main.py", "src/utils.py"]
        project_root = Path("/repo")
        cwd = Path("/elsewhere")

        assert _scope_files_to_cwd(files, project_root, cwd) == []

    def test_scope_helper_returns_unchanged_when_cwd_is_root(self):
        """A cwd equal to project_root leaves the repo-relative list unchanged."""
        files = ["src/main.py", "README.md"]
        root = Path("/repo")

        assert _scope_files_to_cwd(files, root, root) == files
