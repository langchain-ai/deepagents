"""Tests for the project README pull request gate."""

import io
import json
from pathlib import Path

import pytest
from check_project_readmes import (
    ACKNOWLEDGMENT_LABEL,
    PROJECT_READMES,
    find_readme_edits,
    main,
    parse_pr_type,
)

ROOT = Path(__file__).resolve().parents[4]
GATE_JS = ROOT / ".github" / "scripts" / "checks" / "readme-gate.js"

# Every Conventional Commit type this repo uses, minus `docs`. Parametrized
# rather than sampled so widening the exemption (adding `chore`, say) fails
# here instead of silently letting a whole class of PR through.
NON_DOCS_TYPES = (
    "feat",
    "fix",
    "chore",
    "ci",
    "refactor",
    "test",
    "perf",
    "build",
    "revert",
    "style",
    "release",
)


def test_protected_set_matches_the_packages_in_the_repo() -> None:
    """A newly added package's README cannot silently escape the gate.

    `PROJECT_READMES` is hardcoded so the detector runs without a repo
    checkout, which means nothing at runtime notices when a new package
    arrives. This derives the same set from the real source of truth --
    every `libs/**/pyproject.toml` declares `readme = "README.md"` -- so
    adding a package without protecting its README fails here.
    """
    derived = {"README.md"} | {
        str(pyproject.parent.relative_to(ROOT) / "README.md")
        for pyproject in (ROOT / "libs").rglob("pyproject.toml")
    }
    assert PROJECT_READMES == derived


def test_acknowledgment_label_matches_the_javascript_literal() -> None:
    """The label name is duplicated in readme-gate.js and must not drift.

    Nothing at runtime reads the Python constant; the gate reads the JS one.
    If they diverge, applying the label the PR comment names would no longer
    clear the block, with no failing test to say so.
    """
    assert ACKNOWLEDGMENT_LABEL == "readme: acknowledged"
    assert f"'{ACKNOWLEDGMENT_LABEL}'" in GATE_JS.read_text()


def test_non_docs_pr_reports_only_protected_readmes() -> None:
    """Nested test and example READMEs do not trigger the project gate."""
    result = find_readme_edits(
        "fix(code): repair examples",
        [
            "libs/code/README.md",
            "README.md",
            "libs/code/tests/README.md",
            "examples/README.md",
            "libs/README.md",
        ],
    )
    assert result == {
        "pr_type": "fix",
        "readmes": ["README.md", "libs/code/README.md"],
    }


@pytest.mark.parametrize("pr_type", NON_DOCS_TYPES)
def test_only_docs_exempts_protected_readme_edits(pr_type: str) -> None:
    """`docs` is the sole exempt type; every other type still blocks."""
    assert find_readme_edits(f"{pr_type}(code): touch up", ["README.md"]) == {
        "pr_type": pr_type,
        "readmes": ["README.md"],
    }


def test_docs_pr_exempts_protected_readme_edits() -> None:
    """A `docs` Conventional Commit type satisfies the gate."""
    assert find_readme_edits(
        "docs(code): clarify setup",
        ["README.md", "libs/code/README.md"],
    ) == {"pr_type": "docs", "readmes": []}


def test_renaming_a_protected_readme_is_an_edit() -> None:
    """The pre-rename path the workflow supplies still trips the gate.

    Renaming a protected README away would otherwise be a one-line bypass:
    the new path is not in the protected set, so only `previous_filename`
    catches it.
    """
    assert find_readme_edits(
        "feat(code): reorganize docs",
        ["libs/code/READ_ME.md", "libs/code/README.md"],
    )["readmes"] == ["libs/code/README.md"]


def test_unparseable_title_is_not_exempt() -> None:
    """A title with no Conventional Commit prefix fails closed."""
    assert find_readme_edits("update the readme", ["README.md"]) == {
        "pr_type": None,
        "readmes": ["README.md"],
    }


def test_title_type_parser_is_anchored() -> None:
    """Only an exact leading lowercase Conventional Commit type is accepted."""
    assert parse_pr_type(" docs(code): clarify setup") is None
    assert parse_pr_type("Docs(code): clarify setup") is None
    assert parse_pr_type("fix(code): mention docs(code): later") == "fix"


def test_title_type_parser_tolerates_breaking_change_marker() -> None:
    """`!` marks a breaking change and does not change the type.

    The scope and the `!` are independent regex elements, so both the scoped
    and bare forms are pinned: a refactor that folded `!` into the optional
    scope group would keep one working and break the other, wrongly blocking
    a legitimate docs PR.
    """
    assert parse_pr_type("docs(code)!: clarify setup") == "docs"
    assert parse_pr_type("docs!: drop the old guide") == "docs"


def test_main_emits_parseable_result_and_no_stderr(capsys, monkeypatch) -> None:
    """stdout carries only the JSON verdict; stderr stays empty.

    The workflow captures stdout straight into `$GITHUB_OUTPUT` via a
    heredoc, so any stray output would corrupt the step output.
    """
    monkeypatch.setattr("sys.stdin", io.StringIO('["README.md"]'))
    assert main("feat(sdk): update landing page") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"pr_type": "feat", "readmes": ["README.md"]}
    assert captured.err == ""
    # Compact separators: pinned as an exact string because `json.loads`
    # round-trips identically with or without them.
    assert captured.out.strip() == '{"pr_type":"feat","readmes":["README.md"]}'


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('{"README.md": true}', id="object"),
        pytest.param("[1, 2]", id="non-string-elements"),
        pytest.param('"README.md"', id="bare-string"),
        pytest.param("", id="empty-stdin"),
        pytest.param("[", id="truncated-json"),
        pytest.param("null", id="null"),
    ],
)
def test_main_fails_closed_on_invalid_input(payload, capsys, monkeypatch) -> None:
    """Malformed changed-file input cannot be treated as a clean PR."""
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    assert main("fix(repo): update readme") == 2
    captured = capsys.readouterr()
    assert "::error::" in captured.err
    # Nothing on stdout, so `set -euo pipefail` cannot capture a partial
    # result that the workflow would parse as a clean verdict.
    assert captured.out == ""
