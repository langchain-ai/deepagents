"""Tests for the project README pull request gate."""

import json

from check_project_readmes import (
    ACKNOWLEDGMENT_LABEL,
    PROJECT_READMES,
    find_readme_edits,
    main,
    parse_pr_type,
)


def test_known_project_readmes_are_protected() -> None:
    """Root and package metadata READMEs are the protected project set."""
    assert PROJECT_READMES == {
        "README.md",
        "libs/acp/README.md",
        "libs/code/README.md",
        "libs/deepagents/README.md",
        "libs/evals/README.md",
        "libs/partners/daytona/README.md",
        "libs/partners/modal/README.md",
        "libs/partners/quickjs/README.md",
        "libs/partners/runloop/README.md",
        "libs/partners/vercel/README.md",
        "libs/talon/README.md",
    }
    assert ACKNOWLEDGMENT_LABEL == "readme: acknowledged"


def test_non_docs_pr_reports_only_protected_readmes() -> None:
    """Nested test and example READMEs do not trigger the project gate."""
    result = find_readme_edits(
        "fix(code): repair examples",
        [
            "libs/code/README.md",
            "README.md",
            "libs/code/tests/README.md",
            "examples/README.md",
        ],
    )
    assert result == {
        "pr_type": "fix",
        "readmes": ["README.md", "libs/code/README.md"],
    }


def test_docs_pr_exempts_protected_readme_edits() -> None:
    """A `docs` Conventional Commit type satisfies the gate."""
    assert find_readme_edits(
        "docs(code): clarify setup",
        ["README.md", "libs/code/README.md"],
    ) == {"pr_type": "docs", "readmes": []}


def test_title_type_parser_is_anchored() -> None:
    """Only an exact leading Conventional Commit type is accepted."""
    assert parse_pr_type("docs(code)!: clarify setup") == "docs"
    assert parse_pr_type(" docs(code): clarify setup") is None
    assert parse_pr_type("Docs(code): clarify setup") is None
    assert parse_pr_type("fix(code): mention docs(code): later") == "fix"


def test_main_emits_compact_json(capsys, monkeypatch) -> None:
    """The workflow receives a stable JSON object from the helper."""
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO('["README.md"]'))
    assert main("feat(sdk): update landing page") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"pr_type": "feat", "readmes": ["README.md"]}
    assert captured.err == ""


def test_main_fails_closed_on_invalid_input(capsys, monkeypatch) -> None:
    """Malformed changed-file input cannot be treated as a clean PR."""
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO('{"README.md": true}'))
    assert main("fix(repo): update readme") == 2
    assert "::error::" in capsys.readouterr().err
