"""Tests for check_pr_scope_files (PR title scope vs changed package dirs)."""

import json

import pytest
from check_pr_scope_files import (
    DEFAULT_CONFIG,
    changed_packages,
    declared_packages,
    find_offenders,
    main,
    parse_title_scopes,
)

CONFIG = {
    "scopeToLabel": {
        "ci": "infra",
        "cli": "cli",
        "code": "dcode",
        "deepagents-cli": "cli",
        "deepagents-code": "dcode",
        "docs": "documentation",
        "infra": "infra",
        "sdk": "deepagents",
    },
    "fileRules": [
        {"label": "deepagents", "prefix": "libs/deepagents/"},
        {"label": "cli", "prefix": "libs/cli/"},
        {"label": "dcode", "prefix": "libs/code/"},
        {"label": "github_actions", "prefix": ".github/workflows/"},
        {"label": "dependencies", "suffix": "pyproject.toml"},
    ],
}


def test_matching_scope_files_pass() -> None:
    """A package scope that covers the touched package dir is clean."""
    changed = ["libs/code/deepagents_code/app.py"]
    assert find_offenders("fix(code): repair startup", changed, CONFIG) == []


def test_cli_scope_with_code_files_blocks() -> None:
    """`fix(cli):` does not cover files under `libs/code/`."""
    changed = ["libs/code/deepagents_code/app.py"]
    assert find_offenders("fix(cli): repair startup", changed, CONFIG) == [
        {"package": "dcode", "dirs": ["libs/code/"]}
    ]


def test_multi_scope_title_covers_multiple_package_dirs() -> None:
    """Comma-separated scopes cover every matching package label."""
    changed = [
        "libs/cli/deepagents_cli/main.py",
        "libs/code/deepagents_code/app.py",
    ]
    assert find_offenders("feat(cli,code): share option", changed, CONFIG) == []


def test_multi_scope_title_blocks_uncovered_package_dir() -> None:
    """A third package remains an offender when absent from a multi-scope title."""
    changed = [
        "libs/cli/deepagents_cli/main.py",
        "libs/code/deepagents_code/app.py",
        "libs/deepagents/deepagents/graph.py",
    ]
    assert find_offenders("feat(cli,code): share option", changed, CONFIG) == [
        {"package": "deepagents", "dirs": ["libs/deepagents/"]}
    ]


def test_unscoped_title_type_and_non_package_paths_pass() -> None:
    """Non-package scopes and non-package paths are ignored as unscoped."""
    assert (
        find_offenders(
            "ci(infra): tune workflow",
            [".github/workflows/ci.yml", "README.md", "pyproject.toml"],
            CONFIG,
        )
        == []
    )
    assert (
        find_offenders(
            "ci(infra): tune package job",
            ["libs/code/deepagents_code/app.py"],
            CONFIG,
        )
        == []
    )


def test_non_package_path_with_package_scope_passes() -> None:
    """A package scope plus only non-package paths has no touched package offender."""
    assert find_offenders("fix(cli): repair action", ["action.yml"], CONFIG) == []


def test_package_scope_aliases_resolve_to_same_package_label() -> None:
    """Long package-name scopes are aliases for the same labels as short scopes."""
    assert declared_packages("fix(deepagents-code): repair startup", CONFIG) == {
        "dcode"
    }
    assert (
        find_offenders(
            "fix(deepagents-code): repair startup",
            ["libs/code/deepagents_code/app.py"],
            CONFIG,
        )
        == []
    )


def test_parse_title_scopes_variants() -> None:
    """Scopes are parsed from conventional-commit-shaped titles only."""
    assert parse_title_scopes("feat(cli, code): x") == ("cli", "code")
    assert parse_title_scopes("fix(deepagents-code)!: x") == ("deepagents-code",)
    assert parse_title_scopes("fix: x") == ()
    assert parse_title_scopes("not conventional") == ()


def test_changed_packages_returns_touched_package_dirs_only() -> None:
    """Only `libs/**` prefix rules are treated as package dirs."""
    assert changed_packages(
        ["libs/code/deepagents_code/app.py", ".github/workflows/ci.yml"], CONFIG
    ) == {"dcode": ["libs/code/"]}


def test_changed_packages_matches_bare_package_dir() -> None:
    """A path equal to the package dir (no trailing slash) still matches."""
    assert changed_packages(["libs/code"], CONFIG) == {"dcode": ["libs/code/"]}


def test_changed_packages_ignores_prefix_collision() -> None:
    """A sibling dir sharing a name prefix is not treated as the package."""
    assert changed_packages(["libs/codex/app.py"], CONFIG) == {}


def test_breaking_change_title_still_detects_offender() -> None:
    """A breaking-change `!` title parses its scope for offender detection."""
    assert find_offenders(
        "feat(cli)!: drop option", ["libs/code/deepagents_code/app.py"], CONFIG
    ) == [{"package": "dcode", "dirs": ["libs/code/"]}]


def test_partner_package_dir_detected_with_real_config() -> None:
    """Partner packages under `libs/partners/` are package dirs too."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert find_offenders(
        "fix(cli): repair startup",
        ["libs/partners/daytona/langchain_daytona/sandbox.py"],
        config,
    ) == [{"package": "daytona", "dirs": ["libs/partners/daytona/"]}]


def test_partner_scope_aliases_resolve_with_real_config() -> None:
    """`langchain-*` scope aliases map to the same partner package labels."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert declared_packages("fix(langchain-quickjs): x", config) == {"quickjs"}
    assert declared_packages("fix(quickjs): x", config) == {"quickjs"}


def test_non_dict_file_rule_raises() -> None:
    """A non-object `fileRules` entry is config corruption, not a skip."""
    config = {"scopeToLabel": CONFIG["scopeToLabel"], "fileRules": ["libs/code/"]}
    with pytest.raises(ValueError, match="not an object"):
        find_offenders("fix(cli): x", ["libs/code/file.py"], config)


def test_non_string_prefix_file_rule_raises() -> None:
    """A package rule with a malformed `prefix` type fails closed."""
    config = {
        "scopeToLabel": CONFIG["scopeToLabel"],
        "fileRules": [{"label": "dcode", "prefix": 123}],
    }
    with pytest.raises(ValueError, match="non-string label/prefix"):
        find_offenders("fix(cli): x", ["libs/code/file.py"], config)


def test_main_partially_malformed_file_rules_returns_2(capsys, tmp_path) -> None:
    """A single malformed package rule fails closed instead of being dropped."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(
        json.dumps(
            {
                "scopeToLabel": CONFIG["scopeToLabel"],
                "fileRules": [
                    {"label": "dcode", "prefix": 123},
                    {"label": "cli", "prefix": "libs/cli/"},
                ],
            }
        ),
        encoding="utf-8",
    )
    rc = main("fix(cli): x", ["libs/code/file.py"], config_path=config_path)
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_stdout_is_json_array(capsys, tmp_path) -> None:
    """The workflow can strictly parse stdout as JSON."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main(
        "fix(cli): repair startup",
        ["libs/code/deepagents_code/app.py"],
        config_path=config_path,
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == [{"package": "dcode", "dirs": ["libs/code/"]}]
    assert "PR title scope does not cover" in captured.err


def test_main_missing_config_returns_2(capsys, tmp_path) -> None:
    """A missing config fails closed instead of silently passing."""
    rc = main("fix(cli): x", ["libs/code/file.py"], config_path=tmp_path / "nope.json")
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_malformed_config_returns_2(capsys, tmp_path) -> None:
    """Invalid JSON fails closed."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text("{not json", encoding="utf-8")
    rc = main("fix(cli): x", ["libs/code/file.py"], config_path=config_path)
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_empty_file_rules_returns_2(capsys, tmp_path) -> None:
    """Config drift that removes package file rules fails closed."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(
        json.dumps({"scopeToLabel": CONFIG["scopeToLabel"], "fileRules": []}),
        encoding="utf-8",
    )
    rc = main("fix(cli): x", ["libs/code/file.py"], config_path=config_path)
    assert rc == 2
    assert "fileRules" in capsys.readouterr().err


def test_main_missing_scope_map_returns_2(capsys, tmp_path) -> None:
    """Config drift that removes `scopeToLabel` fails closed."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(
        json.dumps({"scopeToLabel": {}, "fileRules": CONFIG["fileRules"]}),
        encoding="utf-8",
    )
    rc = main("fix(cli): x", ["libs/code/file.py"], config_path=config_path)
    assert rc == 2
    assert "scopeToLabel" in capsys.readouterr().err


def test_real_config_has_package_scope_and_dir_mappings() -> None:
    """The committed PR labeler config exposes the maps this check reads."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert declared_packages("fix(cli): x", config) == {"cli"}
    assert declared_packages("fix(code): x", config) == {"dcode"}
    assert changed_packages(["libs/cli/deepagents_cli/main.py"], config) == {
        "cli": ["libs/cli/"]
    }
    assert changed_packages(["libs/code/deepagents_code/app.py"], config) == {
        "dcode": ["libs/code/"]
    }
