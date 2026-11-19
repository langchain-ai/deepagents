"""Tests for check_lockfile_release_scope (release fan-out guard)."""

import json

from check_lockfile_release_scope import (
    DEFAULT_CONFIG,
    bump_worthy_types,
    find_fanout,
    is_bump_worthy,
    main,
    parse_title,
    release_relevant_files,
    touched_components,
)


def _pkg(path: str, component: str) -> dict:
    """Return a package entry that excludes `tests/` and `uv.lock`, like the real config."""
    return {
        "component": component,
        "exclude-paths": [f"{path}/tests", f"{path}/uv.lock"],
    }


# A minimal config mirroring the real release-please-config.json shape: managed
# packages with their exclude-paths, plus the changelog-sections the bump-worthy
# set is derived from.
CONFIG = {
    "changelog-sections": [
        {"type": "feat", "section": "Features"},
        {"type": "fix", "section": "Bug Fixes"},
        {"type": "perf", "section": "Performance Improvements"},
        {"type": "revert", "section": "Reverted Changes"},
        {"type": "chore", "section": "Chores", "hidden": True},
        {"type": "docs", "section": "Documentation", "hidden": True},
        {"type": "refactor", "section": "Refactors", "hidden": True},
    ],
    "packages": {
        "libs/deepagents": _pkg("libs/deepagents", "deepagents"),
        "libs/acp": _pkg("libs/acp", "deepagents-acp"),
        "libs/code": _pkg("libs/code", "deepagents-code"),
        "libs/partners/quickjs": _pkg("libs/partners/quickjs", "langchain-quickjs"),
    },
}

CLEAN = {"multi_component": [], "lockfile_only": []}


def test_lockfile_only_dependents_not_flagged() -> None:
    """Relocking dependents in a single-package feat is not a fan-out."""
    changed = [
        "libs/partners/quickjs/langchain_quickjs/_repl.py",
        "libs/partners/quickjs/pyproject.toml",
        "libs/partners/quickjs/uv.lock",
        "libs/code/uv.lock",
        "libs/acp/uv.lock",
    ]
    assert find_fanout("feat(quickjs)!: bump quickjs-rs", changed, CONFIG) == CLEAN
    assert touched_components(changed, CONFIG) == ["langchain-quickjs"]


def test_pyproject_in_second_package_flagged() -> None:
    """A dependent's `pyproject.toml` is published metadata, so it still fans out."""
    changed = [
        "libs/partners/quickjs/pyproject.toml",
        "libs/partners/quickjs/uv.lock",
        "libs/code/pyproject.toml",
        "libs/code/uv.lock",
    ]
    assert find_fanout("feat(quickjs): bump quickjs-rs", changed, CONFIG) == {
        "multi_component": ["deepagents-code", "langchain-quickjs"],
        "lockfile_only": [],
    }


def test_tests_only_change_in_second_package_not_flagged() -> None:
    """Files under a package's excluded `tests/` dir do not release it."""
    changed = [
        "libs/deepagents/deepagents/graph.py",
        "libs/code/tests/unit_tests/test_graph.py",
        "libs/code/uv.lock",
    ]
    assert find_fanout("feat(sdk): surface subagents", changed, CONFIG) == CLEAN


def test_multi_component_real_files_flagged() -> None:
    """A bump-worthy PR that edits published files in 2+ components is flagged."""
    changed = [
        "libs/code/deepagents_code/models.py",
        "libs/code/pyproject.toml",
        "libs/code/uv.lock",
        "libs/acp/pyproject.toml",
        "libs/acp/uv.lock",
        "libs/deepagents/pyproject.toml",
        "libs/deepagents/uv.lock",
    ]
    fanout = find_fanout("feat(code): add model support", changed, CONFIG)
    assert fanout["multi_component"] == [
        "deepagents",
        "deepagents-acp",
        "deepagents-code",
    ]
    assert fanout["lockfile_only"] == []


def test_non_bump_title_skipped() -> None:
    """A hidden type (chore) does not cut a release, so nothing is flagged."""
    changed = ["libs/acp/pyproject.toml", "libs/code/pyproject.toml"]
    assert find_fanout("chore(deps): bump bounds", changed, CONFIG) == CLEAN


def test_breaking_bang_is_bump_worthy() -> None:
    """The `!` breaking shorthand counts as bump-worthy even for an odd type."""
    changed = ["libs/acp/pyproject.toml", "libs/code/pyproject.toml"]
    assert find_fanout("refactor(acp)!: drop thing", changed, CONFIG)[
        "multi_component"
    ] == ["deepagents-acp", "deepagents-code"]


def test_no_managed_paths_touched() -> None:
    """Changes outside any managed package path produce no offenders."""
    changed = ["examples/deep_research/uv.lock", "README.md"]
    assert find_fanout("feat(docs): example", changed, CONFIG) == CLEAN
    assert touched_components(changed, CONFIG) == []


def test_single_package_real_files_not_multi() -> None:
    """One package with real files does not trip multi-component, even with many files."""
    changed = [
        "libs/code/deepagents_code/models.py",
        "libs/code/pyproject.toml",
        "libs/code/uv.lock",
        "README.md",
    ]
    assert find_fanout("feat(code): models", changed, CONFIG) == CLEAN


def test_touched_components_ignores_excluded_files() -> None:
    """`touched_components` lists only components with release-relevant files."""
    changed = [
        "libs/acp/uv.lock",
        "libs/deepagents/tests/unit_tests/test_x.py",
        "libs/code/pyproject.toml",
        "README.md",
    ]
    assert touched_components(changed, CONFIG) == ["deepagents-code"]


def test_release_relevant_files_matches_release_please_exclusion() -> None:
    """Exclusions match the exact file or anything under it, never a sibling prefix."""
    meta = _pkg("libs/code", "deepagents-code")
    changed = [
        "libs/code/uv.lock",
        "libs/code/uv.lock.bak",
        "libs/code/tests/test_x.py",
        "libs/code/tests_helpers.py",
        "libs/code/sub/uv.lock",
        "libs/codex/main.py",
    ]
    assert release_relevant_files(changed, "libs/code", meta) == [
        "libs/code/uv.lock.bak",
        "libs/code/tests_helpers.py",
        "libs/code/sub/uv.lock",
    ]


def test_release_relevant_files_normalizes_slashes() -> None:
    """Leading/trailing slashes in config paths are ignored, as in release-please."""
    meta = {"exclude-paths": ["/libs/code/uv.lock/", "libs/code/tests/"]}
    changed = ["libs/code/uv.lock", "libs/code/tests/t.py", "libs/code/main.py"]
    assert release_relevant_files(changed, "libs/code/", meta) == ["libs/code/main.py"]


def test_package_without_exclude_paths_counts_every_file() -> None:
    """A package with no `exclude-paths` releases on any file, lockfile included."""
    config = {
        "changelog-sections": [{"type": "feat", "section": "Features"}],
        "packages": {
            "libs/a": {"component": "a"},
            "libs/b": {"component": "b"},
        },
    }
    changed = ["libs/a/main.py", "libs/b/uv.lock"]
    assert find_fanout("feat: x", changed, config)["multi_component"] == ["a", "b"]


def test_parse_title_variants() -> None:
    """Type and breaking marker are extracted across common title shapes."""
    assert parse_title("feat(sdk): x") == ("feat", False)
    assert parse_title("fix: x") == ("fix", False)
    assert parse_title("feat(sdk)!: x") == ("feat", True)
    assert parse_title("revert!: x") == ("revert", True)
    assert parse_title("not a conventional title") == (None, False)


def test_is_bump_worthy() -> None:
    """Bump-worthy covers visible types and the `!` shorthand only."""
    assert is_bump_worthy("feat(sdk): x", CONFIG)
    assert is_bump_worthy("refactor(acp)!: x", CONFIG)
    assert not is_bump_worthy("chore(deps): x", CONFIG)
    assert not is_bump_worthy("not conventional", CONFIG)


def test_bump_types_derived_from_visible_sections() -> None:
    """Bump-worthy set is exactly the non-hidden changelog sections."""
    assert bump_worthy_types(CONFIG) == frozenset({"feat", "fix", "perf", "revert"})


def test_real_config_has_expected_shape() -> None:
    """The committed release-please-config.json still exposes what the check reads."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config.get("packages"), "release-please config has no packages"
    assert bump_worthy_types(config), "release-please config has no visible sections"
    for path, meta in config["packages"].items():
        assert "component" in meta, f"package {path} missing component"


def test_real_config_excludes_lockfile_and_tests() -> None:
    """Every managed package excludes its own `uv.lock` and `tests/` dir.

    A new package added without these exclusions would release on every
    lockfile relock again.
    """
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for path, meta in config["packages"].items():
        excludes = meta.get("exclude-paths", [])
        assert f"{path}/uv.lock" in excludes, f"{path} does not exclude its uv.lock"
        assert f"{path}/tests" in excludes, f"{path} does not exclude its tests"


def test_path_prefix_collision_not_misattributed() -> None:
    """A path that is a string prefix of another must not steal its files."""
    config = {
        "changelog-sections": [{"type": "feat", "section": "Features"}],
        "packages": {
            "libs/acp": {"component": "deepagents-acp"},
            "libs/acp-extra": {"component": "acp-extra"},
        },
    }
    assert touched_components(["libs/acp-extra/main.py"], config) == ["acp-extra"]


def test_package_dir_as_exact_changed_file_not_counted() -> None:
    """A changed entry equal to the package dir itself is not under the package."""
    assert touched_components(["libs/acp"], CONFIG) == []


def test_uppercase_title_is_not_bump_worthy() -> None:
    """Matching is case-sensitive: an uppercase type is treated as non-conventional."""
    assert parse_title("FEAT: x") == (None, False)
    changed = ["libs/acp/pyproject.toml", "libs/code/pyproject.toml"]
    assert find_fanout("FEAT: x", changed, CONFIG) == CLEAN


def test_main_multi_component_stdout_shape(capsys, tmp_path) -> None:
    """Multi-component fan-out goes to stdout as JSON and a summary to stderr.

    The workflow parses stdout as JSON; the stdout/stderr split is load-bearing.
    """
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main(
        "feat(code): model",
        ["libs/code/pyproject.toml", "libs/acp/pyproject.toml"],
        config_path=config_path,
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == {
        "multi_component": ["deepagents-acp", "deepagents-code"],
        "lockfile_only": [],
    }
    assert "multi-component" in captured.err


def test_main_no_offenders_stdout_is_empty_result(capsys, tmp_path) -> None:
    """A clean run still prints empty lists (never empty output)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main("feat(sdk): x", ["libs/acp/uv.lock"], config_path=config_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == CLEAN
    assert captured.err == ""


def test_main_missing_config_returns_2(capsys, tmp_path) -> None:
    """A missing config file fails closed (exit 2), not a silent empty pass."""
    rc = main("feat: x", ["libs/acp/uv.lock"], config_path=tmp_path / "nope.json")
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_malformed_config_returns_2(capsys, tmp_path) -> None:
    """Invalid JSON fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text("{not json", encoding="utf-8")
    rc = main("feat: x", ["libs/acp/uv.lock"], config_path=config_path)
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_empty_packages_returns_2(capsys, tmp_path) -> None:
    """Config drift that empties `packages` fails closed instead of passing all."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(
        json.dumps({"changelog-sections": CONFIG["changelog-sections"], "packages": {}}),
        encoding="utf-8",
    )
    rc = main("feat: x", ["libs/acp/uv.lock"], config_path=config_path)
    assert rc == 2
    assert "packages" in capsys.readouterr().err


def test_main_no_visible_sections_returns_2(capsys, tmp_path) -> None:
    """Config drift that hides every changelog section fails closed."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(
        json.dumps(
            {
                "changelog-sections": [{"type": "chore", "hidden": True}],
                "packages": CONFIG["packages"],
            }
        ),
        encoding="utf-8",
    )
    rc = main("feat: x", ["libs/acp/uv.lock"], config_path=config_path)
    assert rc == 2
    assert "changelog-sections" in capsys.readouterr().err
