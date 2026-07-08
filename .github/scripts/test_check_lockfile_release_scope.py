"""Tests for check_lockfile_release_scope (lockfile-only release fan-out guard)."""

import json
import tomllib

from check_lockfile_release_scope import (
    DEFAULT_CONFIG,
    bump_worthy_types,
    find_offenders,
    main,
    parse_title,
)

# A minimal config mirroring the real release-please-config.json shape: two
# managed packages plus the changelog-sections the bump-worthy set is derived from.
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
        "libs/deepagents": {"component": "deepagents"},
        "libs/cli": {"component": "deepagents-cli"},
        "libs/partners/quickjs": {"component": "langchain-quickjs"},
    },
}


def test_lockfile_only_dependent_is_flagged() -> None:
    """A feat that only churns a dependent's uv.lock is flagged for that package."""
    changed = [
        "libs/deepagents/deepagents/graph.py",
        "libs/deepagents/uv.lock",
        "libs/cli/uv.lock",
        "libs/partners/quickjs/uv.lock",
    ]
    offenders = find_offenders("feat(sdk): surface subagents", changed, CONFIG)
    assert offenders == ["deepagents-cli", "langchain-quickjs"]


def test_owner_with_source_changes_not_flagged() -> None:
    """The package that owns real source edits is never flagged."""
    changed = ["libs/deepagents/deepagents/graph.py", "libs/deepagents/uv.lock"]
    assert find_offenders("feat(sdk): real change", changed, CONFIG) == []


def test_non_bump_title_skipped() -> None:
    """A hidden type (chore) does not cut a release, so nothing is flagged."""
    changed = ["libs/cli/uv.lock", "libs/partners/quickjs/uv.lock"]
    assert find_offenders("chore(deps): relock", changed, CONFIG) == []


def test_breaking_bang_is_flagged() -> None:
    """The `!` breaking shorthand counts as bump-worthy even for an odd type."""
    changed = ["libs/cli/uv.lock"]
    assert find_offenders("refactor(cli)!: drop thing", changed, CONFIG) == [
        "deepagents-cli"
    ]


def test_no_managed_paths_touched() -> None:
    """Changes outside any managed package path produce no offenders."""
    changed = ["examples/deep_research/uv.lock", "README.md"]
    assert find_offenders("feat(docs): example", changed, CONFIG) == []


def test_mixed_lock_and_source_in_dependent_not_flagged() -> None:
    """A dependent with both a source edit and a lockfile bump is a real change."""
    changed = ["libs/cli/deepagents_cli/main.py", "libs/cli/uv.lock"]
    assert find_offenders("feat(cli): real cli change", changed, CONFIG) == []


def test_parse_title_variants() -> None:
    """Type and breaking marker are extracted across common title shapes."""
    assert parse_title("feat(sdk): x") == ("feat", False)
    assert parse_title("fix: x") == ("fix", False)
    assert parse_title("feat(sdk)!: x") == ("feat", True)
    assert parse_title("revert!: x") == ("revert", True)
    assert parse_title("not a conventional title") == (None, False)


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


def test_managed_paths_have_committed_lockfiles() -> None:
    """Every managed package ships a uv.lock, so lockfile churn is a real signal.

    If a package stops committing its lockfile, the guard becomes a no-op for it
    and this test flags that the LOCKFILE_NAMES assumption needs revisiting.
    """
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    repo_root = DEFAULT_CONFIG.parent
    for path, meta in config["packages"].items():
        pyproject = repo_root / path / "pyproject.toml"
        # Sanity: the managed path is a real package directory.
        assert pyproject.exists(), f"{path} ({meta['component']}) has no pyproject.toml"
        with pyproject.open("rb") as f:
            tomllib.load(f)
        assert (repo_root / path / "uv.lock").exists(), (
            f"{path} ({meta['component']}) has no committed uv.lock"
        )


def test_path_prefix_collision_not_misattributed() -> None:
    """A path that is a string prefix of another must not steal its lockfile.

    `_package_files` enforces a `/` boundary; a bare `startswith` would
    misattribute `libs/cli-extra/uv.lock` to `libs/cli`. This pins that.
    """
    config = {
        "changelog-sections": [{"type": "feat", "section": "Features"}],
        "packages": {
            "libs/cli": {"component": "deepagents-cli"},
            "libs/cli-extra": {"component": "cli-extra"},
        },
    }
    offenders = find_offenders("feat: x", ["libs/cli-extra/uv.lock"], config)
    assert offenders == ["cli-extra"]


def test_package_dir_as_exact_changed_file_not_flagged() -> None:
    """A changed entry equal to the package dir itself is not a lockfile."""
    assert find_offenders("feat: x", ["libs/cli"], CONFIG) == []


def test_uppercase_title_is_not_bump_worthy() -> None:
    """Matching is case-sensitive: an uppercase type is treated as non-conventional."""
    assert parse_title("FEAT: x") == (None, False)
    assert find_offenders("FEAT: x", ["libs/cli/uv.lock"], CONFIG) == []


def test_main_happy_path_stdout_is_json_stderr_is_summary(capsys, tmp_path) -> None:
    """main() prints the offenders JSON to stdout and the summary to stderr.

    The workflow parses stdout as JSON; the stdout/stderr split is load-bearing.
    """
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main("feat(sdk): x", ["libs/cli/uv.lock"], config_path=config_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == ["deepagents-cli"]
    assert "Lockfile-only release scope" in captured.err


def test_main_no_offenders_stdout_is_empty_json_array(capsys, tmp_path) -> None:
    """A clean run still prints `[]` to stdout (never empty output)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main("chore(deps): relock", ["libs/cli/uv.lock"], config_path=config_path)
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out) == []


def test_main_missing_config_returns_2(capsys, tmp_path) -> None:
    """A missing config file fails closed (exit 2), not a silent empty pass."""
    rc = main("feat: x", ["libs/cli/uv.lock"], config_path=tmp_path / "nope.json")
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_malformed_config_returns_2(capsys, tmp_path) -> None:
    """Invalid JSON fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text("{not json", encoding="utf-8")
    rc = main("feat: x", ["libs/cli/uv.lock"], config_path=config_path)
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_empty_packages_returns_2(capsys, tmp_path) -> None:
    """Config drift that empties `packages` fails closed instead of passing all."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(
        json.dumps({"changelog-sections": CONFIG["changelog-sections"], "packages": {}}),
        encoding="utf-8",
    )
    rc = main("feat: x", ["libs/cli/uv.lock"], config_path=config_path)
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
    rc = main("feat: x", ["libs/cli/uv.lock"], config_path=config_path)
    assert rc == 2
    assert "changelog-sections" in capsys.readouterr().err
