"""Tests for check_pr_scope_files (PR title scope vs changed package dirs)."""

import json

import pytest
from check_pr_scope_files import (
    DEFAULT_CONFIG,
    _release_files,
    changed_packages,
    declared_packages,
    find_offenders,
    is_release_file,
    is_release_pr_change,
    is_release_title,
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


def test_release_title_with_release_files_bypasses_scope_file_check() -> None:
    """Release PRs can touch generated/version files across package dirs.

    `libs/cli/` is touched and the `deepagents-code` title scope does not cover
    it, so absent the bypass `cli` is a genuine offender. The assertion can
    therefore only pass via the release bypass, not ordinary scope coverage —
    deleting the early-return in `find_offenders` makes this test fail.
    """
    assert is_release_title("release(deepagents-code): 0.1.22")
    assert (
        find_offenders(
            "release(deepagents-code): 0.1.22",
            [
                "libs/code/deepagents_code/_version.py",
                "libs/cli/deepagents_cli/_version.py",
            ],
            CONFIG,
        )
        == []
    )


def test_release_title_with_source_change_still_blocks_mismatch() -> None:
    """An author-controlled release title cannot hide ordinary package edits."""
    assert find_offenders(
        "release(cli): anything",
        ["libs/code/deepagents_code/app.py"],
        CONFIG,
    ) == [{"package": "dcode", "dirs": ["libs/code/"]}]


def test_release_pr_change_requires_release_files_only() -> None:
    """The release bypass is limited to generated/version file patterns."""
    assert is_release_file(".release-please-manifest.json")
    assert is_release_file("libs/code/CHANGELOG.md")
    assert is_release_file("libs/code/pyproject.toml")
    assert is_release_file("libs/code/uv.lock")
    assert is_release_file("libs/code/deepagents_code/_version.py")
    assert is_release_file("libs/partners/daytona/langchain_daytona/_version.py")
    assert not is_release_file("libs/code/deepagents_code/app.py")
    assert not is_release_file("libs/code/docs/CHANGELOG.md")
    assert not is_release_file("libs/code/deepagents_code/nested/_version.py")
    # The manifest match is anchored to the repo root, not matched by name.
    assert not is_release_file("libs/code/.release-please-manifest.json")
    assert not is_release_pr_change(
        "release(cli): anything",
        ["libs/code/deepagents_code/app.py"],
    )
    assert not is_release_pr_change("release(cli): anything", [])


def test_release_file_covers_partner_generated_files() -> None:
    """Partner packages nest one level deeper; their generated files match too."""
    assert is_release_file("libs/partners/daytona/pyproject.toml")
    assert is_release_file("libs/partners/daytona/CHANGELOG.md")
    assert is_release_file("libs/partners/daytona/uv.lock")
    # A non-partner package's CHANGELOG/pyproject live at the package root
    # (depth 3), so the same names at depth 4 are not generated-file locations.
    assert not is_release_file("libs/code/deepagents_code/pyproject.toml")
    assert not is_release_file("libs/code/deepagents_code/CHANGELOG.md")


def test_unmanaged_package_artifacts_do_not_trigger_release_bypass() -> None:
    """Only package roots declared in release-please config are release artifacts."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = ["libs/evals/pyproject.toml"]

    assert not is_release_file("libs/evals/pyproject.toml")
    assert not is_release_file("libs/evals/deepagents_evals/_version.py")
    assert not is_release_pr_change("release(deepagents-code): 0.1.22", changed)
    assert find_offenders("release(deepagents-code): 0.1.22", changed, config) == [
        {"package": "evals", "dirs": ["libs/evals/"]}
    ]


def test_release_files_validates_structure(tmp_path) -> None:
    """`_release_files` fails closed on every malformed release-config shape."""
    cases = [
        ({"packages": {}}, "no non-empty 'packages' map"),
        ({"packages": {"libs/code": "notadict"}}, "malformed"),
        ({"packages": {"libs/code": {"extra-files": "x"}}}, "malformed 'extra-files'"),
        ({"packages": {"libs/code": {"extra-files": [1]}}}, "non-string extra file"),
    ]
    for i, (obj, pattern) in enumerate(cases):
        path = tmp_path / f"release-{i}.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            _release_files(path)

    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="could not read release-please config"):
        _release_files(missing)


def test_release_pr_change_fails_closed_on_malformed_release_config(tmp_path) -> None:
    """The bypass predicate raises (fails closed) on a broken release config."""
    bad = tmp_path / "release-please-config.json"
    bad.write_text(json.dumps({"packages": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no non-empty 'packages' map"):
        is_release_pr_change(
            "release(deepagents-code): 0.1.22",
            ["libs/code/deepagents_code/_version.py"],
            release_config_path=bad,
        )


def test_release_uv_lock_only_still_validates_release_config(tmp_path) -> None:
    """An all-`uv.lock` release PR still reads/validates the release config.

    `uv.lock` short-circuits in `is_release_file`, but the gate must not stand
    down without first validating the release config — otherwise a broken config
    would silently let the bypass fire for a lockfile-only release PR.
    """
    bad = tmp_path / "release-please-config.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read release-please config"):
        is_release_pr_change(
            "release(deepagents-code): 0.1.22",
            ["libs/code/uv.lock", "libs/cli/uv.lock"],
            release_config_path=bad,
        )


def test_main_malformed_release_config_fails_closed_and_attributes_correctly(
    capsys, tmp_path
) -> None:
    """A broken release config returns 2 and is not mis-reported as labeler config."""
    labeler = tmp_path / "pr-labeler-config.json"
    labeler.write_text(json.dumps(CONFIG), encoding="utf-8")
    release = tmp_path / "release-please-config.json"
    release.write_text(json.dumps({"packages": {}}), encoding="utf-8")

    rc = main(
        "release(deepagents-code): 0.1.22",
        ["libs/code/deepagents_code/_version.py"],
        config_path=labeler,
        release_config_path=release,
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "release-please config" in captured.err
    assert "PR labeler config" not in captured.err


def test_main_uv_lock_only_fails_closed_on_bad_release_config(capsys, tmp_path) -> None:
    """All-`uv.lock` release PR returns 2 when the release config cannot be read."""
    labeler = tmp_path / "pr-labeler-config.json"
    labeler.write_text(json.dumps(CONFIG), encoding="utf-8")
    release = tmp_path / "release-please-config.json"
    release.write_text("{not json", encoding="utf-8")

    rc = main(
        "release(deepagents-code): 0.1.22",
        ["libs/code/uv.lock", "libs/cli/uv.lock"],
        config_path=labeler,
        release_config_path=release,
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "release-please config" in captured.err


def test_release_pr_change_covers_repo_wide_lockfiles() -> None:
    """Real partner release PRs regenerate `uv.lock` under `examples/` too.

    Regression for a partner release PR (e.g. `release(langchain-quickjs)`)
    whose changeset includes `examples/*/uv.lock`: the bypass must still fire so
    the release PR is not blocked by cross-dir lockfile churn its title scope
    cannot cover.
    """
    assert is_release_file("examples/async-subagent-server/uv.lock")
    assert is_release_file("examples/llm-wiki/uv.lock")
    assert is_release_pr_change(
        "release(langchain-quickjs): 0.3.1",
        [
            ".release-please-manifest.json",
            "examples/async-subagent-server/uv.lock",
            "examples/llm-wiki/uv.lock",
            "libs/code/uv.lock",
            "libs/evals/uv.lock",
            "libs/partners/quickjs/CHANGELOG.md",
            "libs/partners/quickjs/pyproject.toml",
            "libs/partners/quickjs/uv.lock",
        ],
    )


def test_release_title_ignores_lockfile_churn_for_other_package_dirs() -> None:
    """Release PRs ignore lockfiles in unrelated package dirs when checking scope."""
    changed = [
        ".github/RELEASING.md",
        "libs/code/AGENTS.md",
        "libs/code/CHANGELOG.md",
        "libs/code/deepagents_code/_version.py",
        "libs/code/pyproject.toml",
        "libs/code/uv.lock",
        "libs/evals/uv.lock",
        "libs/talon/uv.lock",
    ]
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert not is_release_pr_change("release(deepagents-code): 0.1.23", changed)
    assert find_offenders("release(deepagents-code): 0.1.23", changed, config) == []


def test_release_title_with_mixed_files_does_not_bypass() -> None:
    """A release artifact cannot launder an accompanying source edit."""
    changed = ["libs/code/CHANGELOG.md", "libs/code/deepagents_code/app.py"]
    assert not is_release_pr_change("release(cli): 0.1.22", changed)
    assert find_offenders("release(cli): 0.1.22", changed, CONFIG) == [
        {"package": "dcode", "dirs": ["libs/code/"]}
    ]


def test_release_bypass_does_not_validate_component() -> None:
    """The bypass keys off title shape + files, not a real component name.

    Pins that an unrecognized component still bypasses when every file is a
    release artifact, so a future tightening (validating the component against
    real package names) is a conscious change rather than a silent one.
    """
    assert (
        find_offenders(
            "release(not-a-real-scope): 9.9.9", ["libs/code/uv.lock"], CONFIG
        )
        == []
    )


def test_is_release_title_boundaries() -> None:
    """Only `release(<scope>):` shapes trigger the bypass title gate."""
    assert is_release_title("release(cli): 1.0.0")
    assert is_release_title("release(): 1.0.0")  # empty scope still matches
    assert not is_release_title("release: 1.0.0")  # scope required
    assert not is_release_title("release(scope)!: 1.0.0")  # breaking marker excluded
    assert not is_release_title("  release(cli): 1.0.0")  # anchored; no leading ws
    assert not is_release_title("Release(cli): 1.0.0")  # case-sensitive


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


def test_main_release_bypass_validates_labeler_config(capsys, tmp_path) -> None:
    """Release artifact bypasses still fail closed on malformed labeler config."""
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(
        json.dumps(
            {
                "scopeToLabel": CONFIG["scopeToLabel"],
                "fileRules": [{"label": "dcode", "prefix": 123}],
            }
        ),
        encoding="utf-8",
    )
    rc = main(
        "release(deepagents-code): 0.1.22",
        ["libs/code/uv.lock"],
        config_path=config_path,
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "fileRules" in captured.err


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


def test_main_clean_stdout_is_empty_json_array(capsys, tmp_path) -> None:
    """A clean PR prints exactly `[]` and nothing on stderr.

    The calling workflow consumes raw stdout: its fail-closed JSON parse blocks
    on empty or non-array output, and its detector-absent branch hard-codes the
    same `[]` sentinel. Pin the exact wire contract here so a future change to
    the clean-path output can't silently desync the gate from the script.
    """
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main(
        "fix(cli): repair startup",
        ["libs/cli/deepagents_cli/main.py"],
        config_path=config_path,
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.strip() == "[]"
    assert captured.err == ""


def test_main_release_bypass_emits_notice(capsys, tmp_path) -> None:
    """A fired release bypass surfaces a `::notice::` so it is not silent.

    The gate standing down must be distinguishable from a genuinely clean PR in
    the Checks UI. The notice goes to stderr so it never corrupts the JSON
    offenders the workflow captures from stdout.
    """
    config_path = tmp_path / "pr-labeler-config.json"
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    rc = main(
        "release(deepagents-code): 0.1.22",
        ["libs/code/uv.lock", "libs/evals/uv.lock"],
        config_path=config_path,
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out.strip() == "[]"
    assert "::notice::" in captured.err


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
