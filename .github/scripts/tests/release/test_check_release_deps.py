"""Tests for release dependency resolution helper."""

import json
import subprocess

import pytest
import tomllib
from check_release_deps import (
    BYPASS_LABEL,
    COMMENT_MARKER,
    FOLLOWUP_LIMIT,
    CheckResult,
    FollowUpConflict,
    PyPIRequestError,
    ResolverFailure,
    _affected_extras,
    _comment_body,
    _compare_with_published,
    _failure_markdown,
    _follow_up_markdown,
    _ranges_may_overlap,
    _relevant_log,
    _toml_value,
    _write_output,
    build_resolver_manifest,
    check_release_dependencies,
    find_follow_up_conflicts,
    is_transient_resolver_error,
    load_release_packages,
    main,
    run_check,
    run_resolver,
)
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def test_build_resolver_manifest_drops_sources_and_preserves_uv_keys() -> None:
    data = {
        "project": {
            "name": "deepagents-code",
            "version": "0.2.0",
            "requires-python": ">=3.11,<4.0",
            "dependencies": [
                "deepagents==0.7.0",
                "langchain>=1.0,<2.0",
                "deepagents-acp>=0.0.8,<0.0.9",
            ],
            "optional-dependencies": {
                "sandbox": ["langchain-daytona>=0.0.8,<0.1.0"],
                "quickjs": ["langchain-quickjs>=0.1.4,<0.2.0"],
            },
        },
        "tool": {
            "uv": {
                "prerelease": "allow",
                "constraint-dependencies": ["example<2"],
                "override-dependencies": ["other==1.0"],
                "sources": {"deepagents": {"path": "../deepagents"}},
            }
        },
    }

    parsed = tomllib.loads(build_resolver_manifest(data))

    assert parsed["project"]["dependencies"] == [
        "deepagents==0.7.0",
        "langchain>=1.0,<2.0",
        "deepagents-acp>=0.0.8,<0.0.9",
    ]
    assert parsed["project"]["optional-dependencies"]["sandbox"] == [
        "langchain-daytona>=0.0.8,<0.1.0"
    ]
    assert parsed["project"]["optional-dependencies"]["quickjs"] == [
        "langchain-quickjs>=0.1.4,<0.2.0"
    ]
    assert parsed["project"]["requires-python"] == ">=3.11,<4.0"
    assert parsed["tool"]["uv"]["prerelease"] == "allow"
    assert parsed["tool"]["uv"]["constraint-dependencies"] == ["example<2"]
    assert parsed["tool"]["uv"]["override-dependencies"] == ["other==1.0"]
    assert "sources" not in parsed["tool"]["uv"]


def test_build_resolver_manifest_requires_project_table() -> None:
    with pytest.raises(ValueError, match="no \\[project\\] table"):
        build_resolver_manifest({"project": "not-a-table"})


def test_check_release_dependencies_writes_each_manifest_as_pyproject(
    monkeypatch,
    tmp_path,
) -> None:
    manifests = [
        "libs/code/pyproject.toml",
        "libs/partners/daytona/pyproject.toml",
    ]
    content = """
[project]
name = "example"
version = "0.1.0"
dependencies = []
""".strip()
    for manifest in manifests:
        path = tmp_path / manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    resolver_paths = []

    def run_resolver(manifest_path, _log_path) -> bool:
        resolver_paths.append(manifest_path)
        assert manifest_path.name == "pyproject.toml"
        assert manifest_path.exists()
        assert manifest_path.read_text(encoding="utf-8") == content
        return True

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {
            "libs/code": "deepagents-code",
            "libs/partners/daytona": "langchain-daytona",
        },
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: manifests,
    )
    monkeypatch.setattr(
        "check_release_deps.build_resolver_manifest", lambda _data: content
    )
    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    result = check_release_dependencies("base-sha", "head-sha")

    assert result.changed is True
    assert result.failures == ()
    assert len(resolver_paths) == len(manifests)
    assert len({path.parent for path in resolver_paths}) == len(manifests)


def test_check_release_dependencies_noop_when_no_manifests_changed(monkeypatch) -> None:
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests", lambda _base, _head, _packages: []
    )

    def run_resolver(_manifest, _log) -> bool:
        pytest.fail("resolver should not run when nothing changed")

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    result = check_release_dependencies("base-sha", "head-sha")

    assert result.changed is False
    assert result.failures == ()
    assert result.followups == ()


def test_run_resolver_allows_prereleases_for_all_extras(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "example"
version = "0.1.0"
dependencies = []
""".strip(),
        encoding="utf-8",
    )
    log = tmp_path / "resolver.log"
    commands = []

    def subprocess_run(args, **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="resolved\n")

    monkeypatch.setattr("check_release_deps.subprocess.run", subprocess_run)

    assert run_resolver(manifest, log) is True

    command = commands[0]
    assert command[:3] == ["uv", "pip", "compile"]
    assert "--no-sources" in command
    assert "--all-extras" in command
    assert command[command.index("--prerelease") + 1] == "allow"
    assert command[-1] == str(manifest)
    assert log.read_text(encoding="utf-8") == "resolved\n"


def test_load_release_packages_resolves_name_then_component_then_path(tmp_path) -> None:
    config = tmp_path / "release-please-config.json"
    config.write_text(
        json.dumps(
            {
                "packages": {
                    "libs/deepagents": {
                        "package-name": "deepagents",
                        "component": "sdk",
                    },
                    "libs/cli": {"component": "cli"},
                    "libs/acp": {},
                    "libs/skip": "not-a-dict",
                }
            }
        ),
        encoding="utf-8",
    )

    packages = load_release_packages(config)

    assert packages == {
        "libs/deepagents": "deepagents",
        "libs/cli": "cli",
        "libs/acp": "libs/acp",
    }


def test_load_release_packages_rejects_empty_packages(tmp_path) -> None:
    config = tmp_path / "release-please-config.json"
    config.write_text(json.dumps({"packages": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="no packages map"):
        load_release_packages(config)


def test_toml_value_renders_scalars_and_collections() -> None:
    assert _toml_value("hello") == '"hello"'
    # bool must render before int (bool is an int subclass).
    assert _toml_value(value=True) == "true"
    assert _toml_value(value=False) == "false"
    assert _toml_value(7) == "7"
    assert _toml_value([]) == "[]"
    assert _toml_value({"key": "val"}) == '{ key = "val" }'
    assert tomllib.loads(f"x = {_toml_value(['a', 'b'])}")["x"] == ["a", "b"]


def test_toml_value_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="Unsupported TOML value"):
        _toml_value(object())


def test_main_requires_both_shas(monkeypatch) -> None:
    monkeypatch.delenv("BASE_SHA", raising=False)
    monkeypatch.delenv("HEAD_SHA", raising=False)

    assert main() == 2


def test_main_fails_closed_on_unexpected_error(monkeypatch) -> None:
    monkeypatch.setenv("BASE_SHA", "base-sha")
    monkeypatch.setenv("HEAD_SHA", "head-sha")

    def boom(_base, _head, *, acked=False, fetcher=None) -> int:
        msg = "kaboom"
        raise RuntimeError(msg)

    monkeypatch.setattr("check_release_deps.run_check", boom)

    assert main() == 2


def test_transient_resolver_error_patterns() -> None:
    assert is_transient_resolver_error("failed to fetch https://pypi.org/simple/pkg")
    assert is_transient_resolver_error("HTTP 503 service unavailable")
    assert is_transient_resolver_error("error sending request for url")
    assert is_transient_resolver_error("the connection was reset")
    assert is_transient_resolver_error("request timed out")
    assert is_transient_resolver_error("status code: 429")
    assert is_transient_resolver_error("HTTP 429 Too Many Requests")
    assert not is_transient_resolver_error(
        "No solution found when resolving dependencies"
    )
    assert not is_transient_resolver_error("version conflict for package foo")


def _optional_deps_manifest(name: str, optional: dict[str, list[str]]) -> dict:
    return {"project": {"name": name, "optional-dependencies": optional}}


def test_affected_extras_direct_hit() -> None:
    data = _optional_deps_manifest(
        "example",
        {"sandbox": ["langchain-daytona>=0.1"], "quickjs": ["langchain-quickjs>=0.1"]},
    )
    log = "no solution found: langchain-daytona>=0.1 is not available"

    assert _affected_extras(data, log) == ("sandbox",)


def test_affected_extras_propagates_transitive_self_extra() -> None:
    # `all` pulls in the package's own `sandbox` extra, so a sandbox conflict
    # marks `all` affected too — and the fixpoint must reach it regardless of the
    # declaration order in the dict.
    data = _optional_deps_manifest(
        "example",
        {"all": ["example[sandbox]"], "sandbox": ["langchain-daytona>=0.1"]},
    )
    log = "langchain-daytona>=0.1 has no matching distribution"

    assert _affected_extras(data, log) == ("all", "sandbox")


def test_affected_extras_word_boundary_avoids_false_positive() -> None:
    # `click` must not match inside `clickhouse-driver`, nor `requests` inside
    # `requests-toolbelt`.
    data = _optional_deps_manifest(
        "example",
        {"cli": ["click>=8"], "http": ["requests>=2"]},
    )
    log = "resolved clickhouse-driver==1.0 and requests-toolbelt==1.0"

    assert _affected_extras(data, log) == ()


def test_affected_extras_returns_empty_for_malformed_project() -> None:
    assert _affected_extras({"project": "not-a-table"}, "log") == ()
    assert _affected_extras({"project": {"name": 123}}, "log") == ()


def test_relevant_log_passes_through_short_logs() -> None:
    assert _relevant_log("line1\nline2\nline3", max_lines=10) == "line1\nline2\nline3"


def test_relevant_log_keeps_last_lines_with_omission_header() -> None:
    log = "\n".join(f"line{index}" for index in range(100))

    result = _relevant_log(log, max_lines=10).splitlines()

    assert result[0] == "... 90 earlier lines omitted ..."
    assert result[-1] == "line99"
    assert len(result) == 11  # omission header + last 10 lines


def test_write_output_appends_heredoc(monkeypatch, tmp_path) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    _write_output("failed", "true")

    assert output.read_text(encoding="utf-8") == (
        "failed<<__FAILED_EOF__\ntrue\n__FAILED_EOF__\n"
    )


def test_write_output_extends_delimiter_on_collision(monkeypatch, tmp_path) -> None:
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    # A resolver log that embeds the default delimiter must not break out of the
    # heredoc block; the delimiter grows until it no longer appears in the value.
    value = "line\n__COMMENT_BODY_EOF__\nmore"

    _write_output("comment_body", value)

    written = output.read_text(encoding="utf-8")
    assert written.startswith("comment_body<<__COMMENT_BODY_EOF___\n")
    assert written.endswith("\n__COMMENT_BODY_EOF___\n")
    # The embedded default delimiter survives verbatim inside the block.
    assert value in written


def test_write_output_noop_without_env(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    # Must not raise when run outside GitHub Actions (e.g. local debugging).
    _write_output("failed", "true")


def _failure(
    *,
    transient: bool = False,
    affected_extras: tuple[str, ...] = (),
    log: str = "No solution found",
) -> ResolverFailure:
    return ResolverFailure(
        manifest_path="libs/code/pyproject.toml",
        package_name="deepagents-code",
        log=log,
        transient=transient,
        affected_extras=affected_extras,
    )


def _conflict(
    *,
    manifest_path: str = "libs/code/pyproject.toml",
    package_name: str = "deepagents-code",
    dependency_name: str = "deepagents",
    requirement: str = "==0.7.0",
    published_constraint: str | None = ">=0.6,<0.7",
) -> FollowUpConflict:
    return FollowUpConflict(
        manifest_path=manifest_path,
        package_name=package_name,
        dependency_name=dependency_name,
        requirement=requirement,
        published_constraint=published_constraint,
        reason=f"published 0.6.12 constrains it to `{published_constraint}`",
    )


def test_failure_markdown_marker_gated_on_flag() -> None:
    failure = _failure()

    with_marker = _failure_markdown([failure], include_marker=True)
    without_marker = _failure_markdown([failure], include_marker=False)

    assert with_marker.startswith(COMMENT_MARKER)
    assert COMMENT_MARKER not in without_marker


def test_failure_markdown_lists_affected_extras_and_bypass_guidance() -> None:
    failure = _failure(affected_extras=("sandbox", "all"))

    markdown = _failure_markdown([failure], include_marker=False)

    assert "`deepagents-code[sandbox]`" in markdown
    assert "`deepagents-code[all]`" in markdown
    # Non-transient failures point at the bypass label.
    assert BYPASS_LABEL in markdown


def test_failure_markdown_transient_suppresses_bypass_guidance() -> None:
    failure = _failure(transient=True, log="failed to fetch")

    markdown = _failure_markdown([failure], include_marker=True)

    assert "transient network" in markdown
    # A purely transient failure must not nudge toward the bypass label.
    assert BYPASS_LABEL not in markdown


def test_check_release_dependencies_reports_failure_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = "libs/code/pyproject.toml"
    content = """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["langchain-daytona==9.9.9"]
""".strip()
    path = tmp_path / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    output = tmp_path / "github_output"
    summary = tmp_path / "github_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text(
            "No solution found: langchain-daytona==9.9.9 is not available",
            encoding="utf-8",
        )
        return False

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    assert run_check("base-sha", "head-sha", acked=False) == 1

    written = output.read_text(encoding="utf-8")
    assert "failed<<" in written
    assert "\ntrue\n" in written
    assert COMMENT_MARKER in written
    assert summary.read_text(encoding="utf-8").startswith(
        "## Release dependency resolution failed"
    )


def test_check_release_dependencies_transient_failure_emits_no_comment(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = "libs/code/pyproject.toml"
    content = """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["langchain-daytona>=0.1"]
""".strip()
    path = tmp_path / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text("error sending request: connection reset", encoding="utf-8")
        return False

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    # The job still fails closed (exit 1) on a transient error, but emits an
    # empty comment_body so the workflow keeps any existing comment untouched
    # rather than posting transient noise.
    assert run_check("base-sha", "head-sha", acked=False) == 1

    written = output.read_text(encoding="utf-8")
    assert "\ntrue\n" in written
    assert COMMENT_MARKER not in written


# --- Follow-up conflict analysis -------------------------------------------


def _pypi_payload(version: str, requires_dist: list[str] | None) -> dict[str, object]:
    return {
        "info": {"name": "pkg", "version": version, "requires_dist": requires_dist},
        "releases": {version: [{"requires_dist": None, "yanked": False}]},
    }


def _write_manifest(tmp_path, manifest: str, content: str) -> None:
    path = tmp_path / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_find_follow_up_conflicts_flags_published_upper_bound(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents==0.7.0"]
""".strip(),
    )
    payloads = {"deepagents-code": _pypi_payload("0.1.49", ["deepagents==0.6.12"])}

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest],
        ["libs/deepagents", "libs/code"],
        fetcher=lambda name: payloads[name],
    )

    assert unavailable == ()
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.dependency_name == "deepagents"
    assert conflict.requirement == "==0.7.0"
    assert conflict.published_constraint == "==0.6.12"
    assert "constrains it to" in conflict.reason


def test_find_follow_up_conflicts_flags_conflicting_published_range(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents>=0.7,<0.8"]
""".strip(),
    )

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )

    def fetch(name: str) -> dict[str, object]:
        return {"deepagents-code": _pypi_payload("0.1.49", ["deepagents>=0.6,<0.7"])}[
            name
        ]

    conflicts, _ = find_follow_up_conflicts(
        [manifest], ["libs/deepagents", "libs/code"], fetcher=fetch
    )
    assert len(conflicts) == 1
    assert conflicts[0].dependency_name == "deepagents"


def test_find_follow_up_conflicts_covers_multiple_partners(
    monkeypatch, tmp_path
) -> None:
    """A core release finds every reverse dependent across managed manifests."""
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.49"
dependencies = ["deepagents==0.7.0"]

[project.optional-dependencies]
daytona = ["langchain-daytona>=0.1.0"]
modal = ["langchain-modal>=0.1.0"]

[tool.uv.sources]
deepagents = { path = "../deepagents", editable = true }
langchain-daytona = { path = "../partners/daytona", editable = true }
langchain-modal = { path = "../partners/modal", editable = true }
""".strip(),
    )
    for partner in ("daytona", "modal"):
        _write_manifest(
            tmp_path,
            f"libs/partners/{partner}/pyproject.toml",
            f"""
[project]
name = "langchain-{partner}"
version = "0.0.8"
dependencies = ["deepagents>=0.7,<0.8"]

[tool.uv.sources]
deepagents = {{ path = "../../deepagents", editable = true }}
""".strip(),
        )
    payloads = {
        "deepagents-code": _pypi_payload(
            "0.1.48",
            [
                "deepagents==0.6.12",
                "langchain-daytona>=0.0.8,<0.1.0",
                "langchain-modal>=0.0.8,<0.1.0",
            ],
        ),
        "langchain-daytona": _pypi_payload("0.0.8", ["deepagents>=0.6,<0.7"]),
        "langchain-modal": _pypi_payload("0.0.8", ["deepagents>=0.6,<0.7"]),
    }

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {
            "libs/code": "deepagents-code",
            "libs/deepagents": "deepagents",
            "libs/partners/daytona": "langchain-daytona",
            "libs/partners/modal": "langchain-modal",
        },
    )

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest],
        [
            "libs/deepagents",
            "libs/code",
            "libs/partners/daytona",
            "libs/partners/modal",
        ],
        fetcher=lambda name: payloads[name],
    )

    assert unavailable == ()
    declaring_packages = {conflict.package_name for conflict in conflicts}
    assert {
        "deepagents-code",
        "langchain-daytona",
        "langchain-modal",
    } <= declaring_packages
    partner_conflicts = [
        conflict
        for conflict in conflicts
        if conflict.package_name.startswith("langchain-")
    ]
    assert all(
        conflict.dependency_name == "deepagents"
        and conflict.published_constraint == "<0.7,>=0.6"
        for conflict in partner_conflicts
    )


def test_find_follow_up_conflicts_keeps_local_edges_and_skips_self_requirements(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = ["deepagents==0.7.0"]
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = [
    "deepagents-code==0.1.0",
    "deepagents==0.7.0",
    "deepagents-acp>=0.1",
]

[tool.uv.sources]
deepagents = { path = "../deepagents" }
""".strip(),
    )
    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {
            "libs/code": "deepagents-code",
            "libs/deepagents": "deepagents",
            "libs/acp": "deepagents-acp",
        },
    )

    fetched = []

    def fetch(name: str) -> dict[str, object]:
        fetched.append(name)
        return _pypi_payload("0.1.0", ["deepagents==0.6.12", "deepagents-acp>=0.1"])

    conflicts, _ = find_follow_up_conflicts(
        [manifest], ["libs/deepagents", "libs/code"], fetcher=fetch
    )

    assert fetched == ["deepagents-code"]
    assert [conflict.dependency_name for conflict in conflicts] == ["deepagents"]


def test_find_follow_up_conflicts_uses_declaring_package_metadata(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/acp/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents-acp"
version = "0.1.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents-acp>=0.1"]
""".strip(),
    )
    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/acp": "deepagents-acp"},
    )

    fetched = []

    def fetch(name: str) -> dict[str, object]:
        fetched.append(name)
        return _pypi_payload("0.1.0", ["deepagents-acp>=0.1"])

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest], ["libs/acp", "libs/code"], fetcher=fetch
    )

    assert fetched == ["deepagents-code"]
    assert conflicts == []
    assert unavailable == ()


def test_find_follow_up_conflicts_reports_unavailable_without_claiming_releases(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents>=0.7"]
""".strip(),
    )
    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )

    def fetch(name: str) -> dict[str, object]:
        msg = f"PyPI request failed for {name}: boom"
        raise PyPIRequestError(msg, transient=True)

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest], ["libs/deepagents", "libs/code"], fetcher=fetch
    )

    assert conflicts == []
    assert unavailable == ("deepagents-code",)


def test_failure_markdown_includes_follow_up_section() -> None:
    failure = _failure(log="No solution found: deepagents==0.7.0 is unavailable")
    conflict = FollowUpConflict(
        manifest_path="libs/code/pyproject.toml",
        package_name="deepagents-code",
        dependency_name="deepagents",
        requirement="==0.7.0",
        published_constraint=">=0.6,<0.7",
        reason="published 0.6.12 constrains it to `>=0.6,<0.7`",
    )

    markdown = _failure_markdown([failure], include_marker=True, followups=[conflict])

    assert "### Follow-up releases needed" in markdown
    assert "`deepagents-code`" in markdown
    assert "`deepagents`" in markdown
    assert "`==0.7.0`" in markdown
    assert "`>=0.6,<0.7`" in markdown
    # The section lands above the raw resolver log.
    assert markdown.index("### Follow-up releases needed") < markdown.index("```text")


def test_failure_markdown_lists_all_partner_conflicts_not_only_first(
    tmp_path, monkeypatch
) -> None:
    failure = _failure(log="No solution found when resolving dependencies")
    conflicts = [
        FollowUpConflict(
            manifest_path="libs/code/pyproject.toml",
            package_name="deepagents-code",
            dependency_name=name,
            requirement=">=0.1.0",
            published_constraint=">=0.6,<0.7",
            reason="published 0.0.8 constrains it to `>=0.6,<0.7`",
        )
        for name in ("langchain-daytona", "langchain-modal", "langchain-runloop")
    ]

    markdown = _failure_markdown([failure], include_marker=False, followups=conflicts)

    for name in ("langchain-daytona", "langchain-modal", "langchain-runloop"):
        assert f"`{name}`" in markdown


def test_publish_candidates_section_stays_version_agnostic() -> None:
    failure = _failure()
    conflict = FollowUpConflict(
        manifest_path="libs/code/pyproject.toml",
        package_name="deepagents-code",
        dependency_name="deepagents",
        requirement="==0.7.0",
        published_constraint=">=0.6,<0.7",
        reason="published 0.6.12 constrains it to `>=0.6,<0.7`",
    )

    markdown = _failure_markdown([failure], include_marker=True, followups=[conflict])

    # Headings and prose never freeze a specific version line into the template;
    # only the data-driven table cells carry concrete pins.
    headings = [line for line in markdown.splitlines() if line.startswith("#")]
    assert all("0.7" not in heading and "0.6" not in heading for heading in headings)


def test_ack_soft_run_writes_sticky_and_exits_zero(monkeypatch, tmp_path) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents==0.7.0"]
""".strip(),
    )
    output = tmp_path / "github_output"
    summary = tmp_path / "github_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text(
            "No solution found: deepagents==0.7.0 is not available",
            encoding="utf-8",
        )
        return False

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)
    monkeypatch.setattr(
        "check_release_deps.fetch_pypi_json",
        lambda name: _pypi_payload("0.1.49", ["deepagents==0.6.12"]),
    )

    assert run_check("base-sha", "head-sha", acked=True) == 0

    written = output.read_text(encoding="utf-8")
    assert "\nfalse\n" in written
    assert COMMENT_MARKER in written
    body_start = written.index(COMMENT_MARKER)
    body = written[body_start:]
    assert "acknowledged bypass" in body
    assert "### Follow-up releases needed after merge" in body
    assert "### Resolution failures (reported, not blocking)" in body
    assert "`deepagents`" in body
    summary_text = summary.read_text(encoding="utf-8")
    assert "acknowledged bypass" in summary_text
    assert COMMENT_MARKER not in summary_text


def test_ack_soft_run_clean_resolve_reports_no_followups(monkeypatch, tmp_path) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents>=0.6"]
""".strip(),
    )
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text("resolved\n", encoding="utf-8")
        return True

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    def fetch(name: str) -> dict[str, object]:
        # Published deepagents-code already allows the required deepagents range.
        return _pypi_payload("0.1.49", ["deepagents>=0.6,<0.8"])

    monkeypatch.setattr("check_release_deps.fetch_pypi_json", fetch)

    assert run_check("base-sha", "head-sha", acked=True) == 0

    written = output.read_text(encoding="utf-8")
    assert "\nfalse\n" in written
    # Nothing to report, so the body is empty: the workflow reads that as
    # "clear any stale sticky" rather than leaving a comment behind that only
    # restates that the label is applied.
    assert COMMENT_MARKER not in written
    assert "Follow-up releases" not in written


def test_transient_unavailable_pypi_does_not_claim_needs_release(
    monkeypatch, tmp_path
) -> None:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        """
[project]
name = "deepagents-code"
version = "0.1.0"
dependencies = ["deepagents>=0.7"]
""".strip(),
    )
    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )

    def fetch(name: str) -> dict[str, object]:
        msg = f"PyPI request failed for {name}: connection reset"
        raise PyPIRequestError(msg, transient=True)

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest], ["libs/deepagents", "libs/code"], fetcher=fetch
    )
    result = CheckResult(
        changed=True,
        followups=tuple(conflicts),
        unavailable=unavailable,
    )

    body = _comment_body(result, acked=True)

    assert conflicts == []
    assert unavailable == ("deepagents-code",)
    assert "### Follow-up releases needed" not in body
    assert "could not be determined" in body
    assert "neither confirmed clean nor confirmed to need a release" in body
    assert "deepagents" in body


def test_ack_clean_run_emits_empty_body_so_stale_sticky_is_cleared() -> None:
    """An acknowledged run with nothing to report clears the comment.

    The workflow deletes the sticky on an empty body, so a release PR that
    resolves cleanly under the label does not keep a comment whose only content
    is "the label is applied."
    """
    result = CheckResult(changed=True)

    assert _comment_body(result, acked=True) == ""


def test_ack_comment_explains_bypass_when_there_is_something_to_report() -> None:
    result = CheckResult(changed=True, followups=(_conflict(),))

    body = _comment_body(result, acked=True)

    assert body.startswith(COMMENT_MARKER)
    assert "acknowledged bypass" in body
    assert BYPASS_LABEL in body
    assert "coordinated release order" in body


def test_ack_transient_failure_with_followups_does_not_claim_all_clear() -> None:
    """The transient-only reassurance must not sit above a conflict table.

    A transient resolver error says nothing about sibling metadata, so claiming
    "no metadata problems were detected" while listing follow-ups would
    contradict the table directly above it.
    """
    result = CheckResult(
        changed=True,
        failures=(_failure(transient=True),),
        followups=(_conflict(),),
    )

    body = _comment_body(result, acked=True)

    assert "Follow-up releases needed" in body
    assert "no metadata problems were detected" not in body
    assert "the follow-ups above are still outstanding" in body


def test_unacked_clean_resolve_reports_followups_and_stays_green() -> None:
    """Follow-up debt survives removing the bypass label.

    Resolution passing only proves the changed package installs. If a
    reverse-dependent's published metadata still caps the new line, dropping the
    report here would delete the sticky and lose the debt while it is still real.
    """
    result = CheckResult(changed=True, followups=(_conflict(),))

    body = _comment_body(result, acked=False)

    assert body.startswith(COMMENT_MARKER)
    assert "Release dependency follow-ups" in body
    assert "Follow-up releases needed" in body


def test_check_result_rejects_findings_when_nothing_changed() -> None:
    with pytest.raises(ValueError, match="cannot carry findings"):
        CheckResult(changed=False, failures=(_failure(transient=False),))


# --- Range overlap heuristic ------------------------------------------------


@pytest.mark.parametrize(
    ("head", "published", "disjoint"),
    [
        # Genuinely disjoint: the published cap excludes the whole new line.
        ("==0.7.0", ">=0.6,<0.7", True),
        (">=0.7", "<0.7", True),
        (">0.7", "<=0.7", True),
        ("~=0.8.0", ">=0.6,<0.7", True),
        ("==1", ">=2", True),
        # Overlapping, so no follow-up release is owed.
        ("==0.7.0", ">=0.6,<0.8", False),
        (">=0.7", ">=0.7", False),
        ("==0.7.0", "==0.7.0", False),
        ("!=0.6", "==0.7.0", False),
        # Wildcards and `!=` yield no probes. Sampling cannot prove disjointness,
        # so these must come back as "may overlap" rather than fabricating a row.
        ("==0.7.*", "==0.7.*", False),
        ("==0.7.*", "!=0.6", False),
        ("!=0.6", "!=0.7", False),
    ],
)
def test_ranges_may_overlap_only_reports_disjoint_when_provable(
    head: str, published: str, disjoint: bool
) -> None:
    """A probe sample can prove overlap but never disjointness.

    Returning "may overlap" suppresses a follow-up row, so an inconclusive
    comparison has to fall on the noisy side (report) rather than the silent one.
    """
    assert (
        _ranges_may_overlap(SpecifierSet(head), SpecifierSet(published)) is not disjoint
    )


def test_identical_wildcard_pins_do_not_fabricate_a_conflict() -> None:
    """Regression: no probes must not read as "disjoint"."""
    conflict = _compare_with_published(
        "libs/code/pyproject.toml",
        "deepagents-code",
        Requirement("deepagents==0.7.*"),
        "0.1.49",
        [Requirement("deepagents==0.7.*")],
    )

    assert conflict is None


# --- PyPI metadata edge cases ----------------------------------------------


def _followup_env(monkeypatch, tmp_path) -> str:
    manifest = "libs/deepagents/pyproject.toml"
    _write_manifest(
        tmp_path,
        manifest,
        '[project]\nname = "deepagents"\nversion = "0.7.0"\ndependencies = []',
    )
    _write_manifest(
        tmp_path,
        "libs/code/pyproject.toml",
        '[project]\nname = "deepagents-code"\nversion = "0.1.0"\n'
        'dependencies = ["deepagents==0.7.0"]',
    )
    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    return manifest


def test_unpublished_declaring_package_is_not_reported_as_indeterminate(
    monkeypatch, tmp_path
) -> None:
    """A 404 means no published release exists, so nothing can conflict.

    Treating it as indeterminate would print "re-run the job" forever for a
    package that simply has not had its first release yet.
    """
    manifest = _followup_env(monkeypatch, tmp_path)

    def fetch(name: str) -> dict[str, object]:
        msg = f"PyPI returned HTTP 404 for {name}"
        raise PyPIRequestError(msg, transient=False, status=404)

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest], ["libs/deepagents", "libs/code"], fetcher=fetch
    )

    assert conflicts == []
    assert unavailable == ()


def test_payload_without_usable_version_is_indeterminate_not_clean(
    monkeypatch, tmp_path
) -> None:
    """A 200 with no `info.version` must not read as "no conflict"."""
    manifest = _followup_env(monkeypatch, tmp_path)

    conflicts, unavailable = find_follow_up_conflicts(
        [manifest],
        ["libs/deepagents", "libs/code"],
        fetcher=lambda _name: {"releases": {}},
    )

    assert conflicts == []
    assert unavailable == ("deepagents-code",)


def test_followup_analysis_crash_does_not_fail_the_gate(monkeypatch, tmp_path) -> None:
    """The advisory analysis must never turn a decided run into exit 2.

    On the acknowledged path an exit-2 crash would leave the release with no way
    through, so a broken fetcher degrades to a warning on the report instead.
    """
    manifest = _followup_env(monkeypatch, tmp_path)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text("resolved\n", encoding="utf-8")
        return True

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    def boom(_name: str) -> dict[str, object]:
        raise RuntimeError("unexpected payload shape")

    monkeypatch.setattr("check_release_deps.fetch_pypi_json", boom)

    assert run_check("base-sha", "head-sha", acked=True) == 0

    written = output.read_text(encoding="utf-8")
    assert "\nfalse\n" in written
    assert "did not complete" in written


def test_followup_table_is_capped_for_a_single_manifest() -> None:
    """FOLLOWUP_LIMIT applies to every conflict set, not just multi-manifest ones.

    An uncapped table can push the comment past GitHub's body limit, which makes
    the create/update call fail outright.
    """
    followups = tuple(
        _conflict(dependency_name=f"dep-{index}") for index in range(FOLLOWUP_LIMIT + 5)
    )

    lines = _follow_up_markdown(followups, acked=False)
    rows = [line for line in lines if line.startswith("| `")]

    assert len(rows) == FOLLOWUP_LIMIT
    assert any("5 more conflicts" in line for line in lines)


def test_transient_only_run_still_writes_a_step_summary(monkeypatch, tmp_path) -> None:
    """The summary keeps transient logs even though the comment drops them.

    The comment is suppressed so a flaky run cannot overwrite a real failure
    explanation; the summary is where the re-run decision gets made.
    """
    manifest = _followup_env(monkeypatch, tmp_path)
    output = tmp_path / "github_output"
    summary = tmp_path / "github_summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(
        "check_release_deps.load_release_packages",
        lambda: {"libs/code": "deepagents-code", "libs/deepagents": "deepagents"},
    )
    monkeypatch.setattr(
        "check_release_deps.changed_manifests",
        lambda _base, _head, _packages: [manifest],
    )
    monkeypatch.setattr(
        "check_release_deps.fetch_pypi_json",
        lambda _name: _pypi_payload("0.1.49", ["deepagents==0.7.0"]),
    )

    def run_resolver(_manifest_path, log_path) -> bool:
        log_path.write_text("error sending request: connection reset", encoding="utf-8")
        return False

    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    assert run_check("base-sha", "head-sha", acked=False) == 1

    written = output.read_text(encoding="utf-8")
    assert COMMENT_MARKER not in written
    assert "transient network or package-index failure" in summary.read_text(
        encoding="utf-8"
    )
