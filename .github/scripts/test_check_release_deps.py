"""Tests for release dependency resolution helper."""

import json
import subprocess
import tomllib

import pytest
from check_release_deps import (
    BYPASS_LABEL,
    COMMENT_MARKER,
    ResolverFailure,
    _affected_extras,
    _failure_markdown,
    _relevant_log,
    _toml_value,
    _write_output,
    build_resolver_manifest,
    check_release_dependencies,
    is_transient_resolver_error,
    load_release_packages,
    main,
    run_resolver,
)


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

    assert check_release_dependencies("base-sha", "head-sha") == 0
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

    assert check_release_dependencies("base-sha", "head-sha") == 0


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

    def boom(_base, _head) -> int:
        msg = "kaboom"
        raise RuntimeError(msg)

    monkeypatch.setattr("check_release_deps.check_release_dependencies", boom)

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

    assert check_release_dependencies("base-sha", "head-sha") == 1

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
    assert check_release_dependencies("base-sha", "head-sha") == 1

    written = output.read_text(encoding="utf-8")
    assert "\ntrue\n" in written
    assert COMMENT_MARKER not in written
