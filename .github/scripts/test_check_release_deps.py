"""Tests for release dependency resolution helper."""

import json
import subprocess
import tomllib

import pytest
from check_release_deps import (
    FilteredManifest,
    PackageBump,
    _toml_value,
    build_filtered_manifest,
    check_release_dependencies,
    detect_package_bumps,
    is_transient_resolver_error,
    load_release_packages,
    main,
    run_resolver,
)


def test_detect_package_bumps_skips_new_manifest(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr("check_release_deps._git_show", lambda _base, _path: None)

    assert detect_package_bumps(["pyproject.toml"], "base-sha") == {}


def test_detect_package_bumps_returns_changed_static_versions(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip(),
        encoding="utf-8",
    )

    base_manifest = """
[project]
name = "deepagents"
version = "0.6.8"
dependencies = []
""".strip()

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr("check_release_deps._git_show", lambda _base, _path: base_manifest)

    bumps = detect_package_bumps(["pyproject.toml"], "base-sha")

    assert bumps["deepagents"] == PackageBump(
        name="deepagents",
        version="0.7.0",
        path="pyproject.toml",
    )


def test_filtered_manifest_removes_only_satisfied_same_pr_pins_and_preserves_uv_keys() -> None:
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
    bumped = {
        "deepagents": PackageBump("deepagents", "0.7.0", "libs/deepagents/pyproject.toml"),
        "langchain-daytona": PackageBump(
            "langchain-daytona",
            "0.0.8",
            "libs/partners/daytona/pyproject.toml",
        ),
    }

    filtered = build_filtered_manifest(data, bumped)
    parsed = tomllib.loads(filtered.content)

    assert parsed["project"]["dependencies"] == [
        "langchain>=1.0,<2.0",
        "deepagents-acp>=0.0.8,<0.0.9",
    ]
    assert parsed["project"]["optional-dependencies"]["sandbox"] == []
    assert parsed["project"]["optional-dependencies"]["quickjs"] == [
        "langchain-quickjs>=0.1.4,<0.2.0"
    ]
    assert parsed["tool"]["uv"]["prerelease"] == "allow"
    assert parsed["tool"]["uv"]["constraint-dependencies"] == ["example<2"]
    assert parsed["tool"]["uv"]["override-dependencies"] == ["other==1.0"]
    assert "sources" not in parsed["tool"]["uv"]
    assert filtered.skipped == (
        "deepagents==0.7.0",
        "sandbox: langchain-daytona>=0.0.8,<0.1.0",
    )


def test_check_release_dependencies_writes_each_filtered_manifest_as_pyproject(
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
        lambda: {"libs/code": "deepagents-code", "libs/partners/daytona": "langchain-daytona"},
    )
    monkeypatch.setattr("check_release_deps.changed_manifests", lambda _base, _head, _packages: manifests)
    monkeypatch.setattr("check_release_deps.detect_package_bumps", lambda _manifests, _base: {})
    monkeypatch.setattr(
        "check_release_deps.build_filtered_manifest",
        lambda _data, _bumped: FilteredManifest(content=content, skipped=()),
    )
    monkeypatch.setattr("check_release_deps.run_resolver", run_resolver)

    assert check_release_dependencies("base-sha", "head-sha") == 0
    assert len(resolver_paths) == len(manifests)
    assert len({path.parent for path in resolver_paths}) == len(manifests)


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
    assert "--all-extras" in command
    assert command[command.index("--prerelease") + 1] == "allow"
    assert command[-1] == str(manifest)
    assert log.read_text(encoding="utf-8") == "resolved\n"


def test_filtered_manifest_keeps_pin_not_satisfied_by_bump() -> None:
    """A pin on a bumped package that the bump does NOT satisfy must be kept.

    This is the core safety guarantee: stripping only happens when the same-PR
    bump actually satisfies the specifier, so a stale pin still reaches the
    resolver and fails.
    """
    data = {
        "project": {
            "name": "deepagents-code",
            "version": "0.2.0",
            "dependencies": ["deepagents==0.6.8"],
        },
    }
    bumped = {"deepagents": PackageBump("deepagents", "0.7.0", "libs/deepagents/pyproject.toml")}

    filtered = build_filtered_manifest(data, bumped)
    parsed = tomllib.loads(filtered.content)

    assert parsed["project"]["dependencies"] == ["deepagents==0.6.8"]
    assert filtered.skipped == ()


def test_filtered_manifest_keeps_unparseable_requirement() -> None:
    """An unparseable dependency string is left in place, never silently dropped."""
    data = {
        "project": {
            "name": "deepagents-code",
            "version": "0.2.0",
            "dependencies": ["deepagents @@@ broken"],
        },
    }
    bumped = {"deepagents": PackageBump("deepagents", "0.7.0", "libs/deepagents/pyproject.toml")}

    filtered = build_filtered_manifest(data, bumped)
    parsed = tomllib.loads(filtered.content)

    assert parsed["project"]["dependencies"] == ["deepagents @@@ broken"]
    assert filtered.skipped == ()


def test_detect_package_bumps_skips_unchanged_version(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    body = """
[project]
name = "deepagents"
version = "0.7.0"
dependencies = []
""".strip()
    manifest.write_text(body, encoding="utf-8")

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr("check_release_deps._git_show", lambda _base, _path: body)

    assert detect_package_bumps(["pyproject.toml"], "base-sha") == {}


def test_detect_package_bumps_skips_renamed_package(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "deepagents-code"
version = "0.7.0"
dependencies = []
""".strip(),
        encoding="utf-8",
    )
    base_manifest = """
[project]
name = "deepagents"
version = "0.6.8"
dependencies = []
""".strip()

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    monkeypatch.setattr("check_release_deps._git_show", lambda _base, _path: base_manifest)

    assert detect_package_bumps(["pyproject.toml"], "base-sha") == {}


def test_detect_package_bumps_skips_dynamic_version(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(
        """
[project]
name = "deepagents"
dynamic = ["version"]
dependencies = []
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("check_release_deps.REPO_ROOT", tmp_path)
    # _git_show should never be consulted once the dynamic head manifest is skipped.
    monkeypatch.setattr(
        "check_release_deps._git_show",
        lambda _base, _path: pytest.fail("base manifest should not be read"),
    )

    assert detect_package_bumps(["pyproject.toml"], "base-sha") == {}


def test_load_release_packages_resolves_name_then_component_then_path(tmp_path) -> None:
    config = tmp_path / "release-please-config.json"
    config.write_text(
        json.dumps(
            {
                "packages": {
                    "libs/deepagents": {"package-name": "deepagents", "component": "sdk"},
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
    assert not is_transient_resolver_error("No solution found when resolving dependencies")
    assert not is_transient_resolver_error("version conflict for package foo")
