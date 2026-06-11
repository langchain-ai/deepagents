"""Tests for release dependency resolution helper."""

import subprocess
import tomllib

from check_release_deps import (
    FilteredManifest,
    PackageBump,
    build_filtered_manifest,
    check_release_dependencies,
    detect_package_bumps,
    is_transient_resolver_error,
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
        canonical_name="deepagents",
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
        "deepagents": PackageBump("deepagents", "deepagents", "0.7.0", "libs/deepagents/pyproject.toml"),
        "langchain-daytona": PackageBump(
            "langchain-daytona",
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


def test_transient_resolver_error_patterns() -> None:
    assert is_transient_resolver_error("failed to fetch https://pypi.org/simple/pkg")
    assert is_transient_resolver_error("HTTP 503 service unavailable")
    assert not is_transient_resolver_error("No solution found when resolving dependencies")
