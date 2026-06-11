"""Tests for release dependency resolution helper."""

import tomllib

from check_release_deps import (
    PackageBump,
    build_filtered_manifest,
    detect_package_bumps,
    is_transient_resolver_error,
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


def test_transient_resolver_error_patterns() -> None:
    assert is_transient_resolver_error("failed to fetch https://pypi.org/simple/pkg")
    assert is_transient_resolver_error("HTTP 503 service unavailable")
    assert not is_transient_resolver_error("No solution found when resolving dependencies")
