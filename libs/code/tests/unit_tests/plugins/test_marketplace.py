from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from deepagents_code.plugins._json import json_object, json_value
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    _redact_url_credentials,
    _root_for_marketplace_file,
    _run_git,
    parse_marketplace_source,
)
from deepagents_code.plugins.models import (
    LocalMarketplaceSource,
    RepositoryMarketplaceSource,
    UrlMarketplaceSource,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://[invalid", "https://[invalid"),
        ("git@github.com:owner/repo.git", "git@github.com:owner/repo.git"),
        (
            "https://user:pass@example.com:8443/repo",
            "https://***@example.com:8443/repo",
        ),
        (
            "https://user:pass@example.com:invalid/repo",
            "https://user:pass@example.com:invalid/repo",
        ),
        (
            "https://example.com/catalog?token=secret&channel=stable",
            "https://example.com/catalog?token=%2A%2A%2A&channel=stable",
        ),
        (
            "https://example.com/api-key/secret/plugins",
            "https://example.com/api-key/***/plugins",
        ),
        ("https://example.com/token", "https://example.com/token"),
        (
            "https://example.com/catalog?channel=stable",
            "https://example.com/catalog?channel=stable",
        ),
        ("https://example.com/catalog", "https://example.com/catalog"),
    ],
)
def test_redact_url_credentials(value: str, expected: str) -> None:
    assert _redact_url_credentials(value) == expected


def test_parse_marketplace_source_repositories() -> None:
    ssh = parse_marketplace_source("git@github.com:owner/repo.git")
    ssh_ref = parse_marketplace_source("git@github.com:owner/repo.git#main")
    generic = parse_marketplace_source("https://git.example.com/team/repo.git#v1")
    azure = parse_marketplace_source(
        "https://dev.azure.com/org/project/_git/plugins#release"
    )
    github = parse_marketplace_source("https://github.com/owner/repo#main")

    assert ssh == RepositoryMarketplaceSource(
        source_type="git", value="git@github.com:owner/repo.git", ref=None
    )
    assert ssh_ref == RepositoryMarketplaceSource(
        source_type="git", value="git@github.com:owner/repo.git", ref="main"
    )
    assert generic == RepositoryMarketplaceSource(
        source_type="git",
        value="https://git.example.com/team/repo.git",
        ref="v1",
    )
    assert azure == RepositoryMarketplaceSource(
        source_type="git",
        value="https://dev.azure.com/org/project/_git/plugins",
        ref="release",
    )
    assert github == RepositoryMarketplaceSource(
        source_type="git",
        value="https://github.com/owner/repo.git",
        ref="main",
    )


def test_parse_marketplace_source_http_json() -> None:
    source = parse_marketplace_source("https://example.com/marketplace.json")
    assert source == UrlMarketplaceSource(
        source_type="url", value="https://example.com/marketplace.json"
    )


@pytest.mark.parametrize("separator", ["#", "@"])
def test_parse_marketplace_source_github_shorthand(separator: str) -> None:
    source = parse_marketplace_source(f"owner/repo{separator}main")
    assert source == RepositoryMarketplaceSource(
        source_type="github", value="owner/repo", ref="main"
    )


def test_parse_marketplace_source_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "catalog"
    directory.mkdir()
    json_file = directory / "marketplace.json"
    json_file.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    explicit = parse_marketplace_source("./catalog")
    bare = parse_marketplace_source("catalog")
    local_file = parse_marketplace_source("./catalog/marketplace.json")

    assert explicit == LocalMarketplaceSource(
        source_type="directory", value=str(directory.resolve())
    )
    assert bare == explicit
    assert local_file == LocalMarketplaceSource(
        source_type="file", value=str(json_file.resolve())
    )


def test_parse_marketplace_source_rejects_non_json_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "marketplace.txt"
    path.write_text("catalog", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MarketplaceError, match=r"must point to a \.json"):
        parse_marketplace_source("./marketplace.txt")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "Please enter"),
        ("https://user:pass@example.com/marketplace.json", "embedded credentials"),
        ("https://github.com/owner/repo/tree/main", "exactly owner/repo"),
        ("https://[invalid", "Invalid marketplace URL"),
        ("./missing", "Path does not exist"),
        ("not a source", "Invalid marketplace source format"),
        ("owner/repo/extra", "Invalid marketplace source format"),
        ("@owner/repo", "Invalid marketplace source format"),
    ],
)
def test_parse_marketplace_source_rejects_invalid_formats(
    value: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(MarketplaceError, match=message):
        parse_marketplace_source(value)


@pytest.mark.parametrize(
    "relative",
    [
        Path(".claude-plugin/marketplace.json"),
        Path(".agents/plugins/marketplace.json"),
        Path(".agents/plugins/api_marketplace.json"),
    ],
)
def test_root_for_marketplace_file(relative: Path, tmp_path: Path) -> None:
    assert _root_for_marketplace_file(tmp_path / relative) == tmp_path


def test_root_for_unconventional_marketplace_file(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    assert _root_for_marketplace_file(path) == tmp_path


def test_run_git_passes_fixed_argv_and_noninteractive_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INHERITED_SETTING", "yes")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/git")
    received: dict[str, object] = {}

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        received["argv"] = argv
        received.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("subprocess.run", run)
    _run_git(["clone", "https://example.com/repo.git", "/tmp/repo"])

    assert received["argv"] == [
        "/usr/bin/git",
        "clone",
        "https://example.com/repo.git",
        "/tmp/repo",
    ]
    env = received["env"]
    assert env == {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    assert received["check"] is False
    assert received["capture_output"] is True
    assert received["text"] is True
    assert received["timeout"] == 120


def test_run_git_requires_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(MarketplaceError, match="Git is required"):
        _run_git(["clone"])


def test_run_git_redacts_nonzero_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/git")
    result = subprocess.CompletedProcess(
        ["git"], 1, "", "failed https://example.com/?token=secret"
    )
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: result)

    with pytest.raises(MarketplaceError) as exc_info:
        _run_git(["clone"])

    assert "secret" not in str(exc_info.value)
    assert "token=%2A%2A%2A" in str(exc_info.value)


@pytest.mark.parametrize(
    "error",
    [
        OSError("cannot execute"),
        subprocess.TimeoutExpired(["git", "clone"], 120),
    ],
)
def test_run_git_wraps_execution_errors(
    error: OSError | subprocess.TimeoutExpired,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/git")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("subprocess.run", fail)
    with pytest.raises(MarketplaceError, match="Failed to run git"):
        _run_git(["clone"])


def test_json_normalization_is_recursive_and_precisely_typed() -> None:
    marker = object()
    value = {"valid": [1, None, {"nested": True}], "invalid": marker, 1: "ignored"}

    assert json_value(value) == {"valid": [1, None, {"nested": True}]}
    assert json_object(value) == {"valid": [1, None, {"nested": True}]}
    assert json_object(["not", "an", "object"]) == {}
