from __future__ import annotations

import io
import json
import os
import subprocess
import urllib.request
from http.client import HTTPMessage
from typing import TYPE_CHECKING

import pytest

from deepagents_code.plugins._json import json_object
from deepagents_code.plugins.marketplace import (
    MarketplaceError,
    _download_marketplace,
    _HttpsOnlyRedirectHandler,
    _redact_url_credentials,
    _run_git,
    materialize_plugin_source,
    parse_marketplace_source,
)
from deepagents_code.plugins.models import (
    GitSubdirectoryPluginSource,
    LocalPluginSource,
    MarketplacePluginEntry,
    PluginMarketplace,
    UrlMarketplaceSource,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://[invalid", "https://***"),
        ("git@github.com:owner/repo.git", "git@github.com:owner/repo.git"),
        (
            "https://user:pass@example.com:8443/repo",
            "https://***@example.com:8443/repo",
        ),
        (
            "https://user:pass@example.com:invalid/repo",
            "https://***",
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


def test_parse_marketplace_source_preserves_credentials() -> None:
    source = parse_marketplace_source("https://user:pass@example.com/marketplace.json")
    assert source == UrlMarketplaceSource(
        source_type="url",
        value="https://user:pass@example.com/marketplace.json",
    )


def test_download_marketplace_rejects_http_redirect() -> None:
    handler = _HttpsOnlyRedirectHandler()
    request = urllib.request.Request("https://example.com/marketplace.json")

    with pytest.raises(MarketplaceError, match="redirect must use https"):
        handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            HTTPMessage(),
            "http://example.com/marketplace.json",
        )


def test_download_marketplace_rejects_non_https_final_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "http://example.com/marketplace.json"

    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> Response:
            return Response(json.dumps({"name": "x", "plugins": []}).encode())

    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: Opener())

    with pytest.raises(MarketplaceError, match="response must use https"):
        _download_marketplace("https://example.com/marketplace.json")


def test_download_marketplace_rejects_plain_http() -> None:
    with pytest.raises(MarketplaceError, match="must use https"):
        _download_marketplace("http://example.com/marketplace.json")


def test_download_marketplace_cache_path_is_opaque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response(io.BytesIO):
        def geturl(self) -> str:
            return "https://example.com/marketplace.json"

    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> Response:
            return Response(json.dumps({"name": "x", "plugins": []}).encode())

    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_DIR", tmp_path / "config"
    )
    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: Opener())

    path = _download_marketplace(
        "https://user:secret@example.com/marketplace.json?token=hidden"
    )

    assert "secret" not in str(path)
    assert "hidden" not in str(path)


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


def test_run_git_redacts_credentials_from_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/git")
    error = subprocess.TimeoutExpired(
        ["git", "clone", "https://user:secret@example.com/repo.git"], 120
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("subprocess.run", fail)
    with pytest.raises(MarketplaceError) as exc_info:
        _run_git(["clone"])

    assert "secret" not in str(exc_info.value)


def _marketplace(
    root: Path, entry: MarketplacePluginEntry, **metadata: str
) -> PluginMarketplace:
    return PluginMarketplace(
        name="my-plugins",
        root=root,
        manifest_path=root / ".claude-plugin" / "marketplace.json",
        metadata=json_object(metadata),
        plugins=(entry,),
    )


@pytest.mark.parametrize("declared", ["plugins/my-plugin", "./plugins/my-plugin"])
def test_materialize_local_source_accepts_bare_and_dotted_paths(
    declared: str, tmp_path: Path
) -> None:
    (tmp_path / "plugins" / "my-plugin").mkdir(parents=True)
    entry = MarketplacePluginEntry(
        name="my-plugin", source=LocalPluginSource(source_type="local", path=declared)
    )

    resolved = materialize_plugin_source(_marketplace(tmp_path, entry), entry)

    assert resolved == tmp_path / "plugins" / "my-plugin"


@pytest.mark.parametrize("plugin_root", ["nested", "./nested"])
def test_materialize_local_source_applies_plugin_root(
    plugin_root: str, tmp_path: Path
) -> None:
    (tmp_path / "nested" / "my-plugin").mkdir(parents=True)
    entry = MarketplacePluginEntry(
        name="my-plugin",
        source=LocalPluginSource(source_type="local", path="my-plugin"),
    )

    resolved = materialize_plugin_source(
        _marketplace(tmp_path, entry, pluginRoot=plugin_root), entry
    )

    assert resolved == tmp_path / "nested" / "my-plugin"


@pytest.mark.parametrize(
    "declared", ["../escape", "./../escape", "/etc", "", "./", "plugins/../../escape"]
)
def test_materialize_local_source_rejects_paths_outside_the_marketplace(
    declared: str, tmp_path: Path
) -> None:
    entry = MarketplacePluginEntry(
        name="my-plugin", source=LocalPluginSource(source_type="local", path=declared)
    )

    assert materialize_plugin_source(_marketplace(tmp_path, entry), entry) is None


@pytest.mark.parametrize("declared", ["skills", "./skills"])
def test_materialize_git_subdir_source_accepts_bare_and_dotted_paths(
    declared: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = tmp_path / "clone"
    (clone / "skills").mkdir(parents=True)
    monkeypatch.setattr(
        "deepagents_code.plugins.marketplace._materialize_plugin_repository",
        lambda *_args, **_kwargs: clone,
    )
    entry = MarketplacePluginEntry(
        name="my-plugin",
        source=GitSubdirectoryPluginSource(
            source_type="git-subdir",
            url="https://github.com/acme-corp/monorepo.git",
            path=declared,
        ),
    )

    resolved = materialize_plugin_source(_marketplace(tmp_path, entry), entry)

    assert resolved == clone / "skills"


def test_materialize_git_subdir_source_rejects_escaping_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.setattr(
        "deepagents_code.plugins.marketplace._materialize_plugin_repository",
        lambda *_args, **_kwargs: clone,
    )
    entry = MarketplacePluginEntry(
        name="my-plugin",
        source=GitSubdirectoryPluginSource(
            source_type="git-subdir",
            url="https://github.com/acme-corp/monorepo.git",
            path="../outside",
        ),
    )

    assert materialize_plugin_source(_marketplace(tmp_path, entry), entry) is None
