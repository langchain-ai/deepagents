"""Tests for `[sandboxes]` config parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from deepagents_code.integrations.sandbox_config import SandboxConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_file_returns_empty_config(tmp_path: Path) -> None:
    config = SandboxConfig.load(tmp_path / "does-not-exist.toml")
    assert config.default is None
    assert dict(config.providers) == {}


def test_parses_default_and_providers(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        [sandboxes]
        default = "acme"

        [sandboxes.providers.acme]
        class_path = "acme_dcode_sandbox:AcmeSandboxProvider"
        working_dir = "/workspace"
        package = "acme-dcode-sandbox"

        [sandboxes.providers.acme.params]
        region = "us-east-1"
        namespace = "dev"
        """,
    )
    config = SandboxConfig.load(path)
    assert config.default == "acme"
    acme = config.providers["acme"]
    assert acme["class_path"] == "acme_dcode_sandbox:AcmeSandboxProvider"
    assert acme["working_dir"] == "/workspace"
    assert acme["package"] == "acme-dcode-sandbox"
    assert config.get_params("acme") == {"region": "us-east-1", "namespace": "dev"}


def test_get_params_for_unknown_provider_is_empty(tmp_path: Path) -> None:
    config = SandboxConfig.load(tmp_path / "missing.toml")
    assert config.get_params("acme") == {}


def test_invalid_toml_returns_empty_config(tmp_path: Path) -> None:
    path = _write(tmp_path, "this is not = valid = toml")
    config = SandboxConfig.load(path)
    assert config.default is None
    assert dict(config.providers) == {}


def test_providers_mapping_is_read_only(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        [sandboxes.providers.acme]
        class_path = "acme_dcode_sandbox:AcmeSandboxProvider"
        """,
    )
    config = SandboxConfig.load(path)
    providers = cast("Any", config.providers)
    try:
        providers["other"] = {}
    except TypeError:
        return
    msg = "providers mapping should be read-only"
    raise AssertionError(msg)
