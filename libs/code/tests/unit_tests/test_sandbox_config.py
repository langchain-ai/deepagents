"""Tests for `[sandboxes]` config parsing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from deepagents_code.integrations.sandbox_config import SandboxConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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


def test_invalid_toml_records_parse_error(tmp_path: Path) -> None:
    """A malformed file degrades to empty but records why, for the caller."""
    path = _write(tmp_path, "this is not = valid = toml")
    config = SandboxConfig.load(path)
    assert config.parse_error is not None
    assert "invalid TOML" in config.parse_error


def test_clean_config_has_no_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, '[sandboxes]\ndefault = "acme"\n')
    assert SandboxConfig.load(path).parse_error is None


def test_missing_file_has_no_parse_error(tmp_path: Path) -> None:
    assert SandboxConfig.load(tmp_path / "missing.toml").parse_error is None


def test_sandboxes_not_a_table_records_parse_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "sandboxes = 1\n")
    config = SandboxConfig.load(path)
    assert dict(config.providers) == {}
    assert config.parse_error is not None
    assert "not a table" in config.parse_error


def test_providers_not_a_table_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, "[sandboxes]\nproviders = 1\n")
    config = SandboxConfig.load(path)
    assert dict(config.providers) == {}


def test_provider_entry_not_a_table_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(tmp_path, '[sandboxes.providers]\nacme = "not-a-table"\n')
    with caplog.at_level(logging.WARNING):
        config = SandboxConfig.load(path)
    assert dict(config.providers) == {}
    assert any("is not a table" in r.message for r in caplog.records)


def test_non_table_params_warns_and_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A non-table `params` is dropped with a visible warning, not silently."""
    path = _write(
        tmp_path,
        """
        [sandboxes.providers.acme]
        class_path = "acme:Provider"
        params = "not-a-table"
        """,
    )
    with caplog.at_level(logging.WARNING):
        config = SandboxConfig.load(path)
    assert config.get_params("acme") == {}
    assert any("non-table 'params'" in r.message for r in caplog.records)


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
