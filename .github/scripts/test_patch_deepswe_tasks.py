"""Tests for the DeepSWE task.toml patcher."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = REPO_ROOT / ".github" / "scripts" / "patch_deepswe_tasks.py"

_VALID_SHA = "cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a"

# Mirrors the shape of a real DeepSWE task.toml: an air-gapped agent
# environment plus a separate air-gapped verifier environment.
_RAW_TASK = """\
schema_version = "1.1"
[task]
name = "datacurve/example"
[metadata]
base_commit_hash = "cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a"
[verifier]
environment_mode = "separate"
[verifier.environment]
allow_internet = false
build_timeout_sec = 1800.0
[agent]
timeout_sec = 5400.0
[environment]
docker_image = "example"
allow_internet = false
build_timeout_sec = 1800.0
mcp_servers = []
[environment.env]
[solution.env]
"""


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "patch_deepswe_tasks", PATCH_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patcher() -> ModuleType:
    return _load_module()


def _write_task(tmp_path: Path, content: str) -> Path:
    task_dir = tmp_path / "example"
    task_dir.mkdir()
    task_toml = task_dir / "task.toml"
    task_toml.write_text(content)
    return task_toml


def test_environment_allow_internet_enabled(patcher, tmp_path):
    task = _write_task(tmp_path, _RAW_TASK)
    status = patcher.patch_task_toml(task)
    assert status.startswith("PATCHED")
    data = tomllib.loads(task.read_text())
    assert data["environment"]["allow_internet"] is True


def test_verifier_stays_air_gapped(patcher, tmp_path):
    """The separate verifier environment must remain offline for grading
    integrity — the whole point of setting the baseline rather than a blunt
    find-and-replace."""
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    assert data["verifier"]["environment"]["allow_internet"] is False


def test_no_agent_network_mode_override(patcher, tmp_path):
    """The fix uses the [environment] baseline, not a per-phase [agent]
    override (which only the e2b sandbox can honor)."""
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    assert "network_mode" not in data.get("agent", {})


def test_collect_hook_added(patcher, tmp_path):
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    collect = data["verifier"]["collect"]
    assert collect and "model.patch" in collect[0]["command"]


def test_output_is_valid_toml(patcher, tmp_path):
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    tomllib.loads(task.read_text())  # must not raise


def test_idempotent(patcher, tmp_path):
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    first = task.read_text()
    status = patcher.patch_task_toml(task)
    assert task.read_text() == first
    assert status.startswith("OK")


def test_already_public_environment_unchanged(patcher, tmp_path):
    # Target the [environment] block specifically (it has docker_image;
    # [verifier.environment] does not).
    content = _RAW_TASK.replace(
        'docker_image = "example"\nallow_internet = false',
        'docker_image = "example"\nallow_internet = true',
    )
    task = _write_task(tmp_path, content)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    assert data["environment"]["allow_internet"] is True
    assert data["verifier"]["environment"]["allow_internet"] is False


def test_environment_build_timeout_bumped(patcher, tmp_path):
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    assert data["environment"]["build_timeout_sec"] == 3600.0


def test_verifier_build_timeout_unchanged(patcher, tmp_path):
    """Harbor derives the verifier-env build timeout from [environment], so we
    bump only that; [verifier.environment] is deliberately left as-is."""
    task = _write_task(tmp_path, _RAW_TASK)
    patcher.patch_task_toml(task)
    data = tomllib.loads(task.read_text())
    assert data["verifier"]["environment"]["build_timeout_sec"] == 1800.0


def test_missing_environment_section_skipped(patcher, tmp_path):
    content = (
        f'[metadata]\nbase_commit_hash = "{_VALID_SHA}"\n'
        "[verifier]\n[verifier.environment]\nallow_internet = false\n"
    )
    task = _write_task(tmp_path, content)
    status = patcher.patch_task_toml(task)
    assert status.startswith("SKIP")
