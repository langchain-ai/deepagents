"""Tests for the local-dataset populate dispatcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "prepare_local_dataset", Path(__file__).with_name("prepare_local_dataset.py")
)
assert _SPEC and _SPEC.loader
pld = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pld)


def test_every_registered_dataset_exists_on_disk() -> None:
    """A stale mapping would populate nothing and Harbor would report 0 tasks."""
    evals_root = Path(__file__).resolve().parents[2] / "libs/evals"
    for dataset_path in pld.ADAPTERS:
        assert (evals_root / dataset_path / "dataset.toml").is_file(), dataset_path


def test_every_registered_adapter_module_exists() -> None:
    """A registered module that isn't importable fails only mid-run, per shard."""
    evals_root = Path(__file__).resolve().parents[2] / "libs/evals"
    for dataset_path, module in pld.ADAPTERS.items():
        main_py = evals_root / Path(*module.split(".")).with_suffix(".py")
        assert main_py.is_file(), f"{dataset_path} -> {module} ({main_py})"


def test_resolves_each_known_dataset() -> None:
    assert (
        pld.resolve_adapter("datasets/context-retrieval-evals")
        == "harbor_adapters.contextbench.main"
    )
    assert pld.resolve_adapter("datasets/drbench-evals") == "harbor_adapters.drbench.main"


def test_tolerates_a_trailing_slash() -> None:
    assert pld.resolve_adapter("datasets/drbench-evals/") == "harbor_adapters.drbench.main"


def test_unknown_dataset_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="No populate adapter registered"):
        pld.resolve_adapter("datasets/some-new-thing")


@pytest.mark.parametrize(
    "bad",
    [
        "datasets/../etc/passwd",
        "datasets/~/secrets",
        "/etc/passwd",
        "../datasets/x",
        "not-datasets/x",
        "datasets/x; rm -rf /",
        "datasets/x $(id)",
        "datasets/x\nrm -rf /",
    ],
)
def test_rejects_malformed_paths(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid local Harbor dataset path"):
        pld.resolve_adapter(bad)


def test_main_reports_unknown_dataset_as_a_workflow_error(capsys) -> None:
    assert pld.main(["datasets/some-new-thing"]) == 1
    assert "::error::" in capsys.readouterr().out
