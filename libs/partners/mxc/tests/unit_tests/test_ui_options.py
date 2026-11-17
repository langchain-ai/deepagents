"""The optional UI settings: windows, clipboard and input injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain_mxc import MXCSandbox
from langchain_mxc._policy import CLIPBOARD_LEVELS, HostInfo, SandboxPolicy

if TYPE_CHECKING:
    from pathlib import Path

LOCKED_DOWN = {"disable": True, "clipboard": "none", "injection": False}


@pytest.fixture
def dummy_executor(tmp_path: Path) -> Path:
    exe = tmp_path / "mxc-exec-dummy"
    exe.write_bytes(b"")
    return exe


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    return root


def test_defaults_are_mxc_default_deny(root: Path, dummy_executor: Path) -> None:
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    assert sb.build_request("echo", 1)["ui"] == LOCKED_DOWN
    assert sb.policy.allow_ui is False
    assert sb.policy.clipboard == "none"
    assert sb.policy.allow_input_injection is False


def test_options_loosen_the_ui_block_only(root: Path, dummy_executor: Path) -> None:
    locked = MXCSandbox(root_dir=root, executor_path=dummy_executor)
    opened = MXCSandbox(root_dir=root, executor_path=dummy_executor, allow_ui=True, clipboard="read", allow_input_injection=True)

    assert opened.build_request("echo", 1)["ui"] == {"disable": False, "clipboard": "read", "injection": True}

    a, b = locked.build_request("echo", 1), opened.build_request("echo", 1)
    assert a["filesystem"]["readonlyPaths"] == b["filesystem"]["readonlyPaths"]
    assert a["filesystem"].get("deniedPaths") == b["filesystem"].get("deniedPaths")
    assert a["network"] == b["network"]
    assert a["containment"] == b["containment"] == "process"


@pytest.mark.parametrize("level", CLIPBOARD_LEVELS)
def test_every_documented_clipboard_level_is_accepted(root: Path, dummy_executor: Path, level: str) -> None:
    sb = MXCSandbox(root_dir=root, executor_path=dummy_executor, clipboard=level)
    assert sb.build_request("echo", 1)["ui"]["clipboard"] == level


@pytest.mark.parametrize("bad", ["readwrite", "READ", "", "yes"])
def test_unknown_clipboard_level_is_rejected_at_construction(root: Path, dummy_executor: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="clipboard must be one of none, read, write, all"):
        MXCSandbox(root_dir=root, executor_path=dummy_executor, clipboard=bad)


def test_policy_level_defaults_and_coercion(tmp_path: Path) -> None:
    policy = SandboxPolicy.create(
        root_dir=str(tmp_path), scratch_dir=str(tmp_path / "s"), host=HostInfo.current(), allow_ui=1, allow_input_injection=""
    )
    ui = policy.build_request("echo", 1)["ui"]
    assert ui == {"disable": False, "clipboard": "none", "injection": False}
