"""Live local and Cloud microVM coverage for the Deep Agents backend."""

from __future__ import annotations

import os
from typing import Literal

import pytest
import smol

from langchain_smol import SmolSandbox


@pytest.mark.parametrize("target", ["local", "cloud"])
def test_sandbox_live_vm(target: Literal["local", "cloud"]) -> None:
    """Exercise command, high-level file tools, and missing-file handling."""
    enabled = (
        os.environ.get("SMOL_LOCAL_LIVE") == "1"
        if target == "local"
        else os.environ.get("SMOL_CLOUD_LIVE") == "1"
        or bool(os.environ.get("SMOL_CLOUD_TOKEN"))
    )
    if not enabled:
        pytest.skip(f"{target} live VM was not enabled")

    machine = smol.Machine.create(
        smol.MachineConfig(image="python:3.12-alpine", network=target == "cloud"),
        smol.ConnectOptions(target=target),
    )
    try:
        sandbox = SmolSandbox(machine=machine)
        assert sandbox.execute("python3 -c 'print(6 * 7)'").output.strip() == "42"
        assert (
            sandbox.write("/workspace/nested/hello.txt", "hello from VM\n").error
            is None
        )
        content = sandbox.read("/workspace/nested/hello.txt")
        assert content.error is None
        assert content.file_data is not None
        assert content.file_data["content"] == "hello from VM"
        assert (
            sandbox.download_files(["/workspace/absent.txt"])[0].error
            == "file_not_found"
        )
        assert sandbox.execute("mkdir -p /workspace/test-directory").exit_code == 0
        assert (
            sandbox.download_files(["/workspace/test-directory"])[0].error
            == "is_directory"
        )
    finally:
        machine.delete()
