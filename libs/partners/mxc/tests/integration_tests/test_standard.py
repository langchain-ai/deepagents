"""Shared Deep Agents sandbox conformance suite, run against the real executor."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_mxc import MXCSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

# Created at import so `sandbox_root_dir` is known before any fixture runs.
_ROOT = os.path.realpath(tempfile.mkdtemp(prefix="langchain-mxc-standard-"))


class _ConformanceSandbox(MXCSandbox):
    """`MXCSandbox` with file tools in non-virtual mode, for the shared suite only.

    The shared suite addresses the same files through the file tools and
    through `execute` using real absolute paths, which virtual mode (the only
    mode `MXCSandbox` exposes) would map under `root_dir`. Shell commands are
    still OS-enforced to `root_dir`; only the host-side file tools are widened,
    and only inside this test module.
    """

    def __init__(self, root_dir: str) -> None:
        super().__init__(root_dir=root_dir, max_output_bytes=10_000_000)
        self.virtual_mode = False


@pytest.mark.requires_mxc
@pytest.mark.skipif(sys.platform == "win32", reason="the shared suite uses POSIX commands (cat, wc, chmod, mkdir -p)")
# `SandboxIntegrationTests` declares its class-scoped fixtures as instance methods, which
# pytest deprecates; the shared suite's fixtures cannot be changed from here.
@pytest.mark.filterwarnings("ignore:Class-scoped fixture:pytest.PytestRemovedIn10Warning")
class TestMXCSandboxStandard(SandboxIntegrationTests):
    @property
    def sandbox_root_dir(self) -> str:
        return _ROOT + "/"

    @pytest.fixture(scope="class")
    def sandbox(self, executor: Path) -> Iterator[MXCSandbox]:  # noqa: ARG002 - requesting it applies the skip
        Path(_ROOT).mkdir(parents=True, exist_ok=True)
        backend = _ConformanceSandbox(root_dir=_ROOT)
        try:
            yield backend
        finally:
            shutil.rmtree(_ROOT, ignore_errors=True)
