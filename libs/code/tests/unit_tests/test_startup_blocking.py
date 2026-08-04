"""Smoke tests for blocking calls during server startup."""

from __future__ import annotations

import asyncio
import builtins
import sys
import time
from typing import TYPE_CHECKING

import pytest
from blockbuster import BlockingError, blockbuster_ctx

import deepagents_code
from deepagents_code.mcp_tools import _warm_mcp_adapter_imports

if TYPE_CHECKING:
    from types import ModuleType


async def test_mcp_auth_import_is_warmed_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warmup routes the `mcp_auth` first-import off the server event loop.

    Importing `mcp_auth` today does block on the loop (its `httpx` -> `rich`
    chain calls `os.getcwd()` at import time). Rather than couple to that exact
    transitive culprit, this test injects a *synthetic* block via a first-import
    hook, so it stays a stable regression guard for the routing — that
    `_warm_mcp_adapter_imports` consumes the cold import in a worker thread —
    even as dependencies shift. It does not itself prove the real chain blocks.
    If warmup stopped importing `mcp_auth`, the on-loop import below would hit
    the synthetic block and raise `BlockingError` outside `pytest.raises`,
    failing the test.
    """
    monkeypatch.delitem(sys.modules, "deepagents_code.mcp_auth", raising=False)
    monkeypatch.delattr(deepagents_code, "mcp_auth", raising=False)

    original_import = builtins.__import__

    def _blocking_first_auth_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        cold_auth_import = "deepagents_code.mcp_auth" not in sys.modules and (
            name == "deepagents_code.mcp_auth"
            or (name == "deepagents_code" and "mcp_auth" in fromlist)
        )
        if cold_auth_import:
            # Synthetic stand-in for `mcp_auth`'s real import-time blocking: a
            # call Blockbuster rejects on the loop but allows in a worker
            # thread. Fires only on the first (cold) import.
            time.sleep(0)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocking_first_auth_import)

    with blockbuster_ctx() as blockbuster:
        try:
            with pytest.raises(BlockingError):
                time.sleep(0)  # noqa: ASYNC251  # verifies Blockbuster is active

            await asyncio.to_thread(_warm_mcp_adapter_imports)

            from deepagents_code.mcp_auth import build_oauth_provider

            assert callable(build_oauth_provider)
        finally:
            blockbuster.deactivate()
