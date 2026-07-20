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
    """The MCP auth startup import remains safe on the server event loop."""
    for module_name in tuple(sys.modules):
        if module_name == "deepagents_code.mcp_auth" or module_name.startswith(
            "filelock."
        ):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.delitem(sys.modules, "filelock", raising=False)
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
            # Model a future transitive dependency that blocks when `mcp_auth`
            # first imports it. The call is allowed only in the worker thread.
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
