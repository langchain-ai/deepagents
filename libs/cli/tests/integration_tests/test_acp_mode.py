"""ACP client code adapted from the python-sdk examples.

See: https://github.com/agentclientprotocol/python-sdk/blob/main/examples/client.py
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acp import PROTOCOL_VERSION, Client, RequestError, connect_to_agent
from acp.schema import ClientCapabilities, Implementation

if TYPE_CHECKING:
    from acp.core import ClientSideConnection


class _AcpSmokeClient(Client):
    async def request_permission(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "session/request_permission"
        raise RequestError.method_not_found(msg)

    async def write_text_file(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "fs/write_text_file"
        raise RequestError.method_not_found(msg)

    async def read_text_file(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "fs/read_text_file"
        raise RequestError.method_not_found(msg)

    async def create_terminal(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "terminal/create"
        raise RequestError.method_not_found(msg)

    async def terminal_output(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "terminal/output"
        raise RequestError.method_not_found(msg)

    async def release_terminal(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "terminal/release"
        raise RequestError.method_not_found(msg)

    async def wait_for_terminal_exit(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "terminal/wait_for_exit"
        raise RequestError.method_not_found(msg)

    async def kill_terminal(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002  # required by ACP Client protocol
        msg = "terminal/kill"
        raise RequestError.method_not_found(msg)

    async def ext_method(self, method: str, params: dict) -> dict:  # noqa: ARG002
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:  # noqa: ARG002
        raise RequestError.method_not_found(method)


async def test_cli_acp_mode_starts_session_and_exits() -> None:
    """Test that the CLI can start in ACP mode, initialize a session, and exit."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = await asyncio.create_subprocess_exec(
        "deepagents",
        "--acp",
        "--no-mcp",
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
        stderr=aio_subprocess.PIPE,
        env=env,
    )

    assert proc.stdin is not None
    assert proc.stdout is not None

    conn: ClientSideConnection = connect_to_agent(
        _AcpSmokeClient(), proc.stdin, proc.stdout
    )

    try:
        await asyncio.wait_for(
            conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(
                    name="test-client",
                    title="Test Client",
                    version="0.0.0",
                ),
            ),
            timeout=15,
        )

        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=str(Path.cwd())),
            timeout=15,
        )
        assert session.session_id is not None
    finally:
        if proc.returncode is None:
            proc.terminate()
            with contextlib.suppress(ProcessLookupError):
                await asyncio.wait_for(proc.wait(), timeout=10)

        if proc.stderr is not None:
            stderr_output = await proc.stderr.read()
            if stderr_output:
                print(f"ACP subprocess stderr:\n{stderr_output.decode()}")  # noqa: T201
