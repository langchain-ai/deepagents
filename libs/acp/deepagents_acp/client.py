import asyncio
import asyncio.subprocess as aio_subprocess
import contextlib
import logging
import os
import sys

from acp import (
    Client,
    ClientSideConnection,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    RequestError,
    SessionNotification,
    text_block,
    PROTOCOL_VERSION,
)
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    Implementation,
    ResourceContentBlock,
    TextContentBlock,
)


class ExampleClient(Client):
    async def requestPermission(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("session/request_permission")

    async def writeTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/write_text_file")

    async def readTextFile(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("fs/read_text_file")

    async def createTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/create")

    async def terminalOutput(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/output")

    async def releaseTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/release")

    async def waitForTerminalExit(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def killTerminal(self, params):  # type: ignore[override]
        raise RequestError.method_not_found("terminal/kill")

    async def sessionUpdate(self, params: SessionNotification) -> None:
        update = params.update
        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text: str
        if isinstance(content, TextContentBlock):
            text = content.text
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
        else:
            text = "<content>"

        print(f"| Agent: {text}")

    async def extMethod(self, method: str, params: dict) -> dict:  # noqa: ARG002
        raise RequestError.method_not_found(method)

    async def extNotification(self, method: str, params: dict) -> None:  # noqa: ARG002
        raise RequestError.method_not_found(method)


async def read_console(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def interactive_loop(conn: ClientSideConnection, session_id: str) -> None:
    while True:
        try:
            line = await read_console("> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            break

        if not line:
            continue

        try:
            await conn.prompt(
                PromptRequest(
                    sessionId=session_id,
                    prompt=[text_block(line)],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("Prompt failed: %s", exc)


async def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO)

    # if len(argv) < 2:
    #     print("Usage: python examples/client.py AGENT_PROGRAM [ARGS...]", file=sys.stderr)
    #     return 2
    #
    # program = argv[1]
    # args = argv[2:]
    #
    # program_path = Path(program)
    # spawn_program = program
    # spawn_args = args
    #
    # if program_path.exists() and not os.access(program_path, os.X_OK):
    #     spawn_program = sys.executable
    #     spawn_args = [str(program_path), *args]

    # agent_process = subprocess.Popen(
    #     [sys.executable, "-m", "deepagents_acp.server"],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )

    spawn_args = [sys.executable, "-m", "deepagents_acp.server"]


    proc = await asyncio.create_subprocess_exec(
        *spawn_args,
        stdin=aio_subprocess.PIPE,
        stdout=aio_subprocess.PIPE,
    )

    if proc.stdin is None or proc.stdout is None:
        print("Agent process does not expose stdio pipes", file=sys.stderr)
        return 1

    client_impl = ExampleClient()
    conn = ClientSideConnection(lambda _agent: client_impl, proc.stdin, proc.stdout)

    response = await conn.initialize(
        InitializeRequest(
            protocolVersion=PROTOCOL_VERSION,
            clientCapabilities=ClientCapabilities(),
            clientInfo=Implementation(name="example-client", title="Example Client", version="0.1.0"),
        )
    )
    print(response)
    session = await conn.newSession(NewSessionRequest(mcpServers=[], cwd=os.getcwd()))

    await interactive_loop(conn, session.sessionId)

    if proc.returncode is None:
        proc.terminate()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
