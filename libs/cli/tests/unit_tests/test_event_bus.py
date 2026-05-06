"""Unit tests for alpha external event ingress."""

from __future__ import annotations

import asyncio
import socket
import tempfile
from pathlib import Path

import pytest

from deepagents_cli.command_registry import BypassTier
from deepagents_cli.event_bus import (
    ExternalEvent,
    UnixSocketEventSource,
    decode_external_event,
)


class TestDecodeExternalEvent:
    """Validate the JSON-lines external event envelope."""

    def test_decodes_command_event(self) -> None:
        """Command events should parse into typed external events."""
        event = decode_external_event(
            b'{"kind":"command","payload":"/force-clear","bypass":"always"}\n',
            source="test",
        )

        assert event.kind == "command"
        assert event.payload == "/force-clear"
        assert event.bypass is BypassTier.ALWAYS
        assert event.source == "test"

    def test_rejects_empty_payload(self) -> None:
        """Payload must contain command or prompt text."""
        with pytest.raises(ValueError, match="payload"):
            decode_external_event(b'{"kind":"prompt","payload":" "}\n', source="test")


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="requires Unix sockets")
class TestUnixSocketEventSource:
    """Exercise the alpha local socket source."""

    async def test_forwards_json_lines_to_sink(self, tmp_path: Path) -> None:
        """Events sent over the socket should reach the configured sink."""
        del tmp_path
        tmp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[ExternalEvent] = []

        async def sink(event: ExternalEvent) -> None:
            await asyncio.sleep(0)
            received.append(event)

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b'{"kind":"command","payload":"/force-clear"}\n')
            await writer.drain()

            response = await reader.readline()

            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert response == b'{"ok":true}\n'
        assert [event.payload for event in received] == ["/force-clear"]
        assert not path.exists()

    async def test_start_refuses_existing_regular_file(self, tmp_path: Path) -> None:
        """Startup must not delete a non-socket file at the configured path."""
        path = tmp_path / "events.sock"
        path.write_text("do not delete")
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:
            await asyncio.sleep(0)
            raise AssertionError(event)

        with pytest.raises(FileExistsError, match="non-socket"):
            await source.start(sink)

        assert path.read_text() == "do not delete"
