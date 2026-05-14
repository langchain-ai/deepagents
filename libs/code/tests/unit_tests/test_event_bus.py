"""Unit tests for the external event ingress."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path

import pytest

from deepagents_code.command_registry import BypassTier
from deepagents_code.event_bus import (
    _MAX_LINE_BYTES,
    ExternalEvent,
    UnixSocketEventSource,
    decode_external_event,
    default_unix_socket_path,
)

# Unix socket paths are capped at ~104 bytes on macOS / ~108 on Linux. Pytest's
# default `tmp_path` lives under `/var/folders/...` on macOS which routinely
# exceeds that limit. The helper below binds the socket inside a short-path
# temp dir while still letting the test's other artifacts use `tmp_path`.
_SHORT_TMP_ROOT = "/tmp"  # short path required for AF_UNIX limit


def _short_tmp_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(dir=_SHORT_TMP_ROOT)


class TestExternalEventInvariants:
    """Direct construction must enforce envelope invariants."""

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown external event kind"):
            ExternalEvent(kind="reboot", payload="x", source="t")  # type: ignore[arg-type]

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ExternalEvent(kind="prompt", payload="   ", source="t")

    def test_rejects_unknown_signal_payload(self) -> None:
        with pytest.raises(ValueError, match="Unknown external signal"):
            ExternalEvent(kind="signal", payload="reboot", source="t")

    def test_accepts_known_signal(self) -> None:
        event = ExternalEvent(kind="signal", payload="interrupt", source="t")
        assert event.payload == "interrupt"

    def test_accepts_force_clear_signal(self) -> None:
        event = ExternalEvent(kind="signal", payload="force-clear", source="t")
        assert event.payload == "force-clear"


class TestDecodeExternalEvent:
    """Validate the JSON-lines external event envelope."""

    def test_decodes_command_event(self) -> None:
        event = decode_external_event(
            b'{"kind":"command","payload":"/force-clear","bypass":"always"}\n',
            source="test",
        )
        assert event.kind == "command"
        assert event.payload == "/force-clear"
        assert event.bypass is BypassTier.ALWAYS
        assert event.source == "test"
        assert event.correlation_id is None

    def test_decodes_correlation_id_round_trip(self) -> None:
        event = decode_external_event(
            b'{"kind":"prompt","payload":"hi","correlation_id":"req-1"}\n',
            source="test",
        )
        assert event.correlation_id == "req-1"

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="valid JSON"):
            decode_external_event(b"not json\n", source="t")

    def test_rejects_json_array(self) -> None:
        with pytest.raises(TypeError, match="JSON object"):
            decode_external_event(b"[1, 2, 3]\n", source="t")

    def test_rejects_missing_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            decode_external_event(b'{"payload":"x"}\n', source="t")

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            decode_external_event(b'{"kind":"reboot","payload":"x"}\n', source="t")

    def test_rejects_non_string_payload(self) -> None:
        with pytest.raises(ValueError, match="payload"):
            decode_external_event(b'{"kind":"prompt","payload":123}\n', source="t")

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="payload"):
            decode_external_event(b'{"kind":"prompt","payload":" "}\n', source="t")

    def test_rejects_invalid_bypass(self) -> None:
        with pytest.raises(ValueError, match="bypass"):
            decode_external_event(
                b'{"kind":"prompt","payload":"x","bypass":"nope"}\n',
                source="t",
            )

    def test_rejects_non_string_correlation_id(self) -> None:
        with pytest.raises(ValueError, match="correlation_id"):
            decode_external_event(
                b'{"kind":"prompt","payload":"x","correlation_id":42}\n',
                source="t",
            )

    def test_rejects_unknown_signal_payload(self) -> None:
        with pytest.raises(ValueError, match="Unknown external signal"):
            decode_external_event(
                b'{"kind":"signal","payload":"reboot"}\n',
                source="t",
            )

    def test_accepts_known_signal(self) -> None:
        event = decode_external_event(
            b'{"kind":"signal","payload":"interrupt"}\n',
            source="t",
        )
        assert event.kind == "signal"
        assert event.payload == "interrupt"


class TestDefaultUnixSocketPath:
    """`default_unix_socket_path` resolution."""

    def test_uses_xdg_runtime_dir_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        path = default_unix_socket_path()
        assert path.parent == tmp_path / "deepagents"
        assert path.name == f"events-{os.getpid()}.sock"

    def test_falls_back_to_tempdir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        path = default_unix_socket_path()
        assert path.parent.name == "deepagents"
        assert path.parent.parent == Path(tempfile.gettempdir())


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="requires Unix sockets")
class TestUnixSocketEventSource:
    """Exercise the local socket source end-to-end."""

    async def test_forwards_json_lines_to_sink(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[ExternalEvent] = []

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
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
        assert [e.payload for e in received] == ["/force-clear"]
        assert not path.exists()

    async def test_socket_has_restrictive_permissions(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event

        await source.start(sink)
        try:
            mode = path.stat().st_mode & 0o777
            assert mode == 0o600, f"socket mode is {oct(mode)}, expected 0o600"
        finally:
            await source.stop()
            tmp_dir.cleanup()

    async def test_echoes_correlation_id_in_ack(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b'{"kind":"prompt","payload":"hi","correlation_id":"req-7"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert response == {"ok": True, "correlation_id": "req-7"}

    async def test_nacks_malformed_envelope_and_keeps_listening(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[ExternalEvent] = []

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            received.append(event)

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b"not json\n")
            await writer.drain()
            nack = json.loads(await reader.readline())
            assert nack["ok"] is False
            assert "JSON" in nack["error"]

            writer.write(b'{"kind":"prompt","payload":"valid"}\n')
            await writer.drain()
            ack = json.loads(await reader.readline())
            assert ack == {"ok": True}

            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert [e.payload for e in received] == ["valid"]

    async def test_nack_includes_correlation_id_when_present(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b'{"kind":"reboot","payload":"x","correlation_id":"r-9"}\n')
            await writer.drain()
            nack = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert nack["ok"] is False
        assert nack["correlation_id"] == "r-9"

    async def test_sink_failure_responds_with_nack(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event
            msg = "boom"
            raise RuntimeError(msg)

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b'{"kind":"prompt","payload":"x"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert response["ok"] is False
        assert "boom" in response["error"]

    async def test_oversized_line_responds_with_nack(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[ExternalEvent] = []

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            received.append(event)

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b"x" * (_MAX_LINE_BYTES + 1) + b"\n")
            await writer.drain()
            response = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert response == {"ok": False, "error": "line exceeds read limit"}
        assert received == []

    async def test_handles_multiple_events_per_connection(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[str] = []

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            received.append(event.payload)

        await source.start(sink)
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            for payload in ("first", "second", "third"):
                writer.write(
                    json.dumps({"kind": "prompt", "payload": payload}).encode() + b"\n"
                )
                await writer.drain()
                ack = json.loads(await reader.readline())
                assert ack["ok"] is True
            writer.close()
            await writer.wait_closed()
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert received == ["first", "second", "third"]

    async def test_handles_concurrent_clients(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)
        received: list[str] = []

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            received.append(event.payload)

        await source.start(sink)

        async def send(payload: str) -> dict[str, object]:
            r, w = await asyncio.open_unix_connection(str(path))
            w.write(json.dumps({"kind": "prompt", "payload": payload}).encode() + b"\n")
            await w.drain()
            ack = json.loads(await r.readline())
            w.close()
            await w.wait_closed()
            return ack

        try:
            results = await asyncio.gather(send("a"), send("b"), send("c"))
        finally:
            await source.stop()
            tmp_dir.cleanup()

        assert all(r["ok"] is True for r in results)
        assert sorted(received) == ["a", "b", "c"]

    async def test_recovers_from_stale_socket_file(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        # Pre-create a real socket at the path to simulate a previous crash.
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stale.bind(str(path))
        finally:
            stale.close()

        assert path.exists()
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event

        try:
            await source.start(sink)
            assert path.exists()
        finally:
            await source.stop()
            tmp_dir.cleanup()

    async def test_start_refuses_existing_regular_file(self, tmp_path: Path) -> None:
        path = tmp_path / "events.sock"
        path.write_text("do not delete")
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event
            msg = "should not reach sink"
            raise AssertionError(msg)

        with pytest.raises(FileExistsError, match="non-socket"):
            await source.start(sink)

        assert path.read_text() == "do not delete"

    async def test_start_twice_raises(self) -> None:
        tmp_dir = _short_tmp_dir()
        path = Path(tmp_dir.name) / "events.sock"
        source = UnixSocketEventSource(path)

        async def sink(event: ExternalEvent) -> None:  # noqa: RUF029
            del event

        await source.start(sink)
        try:
            with pytest.raises(RuntimeError, match="already started"):
                await source.start(sink)
        finally:
            await source.stop()
            tmp_dir.cleanup()

    async def test_stop_is_idempotent_without_start(self, tmp_path: Path) -> None:
        source = UnixSocketEventSource(tmp_path / "events.sock")
        await source.stop()
        await source.stop()  # second call must not raise

    async def test_serve_forever_requires_start(self) -> None:
        tmp_dir = _short_tmp_dir()
        source = UnixSocketEventSource(Path(tmp_dir.name) / "events.sock")
        try:
            with pytest.raises(RuntimeError, match="before start"):
                await source.serve_forever()
        finally:
            tmp_dir.cleanup()
