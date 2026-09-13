"""Loopback-only socket tests for bounded relay forwarding and shutdown."""

import importlib.util
import socket
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "local_relay", Path(__file__).parents[2] / "local_relay.py"
)
assert SPEC and SPEC.loader
relay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relay)


class RelayTests(unittest.TestCase):
    """Exercise real TCP streams without Docker or external network access."""

    def test_fixed_endpoints(self) -> None:
        """Production exposes only host IPv4 loopback and one fixed upstream."""
        self.assertEqual(relay.LISTEN, ("127.0.0.1", 8765))
        self.assertEqual(relay.UPSTREAM, ("172.30.13.3", 8080))
        self.assertEqual(relay.MAX_CONNECTIONS, 64)

    def test_bidirectional_half_close_and_shutdown(self) -> None:
        """Raw upgrade bytes and half-closes survive; idle workers join promptly."""
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(2)
            with (
                patch.object(relay, "LISTEN", ("127.0.0.1", 0)),
                patch.object(relay, "UPSTREAM", server.getsockname()),
                relay.LocalRelay() as instance,
            ):
                with socket.create_connection(
                    instance._listener.getsockname(), 2
                ) as client:
                    upstream, _ = server.accept()
                    with upstream:
                        upstream.settimeout(2)
                        client.sendall(b"GET / HTTP/1.1\r\nUpgrade: websocket\r\n\r\n")
                        client.shutdown(socket.SHUT_WR)
                        received = bytearray()
                        while data := upstream.recv(1024):
                            received.extend(data)
                        self.assertIn(b"Upgrade: websocket", received)
                        upstream.sendall(b"\x00\xffresponse")
                        upstream.shutdown(socket.SHUT_WR)
                        received = bytearray()
                        while data := client.recv(1024):
                            received.extend(data)
                        self.assertEqual(received, b"\x00\xffresponse")
                with socket.create_connection(
                    instance._listener.getsockname(), 2
                ) as client:
                    upstream, _ = server.accept()
                    with upstream:
                        started = time.monotonic()
                        instance.__exit__()
                        self.assertLess(time.monotonic() - started, 1)
                        self.assertFalse(instance._workers)

    def test_capacity_and_idle_deadline(self) -> None:
        """Excess clients close rather than growing the worker pool."""
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(2)
            with (
                patch.object(relay, "LISTEN", ("127.0.0.1", 0)),
                patch.object(relay, "UPSTREAM", server.getsockname()),
                patch.object(relay, "MAX_CONNECTIONS", 1),
                patch.object(relay, "IDLE_TIMEOUT", 0.3),
                relay.LocalRelay() as instance,
                socket.create_connection(instance._listener.getsockname(), 2) as first,
            ):
                upstream, _ = server.accept()
                with upstream:
                    with socket.create_connection(
                        instance._listener.getsockname(), 2
                    ) as excess:
                        self.assertEqual(excess.recv(1), b"")
                    self.assertEqual(first.recv(1), b"")


if __name__ == "__main__":
    unittest.main()
