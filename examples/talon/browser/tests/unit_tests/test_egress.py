"""Network-free checks for proxy destination policy, framing, and resource bounds."""

from __future__ import annotations

import importlib
import ipaddress
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
egress = importlib.import_module("egress")


def stalled_resolver(sender, host, port, denied) -> None:
    """Simulate a DNS lookup that never completes."""
    time.sleep(60)


def synchronous_resolver(host, port, denied, deadline):
    """Keep destination and dialing checks independent of child-process mocking."""
    return egress.resolve(host, port, denied)


def answer(ip: str, port: int = 443) -> tuple:
    """Produce a resolver answer without opening a network connection."""
    ipv6 = ":" in ip
    address = (ip, port, 0, 0) if ipv6 else (ip, port)
    return (
        socket.AF_INET6 if ipv6 else socket.AF_INET,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        address,
    )


def request(
    target: str = "http://example.com/path?q=value", headers: str = ""
) -> bytes:
    """Build an absolute-form HTTP request."""
    return f"GET {target} HTTP/1.1\r\n{headers}\r\n".encode("ascii")


class DestinationTests(unittest.TestCase):
    def setUp(self) -> None:
        resolver = patch.object(
            egress, "bounded_resolve", side_effect=synchronous_resolver
        )
        resolver.start()
        self.addCleanup(resolver.stop)

    def test_rejects_nonpublic_and_special_addresses(self) -> None:
        addresses = (
            "0.0.0.0",
            "10.1.2.3",
            "127.0.0.1",
            "169.254.169.254",
            "172.30.14.3",
            "192.168.0.1",
            "100.100.100.200",
            "192.0.2.1",
            "198.18.0.1",
            "224.0.0.1",
            "240.0.0.1",
            "255.255.255.255",
            "::",
            "::1",
            "fc00::1",
            "fe80::1",
            "fec0::1",
            "ff0e::1",
            "::ffff:8.8.8.8",
            "::ffff:127.0.0.1",
            "2001:db8::1",
            "64:ff9b::808:808",
            "2002:808:808::1",
            "2001::1",
            "168.63.129.16",
            "192.0.0.8",
        )
        for ip in addresses:
            with (
                self.subTest(ip=ip),
                patch.object(egress.socket, "getaddrinfo", return_value=[answer(ip)]),
                self.assertRaisesRegex(egress.ProxyError, "forbidden_destination"),
            ):
                egress.resolve("example.com", 443, egress.deny_networks())

    def test_all_mixed_answers_rejected_before_dial(self) -> None:
        for ips in (("8.8.8.8", "10.0.0.1"), ("::1", "2606:4700:4700::1111")):
            with (
                self.subTest(ips=ips),
                patch.object(
                    egress.socket,
                    "getaddrinfo",
                    return_value=[answer(ip) for ip in ips],
                ),
                patch.object(egress.socket, "socket") as factory,
            ):
                with self.assertRaisesRegex(egress.ProxyError, "forbidden_destination"):
                    egress.dial("example.com", 443, (), time.monotonic() + 30)
                factory.assert_not_called()

    def test_dials_exact_vetted_ipv4_and_ipv6_without_reresolution(self) -> None:
        for ip in ("8.8.8.8", "2606:4700:4700::1111"):
            resolved = answer(ip)
            with (
                self.subTest(ip=ip),
                patch.object(
                    egress.socket,
                    "getaddrinfo",
                    side_effect=[[resolved], [answer("127.0.0.1")]],
                ) as resolver,
                patch.object(egress.socket, "socket") as factory,
            ):
                sock = egress.dial("example.com", 443, (), time.monotonic() + 30)
                resolver.assert_called_once()
                factory.assert_called_once_with(
                    resolved[0], socket.SOCK_STREAM, socket.IPPROTO_TCP
                )
                sock.connect.assert_called_once_with(resolved[4])
                self.assertLessEqual(sock.settimeout.call_args.args[0], 30)

    def test_runtime_cidrs_extend_defaults(self) -> None:
        denied = egress.deny_networks("8.8.8.0/24,2606:4700::/32")
        self.assertIn(ipaddress.ip_network("168.63.129.16/32"), denied)
        for ip in ("8.8.8.8", "2606:4700:4700::1111"):
            with (
                patch.object(egress.socket, "getaddrinfo", return_value=[answer(ip)]),
                self.assertRaises(egress.ProxyError),
            ):
                egress.resolve("example.com", 443, denied)
        with self.assertRaises(egress.ProxyError):
            egress.deny_networks("not-a-cidr")

    def test_failure_and_expired_deadline_close_socket(self) -> None:
        with (
            patch.object(
                egress.socket, "getaddrinfo", return_value=[answer("8.8.8.8")]
            ),
            patch.object(egress.socket, "socket") as factory,
        ):
            factory.return_value.connect.side_effect = OSError("sensitive details")
            with self.assertRaisesRegex(egress.ProxyError, "upstream_unavailable"):
                egress.dial("example.com", 443, (), time.monotonic() + 30)
            factory.return_value.close.assert_called_once()
            factory.reset_mock()
            with self.assertRaisesRegex(egress.ProxyError, "timeout"):
                egress.dial("example.com", 443, (), time.monotonic() - 1)
            factory.return_value.connect.assert_not_called()
            factory.return_value.close.assert_called_once()


class BoundedResolverTests(unittest.TestCase):
    def test_spawn_returns_exact_numeric_addresses_and_applies_policy(self) -> None:
        for ip in ("8.8.8.8", "2606:4700:4700::1111"):
            with self.subTest(ip=ip):
                resolved = answer(ip)
                self.assertEqual(
                    egress.bounded_resolve(ip, 443, (), time.monotonic() + 5),
                    [(resolved[0], resolved[4])],
                )
        with self.assertRaisesRegex(egress.ProxyError, "forbidden_destination"):
            egress.bounded_resolve(
                "8.8.8.8",
                443,
                (ipaddress.ip_network("8.8.8.0/24"),),
                time.monotonic() + 5,
            )

    def test_stalled_dns_times_out_and_reaps_child_before_dial(self) -> None:
        context = egress.multiprocessing.get_context("spawn")
        process = context.Process
        children = []

        def spawn(*args, **kwargs):
            child = process(*args, **kwargs)
            children.append(child)
            return child

        started = time.monotonic()
        with (
            patch.object(egress, "_resolve_worker", stalled_resolver),
            patch.object(context, "Process", side_effect=spawn),
            patch.object(egress.socket, "socket") as factory,
            self.assertRaisesRegex(egress.ProxyError, "timeout"),
        ):
            egress.dial("example.com", 443, (), started + 0.3)
        self.assertLess(time.monotonic() - started, 2)
        factory.assert_not_called()
        self.assertEqual(len(children), 1)
        self.assertTrue(children[0]._closed)
        self.assertNotIn(children[0], egress.multiprocessing.active_children())

    def test_expired_deadline_does_not_spawn(self) -> None:
        with (
            patch.object(egress.multiprocessing, "get_context") as context,
            self.assertRaisesRegex(egress.ProxyError, "timeout"),
        ):
            egress.bounded_resolve("example.com", 443, (), time.monotonic() - 1)
        context.assert_not_called()

    def test_worker_sanitizes_errors_and_closes_pipe(self) -> None:
        for error, expected in (
            (OSError("secret destination"), "upstream_unavailable"),
            (RuntimeError("secret destination"), "upstream_unavailable"),
            (egress.ProxyError("secret destination"), "upstream_unavailable"),
            (egress.ProxyError("forbidden_destination"), "forbidden_destination"),
        ):
            sender = MagicMock()
            with patch.object(egress, "resolve", side_effect=error):
                egress._resolve_worker(sender, "example.com", 443, ())
            sender.send.assert_called_once_with((expected, []))
            sender.close.assert_called_once()

    def test_child_exit_without_result_is_safe_and_cleaned_up(self) -> None:
        context = MagicMock()
        receiver, sender = MagicMock(), MagicMock()
        context.Pipe.return_value = receiver, sender
        receiver.recv.side_effect = EOFError("secret")
        with (
            patch.object(egress.multiprocessing, "get_context", return_value=context),
            self.assertRaisesRegex(egress.ProxyError, "^upstream_unavailable$"),
        ):
            egress.bounded_resolve("example.com", 443, (), time.monotonic() + 5)
        receiver.close.assert_called_once()
        context.Process.return_value.terminate.assert_called_once()
        context.Process.return_value.close.assert_called_once()


class ParsingTests(unittest.TestCase):
    def test_rewrites_origin_and_strips_hop_and_proxy_headers(self) -> None:
        parsed = egress.parse_request(
            request(
                headers=(
                    "Host: example.com\r\nConnection: keep-alive, X-Remove\r\n"
                    "X-Remove: secret\r\nProxy-Authorization: secret\r\n"
                    "Proxy-Connection: keep-alive\r\nAuthorization: runtime-secret\r\n"
                )
            )
        )
        self.assertEqual(
            (parsed.host, parsed.port, parsed.connect), ("example.com", 80, False)
        )
        self.assertTrue(parsed.headers.startswith(b"GET /path?q=value HTTP/1.1\r\n"))
        self.assertIn(b"Connection: close\r\n", parsed.headers)
        self.assertIn(b"authorization: runtime-secret\r\n", parsed.headers)
        self.assertNotIn(b"Proxy-", parsed.headers)
        self.assertNotIn(b"x-remove", parsed.headers)
        self.assertNotIn(b"keep-alive", parsed.headers)

    def test_connect_ipv4_ipv6_and_ports(self) -> None:
        for target in ("example.com:443", "8.8.8.8:80", "[2606:4700:4700::1111]:443"):
            parsed = egress.parse_request(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
            self.assertTrue(parsed.connect)
        for target in (
            "example.com:22",
            "example.com:8080",
            "example.com",
            "example.com:0",
        ):
            with self.subTest(target=target), self.assertRaises(egress.ProxyError):
                egress.parse_request(f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())

    def test_invalid_targets_and_authorities(self) -> None:
        targets = (
            "https://example.com/",
            "/origin",
            "http://user:pass@example.com/",
            "http://example.com@127.0.0.1/",
            "http://example.com:/",
            "http://example.com:65536/",
            "http://example.com:22/",
            "http://bad_host/",
            "http://-bad.example/",
            "http://example..com/",
            "http://example.com/#fragment",
            "http://example.com/#",
            "http://example.com\\@127.0.0.1/",
            "http://[fe80::1%25eth0]/",
            "http://example.com//other",
            "http://example.com/\tpath",
        )
        for target in targets:
            with (
                self.subTest(target=target),
                self.assertRaises((egress.ProxyError, ValueError)),
            ):
                egress.parse_request(request(target))

    def test_rejects_ambiguous_framing_and_host(self) -> None:
        headers = (
            "Transfer-Encoding: chunked\r\n",
            "Content-Length: 0\r\nContent-Length: 0\r\n",
            "Content-Length: 1,1\r\n",
            "Content-Length: -1\r\n",
            "Content-Length: +1\r\n",
            "Content-Length: 9999999999999999999999\r\n",
            "Host: evil.example\r\n",
            "Host: example.com\r\nHost: example.com\r\n",
            "Connection: content-length\r\n",
            "Connection: host\r\n",
            "Expect: 100-continue\r\n",
            "Upgrade: websocket\r\n",
            " Folded: yes\r\n",
            "Bad : value\r\n",
            "X: yes\nInjected: yes\r\n",
        )
        for header in headers:
            with self.subTest(header=header), self.assertRaises(egress.ProxyError):
                egress.parse_request(request(headers=header))


class ResourceTests(unittest.TestCase):
    def test_header_limit_and_deadline(self) -> None:
        client = MagicMock()
        client.recv.side_effect = lambda size: b"x" * size
        with self.assertRaisesRegex(egress.ProxyError, "headers_too_large"):
            egress._read_headers(client, time.monotonic() + 30)
        self.assertEqual(
            sum(call.args[0] for call in client.recv.call_args_list), egress.MAX_HEADERS
        )
        self.assertTrue(
            all(call.args[0] <= 30 for call in client.settimeout.call_args_list)
        )
        client.reset_mock()
        with self.assertRaisesRegex(egress.ProxyError, "timeout"):
            egress._read_headers(client, time.monotonic() - 1)
        client.recv.assert_not_called()

    def test_body_is_bounded_and_pipeline_is_not_forwarded(self) -> None:
        client, upstream = MagicMock(), MagicMock()
        parsed = egress.parse_request(request(headers="Content-Length: 3\r\n"))
        client.recv.return_value = b"abc"
        egress._http_body(client, upstream, parsed, b"", time.monotonic() + 30)
        client.recv.assert_called_once_with(3)
        self.assertEqual(upstream.sendall.call_args.args[0], b"abc")
        upstream.reset_mock()
        with self.assertRaisesRegex(egress.ProxyError, "bad_request"):
            egress._http_body(
                client,
                upstream,
                parsed,
                b"abcGET / HTTP/1.1\r\n\r\n",
                time.monotonic() + 30,
            )
        upstream.sendall.assert_not_called()

    def test_tunnel_chunk_and_timeout(self) -> None:
        client, upstream = MagicMock(), MagicMock()
        client.recv.return_value = b"data"
        with (
            patch.object(
                egress.select, "select", side_effect=[([client], [], []), ([], [], [])]
            ),
            self.assertRaisesRegex(egress.ProxyError, "timeout"),
        ):
            egress._tunnel(client, upstream, time.monotonic() + 30)
        client.recv.assert_called_once_with(egress.CHUNK_SIZE)
        upstream.sendall.assert_called_once_with(b"data")

    def test_saturated_workers_rejected_without_thread(self) -> None:
        server = object.__new__(egress.EgressProxy)
        server._slots = threading.BoundedSemaphore(1)
        server._slots.acquire()
        client = MagicMock()
        with patch.object(
            egress.socketserver.ThreadingMixIn, "process_request"
        ) as spawn:
            server.process_request(client, ("172.30.14.2", 1234))
        spawn.assert_not_called()
        self.assertIn(b'"code":"busy"', client.sendall.call_args.args[0])
        client.close.assert_called_once()

    def test_worker_releases_capacity_on_failure(self) -> None:
        server = object.__new__(egress.EgressProxy)
        server._slots = threading.BoundedSemaphore(1)
        server._slots.acquire()
        with (
            patch.object(
                egress.socketserver.ThreadingMixIn,
                "process_request_thread",
                side_effect=RuntimeError,
            ),
            self.assertRaises(RuntimeError),
        ):
            server.process_request_thread(MagicMock(), ("172.30.14.2", 1234))
        self.assertTrue(server._slots.acquire(blocking=False))

    def test_safe_error_does_not_expose_destination_or_exception(self) -> None:
        client = MagicMock()
        client.recv.return_value = request("http://private.example/secret?token=secret")
        server = MagicMock()
        server.denied = ()
        with patch.object(
            egress, "dial", side_effect=OSError("private.example secret")
        ):
            egress.ProxyHandler(client, ("172.30.14.2", 1234), server)
        output = client.sendall.call_args.args[0]
        self.assertIn(b'"code":"upstream_unavailable"', output)
        self.assertNotIn(b"secret", output)
        self.assertNotIn(b"private.example", output)

    def test_new_http_request_revalidates_redirect_destination(self) -> None:
        server = MagicMock()
        server.denied = egress.deny_networks()
        upstream = MagicMock()
        upstream.__enter__.return_value = upstream
        upstream.recv.side_effect = [
            b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/\r\nContent-Length: 0\r\n\r\n",
            b"",
        ]
        first, redirected = MagicMock(), MagicMock()
        first.recv.return_value = request()
        redirected.recv.return_value = request("http://127.0.0.1/")
        with (
            patch.object(
                egress.socket,
                "getaddrinfo",
                side_effect=[[answer("8.8.8.8", 80)], [answer("127.0.0.1", 80)]],
            ),
            patch.object(egress.socket, "socket", return_value=upstream) as factory,
            patch.object(egress, "bounded_resolve", side_effect=synchronous_resolver),
        ):
            egress.ProxyHandler(first, ("172.30.14.2", 1), server)
            egress.ProxyHandler(redirected, ("172.30.14.2", 2), server)
        factory.assert_called_once()
        self.assertIn(b"302 Found", first.sendall.call_args.args[0])
        self.assertIn(
            b'"code":"forbidden_destination"', redirected.sendall.call_args.args[0]
        )


if __name__ == "__main__":
    unittest.main()
