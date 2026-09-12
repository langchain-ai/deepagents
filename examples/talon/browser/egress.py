"""Bounded browsing proxy; deploy behind an internal-only network and host firewall."""

from __future__ import annotations

import ipaddress
import json
import multiprocessing
import os
import re
import select
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from urllib.parse import urlsplit

LISTEN = ("0.0.0.0", 8080)
MAX_HEADERS = 16 * 1024
CHUNK_SIZE = 16 * 1024
MAX_WORKERS = 32
CONNECTION_TIMEOUT = 30.0
DEFAULT_DENY_CIDRS = (
    "168.63.129.16/32",
    "64:ff9b::/96",
    "64:ff9b:1::/48",
    "2002::/16",
    "2001::/32",
)
Network = ipaddress.IPv4Network | ipaddress.IPv6Network
Address = tuple[str, int] | tuple[str, int, int, int]
TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
STATUS = {
    "bad_request": 400,
    "forbidden_destination": 403,
    "headers_too_large": 431,
    "upstream_unavailable": 502,
    "busy": 503,
    "timeout": 504,
}


class ProxyError(Exception):
    """Expose only a fixed, non-sensitive error code."""


@dataclass(frozen=True)
class Request:
    """A validated single request and its destination."""

    host: str
    port: int
    connect: bool
    headers: bytes
    length: int


def deny_networks(value: str = "") -> tuple[Network, ...]:
    """Extend immutable platform/translation exclusions with runtime CIDRs."""
    try:
        return tuple(
            ipaddress.ip_network(cidr.strip())
            for cidr in (*DEFAULT_DENY_CIDRS, *value.split(","))
            if cidr.strip()
        )
    except ValueError:
        raise ProxyError("bad_request") from None


def _authority(value: str, default_port: int | None = None) -> tuple[str, int]:
    if not value or any(char in value for char in "@/%?#\\"):
        raise ProxyError("bad_request")
    if any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ProxyError("bad_request")
    match = re.fullmatch(r"\[([0-9a-fA-F:.]+)\](?::([0-9]{1,5}))?", value)
    if value.startswith("["):
        if not match:
            raise ProxyError("bad_request")
        host = str(ipaddress.IPv6Address(match[1]))
        port_text = match[2]
    else:
        host, separator, port_text = value.partition(":")
        host = host.lower().removesuffix(".")
        if len(host) > 253 or not all(LABEL.fullmatch(p) for p in host.split(".")):
            raise ProxyError("bad_request")
        if separator and not re.fullmatch(r"[0-9]{1,5}", port_text):
            raise ProxyError("bad_request")
    port = int(port_text) if port_text else default_port
    if port is None or not 1 <= port <= 65535:
        raise ProxyError("bad_request")
    if port not in (80, 443):
        raise ProxyError("forbidden_destination")
    return host, port


def _fields(lines: list[str]) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if not separator or not TOKEN.fullmatch(name):
            raise ProxyError("bad_request")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ProxyError("bad_request")
        fields.setdefault(name.lower(), []).append(value.strip())
    return fields


def _framing(fields: dict[str, list[str]]) -> tuple[int, set[str]]:
    if any(name in fields for name in ("transfer-encoding", "expect", "upgrade")):
        raise ProxyError("bad_request")
    lengths = fields.get("content-length", ["0"])
    if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,19}", lengths[0]):
        raise ProxyError("bad_request")
    tokens = {
        token.strip().lower()
        for value in fields.get("connection", [])
        for token in value.split(",")
    }
    if any(not TOKEN.fullmatch(token) for token in tokens):
        raise ProxyError("bad_request")
    if tokens & {"host", "content-length", "transfer-encoding"}:
        raise ProxyError("bad_request")
    return int(lengths[0]), tokens


def parse_request(raw: bytes) -> Request:
    """Validate framing and rewrite absolute HTTP requests to origin form."""
    if len(raw) > MAX_HEADERS:
        raise ProxyError("headers_too_large")
    if not raw.endswith(b"\r\n\r\n"):
        raise ProxyError("bad_request")
    lines = raw[:-4].decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] not in ("HTTP/1.0", "HTTP/1.1"):
        raise ProxyError("bad_request")
    method, target, _ = parts
    if any(ord(char) <= 32 or ord(char) >= 127 for char in target):
        raise ProxyError("bad_request")
    fields = _fields(lines[1:])
    length, tokens = _framing(fields)
    if method == "CONNECT":
        host, port = _authority(target)
        if length:
            raise ProxyError("bad_request")
        origin = ""
    else:
        if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
            raise ProxyError("bad_request")
        url = urlsplit(target)
        if url.scheme != "http" or url.fragment or "#" in target or "\\" in target:
            raise ProxyError("bad_request")
        host, port = _authority(url.netloc, 80)
        origin = (url.path or "/") + ("?" + url.query if url.query else "")
        if origin.startswith("//"):
            raise ProxyError("bad_request")
    hosts = fields.get("host", [])
    if len(hosts) > 1 or (hosts and _authority(hosts[0], port) != (host, port)):
        raise ProxyError("bad_request")
    authority = f"[{host}]" if ":" in host else host
    authority = f"{authority}:{port}"
    output = [f"{method} {origin} HTTP/1.1", f"Host: {authority}"]
    excluded = tokens | {
        "host",
        "connection",
        "proxy-connection",
        "proxy-authorization",
        "proxy-authenticate",
        "keep-alive",
        "te",
        "trailer",
        "content-length",
    }
    output.extend(
        f"{name}: {value}"
        for name, values in fields.items()
        if name not in excluded
        for value in values
    )
    output.extend((f"Content-Length: {length}", "Connection: close", "", ""))
    return Request(
        host, port, method == "CONNECT", "\r\n".join(output).encode("latin-1"), length
    )


def resolve(
    host: str, port: int, denied: tuple[Network, ...]
) -> list[tuple[int, Address]]:
    """Reject the entire DNS answer set if any address is unsafe."""
    answers = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    vetted: list[tuple[int, Address]] = []
    for family, kind, protocol, _, address in answers:
        ip = ipaddress.ip_address(address[0])
        if (
            family not in (socket.AF_INET, socket.AF_INET6)
            or kind != socket.SOCK_STREAM
            or protocol != socket.IPPROTO_TCP
            or not ip.is_global
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_loopback
            or ip.is_link_local
            or getattr(ip, "is_site_local", False)
            or getattr(ip, "ipv4_mapped", None) is not None
            or (family == socket.AF_INET6 and (address[2] or address[3]))
            or any(ip in network for network in denied)
        ):
            raise ProxyError("forbidden_destination")
        vetted.append((family, address))
    if not vetted:
        raise ProxyError("upstream_unavailable")
    return vetted


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProxyError("timeout")
    return remaining


def _safe_resolve(
    host: str, port: int, denied: tuple[Network, ...]
) -> list[tuple[int, Address]]:
    try:
        return resolve(host, port, denied)
    except ProxyError:
        raise
    except Exception as error:
        raise ProxyError("upstream_unavailable") from error


def _resolve_worker(
    sender: Connection, host: str, port: int, denied: tuple[Network, ...]
) -> None:
    try:
        try:
            result = _safe_resolve(host, port, denied)
        except ProxyError as error:
            code = (
                "forbidden_destination"
                if str(error) == "forbidden_destination"
                else "upstream_unavailable"
            )
            sender.send((code, []))
        else:
            sender.send((None, result))
    except OSError:
        pass
    finally:
        sender.close()


def bounded_resolve(
    host: str, port: int, denied: tuple[Network, ...], deadline: float
) -> list[tuple[int, Address]]:
    """Resolve within the request deadline and reap the isolated DNS worker."""
    _remaining(deadline)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_resolve_worker, args=(sender, host, port, denied))
    try:
        process.start()
        sender.close()
        if not receiver.poll(_remaining(deadline)):
            raise ProxyError("timeout")
        code, addresses = receiver.recv()
        _remaining(deadline)
        if code is not None:
            raise ProxyError(code)
        return addresses
    except (OSError, EOFError):
        raise ProxyError("upstream_unavailable") from None
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def dial(
    host: str, port: int, denied: tuple[Network, ...], deadline: float
) -> socket.socket:
    """Connect only to previously vetted socket addresses, without resolving again."""
    addresses = bounded_resolve(host, port, denied, deadline)
    for family, address in addresses:
        upstream = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        try:
            upstream.settimeout(_remaining(deadline))
            upstream.connect(address)
            return upstream
        except OSError:
            upstream.close()
        except ProxyError:
            upstream.close()
            raise
    raise ProxyError("upstream_unavailable")


def _receive(sock: socket.socket, size: int, deadline: float) -> bytes:
    sock.settimeout(_remaining(deadline))
    return sock.recv(size)


def _send(sock: socket.socket, data: bytes, deadline: float) -> None:
    sock.settimeout(_remaining(deadline))
    sock.sendall(data)


def _read_headers(client: socket.socket, deadline: float) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        if len(data) >= MAX_HEADERS:
            raise ProxyError("headers_too_large")
        chunk = _receive(client, min(4096, MAX_HEADERS - len(data)), deadline)
        if not chunk:
            raise ProxyError("bad_request")
        data.extend(chunk)
    headers, rest = bytes(data).split(b"\r\n\r\n", 1)
    return headers + b"\r\n\r\n", rest


def _http_body(
    client: socket.socket,
    upstream: socket.socket,
    request: Request,
    initial: bytes,
    deadline: float,
) -> None:
    if len(initial) > request.length:
        raise ProxyError("bad_request")
    _send(upstream, request.headers, deadline)
    _send(upstream, initial, deadline)
    remaining = request.length - len(initial)
    while remaining:
        chunk = _receive(client, min(CHUNK_SIZE, remaining), deadline)
        if not chunk:
            raise ProxyError("bad_request")
        _send(upstream, chunk, deadline)
        remaining -= len(chunk)


def _tunnel(client: socket.socket, upstream: socket.socket, deadline: float) -> None:
    readers = [client, upstream]
    while readers:
        ready, _, _ = select.select(readers, [], [], _remaining(deadline))
        if not ready:
            raise ProxyError("timeout")
        for source in ready:
            target = upstream if source is client else client
            data = _receive(source, CHUNK_SIZE, deadline)
            if data:
                _send(target, data, deadline)
            else:
                readers.remove(source)
                target.shutdown(socket.SHUT_WR)


def _error(client: socket.socket, code: str) -> None:
    body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("ascii")
    response = (
        f"HTTP/1.1 {STATUS[code]} Proxy Error\r\n"
        "Content-Type: application/json\r\nConnection: close\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode("ascii") + body
    try:
        client.settimeout(0.2)
        client.sendall(response)
    except OSError:
        pass


class ProxyHandler(socketserver.BaseRequestHandler):
    """Serve one HTTP request or a bounded opaque CONNECT tunnel."""

    def handle(self) -> None:
        """Close silently after response bytes have begun, without leaking errors."""
        started = False
        deadline = time.monotonic() + CONNECTION_TIMEOUT
        try:
            raw, initial = _read_headers(self.request, deadline)
            request = parse_request(raw)
            with dial(
                request.host, request.port, self.server.denied, deadline
            ) as upstream:
                if request.connect:
                    started = True
                    _send(
                        self.request,
                        b"HTTP/1.1 200 Connection Established\r\n\r\n",
                        deadline,
                    )
                    _send(upstream, initial, deadline)
                    _tunnel(self.request, upstream, deadline)
                else:
                    _http_body(self.request, upstream, request, initial, deadline)
                    while chunk := _receive(upstream, CHUNK_SIZE, deadline):
                        started = True
                        _send(self.request, chunk, deadline)
        except (ProxyError, OSError, ValueError) as error:
            if not started:
                code = "upstream_unavailable"
                if isinstance(error, ProxyError):
                    code = str(error)
                elif isinstance(error, TimeoutError):
                    code = "timeout"
                elif isinstance(error, ValueError):
                    code = "bad_request"
                _error(self.request, code)


class EgressProxy(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Bound workers without an unbounded executor queue or request logging."""

    daemon_threads = False
    request_queue_size = MAX_WORKERS
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int] = LISTEN,
        *,
        denied: tuple[Network, ...] = (),
        bind_and_activate: bool = True,
    ) -> None:
        self.denied = deny_networks() + denied
        self._slots = threading.BoundedSemaphore(MAX_WORKERS)
        super().__init__(address, ProxyHandler, bind_and_activate=bind_and_activate)

    def process_request(self, request: socket.socket, client_address: Address) -> None:
        """Reject excess clients before allocating a worker."""
        if not self._slots.acquire(blocking=False):
            _error(request, "busy")
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: Address
    ) -> None:
        """Return capacity even when handling or socket cleanup fails."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request: socket.socket, client_address: Address) -> None:
        """Suppress standard traceback logging of request-related failures."""


def main() -> None:
    """Read runtime policy and listen on the fixed internal proxy endpoint."""
    try:
        denied = deny_networks(os.environ.get("BROWSER_DENY_CIDRS", ""))
        with EgressProxy(denied=denied) as server:
            server.serve_forever()
    except (ProxyError, OSError):
        raise SystemExit('{"error":{"code":"proxy_startup_failed"}}') from None


if __name__ == "__main__":
    main()
