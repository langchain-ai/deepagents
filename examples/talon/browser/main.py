"""Fail-closed viewer and authenticated control scaffolding for the browser sidecar."""

from __future__ import annotations

import hmac
import json
import os
import signal
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


class Bridge(ThreadingHTTPServer):
    """Serve one fixed route table without forwarding caller-selected paths."""

    daemon_threads = True
    request_queue_size = 8

    def __init__(
        self, address: tuple[str, int], *, control: bool, token: str, steel: str
    ) -> None:
        self.control = control
        self.token = token
        self.steel = steel
        self.workers = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, client_address) -> None:
        if not self.workers.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            request.settimeout(5)
            super().process_request(request, client_address)
        except BaseException:
            self.workers.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.workers.release()

    def handle_error(self, request, client_address) -> None:
        pass

    def healthy(self) -> bool:
        try:
            with build_opener(ProxyHandler({})).open(
                self.steel + "/v1/sessions", timeout=2
            ) as response:
                return response.status == 200
        except (OSError, URLError, ValueError):
            return False


class Handler(BaseHTTPRequestHandler):
    """Never expose upstream routes, browser metadata, or exception bodies."""

    server: Bridge

    def log_message(self, format: str, *args: object) -> None:
        pass

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        self.reply(code, {"error": "invalid_request"})

    def reply(self, status: int, result: dict[str, str]) -> None:
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.headers.get("Upgrade"):
            self.reply(404, {"error": "not_found"})
            return
        if self.path == "/health":
            healthy = self.server.healthy()
            self.reply(
                200 if healthy else 503,
                {"status": "ready" if healthy else "unavailable"},
            )
            return
        if not self.server.control or self.path != "/internal/browser/status":
            self.reply(404, {"error": "not_found"})
            return
        authorization = self.headers.get_all("Authorization", [])
        if len(authorization) != 1 or not hmac.compare_digest(
            authorization[0].encode(), ("Bearer " + self.server.token).encode()
        ):
            self.reply(401, {"error": "unauthorized"})
            return
        self.reply(200, {"status": "foundation_only"})


def read_token(path: Path) -> str:
    """Require the deployment's owner-readable runtime credential."""
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o400:
        raise ValueError("invalid_token_file")
    value = path.read_text().strip()
    if len(value) != 43 or not all(
        character.isascii() and (character.isalnum() or character in "-_")
        for character in value
    ):
        raise ValueError("invalid_token")
    return value


def main() -> None:
    """Bind separate interfaces and discard all process state on shutdown."""
    token = read_token(Path(os.environ["TALON_BROWSER_TOKEN_FILE"]))
    steel = os.environ["TALON_BROWSER_STEEL_URL"]
    if steel != "http://172.30.14.2:3000":
        raise ValueError("invalid_steel_origin")
    servers = []
    for name, control in (("CONTROL", True), ("VIEWER", False)):
        address = (
            os.environ[f"TALON_BROWSER_{name}_HOST"],
            int(os.environ[f"TALON_BROWSER_{name}_PORT"]),
        )
        server = Bridge(address, control=control, token=token, steel=steel)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    stopped = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: stopped.set())
    stopped.wait()
    for server in servers:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError):
        raise SystemExit("browser_bridge_start_failed") from None
