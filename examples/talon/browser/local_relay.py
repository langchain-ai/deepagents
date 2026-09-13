"""Bounded host-loopback TCP forwarding for the opt-in browser viewer."""

from __future__ import annotations

import select
import socket
import threading
import time

LISTEN = ("127.0.0.1", 8765)
UPSTREAM = ("172.30.13.3", 8080)
MAX_CONNECTIONS = 64
BUFFER_SIZE = 65536
CONNECT_TIMEOUT = 3.0
IDLE_TIMEOUT = 60.0
SESSION_TIMEOUT = 3600.0


class LocalRelay:
    """Own a fixed-destination listener and at most 64 forwarding workers."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._listener: socket.socket | None = None
        self._acceptor: threading.Thread | None = None

    def __enter__(self) -> LocalRelay:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._listener.bind(LISTEN)
            self._listener.listen(MAX_CONNECTIONS)
            self._listener.settimeout(0.1)
            self._acceptor = threading.Thread(target=self._accept, daemon=True)
            self._acceptor.start()
        except BaseException:
            self._listener.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._acceptor is not None:
            self._acceptor.join()
        if self._listener is not None:
            self._listener.close()
        with self._lock:
            workers = tuple(self._workers)
        for worker in workers:
            worker.join()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                if len(self._workers) >= MAX_CONNECTIONS or self._stop.is_set():
                    client.close()
                    continue
                worker = threading.Thread(
                    target=self._forward, args=(client,), daemon=True
                )
                self._workers.add(worker)
                try:
                    worker.start()
                except RuntimeError:
                    self._workers.discard(worker)
                    client.close()

    def _forward(self, client: socket.socket) -> None:
        try:
            with client, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as upstream:
                upstream.settimeout(CONNECT_TIMEOUT)
                upstream.connect(UPSTREAM)
                client.setblocking(False)
                upstream.setblocking(False)
                self._pump(client, upstream)
        except OSError:
            pass
        finally:
            with self._lock:
                self._workers.discard(threading.current_thread())

    def _pump(self, client: socket.socket, upstream: socket.socket) -> None:
        peers = {client: upstream, upstream: client}
        pending = {client: bytearray(), upstream: bytearray()}
        reading = set(peers)
        started = activity = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - activity >= IDLE_TIMEOUT or now - started >= SESSION_TIMEOUT:
                return
            readers = [
                sock for sock in reading if len(pending[peers[sock]]) < BUFFER_SIZE
            ]
            writers = [sock for sock in peers if pending[sock]]
            if not readers and not writers:
                return
            readable, writable, _ = select.select(readers, writers, [], 0.1)
            for sock in readable:
                target = peers[sock]
                try:
                    data = sock.recv(BUFFER_SIZE - len(pending[target]))
                except BlockingIOError:
                    continue
                if data:
                    pending[target].extend(data)
                    activity = time.monotonic()
                else:
                    reading.remove(sock)
                    if not pending[target]:
                        target.shutdown(socket.SHUT_WR)
            for sock in writable:
                try:
                    count = sock.send(pending[sock])
                except BlockingIOError:
                    continue
                if count == 0:
                    return
                del pending[sock][:count]
                activity = time.monotonic()
                if not pending[sock] and peers[sock] not in reading:
                    sock.shutdown(socket.SHUT_WR)
