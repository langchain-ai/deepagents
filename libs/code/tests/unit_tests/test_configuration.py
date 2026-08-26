"""Managed configuration provider, precedence, merge, and write tests."""

from __future__ import annotations

import argparse
import os
import socket
import ssl
import sys
import time
from email.message import Message
from http.client import BadStatusLine, IncompleteRead, LineTooLong
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self
from urllib.error import HTTPError, URLError

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    OptionKind,
    _emit_ranked_diagnostics,
    _ranked_source,
)
from deepagents_code.configuration.paths import managed_config_path
from deepagents_code.configuration.providers import (
    REMOTE_MANAGED_CONFIG_MAX_BYTES,
    RemoteTomlProvider,
    TomlFileProvider,
)
from deepagents_code.configuration.resolver import merge_toml_tables
from deepagents_code.configuration.service import (
    ConfigSources,
    ManagedConfigError,
    invalidate_config_sources,
    require_healthy_managed_config,
)
from deepagents_code.configuration.types import ProviderHealth
from deepagents_code.configuration.writer import update_user_config
from unit_tests.conftest import redirect_managed_config, resolve_option_for_test

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from urllib.request import Request

    from deepagents_code.configuration.types import TomlSnapshot


def _resolve(
    option: ConfigOption,
    *,
    toml_data: dict[str, Any],
    managed_toml_data: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Resolve `option` through production code as `(value, source)`.

    Thin alias for the shared `resolve_option_for_test`; see it for why these
    resolutions must not be rebuilt in the test suite.
    """
    return resolve_option_for_test(
        option, toml_data=toml_data, managed_toml_data=managed_toml_data
    )


@pytest.mark.parametrize(
    ("platform", "environ", "expected"),
    [
        (
            "darwin",
            {},
            Path("/Library/Application Support/dcode/managed_config.toml"),
        ),
        ("linux", {}, Path("/etc/dcode/managed_config.toml")),
        (
            "win32",
            {"ProgramData": "D:/SharedData"},
            Path("D:/SharedData/dcode/managed_config.toml"),
        ),
        (
            "win32",
            {},
            Path("C:/ProgramData/dcode/managed_config.toml"),
        ),
    ],
)
def test_managed_config_path_is_fixed_by_platform(
    platform: str,
    environ: dict[str, str],
    expected: Path,
) -> None:
    """Managed config uses an administrator-owned OS path."""
    assert managed_config_path(platform=platform, environ=environ) == expected


def test_managed_config_path_windows_ignores_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redefined `%ProgramData%` must not redirect the managed-config lookup.

    An unprivileged user can set `ProgramData` in their own shell; if the app
    trusted it, the admin policy file could be replaced or silently dropped.
    Production (no injected environ) resolves via the registry or the
    hardcoded default — never the process environment.
    """
    from deepagents_code.configuration import paths

    monkeypatch.setattr(
        paths, "_program_data_from_registry", lambda: (None, "registry unreadable")
    )
    monkeypatch.setenv("ProgramData", "C:/attacker/fake")
    monkeypatch.setenv("PROGRAMDATA", "C:/attacker/fake")
    assert managed_config_path(platform="win32") == Path(
        "C:/ProgramData/dcode/managed_config.toml"
    )


def test_provider_status_remote_source_is_keyword_only() -> None:
    """The exported status keeps its original four positional parameters."""
    from inspect import Parameter, signature

    from deepagents_code.configuration import ProviderStatus

    source = "https://config.example.com/policy.toml"
    status = ProviderStatus(
        "managed config",
        None,
        ProviderHealth.OK,
        None,
        remote_source=source,
    )

    assert status.remote_source == source
    assert signature(ProviderStatus).parameters["remote_source"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert ProviderStatus.__match_args__ == ("name", "path", "health", "detail")


def test_registry_program_data_outranks_a_poisoned_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful registry read wins over a redefined `%ProgramData%`.

    The sibling test proves the *fallback* ignores the environment. This is the
    real attack: a machine with a relocated ProgramData, where the user exports
    a path of their own and the registry holds the true one.
    """
    from deepagents_code.configuration import paths

    monkeypatch.setattr(
        paths, "_program_data_from_registry", lambda: ("D:/RealShared", None)
    )
    monkeypatch.setenv("ProgramData", "C:/attacker/fake")
    monkeypatch.setenv("PROGRAMDATA", "C:/attacker/fake")
    assert managed_config_path(platform="win32") == Path(
        "D:/RealShared/dcode/managed_config.toml"
    )


def test_user_config_writers_share_one_lock_object() -> None:
    """A second lock for the same file would not mutually exclude.

    The hazard is the whole-file replace, so a `[effort]` write in
    `model_config` and a `[ui]` write through the shared writer must contend on
    the same object. This is the invariant those docstrings rest on.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration.writer import USER_CONFIG_WRITE_LOCK

    assert model_config._config_write_lock is USER_CONFIG_WRITE_LOCK


def test_toml_provider_distinguishes_missing_corrupt_and_empty(
    tmp_path: Path,
) -> None:
    """TOML snapshots keep missing, invalid, and valid-empty states distinct."""
    path = tmp_path / "managed.toml"
    provider = TomlFileProvider("managed config", path)
    assert provider.load().status.health is ProviderHealth.MISSING

    path.write_text("[broken", encoding="utf-8")
    assert provider.load().status.health is ProviderHealth.CORRUPT

    path.write_text("", encoding="utf-8")
    snapshot = provider.load()
    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {}


def test_toml_provider_marks_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operating-system read failure is not mistaken for a missing file."""
    path = tmp_path / "managed.toml"
    provider = TomlFileProvider("managed config", path)

    def denied(*_args: object, **_kwargs: object) -> Iterator[bytes]:
        raise PermissionError

    monkeypatch.setattr(Path, "open", denied)
    assert provider.load().status.health is ProviderHealth.UNREADABLE


class _RemoteResponse:
    """Minimal context-managed response for remote provider tests."""

    def __init__(
        self,
        payload: bytes,
        *,
        content_length: str | None = None,
        status: int = 200,
        chunked: bool = False,
    ) -> None:
        self._stream = BytesIO(payload)
        self.closed = Event()
        self.status = status
        self.chunked = chunked
        self.read_timeouts: list[float | None] = []
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        elif chunked:
            # Production accepts chunked framing without a declared length,
            # because `http.client` raises `IncompleteRead` on a short body.
            self.headers["Transfer-Encoding"] = "chunked"
        else:
            # A default-constructed double must model a complete response, or
            # every test would exercise the undelimited-body rejection.
            self.headers["Content-Length"] = str(len(payload))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Record that production cleanup released the fake response."""
        self.closed.set()

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def settimeout(self, value: float | None) -> None:
        self.read_timeouts.append(value)

    def shutdown(self, how: int) -> None:
        """Model socket shutdown by releasing an in-flight fake read."""
        assert how == socket.SHUT_RDWR
        self.close()


class _TrackedErrorBody(BytesIO):
    """Byte stream that records when `HTTPError.close` releases its body."""

    def __init__(self, payload: bytes) -> None:
        """Initialize a tracked response body."""
        super().__init__(payload)
        self.closed_event = Event()

    def close(self) -> None:
        """Record closure before releasing the byte buffer."""
        self.closed_event.set()
        super().close()


class _IncompleteRemoteResponse(_RemoteResponse):
    """Response whose HTTP framing reports an interrupted body."""

    def read(self, size: int = -1) -> bytes:
        del size
        partial = b'[startup]\nmode = "manual"\n'
        raise IncompleteRead(partial, 8)


_CONTROL_CHARACTER_REJECTION = "must not contain whitespace or control characters"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("http://config.example.com/policy.toml", "must be an absolute HTTPS URL"),
        ("https://user@example.com/policy.toml", "must not contain credentials"),
        (
            "https://config.example.com/policy.toml?token=secret",
            "must not contain a query string or fragment",
        ),
        (
            "https://config.example.com/policy.toml#fragment",
            "must not contain a query string or fragment",
        ),
        ("https://config.example.com/policy file.toml", _CONTROL_CHARACTER_REJECTION),
        ("https://config.example.com/policy\tfile.toml", _CONTROL_CHARACTER_REJECTION),
        (
            "https://config.example.com/policy\x01file.toml",
            _CONTROL_CHARACTER_REJECTION,
        ),
        (
            "https://config.example.com/policy\x7ffile.toml",
            _CONTROL_CHARACTER_REJECTION,
        ),
        (
            "https://config.example.com/policy\u2028file.toml",
            _CONTROL_CHARACTER_REJECTION,
        ),
        (
            "https://config.example.com/policy\u200bfile.toml",
            _CONTROL_CHARACTER_REJECTION,
        ),
        (
            "https://config.example.com/policy-é.toml",
            "must contain only ASCII URI characters",
        ),
    ],
)
def test_remote_toml_provider_rejects_unsafe_urls(
    source: str,
    expected: str,
    tmp_path: Path,
) -> None:
    """Remote policy only accepts credential-free absolute HTTPS URLs.

    Each rejection is pinned to its own message: asserting only the health enum
    would let all five collapse into one string that names none of them.
    """
    snapshot = RemoteTomlProvider(
        "managed config", source, tmp_path / "managed.toml"
    ).load()
    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert expected in (snapshot.status.detail or "")
    assert "secret" not in (snapshot.status.detail or "")
    assert snapshot.status.remote_source is None


def test_remote_toml_provider_loads_policy_without_environment_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The source is fetched directly and parsed without proxy inheritance."""
    from urllib.request import ProxyHandler

    from deepagents_code.configuration import providers

    captured: dict[str, object] = {}

    class Opener:
        def open(self, request: object, *, timeout: int) -> _RemoteResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    def build(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build)
    provider = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    )
    snapshot = provider.load()

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}
    assert captured["timeout"] == providers.REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS
    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    assert isinstance(handlers[0], ProxyHandler)
    assert vars(handlers[0])["proxies"] == {}


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (TimeoutError(), "timed out"),
        (URLError("dns failed"), "could not be read"),
        # `HTTPException` siblings are not `OSError`s, so each one escapes
        # every other arm unless the base class is caught.
        (BadStatusLine("garbage"), "could not be read"),
        (LineTooLong("header line"), "could not be read"),
        (IncompleteRead(b"partial", 8), "could not be read"),
        (
            HTTPError(
                "https://config.example.com/policy.toml",
                302,
                "Found",
                Message(),
                None,
            ),
            "refused a redirect (HTTP 302)",
        ),
        (
            HTTPError(
                "https://config.example.com/policy.toml",
                503,
                "Unavailable",
                Message(),
                None,
            ),
            "HTTP 503",
        ),
    ],
)
def test_remote_toml_provider_reports_safe_fetch_failure(
    failure: Exception,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network failures become safe unreadable provider states."""
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert expected in (snapshot.status.detail or "")
    assert "dns failed" not in (snapshot.status.detail or "")


def test_remote_toml_provider_closes_http_error_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP status failure releases its response and connection promptly."""
    from deepagents_code.configuration import providers

    response = _TrackedErrorBody(b"untrusted error body")
    failure = HTTPError(
        "https://config.example.com/policy.toml",
        503,
        "Unavailable",
        Message(),
        response,
    )

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert response.closed


def test_remote_toml_provider_times_out_when_deadline_passes_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect that lands at the deadline fails instead of reading the body."""
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monotonic_values = iter([0.0, providers.REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS])
    monkeypatch.setattr(providers.time, "monotonic", lambda: next(monotonic_values))
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")


def test_remote_toml_provider_bounds_a_stalled_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS, TLS, and response headers share the end-to-end deadline."""
    from deepagents_code.configuration import providers

    entered = Event()
    release = Event()
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            entered.set()
            assert release.wait(timeout=1)
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers, "REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    try:
        snapshot = RemoteTomlProvider(
            "managed config",
            "https://config.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
    finally:
        release.set()

    assert entered.is_set()
    assert time.monotonic() - started < 0.5
    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")
    assert response.closed.wait(timeout=1)


def test_remote_toml_provider_does_not_accumulate_stalled_open_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reloads of the same URL fail closed behind its stuck open."""
    from deepagents_code.configuration import providers

    entered = Event()
    release = Event()
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')
    open_count = 0

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            nonlocal open_count
            assert timeout > 0
            open_count += 1
            entered.set()
            assert release.wait(timeout=1)
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers, "REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS", 0.05)
    provider = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    )
    try:
        snapshots = [provider.load() for _ in range(5)]
    finally:
        release.set()

    assert entered.is_set()
    assert open_count == 1
    assert all(
        snapshot.status.health is ProviderHealth.UNREADABLE for snapshot in snapshots
    )
    # The refusal names this process's stuck fetch, not the administrator's
    # server: the source may be answering perfectly well, and a private class
    # name is not an operator-facing cause.
    assert any(
        snapshot.status.detail
        == "an earlier fetch of the remote source has not returned yet"
        for snapshot in snapshots[1:]
    )
    assert not any(
        "_RemoteOpenInProgressError" in (snapshot.status.detail or "")
        for snapshot in snapshots
    )
    assert response.closed.wait(timeout=1)


def test_remote_toml_provider_recovers_on_new_source_after_stalled_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A descriptor change to a healthy URL succeeds behind a stuck open."""
    from deepagents_code.configuration import providers

    stall_started = Event()
    release = Event()
    healthy_response = _RemoteResponse(b'[startup]\nmode = "manual"\n')

    class Opener:
        def open(self, request: Request, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            if "stalled" in request.full_url:
                stall_started.set()
                release.wait()
                msg = "stalled host never answered"
                raise OSError(msg)
            return healthy_response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers, "REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS", 0.05)
    try:
        stalled = RemoteTomlProvider(
            "managed config",
            "https://stalled.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
        # The worker reports the stall past the tiny deadline, so `load()`
        # returns first; the deadline expiring proves the open was entered.
        assert stalled.status.health is ProviderHealth.UNREADABLE
        assert "timed out" in (stalled.status.detail or "")

        # The abandoned open to the stalled host is still stuck, but the slot
        # is per destination, so the new source gets its own worker.
        snapshot = RemoteTomlProvider(
            "managed config",
            "https://config.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
    finally:
        release.set()

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}


def test_remote_toml_provider_closes_late_http_error_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status failure arriving after timeout still releases its response."""
    from deepagents_code.configuration import providers

    entered = Event()
    release = Event()
    response = _TrackedErrorBody(b"untrusted error body")
    failure = HTTPError(
        "https://config.example.com/policy.toml",
        503,
        "Unavailable",
        Message(),
        response,
    )

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            entered.set()
            assert release.wait(timeout=1)
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers, "REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS", 0.05)
    try:
        snapshot = RemoteTomlProvider(
            "managed config",
            "https://config.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
    finally:
        release.set()

    assert entered.is_set()
    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")
    assert response.closed_event.wait(timeout=1)


def test_remote_toml_provider_rejects_late_empty_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An EOF received after the deadline is not accepted as empty policy."""
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b"")

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monotonic_values = iter([0.0, 0.0, 0.0, 5.0])
    monkeypatch.setattr(providers.time, "monotonic", lambda: next(monotonic_values))
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")


def test_remote_toml_provider_bounds_reads_by_remaining_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header time reduces every subsequent socket-read timeout."""
    from deepagents_code.configuration import providers

    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout == providers.REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monotonic_values = iter([10.0, 10.0, 12.0, 13.0, 14.0, 14.5, 14.75])
    monkeypatch.setattr(providers.time, "monotonic", lambda: next(monotonic_values))
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.OK
    assert response.read_timeouts == [2.0, 0.5]


def test_remote_toml_provider_bounds_a_stalled_chunked_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drip-fed chunk line cannot reset socket timeouts indefinitely."""
    from deepagents_code.configuration import providers

    entered = Event()

    class StalledChunkedResponse(_RemoteResponse):
        def read1(self, size: int = -1) -> bytes:
            del size
            entered.set()
            assert self.closed.wait(timeout=1)
            return b""

    response = StalledChunkedResponse(b"", chunked=True)
    monkeypatch.setattr(providers, "REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert entered.is_set()
    assert time.monotonic() - started < 0.5
    assert response.closed.is_set()
    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")


def test_remote_toml_provider_handles_read_worker_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread exhaustion must produce managed-config health, not a traceback."""
    import threading

    class SelectiveThread(threading.Thread):
        def start(self) -> None:
            if self.name == "managed-config-read":
                msg = "can't start new thread"
                raise RuntimeError(msg)
            super().start()

    monkeypatch.setattr(threading, "Thread", SelectiveThread)
    snapshot = _remote_snapshot(
        _RemoteResponse(b'[startup]\nmode = "manual"\n'),
        tmp_path,
        monkeypatch,
    )

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "could not be read (OSError)" in (snapshot.status.detail or "")


@pytest.mark.parametrize(
    "response",
    [
        _RemoteResponse(
            b'[startup]\nmode = "manual"\n',
            content_length="36",
        ),
        _IncompleteRemoteResponse(b""),
    ],
)
def test_remote_toml_provider_rejects_incomplete_response(
    response: _RemoteResponse,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated body cannot replace the last complete policy generation."""
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "could not be read" in (snapshot.status.detail or "")


@pytest.mark.parametrize(
    ("build_response", "expected"),
    [
        (
            lambda: _RemoteResponse(
                b"x", content_length=str(REMOTE_MANAGED_CONFIG_MAX_BYTES + 1)
            ),
            "size limit",
        ),
        (
            lambda: _RemoteResponse(
                b"x" * (REMOTE_MANAGED_CONFIG_MAX_BYTES + 1), chunked=True
            ),
            "size limit",
        ),
        (
            lambda: _RemoteResponse(
                b'[startup]\nmode = "manual"\n', content_length="abc"
            ),
            "invalid body length",
        ),
        (
            lambda: _RemoteResponse(
                b'[startup]\nmode = "manual"\n', content_length="-1"
            ),
            "invalid body length",
        ),
        (
            lambda: _RemoteResponse(b"[mcp]\n", content_length=""),
            "invalid body length",
        ),
        (lambda: _RemoteResponse(b"\xff"), "not UTF-8"),
        (lambda: _RemoteResponse(b"[broken"), "invalid TOML"),
        (
            lambda: _RemoteResponse(
                b'[managed_config]\nsource = "https://nested.example"\n'
            ),
            "must not declare",
        ),
    ],
)
def test_remote_toml_provider_rejects_invalid_response(
    build_response: Callable[[], _RemoteResponse],
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote bodies are bounded, UTF-8 TOML, and cannot chain sources.

    The responses are built per test rather than at collection: each wraps a
    stateful `BytesIO`, and a reused instance would read as exhausted and
    report the wrong rejection.
    """
    from deepagents_code.configuration import providers

    response = build_response()

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert expected in (snapshot.status.detail or "")


def test_remote_toml_provider_reports_parser_recursion_as_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pathologically nested TOML cannot escape the managed-config status."""
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b"value = [[[[0]]]]\n")

    def recurse(_source: str) -> dict[str, object]:
        raise RecursionError

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers.tomllib, "loads", recurse)
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "invalid TOML" in (snapshot.status.detail or "")


def test_managed_provider_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present corrupt managed policy produces a real startup-gate error."""
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


@pytest.mark.parametrize(
    ("managed_toml", "expected"),
    [
        (
            (
                '[managed_config]\nsource = "https://config.example.com/policy.toml"\n'
                "[ui]\nshow_scrollbar = true\n"
            ),
            "cannot contain local policy keys",
        ),
        ("[managed_config]\nsource = 5\n", "must be a non-empty string"),
        (
            (
                '[managed_config]\nsource = "https://config.example.com/policy.toml"\n'
                "extra = true\n"
            ),
            "must contain only a string source",
        ),
        ("managed_config = 5\n", "must contain only a string source"),
        ("[managed_config]\n", "must contain only a string source"),
        ('[managed_config]\nsource = "   "\n', "must be a non-empty string"),
    ],
)
def test_remote_managed_descriptor_must_be_exclusive(
    managed_toml: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed or mixed remote descriptors fail before any network request.

    Each shape has its own diagnostic: asserting only the health enum would let
    them collapse into one message that names none of them.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.providers import RemoteTomlProvider

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setattr(
        RemoteTomlProvider,
        "load",
        lambda _self: pytest.fail("remote provider should not run"),
    )

    with pytest.raises(ManagedConfigError):
        require_healthy_managed_config(refresh=True)
    status = service.managed_health(refresh=True).status
    assert status.health is ProviderHealth.CORRUPT
    assert expected in (status.detail or "")
    # The anchor file is what is broken here, so these must not claim a remote
    # source and send the operator to a document that was never fetched.
    assert status.remote_source is None


def test_remote_managed_policy_outranks_lower_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid remote response participates as the ordinary managed tier."""
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service
    from deepagents_code.configuration.providers import RemoteTomlProvider
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setattr(
        RemoteTomlProvider,
        "load",
        lambda self: TomlSnapshot(
            {"startup": {"mode": "manual"}},
            ProviderStatus(self.name, self.path, ProviderHealth.OK),
        ),
    )
    service.invalidate_config_sources()

    require_healthy_managed_config(refresh=True)
    option = get_option("startup.mode")
    assert option is not None
    assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}
    assert _resolve(
        option,
        toml_data={"startup": {"mode": "yolo"}},
        managed_toml_data=service.get_managed_snapshot().data,
    ) == ("manual", "managed config")


def test_remote_opener_installs_both_destination_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real opener rejects redirects, not just environment proxies.

    The safe-failure parametrization above stubs an `HTTPError(302)`, so it
    pins the message mapping and not the handler that makes a real 302 raise
    at all. Without this test, deleting `_RejectRedirects` from the opener
    turns an administrator-pinned host into attacker-chosen egress with the
    whole suite still green.
    """
    from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

    from deepagents_code.configuration.providers import _build_remote_opener

    # With no proxy in the environment `build_opener` installs no `ProxyHandler`
    # either way, so asserting its absence would pass whether or not production
    # passes `ProxyHandler({})`. Configuring one is what gives the assertion
    # teeth -- and the control below proves this environment would otherwise
    # route through it.
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid:3128")
    monkeypatch.setenv("https_proxy", "https://proxy.invalid:3128")
    # `handlers` is set in `OpenerDirector.__init__` but absent from typeshed,
    # so read it the way the proxy assertion above reads `proxies`.
    control: list[Any] = vars(build_opener())["handlers"]
    proxied = [h for h in control if isinstance(h, ProxyHandler)]
    assert proxied, "expected a default opener to install an env ProxyHandler"
    assert hasattr(proxied[0], "https_open")

    handlers: list[Any] = vars(_build_remote_opener())["handlers"]
    # `ProxyHandler({})` registers no `*_open` method, so `build_opener` never
    # adds it to the chain. Passing it still does the work: it displaces the
    # default `ProxyHandler` class, whose `getproxies()` would install
    # `HTTPS_PROXY`. Absence is therefore the assertion that env proxies lost.
    assert not any(isinstance(h, ProxyHandler) for h in handlers)
    redirectors = [h for h in handlers if isinstance(h, HTTPRedirectHandler)]
    assert len(redirectors) == 1
    assert (
        redirectors[0].redirect_request(
            None,
            None,
            302,
            "Found",
            Message(),
            "https://elsewhere.example/policy.toml",
        )
        is None
    )


def test_remote_descriptor_source_is_the_url_fetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The descriptor's URL reaches the request, normalized and unsubstituted.

    Every other service-level remote test stubs `RemoteTomlProvider.load`
    wholesale, so nothing ties `[managed_config].source` in the local trust
    anchor to the host actually contacted. This walks descriptor to request.
    """
    from deepagents_code.configuration import providers, service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "  https://Config.Example.com./policy.toml  "\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    captured: dict[str, object] = {}

    class Opener:
        def open(self, request: Request, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            captured["url"] = request.full_url
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    require_healthy_managed_config(refresh=True)

    assert captured["url"] == "https://config.example.com/policy.toml"
    assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}


def test_remote_failure_names_the_url_not_the_trust_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote outage blames the URL, never "repair or remove" the anchor.

    The local file holds only the source, so it is not broken, and removing it
    would drop all policy. The URL is safe to print: validation has already
    rejected credentials, query strings, and fragments.
    """
    from deepagents_code.configuration import providers, service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

    failure = URLError("dns failed")

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    with pytest.raises(ManagedConfigError) as caught:
        require_healthy_managed_config(refresh=True)

    message = str(caught.value)
    assert "https://config.example.com/policy.toml" in message
    assert "managed-config source is reachable" in message
    assert "repair or remove" not in message
    assert "dns failed" not in message


def test_doctor_remote_failure_names_the_url_not_the_trust_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doctor directs remote-policy repair at the source, not its descriptor."""
    from deepagents_code.configuration import providers, service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    failure = URLError("dns failed")

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    item = _managed_config_diagnostic()

    assert item.ok is False
    assert str(managed) in item.value
    assert "https://config.example.com/policy.toml" in item.value
    assert "managed-config source is reachable" in item.value
    assert "repair or remove" not in item.value
    assert "dns failed" not in item.value


def test_doctor_shows_a_healthy_remote_source_with_no_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A working remote deployment reads as working, and names its URL.

    This row is how an administrator confirms the fetch succeeded, so it must
    show the anchor and the source it resolved to, and offer no repair hint.
    """
    from deepagents_code.configuration import providers, service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()
    try:
        item = _managed_config_diagnostic()
    finally:
        service.invalidate_config_sources()

    assert item.ok is True
    assert f"{managed} -> https://config.example.com/policy.toml" in item.value
    assert "ask your administrator" not in item.value.lower()


def test_corrupt_remote_policy_directs_admin_to_repair_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable source with invalid TOML needs document repair."""
    from deepagents_code.configuration import providers, service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b"[broken")

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    with pytest.raises(ManagedConfigError) as caught:
        require_healthy_managed_config(refresh=True)

    message = str(caught.value)
    assert "repair the managed-config document published there" in message
    assert "source is reachable" not in message
    assert "repair or remove" not in message


def test_doctor_corrupt_remote_policy_directs_admin_to_repair_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doctor distinguishes corrupt remote content from an unreachable host."""
    from deepagents_code.configuration import providers, service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b"[broken")

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    item = _managed_config_diagnostic()

    assert item.ok is False
    assert "repair the published document" in item.value
    assert "source is reachable" not in item.value
    assert "repair or remove" not in item.value


def test_rejected_remote_url_is_never_echoed_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source rejected by validation stays out of operator-visible text.

    `remote_source` is set only after `_validate_remote_url` accepts the URL,
    so a descriptor carrying a query token cannot leak it through the error.
    """
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/p.toml?t=secret"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()

    with pytest.raises(ManagedConfigError) as caught:
        require_healthy_managed_config(refresh=True)

    message = str(caught.value)
    assert "secret" not in message
    assert "query string" in message


def test_failed_remote_reload_keeps_previous_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote outage never evicts the last enforceable in-process snapshot."""
    from deepagents_code.configuration import service
    from deepagents_code.configuration.providers import RemoteTomlProvider
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    snapshots = iter(
        [
            TomlSnapshot(
                {"startup": {"mode": "manual"}},
                ProviderStatus("managed config", managed, ProviderHealth.OK),
            ),
            TomlSnapshot(
                {},
                ProviderStatus(
                    "managed config",
                    managed,
                    ProviderHealth.UNREADABLE,
                    "remote source timed out",
                ),
            ),
        ]
    )
    monkeypatch.setattr(RemoteTomlProvider, "load", lambda _self: next(snapshots))
    service.invalidate_config_sources()

    require_healthy_managed_config(refresh=True)
    with pytest.raises(ManagedConfigError):
        require_healthy_managed_config(refresh=True)
    assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}


def test_managed_model_allowlist_replaces_user_list() -> None:
    """Managed allowlists replace rather than union with user grants."""
    from deepagents_code.config_manifest import get_option

    option = get_option("models.allowed")
    assert option is not None
    value, source = _resolve(
        option,
        toml_data={"models": {"allowed": ["openai:gpt-5.6-terra"]}},
        managed_toml_data={"models": {"allowed": ["anthropic:claude-sonnet-5"]}},
    )

    assert value == ("anthropic:claude-sonnet-5",)
    assert source == "managed config"


def test_explicit_model_config_path_excludes_managed_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit-path tooling reads remain isolated from machine policy."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    explicit = tmp_path / "isolated.toml"
    explicit.write_text(
        '[models]\nallowed = ["openai:isolated"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nallowed = ["anthropic:managed"]\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        config = model_config.ModelConfig.load(explicit)
        assert config.allowed_models == ("openai:isolated",)
        assert config.allowed_models_source == "config.toml"
    finally:
        invalidate_config_sources()


def test_malformed_managed_model_allowlist_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed administrator list blocks startup."""
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\nallowed = ["openai:gpt", "broken"]\n')
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError, match=r"models\.allowed"):
            require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


def test_rejected_model_allowlist_reload_keeps_previous_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed reload cannot replace the last enforceable model ceiling."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nallowed = ["anthropic:allowed"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    model_config.clear_caches()
    try:
        require_healthy_managed_config(refresh=True)
        assert model_config.ModelConfig.load().allowed_models == ("anthropic:allowed",)

        managed.write_text(
            '[models]\nallowed = ["not-qualified"]\n',
            encoding="utf-8",
        )
        with pytest.raises(ManagedConfigError, match=r"models\.allowed"):
            require_healthy_managed_config(refresh=True)

        model_config.clear_caches()
        assert model_config.ModelConfig.load().allowed_models == ("anthropic:allowed",)
    finally:
        invalidate_config_sources()
        model_config.clear_caches()


@pytest.mark.parametrize("field", ["default", "recent", "auto_classifier"])
def test_managed_model_field_must_be_in_managed_model_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A contradictory managed `[models]` value is unenforceable policy.

    Parametrized over all three fields the check covers. `auto_classifier` is
    the one that matters most: it is itself an enforced managed key, so an
    administrator pinning a classifier outside their own allowlist would
    otherwise produce policy that contradicts itself.
    """
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nallowed = ["anthropic:claude-sonnet-5"]\n'
        f'{field} = "openai:gpt-5.6-terra"\n'
    )
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError, match=rf"models\.{field}"):
            require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


def test_managed_allowlist_permits_consistent_model_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed policy consistent with its own ceiling starts normally."""
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nallowed = ["anthropic:claude-sonnet-5"]\n'
        'default = "anthropic:claude-sonnet-5"\n'
        'auto_classifier = "anthropic:claude-sonnet-5"\n'
    )
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


def test_managed_allowlist_wildcard_permits_consistent_model_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed provider wildcard admits model fields from that provider."""
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nallowed = ["anthropic:*"]\ndefault = "anthropic:claude-sonnet-5"\n'
    )
    redirect_managed_config(monkeypatch, managed)
    invalidate_config_sources()
    try:
        require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


def test_deep_merge_tracks_managed_leaf_provenance() -> None:
    """Ordinary tables merge per leaf while managed values win conflicts."""
    merged, provenance = merge_toml_tables(
        {"providers": {"acme": {"api_url": "user", "model": "small"}}},
        {"providers": {"acme": {"api_url": "managed"}}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged == {"providers": {"acme": {"api_url": "managed", "model": "small"}}}
    assert provenance == {
        "providers.acme.api_url": "managed config",
        "providers.acme.model": "config.toml",
    }


def test_resolve_managed_beats_env_and_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid managed scalar outranks both environment and user TOML."""
    option = ConfigOption(
        key="feature.enabled",
        group="Test",
        summary="test",
        default=False,
        env_var="FEATURE_ENABLED",
        toml_keys=("feature", "enabled"),
        kind=OptionKind.BOOL,
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_FEATURE_ENABLED", "true")
    assert _resolve(
        option,
        toml_data={"feature": {"enabled": True}},
        managed_toml_data={"feature": {"enabled": False}},
    ) == (False, "managed config")


def test_resolve_skips_one_wrong_typed_managed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed key falls through without discarding valid siblings."""
    option = ConfigOption(
        key="feature.enabled",
        group="Test",
        summary="test",
        default=False,
        env_var="FEATURE_ENABLED",
        toml_keys=("feature", "enabled"),
        kind=OptionKind.BOOL,
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_FEATURE_ENABLED", "true")
    assert _resolve(
        option,
        toml_data={"feature": {"enabled": False}},
        managed_toml_data={"feature": {"enabled": "not-a-bool", "other": 1}},
    ) == (True, "env (DEEPAGENTS_CODE_FEATURE_ENABLED)")


def test_managed_allow_list_empty_replaces_user_grants() -> None:
    """An explicit managed empty grant list enforces lockdown."""
    option = ConfigOption(
        key="mcp.approved_servers",
        group="Test",
        summary="test",
        toml_keys=("mcp", "approved_servers"),
        kind=OptionKind.STRUCTURED,
    )
    assert _resolve(
        option,
        toml_data={"mcp": {"approved_servers": ["user-grant"]}},
        managed_toml_data={"mcp": {"approved_servers": []}},
    ) == ([], "managed config")


def test_user_writer_never_modifies_managed_config(tmp_path: Path) -> None:
    """Central writes preserve sibling user tables and leave policy untouched."""
    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("[ui]\ntheme = 'user'\n", encoding="utf-8")
    managed.write_text("[ui]\ntheme = 'managed'\n", encoding="utf-8")

    def mutate(data: dict[str, object]) -> bool:
        data["runtime"] = {"recursion_limit": 42}
        return True

    result = update_user_config(mutate, config_path=user)
    assert result.ok
    assert result.changed
    assert 'theme = "user"' in user.read_text(encoding="utf-8")
    assert "recursion_limit = 42" in user.read_text(encoding="utf-8")
    assert managed.read_text(encoding="utf-8") == "[ui]\ntheme = 'managed'\n"


def test_user_table_cannot_shadow_managed_scalar() -> None:
    """A shape-colliding user table yields to a managed scalar and provenance.

    Regression: skipping the managed scalar let typed readers reject the
    surviving user table and fall back to the built-in default, so the managed
    value was never enforced.
    """
    merged, provenance = merge_toml_tables(
        {"threads": {"relative_time": {"user": "table"}}, "other": {"a": 1}},
        {"threads": {"relative_time": False}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged == {"threads": {"relative_time": False}, "other": {"a": 1}}
    assert provenance["threads.relative_time"] == "managed config"


def test_managed_wrong_typed_table_keeps_valid_user_siblings() -> None:
    """A malformed managed table does not erase a valid lower table."""
    merged, provenance = merge_toml_tables(
        {"sandboxes": {"default": "user", "providers": {"acme": {"token": "x"}}}},
        {"sandboxes": "not-a-table", "runtime": {"recursion_limit": 42}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged["sandboxes"] == {
        "default": "user",
        "providers": {"acme": {"token": "x"}},
    }
    assert merged["runtime"] == {"recursion_limit": 42}
    assert provenance["sandboxes.default"] == "config.toml"
    assert provenance["runtime.recursion_limit"] == "managed config"


@pytest.mark.parametrize(
    ("managed_value", "expected"),
    [
        pytest.param(5, ["user-denied"], id="unreadable-type-keeps-user-denies"),
        pytest.param({"a": 1}, ["user-denied"], id="table-cannot-replace-a-deny-list"),
        pytest.param(
            "managed-denied, other",
            ["user-denied", "managed-denied", "other"],
            id="comma-string-spelling-unions",
        ),
        pytest.param(
            ["managed-denied"],
            ["user-denied", "managed-denied"],
            id="array-spelling-unions",
        ),
    ],
)
def test_managed_deny_list_layers_union_or_keep_the_user_denies(
    managed_value: object, expected: list[str]
) -> None:
    """A deny list accumulates in both spellings and never loses a lower denial.

    A bare comma-separated string is a documented deny-list spelling that both
    runtime readers split (`mcp_disabled._strict_entries` and
    `model_config._toml_str_list`). The merge dropped it in favor of the user's
    array, so `dcode config` reported denials the runtime did not use and the
    provenance credited the user's file for a leaf managed policy controls. A
    value that cannot hold names at all still leaves the user's list intact.
    """
    merged, provenance = merge_toml_tables(
        {"mcp": {"disabled_servers": ["user-denied"]}},
        {"mcp": {"disabled_servers": managed_value}},
        lower_source="config.toml",
        higher_source="managed config",
        union_paths=frozenset({("mcp", "disabled_servers")}),
    )
    assert merged["mcp"]["disabled_servers"] == expected
    if len(expected) > 1:
        assert provenance["mcp.disabled_servers"] == "managed config + config.toml"


def test_managed_mcp_lockdown_replaces_grants_and_unions_denies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed empty grants lock down approvals while every deny source accumulates."""
    from deepagents_code import _env_vars, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        '[mcp]\ndisabled_project_servers = ["user-denied"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[mcp]\nenabled_project_server_approvals = []\n"
        'disabled_project_servers = ["managed-denied"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setenv(
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "env-granted",
    )
    monkeypatch.setenv(_env_vars.DISABLED_PROJECT_MCP_SERVERS, "env-denied")
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.enabled == frozenset()
    assert trust.approvals == frozenset()
    assert trust.disabled == frozenset({"user-denied", "managed-denied", "env-denied"})


def test_wrong_typed_managed_mcp_allow_list_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed grant key denies rather than leaving grants in force.

    This inverts an earlier decision to skip the key. Skipping it read the
    presence of an allow list as absence, so a quoted string instead of an array
    silently kept both the user's remembered approvals and the
    `DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS` bypass the list exists to remove —
    a malformed narrowing policy widened access. `read_error` reports the cause,
    so this is a visible failure, not a silent lockdown, and it matches
    `disabled_project_servers`, which already fails closed here.
    """
    from deepagents_code import _env_vars, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\nenabled_project_server_approvals = "wrong"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setenv(
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "env-granted",
    )
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.enabled == frozenset()
    assert trust.approvals == frozenset()
    assert trust.read_error is not None
    assert "enabled_project_server_approvals" in trust.read_error


def test_unusable_managed_policy_denies_env_granted_project_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable managed file must not restore the env bypass.

    Regression: `managed_approvals_explicit` was only assignable on the usable
    branch, so a corrupt managed file left `DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS`
    grants in force — corrupting the file converted a managed suppression into a
    permit. A deny list that cannot be read denies everything.
    """
    from deepagents_code import _env_vars, model_config
    from deepagents_code.configuration import service

    monkeypatch.setenv(
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "env-granted",
    )
    _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.enabled == frozenset()
    assert trust.read_error is not None


def test_custom_mcp_config_path_is_isolated_from_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit custom config paths retain their test and embedding isolation seam."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "custom.toml"
    user.write_text(
        '[mcp]\ndisabled_project_servers = ["custom-denied"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\ndisabled_project_servers = ["managed-denied"]\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists(user)
    finally:
        service.invalidate_config_sources()

    assert trust.disabled == frozenset({"custom-denied"})


def test_managed_structured_preferences_reach_runtime_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed thread and warning tables override default-path runtime reads."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        "[threads]\nrelative_time = true\n[warnings]\nsuppress = []\n",
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[threads]\nrelative_time = false\n[warnings]\nsuppress = ["ripgrep"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
        assert model_config.is_warning_suppressed("ripgrep") is True
        assert model_config.load_thread_relative_time(user) is True
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a 0o000 file, so the unreadable case cannot be staged",
)
def test_managed_models_survive_an_unreadable_default_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed models remain available when the user config has no read bits."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[models]\ndefault = "user-model"\n', encoding="utf-8")
    user.chmod(0o000)
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\ndefault = "managed-model"\n'
        '[models.providers.acme]\nmodels = ["managed-model"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = model_config.ModelConfig.load()
        assert config.default_model == "managed-model"
        assert config.providers["acme"]["models"] == ["managed-model"]
    finally:
        user.chmod(0o644)
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_skill_dirs_outrank_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed skill containment roots cannot be replaced through the environment."""
    from deepagents_code import config, model_config
    from deepagents_code._env_vars import EXTRA_SKILLS_DIRS
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    managed_dir = tmp_path / "managed-skills"
    env_dir = tmp_path / "env-skills"
    managed.write_text(
        f'[skills]\nextra_allowed_dirs = ["{managed_dir}"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setenv(EXTRA_SKILLS_DIRS, str(env_dir))
    service.invalidate_config_sources()
    try:
        settings = config.Settings.from_environment(start_path=tmp_path)
        assert settings.extra_skills_dirs == [managed_dir]
    finally:
        service.invalidate_config_sources()


def test_invalid_managed_scalar_keeps_valid_user_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed scalar does not erase a valid user preference."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[threads]\nrelative_time = false\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[threads]\nrelative_time = "invalid"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()


def test_managed_table_at_scalar_path_keeps_valid_user_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed table cannot replace a user model specification."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[models]\ndefault = "user:model"\n', encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[models.default]\ninvalid = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        assert model_config.ModelConfig.load().default_model == "user:model"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_scalar_enforced_over_user_table_shape_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user `[threads.relative_time]` table cannot neutralize managed policy."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[threads.relative_time]\nnested = true\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[threads]\nrelative_time = false\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()


@pytest.mark.parametrize(
    "managed_toml",
    [
        pytest.param('threads = "bad"\n', id="manifest-parent"),
        pytest.param('themes = "bad"\n', id="structured-table"),
        pytest.param('mcp = "bad"\n', id="security-adjacent-parent"),
    ],
)
def test_non_table_known_managed_section_stops_startup(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar cannot replace a known managed section and erase user settings."""
    from deepagents_code.configuration import service
    from deepagents_code.configuration.service import ManagedPolicyError

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedPolicyError):
            require_healthy_managed_config(refresh=True)
    finally:
        service.invalidate_config_sources()


def _sources(managed: dict[str, object], user: dict[str, object]) -> ConfigSources:
    """Build `ConfigSources` from two literal tables, both reported healthy."""
    from deepagents_code.configuration.types import (
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    def snapshot(name: str, data: dict[str, object]) -> TomlSnapshot:
        return TomlSnapshot(
            data,
            ProviderStatus(name, None, ProviderHealth.OK),
        )

    return ConfigSources(
        managed=snapshot("managed config", managed),
        user=snapshot("config.toml", user),
    )


@pytest.mark.parametrize(
    "colliding_user_value",
    [
        pytest.param({"nested": True}, id="scalar-only-table"),
        pytest.param({"nested": {}}, id="table-with-empty-table"),
        pytest.param({"nested": {"deep": True}}, id="table-with-nested-table"),
        pytest.param({"a": {"b": {"c": 1}}}, id="deeply-nested-table"),
    ],
)
def test_managed_scalar_beats_a_user_table_at_any_depth(
    colliding_user_value: dict[str, object],
) -> None:
    """A valid managed scalar wins however deeply the user table nests.

    Regression: the merge kept any user table that held a non-empty nested
    table, so adding one level of nesting to `[threads.relative_time]` let a
    user defeat a managed `relative_time = false`. Typed readers then rejected
    the surviving table and fell back to the built-in default, which silently
    voided administrator policy.

    Driven through `ConfigSources.merged` on purpose: calling the merger
    directly with a hand-passed validator would stay green if `merged` stopped
    passing one.
    """
    sources = _sources(
        {"threads": {"relative_time": False}},
        {"threads": {"relative_time": colliding_user_value}},
    )
    merged, provenance = sources.merged()
    assert merged == {"threads": {"relative_time": False}}
    assert provenance["threads.relative_time"] == "managed config"


def test_invalid_managed_scalar_keeps_a_nested_user_table() -> None:
    """A wrong-typed managed scalar must not discard a valid user subtree."""
    sources = _sources(
        {"threads": {"relative_time": "not-a-bool"}},
        {"threads": {"relative_time": {"a": {"b": 1}}}},
    )
    merged, _ = sources.merged()
    assert merged == {"threads": {"relative_time": {"a": {"b": 1}}}}


def test_managed_policy_survives_a_corrupt_default_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt user config drops only the user layer, never managed policy.

    Regression: the shared reader raised `OSError` on an unusable user file
    before consulting the managed layer, so every caller fell back to built-in
    defaults. The user owns that file, which made one invalid byte an
    unprivileged way to switch administrator policy off.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("this is not [valid toml\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[startup]\nmode = "auto"\n[threads]\nrelative_time = false\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_startup_mode() == "auto"
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()
        model_config.invalidate_thread_config_cache()


def test_malformed_managed_deny_list_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong-typed managed deny list reports an error instead of allowing all.

    Regression: the managed branch discarded the malformed flag, so an
    administrator typo produced an empty deny set with no signal, while the
    same typo in the user file failed closed.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[mcp]\ndisabled_project_servers = 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
        assert "disabled_project_servers" in trust.read_error
    finally:
        service.invalidate_config_sources()


def test_non_table_managed_mcp_section_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar managed `[mcp]` cannot silently void the deny list."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('mcp = "locked"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
    finally:
        service.invalidate_config_sources()


def test_non_table_managed_mcp_section_revokes_the_env_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupting managed policy must not convert a suppression into a permit."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('mcp = "locked"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setenv(
        model_config._env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "evil-server",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.enabled == frozenset()
        assert trust.approvals == frozenset()
        assert trust.read_error is not None
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("all", ["__ALL__"], id="string-all"),
        pytest.param(["all"], ["__ALL__"], id="list-all"),
    ],
)
def test_shell_allow_list_sentinels_agree_across_spellings(
    raw: object,
    expected: list[str],
) -> None:
    """The TOML array honors the same sentinels as the comma-separated string.

    Regression: the list branch bypassed the parser, so a managed
    `allow_list = ["all"]` permitted one literal command named `all` instead
    of every command, silently inverting the administrator's intent.
    """
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, raw, source="managed config") == expected


def test_shell_allow_list_rejects_all_combined_with_commands() -> None:
    """`all` stays exclusive in the array form, matching the string form."""
    from deepagents_code.config_manifest import _INVALID, _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, ["all", "git"], source="managed config") is _INVALID


def test_shell_allow_list_array_expands_recommended() -> None:
    """A `recommended` element expands to the curated safe set."""
    from deepagents_code.config import RECOMMENDED_SAFE_SHELL_COMMANDS
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    resolved = _coerce_toml(option, ["recommended", "make"], source="managed config")
    assert isinstance(resolved, list)
    assert set(RECOMMENDED_SAFE_SHELL_COMMANDS) <= set(resolved)
    assert "make" in resolved


def test_shell_allow_list_array_preserves_comma_in_entry() -> None:
    """An element containing a comma stays a single command.

    Regression: the TOML array was joined with commas and reparsed as the
    string form, so `["my,tool"]` resolved to `["my", "tool"]` — auto-approving
    two executables the administrator never listed.
    """
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, ["my,tool", "git"], source="managed config") == [
        "my,tool",
        "git",
    ]


def test_corrupt_managed_config_does_not_empty_the_mcp_deny_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken managed file must not re-enable administrator-denied servers.

    An unusable snapshot carries an empty table, so returning it would read as
    "nothing is denied" and silently undo every managed deny.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            mcp_disabled.get_disabled_servers()
        # The user-facing predicate must fail closed rather than propagate.
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    "value",
    ['"github"', "5", "true", "{ github = true }"],
    ids=["string", "int", "bool", "table"],
)
def test_unusable_managed_deny_list_fails_closed(
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed deny list that cannot hold names must deny, never allow.

    Regression: `_coerce_entries` reported every non-list as "key absent", so
    the lookup fell through to an empty set with no log at all and every
    administrator-denied server started.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(f"[mcp]\ndisabled_servers = {value}\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        # The file parses, so the startup gate cannot catch this.
        require_healthy_managed_config(refresh=True)
        if value == '"github"':
            # A bare string parses as the comma-separated form, matching
            # `[mcp].disabled_project_servers`, so it denies rather than fails.
            assert mcp_disabled.get_disabled_servers() == {"github"}
        else:
            with pytest.raises(ManagedConfigError):
                mcp_disabled.get_disabled_servers()
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_managed_deny_list_string_splits_on_commas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`disabled_servers = "a, b"` denies two servers, not one bogus name."""
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\ndisabled_servers = "github, linear"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"github", "linear"}
    finally:
        service.invalidate_config_sources()


def test_non_table_managed_mcp_section_cannot_void_the_deny_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar managed `[mcp]` shadows the deny list, so it must fail closed."""
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('mcp = "github"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            mcp_disabled.get_disabled_servers()
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_failed_reload_keeps_the_last_healthy_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed file that breaks mid-session must not empty the cached policy.

    Refreshing used to cache the failed load, so every later reader saw an
    empty managed table and treated enforced denies as absent.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[mcp]\ndisabled_servers = ["denied"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"denied"}

        managed.write_text("[broken", encoding="utf-8")
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)

        # The failed refresh reported the error but left policy in force.
        assert mcp_disabled.get_disabled_servers() == {"denied"}
    finally:
        service.invalidate_config_sources()


def test_rejected_reload_keeps_the_last_enforceable_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload rejected on policy grounds must not replace the cached policy.

    Regression: a parseable-but-unenforceable file has health `OK`, so
    `get_managed_snapshot(refresh=True)` cached it before
    `require_healthy_managed_config` rejected it. The reload kept the
    previous settings, but the process-wide cache already held the rejected
    file, so a later non-refresh reader observed it and re-enabled a managed
    MCP deny the edit had removed.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        'startup.mode = "manual"\n[mcp]\ndisabled_servers = ["denied"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"denied"}

        # Enforced-key violation plus removal of the managed deny, in one
        # parseable edit. The reload is blocked, but the deny must stay.
        managed.write_text("startup.mode = 5\n", encoding="utf-8")
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)

        assert mcp_disabled.get_disabled_servers() == {"denied"}
    finally:
        service.invalidate_config_sources()


def test_union_paths_rebase_onto_an_option_subtree() -> None:
    """Deny-list paths must still match when a merge starts below the root."""
    from deepagents_code.configuration.service import UNION_PATHS, union_paths_under

    assert ("disabled_servers",) in union_paths_under(("mcp",))
    # Rebasing strips exactly the prefix, so an absolute path never survives
    # into a subtree merge — passing one there matches nothing and silently
    # replaces a deny list.
    assert ("mcp", "disabled_servers") not in union_paths_under(("mcp",))
    assert union_paths_under(("mcp",)) == frozenset(
        {("disabled_servers",), ("disabled_project_servers",)}
    )
    assert union_paths_under(("models",)) == frozenset()
    assert all(len(path) > 1 for path in UNION_PATHS)


def test_merged_unions_deny_lists_across_layers() -> None:
    """Both layers' deny entries survive, and provenance names both sources.

    Regression: nothing drove the merger's union branch, so a change to the
    `UNION_PATHS` match or the dedupe would have replaced a managed deny list
    with the user's — a fail-open — while every test stayed green.
    """
    sources = _sources(
        {"mcp": {"disabled_servers": ["managed-denied", "shared"]}},
        {"mcp": {"disabled_servers": ["user-denied", "shared"]}},
    )
    data, provenance = sources.merged()
    assert sorted(data["mcp"]["disabled_servers"]) == [
        "managed-denied",
        "shared",
        "user-denied",
    ]
    assert provenance["mcp.disabled_servers"] == "managed config + config.toml"


def test_merged_keeps_a_managed_deny_list_when_the_user_has_none() -> None:
    """A user layer with no deny list cannot dilute the managed one."""
    sources = _sources({"mcp": {"disabled_servers": ["denied"]}}, {"mcp": {}})
    data, provenance = sources.merged()
    assert data["mcp"]["disabled_servers"] == ["denied"]
    assert provenance["mcp.disabled_servers"] == "managed config"


def test_merged_purges_provenance_for_leaves_the_merge_removed() -> None:
    """A replaced user table leaves no provenance behind.

    Regression: the nested merge kept the parent-scope entry, so
    `dcode config --json --verbose` reported `threads.relative_time.x` as
    user-controlled after a managed scalar had removed that path.
    """
    sources = _sources(
        {"threads": {"relative_time": False}},
        {"threads": {"relative_time": {"x": 1}}},
    )
    data, provenance = sources.merged()
    assert data["threads"]["relative_time"] is False
    assert provenance == {"threads.relative_time": "managed config"}


def test_a_quoted_dotted_key_cannot_drop_or_misattribute_a_leaf() -> None:
    """A TOML key containing dots must not collide with a nested path.

    `tomllib.loads('"a.b" = 1')` yields the single key `a.b`. Provenance keyed
    by dotted string could not tell that from the nested path `a` → `b`, so a
    user who wrote a quoted dotted key made the administrator's audit view drop
    a live sibling leaf (`_drop_ancestor_entries` treated `a` as an ancestor of
    `a.b`) or credit managed policy for a value the user still controls.
    """
    merged, provenance = merge_toml_tables(
        {"a": "user-scalar", "x": 1},
        {"a.b": "managed-flat"},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged == {"a": "user-scalar", "x": 1, "a.b": "managed-flat"}
    # The user's `a` is still effective, so it must still be attributed.
    assert provenance["a"] == "config.toml"
    assert provenance["x"] == "config.toml"
    assert provenance["a.b"] == "managed config"


def test_managed_snapshot_rejects_data_it_could_not_have_read() -> None:
    """An unhealthy snapshot must never carry a table.

    Every reader treats an empty managed table as "nothing declared", so a
    snapshot that reports a failure while carrying values would have both
    meanings at once.
    """
    from deepagents_code.configuration.types import (
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    corrupt = ProviderStatus("managed config", None, ProviderHealth.CORRUPT)
    with pytest.raises(ValueError, match="must carry no data"):
        TomlSnapshot({"startup": {"mode": "yolo"}}, corrupt)
    # The empty pairing is the legitimate one.
    assert TomlSnapshot({}, corrupt).data == {}


def test_writer_reports_a_mis_encoded_config_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A config that is not UTF-8 is reported, not raised.

    `tomllib` decodes the bytes itself, so the failure is a
    `UnicodeDecodeError`, which the read guard did not catch. It escaped past
    every caller's error handling and lost the real reason.
    """
    from deepagents_code.configuration.writer import update_user_config

    target = tmp_path / "config.toml"
    target.write_bytes('[ui]\ntheme = "dark"\n'.encode("utf-16"))
    result = update_user_config(
        lambda data: bool(data.setdefault("ui", {})), config_path=target
    )
    assert result.ok is False
    assert result.error is not None
    assert str(target) in result.error


def test_write_result_rejects_a_failure_with_no_detail() -> None:
    """Callers branch on `ok` alone, so a failure must carry something to act on."""
    from deepagents_code.configuration.writer import WriteResult

    with pytest.raises(ValueError, match="error detail"):
        WriteResult(False, False, None)


def test_write_result_rejects_a_change_on_a_failed_write() -> None:
    """A failed write cannot report that it changed the file."""
    from deepagents_code.configuration.writer import WriteResult

    with pytest.raises(ValueError, match="cannot have changed"):
        WriteResult(False, True, "boom")


def test_write_result_accepts_only_the_three_real_outcomes() -> None:
    """The guard must accept every real outcome and reject the impossible ones."""
    from deepagents_code.configuration.writer import WriteResult

    assert WriteResult(True, True).changed is True
    assert WriteResult(True, False).changed is False
    assert WriteResult(False, False, "boom").ok is False
    # A success carrying an error detail is the fourth combination, and it
    # describes no transaction the writer can perform.
    with pytest.raises(ValueError, match="cannot carry an error"):
        WriteResult(True, True, "boom")
    with pytest.raises(ValueError, match="must carry an error"):
        WriteResult(False, False)


def test_writer_reports_caller_bugs_separately_from_disk_errors(
    tmp_path: Path,
) -> None:
    """A bug in the caller's closure must not read as a filesystem failure."""
    config_path = tmp_path / "config.toml"

    def broken(_data: dict[str, object]) -> bool:
        msg = "bug in caller"
        raise TypeError(msg)

    with pytest.raises(TypeError, match="bug in caller"):
        update_user_config(broken, config_path=config_path)

    unchanged = update_user_config(lambda _data: False, config_path=config_path)
    assert unchanged.ok is True
    assert unchanged.changed is False
    assert not config_path.exists()


def test_writer_reports_an_unparseable_existing_config_as_an_error(
    tmp_path: Path,
) -> None:
    """A corrupt file is refused so sibling sections are not truncated."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[broken", encoding="utf-8")

    result = update_user_config(
        lambda data: data.setdefault("ui", {}).update(theme="x") or True,
        config_path=config_path,
    )
    assert result.ok is False
    assert result.error is not None
    assert "could not update" in result.error


def test_reload_keeps_a_user_shell_allow_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/reload` must not discard `[shell].allow_list` from the user's config.

    Regression: `_reload_values` resolved the option with `toml_data={}`, so it
    saw only env and managed layers. `Settings.from_environment` reads the user
    layer, so a reload reset the allow list to `None` and reported a change that
    never happened. `skills.extra_allowed_dirs` in the same function already
    read its user layer, which is what made the omission clearly unintentional.
    """
    from deepagents_code import model_config
    from deepagents_code.config import Settings
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[shell]\nallow_list = ["git status"]\n', encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        runtime = Settings.from_environment(start_path=tmp_path)
        before = runtime.shell_allow_list
        assert before is not None
        runtime.reload_from_environment(start_path=tmp_path)
        assert runtime.shell_allow_list == before
    finally:
        service.invalidate_config_sources()


def test_managed_shell_allow_list_still_wins_a_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading the user layer on reload must not cost managed precedence."""
    from deepagents_code import model_config
    from deepagents_code.config import Settings
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[shell]\nallow_list = ["git status"]\n', encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        runtime = Settings.from_environment(start_path=tmp_path)
        runtime.reload_from_environment(start_path=tmp_path)
        assert runtime.shell_allow_list == ["ls"]
    finally:
        service.invalidate_config_sources()


def test_every_enforced_managed_key_resolves_to_a_manifest_option() -> None:
    """Pin `ENFORCED_MANAGED_KEYS` to the manifest.

    `managed_policy_violations` skips a key whose option it cannot resolve, and
    it skips silently: no log, no violation, no failure. So renaming or
    regrouping a manifest key would quietly drop fail-closed enforcement for a
    privilege-granting setting while every other test stayed green. Nothing
    pinned the tuple before this test.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.service import ENFORCED_MANAGED_KEYS

    unresolved = [key for key in ENFORCED_MANAGED_KEYS if get_option(key) is None]
    assert unresolved == []
    without_toml_keys = []
    for key in ENFORCED_MANAGED_KEYS:
        option = get_option(key)
        assert option is not None
        if not option.toml_keys:
            without_toml_keys.append(key)
    assert without_toml_keys == []


def test_enforced_managed_keys_actually_produce_violations() -> None:
    """Every enforced key must reject a value the manifest cannot apply.

    Resolving to an option is necessary but not sufficient: a `STRUCTURED`
    option always reports its managed value as managed-sourced, so listing one
    here would imply enforcement that never fires. This asserts each key can
    really produce a violation.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.service import (
        ENFORCED_MANAGED_KEYS,
        managed_policy_violations,
    )

    unenforceable = []
    for key in ENFORCED_MANAGED_KEYS:
        option = get_option(key)
        assert option is not None
        toml_keys = option.toml_keys
        assert toml_keys
        managed: dict[str, Any] = {}
        node: dict[str, Any] = managed
        for part in toml_keys[:-1]:
            child: dict[str, Any] = {}
            node[part] = child
            node = child
        # A table is never a valid value for any manifest scalar kind.
        node[toml_keys[-1]] = {"not": "a scalar"}
        if key not in managed_policy_violations(managed):
            unenforceable.append(key)
    assert unenforceable == []


def _managed_policy_args() -> argparse.Namespace:
    """Return a namespace shaped like the parsed agent-launch arguments.

    `sandbox` is `"none"` rather than `None` because that is what `parse_args`
    produces for an omitted `--sandbox` (the argument declares
    `default="none"`). Using `None` here made the "does not force a sandbox"
    regression tests pass against code that forced one.

    Every field managed policy revokes starts at a *user-set* value, never at
    the empty default. `interpreter_tools=None` made the assertion that managed
    `interpreter.ptc` clears it unfalsifiable: the field already held `None`, so
    a regression that stopped clearing a user's `--interpreter-tools all` passed
    the test. The same held for `interpreter`.
    """
    return argparse.Namespace(
        model=None,
        auto_classifier_model=None,
        interpreter=False,
        recursion_limit=None,
        sandbox="none",
        interpreter_tools="all",
        shell_allow_list="all",
        auto_approve=False,
        yolo=True,
    )


@pytest.mark.parametrize(
    ("managed_toml", "expected_exit"),
    [
        ('[startup]\nmode = "YOLO"\n', True),
        ("[runtime]\nrecursion_limit = 3\n", True),
        ("[shell]\nallow_list = 5\n", True),
        ("[skills]\nextra_allowed_dirs = 5\n", True),
        ("[models]\nauto_classifier = 4\n", True),
        ("[sandboxes]\ndefault = 5\n", True),
        ("[interpreter]\nptc = 5\n", True),
        ("[interpreter]\nenable_interpreter = 5\n", True),
        ('[startup]\nyolo_switcher = "false"\n', True),
        ('[interpreter]\nptc_acknowledge_unsafe = "yes"\n', True),
        ('[tracing]\nlangsmith_redact = "yes"\n', True),
        # A scalar where the table belongs shadows the key it should hold.
        ('startup = "manual"\n', True),
        ('shell = "ls"\n', True),
        ("skills = 5\n", True),
        ('[startup]\nmode = "manual"\n', False),
        ("[runtime]\nrecursion_limit = 500\n", False),
    ],
    ids=[
        "bad-startup-mode",
        "out-of-range-limit",
        "bad-shell-allow-list",
        "bad-skills-dirs",
        "bad-auto-classifier",
        "bad-sandbox",
        "bad-ptc",
        "bad-interpreter-toggle",
        "bad-yolo-switcher",
        "bad-ptc-acknowledge",
        "bad-langsmith-redact",
        "shadowed-startup",
        "shadowed-shell",
        "shadowed-skills",
        "valid-startup-mode",
        "valid-limit",
    ],
)
def test_rejected_managed_privilege_value_stops_the_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_toml: str,
    expected_exit: bool,
) -> None:
    """A privilege key the manifest rejects must not resolve in the user's favor.

    Skipping the value left `--yolo` and `--shell-allow-list all` in force, so
    an administrator typo granted exactly the escalation policy forbade.

    Regression: a *shadowed* path (`startup = "manual"` instead of `[startup]`
    plus `mode`) read as "the administrator wrote nothing", so the same typo
    one level up still granted the escalation, silently.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        if expected_exit:
            with pytest.raises(SystemExit) as excinfo:
                main._apply_managed_runtime_exceptions(args)
            assert excinfo.value.code == 78
        else:
            main._apply_managed_runtime_exceptions(args)
    finally:
        service.invalidate_config_sources()


def test_managed_auto_mode_does_not_set_the_interactive_only_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed policy revokes flags; it never sets `--auto-approve`.

    Regression: assigning the flag positively made every headless launch exit 2
    with "--auto-approve is only supported in interactive mode", naming a flag
    the user never passed. That launch now warns and continues instead, and the
    warning keys off a parse-time capture that a positive value here would not
    reach — but the flag would still misreport user intent to every other
    reader of `args`. `_resolve_approval_mode` already ends at
    `coerce_approval_mode(load_startup_mode())`, which reads merged managed
    policy, so the positive value needs no flag.
    """
    from deepagents_code import main, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "auto"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args).value == "auto"
        assert args.auto_approve is False
        assert args.yolo is True
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_startup_mode_masks_a_cli_approval_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed mode revokes `--yolo`, which the ACP launch once kept.

    Regression: `_apply_managed_runtime_policy` revoked the raw approval flags
    managed `startup.mode` masked, but `_apply_managed_runtime_exceptions` does
    not, and the ACP branch read the raw flags — `dcode --acp --yolo` launched
    with approvals disabled despite managed `startup.mode = "manual"`. The
    branch now resolves the mode instead, so the managed tier masks the
    non-durable CLI value.
    """
    from deepagents_code import main, model_config
    from deepagents_code.approval_mode import ApprovalMode
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args) is ApprovalMode.MANUAL
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_cli_approval_flag_outranks_user_startup_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmanaged `--yolo` still masks a user-configured mode.

    The CLI tier is the non-durable one: it masks the durable user tier, and
    only a managed tier masks it. An administrator's empty `managed.toml`
    installs the tier but declares no mode, so it changes nothing.
    """
    from deepagents_code import main, model_config
    from deepagents_code.approval_mode import ApprovalMode
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args) is ApprovalMode.YOLO
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_sandbox_default_does_not_force_a_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sandboxes.default` names a backend; it does not turn sandboxing on.

    Assigning it unconditionally forced every launch into a remote sandbox,
    which is not what the key documents. Both spellings of "no sandbox" have to
    be left alone: an omitted `--sandbox` arrives as `"none"` from argparse, and
    an explicit `--sandbox none` arrives the same way.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[sandboxes]\ndefault = "modal"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        for unsandboxed in ("none", None):
            args = _managed_policy_args()
            args.sandbox = unsandboxed
            main._apply_managed_runtime_exceptions(args)
            assert args.sandbox == unsandboxed
    finally:
        service.invalidate_config_sources()


def test_unavailable_managed_sandbox_leaves_an_unsandboxed_launch_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch that asked for no sandbox must not die on a managed backend.

    Regression: the guard checked only `None`, so a bare `dcode` reached the
    availability check and exited 78 over a backend it was never going to use.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[sandboxes]\ndefault = "not-a-real-provider"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_exceptions(args)
        assert args.sandbox == "none"
    finally:
        service.invalidate_config_sources()


def test_unavailable_managed_sandbox_stops_a_sandboxed_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed backend no provider answers to must not reach the factory.

    `parse_args` validates `--sandbox`, but it runs before managed policy is
    applied, so the managed value skipped `is_available` entirely.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[sandboxes]\ndefault = "not-a-real-provider"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.sandbox = "not-a-real-provider"
    try:
        with pytest.raises(SystemExit) as excinfo:
            main._apply_managed_runtime_exceptions(args)
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


def _managed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed_toml: str
) -> Path:
    """Point every layer at `tmp_path` with `managed_toml` as managed policy.

    Returns:
        The managed file path.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    return managed


@pytest.mark.parametrize(
    "argv",
    [
        ["dcode", "config", "path"],
        ["dcode", "doctor"],
        ["dcode", "--help"],
        ["dcode", "help"],
    ],
    ids=["config", "doctor", "help-flag", "help-command"],
)
def test_diagnostic_commands_run_with_unusable_managed_policy(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The administrator must keep the tools that explain a broken policy file.

    If the startup gate moved above these early returns, the only commands that
    report the managed path and its parse health would be the ones the broken
    file blocks.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    monkeypatch.setattr(sys, "argv", argv)
    # Some of these exit and some return; neither may be the policy gate.
    exit_code: object = None
    try:
        try:
            main.cli_main()
        except SystemExit as exc:
            exit_code = exc.code
    finally:
        service.invalidate_config_sources()
    assert exit_code != 78


def test_agent_launch_commands_stop_on_unusable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that runs the agent must not start without enforceable policy."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    monkeypatch.setattr(sys, "argv", ["dcode", "tools", "list"])
    try:
        with pytest.raises(SystemExit) as excinfo:
            main.cli_main()
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


@pytest.mark.self_managed_update_check
@pytest.mark.parametrize(
    "managed_toml",
    [
        "[broken",
        "[update]\ncheck = 5\nauto_update = 5\n",
        '[update]\ncheck = "false"\nauto_update = "false"\n',
    ],
    ids=["unparseable", "wrong-type", "quoted-boolean"],
)
def test_update_settings_fail_closed_on_any_managed_policy_error(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature that reaches the network turns off on any policy error.

    A present-but-unreadable value takes the same branch as an unreadable file.
    Falling through to the lower layers inverted the risk: *deleting* the
    managed file forced auto-update off, while a typo like `auto_update =
    "false"` handed the decision back to the user's own preference — so an
    administrator locking down a fleet silently kept the permissive default.

    A user layer is written underneath so this proves the managed error wins,
    rather than agreeing with the built-in default by coincidence.
    """
    from deepagents_code import model_config, update_check
    from deepagents_code.configuration import service

    managed = _managed_only(tmp_path, monkeypatch, managed_toml)
    assert managed.exists()
    model_config.DEFAULT_CONFIG_PATH.write_text(
        "[update]\ncheck = true\nauto_update = true\n", encoding="utf-8"
    )
    service.invalidate_config_sources()
    try:
        assert update_check.is_update_check_enabled() is False
        assert update_check.is_auto_update_enabled() is False
    finally:
        service.invalidate_config_sources()


@pytest.mark.self_managed_update_check
def test_managed_update_policy_outranks_the_user_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[update].check = false` cannot be re-enabled locally."""
    from deepagents_code import update_check
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[update]\ncheck = false\n")
    try:
        assert update_check.is_update_check_enabled() is False
    finally:
        service.invalidate_config_sources()


def test_reenabling_a_managed_denied_server_reports_the_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save succeeds, and the user is told policy keeps the server off."""
    from deepagents_code import mcp_disabled
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[mcp]\ndisabled_servers = ["github"]\n')
    try:
        ok, detail = mcp_disabled.set_server_disabled("github", False)
        assert ok is True
        assert detail is not None
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_reenabling_a_server_fails_closed_when_policy_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-enabling must not proceed while the deny list cannot be read."""
    from deepagents_code import mcp_disabled
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        ok, detail = mcp_disabled.set_server_disabled("github", False)
        assert ok is False
        assert detail is not None
    finally:
        service.invalidate_config_sources()


def test_managed_sandbox_settings_survive_an_unusable_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sandbox selection is a containment boundary, so policy outlives a typo."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service
    from deepagents_code.integrations import sandbox_config
    from deepagents_code.integrations.sandbox_config import SandboxConfig

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[sandboxes]\ndefault = "modal"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    # `sandbox_config` binds the default path at import, so patching
    # `model_config` alone leaves it reading the real user config.
    monkeypatch.setattr(sandbox_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = SandboxConfig.load()
        assert config.default == "modal"
        # The user layer failed, and that is reported without dropping policy.
        assert config.parse_error is not None
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_async_subagents_survive_an_unusable_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[async_subagents]` table still defines its agents."""
    from deepagents_code import model_config
    from deepagents_code.agent import load_async_subagents
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'url = "https://example.langsmith.dev"\n'
        'graph_id = "agent"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        subagents = load_async_subagents()
        assert [entry["name"] for entry in subagents] == ["researcher"]
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_doctor_reports_managed_parse_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` must explain the file that just stopped the launch."""
    from deepagents_code.configuration import service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        item = _managed_config_diagnostic()
        assert item.ok is False
        assert str(managed) in item.value
        assert "administrator" in item.value
    finally:
        service.invalidate_config_sources()


def test_a_failed_ui_write_reports_its_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure must tell the user why, not just that it failed.

    Regression: the toast dropped `WriteResult.error` and sent the detail to a
    logger that has no handler in the TUI outside debug mode, so a read-only home
    directory and a full disk produced the same message.
    """
    from deepagents_code import app, model_config
    from deepagents_code.configuration import service, writer

    target = tmp_path / "config.toml"
    target.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", target)
    monkeypatch.setattr(writer, "DEFAULT_CONFIG_PATH", target, raising=False)

    def fail(*_args: object, **_kwargs: object) -> writer.WriteResult:
        return writer.WriteResult(
            False, False, f"could not update {target}: [Errno 13] Permission denied"
        )

    monkeypatch.setattr(app, "update_user_config", fail, raising=False)
    monkeypatch.setattr(writer, "update_user_config", fail)
    service.invalidate_config_sources()
    try:
        result = app._save_ui_bool_result(
            toml_key="show_message_timestamps",
            option_key="display.show_message_timestamps",
            value=True,
            failure_message="Timestamps toggled for this session.",
        )
    finally:
        service.invalidate_config_sources()

    assert result.ok is False
    assert result.message is not None
    assert "Permission denied" in result.message


def test_a_failed_mcp_toggle_reports_its_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`set_server_disabled` must surface the writer's reason.

    Regression: `_save_disabled_entry` returned a bare `bool`, so the caller
    reported "could not write <path>" and dropped "Permission denied", "No space
    left on device", and the missing-`tomli_w` case the writer catches.
    """
    from deepagents_code import mcp_disabled
    from deepagents_code.configuration import writer

    target = tmp_path / "config.toml"
    target.write_text("", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> writer.WriteResult:
        return writer.WriteResult(
            False, False, f"could not update {target}: [Errno 28] No space left"
        )

    monkeypatch.setattr(writer, "update_user_config", fail)
    ok, detail = mcp_disabled.set_server_disabled(
        "srv", disabled=True, config_path=target
    )

    assert ok is False
    assert detail is not None
    assert "No space left" in detail


def test_both_layer_read_errors_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt user file and a corrupt managed file must both be named.

    Regression: `read_error` was a single string assigned at seven sites. The
    user-layer branches run first and the managed-layer branches second, so the
    managed message overwrote the user one and a user with both problems was
    told about the managed file only.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[also-broken", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.read_error is not None
    assert str(user) in trust.read_error
    assert str(managed) in trust.read_error


def test_thread_config_is_not_cached_while_the_user_config_is_broken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded thread config must not outlive the repair.

    Regression: `_load_effective_config_data` stopped raising for a bad user file
    on the default path — it logs and returns managed-only data — so the
    `except (OSError, TOMLDecodeError)` guard that used to prevent caching went
    dead. The defaults-only result was then cached for the process lifetime and
    survived the user fixing `config.toml`, because nothing on the read path
    calls `invalidate_thread_config_cache`.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        degraded = model_config.load_thread_config()
        assert degraded.sort_order == "updated_at"

        user.write_text('[threads]\nsort_order = "created_at"\n', encoding="utf-8")
        service.invalidate_config_sources()

        repaired = model_config.load_thread_config()
        assert repaired.sort_order == "created_at"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_auto_classifier_does_not_set_the_acp_incompatible_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed policy must not set `--auto-classifier-model`.

    Regression: assigning the flag made every ACP launch exit 2 on the
    `--auto-classifier-model` approval gate, naming a flag the user never
    passed. `build_server_config` falls through to
    `resolve_auto_classifier_model_with_source` when the flag is unset, and that
    already reads managed policy at top precedence, so the positive value needs
    no flag — the same reasoning the `startup.mode` block uses.
    """
    from deepagents_code import main, model_config
    from deepagents_code.config_manifest import (
        resolve_auto_classifier_model_with_source,
    )
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nauto_classifier = "openai:gpt-4o-mini"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_exceptions(args)
        assert args.auto_classifier_model is None
        # The value still reaches the runtime through the manifest resolver.
        sources = service.get_config_sources()
        resolved, source = resolve_auto_classifier_model_with_source(
            toml_data=dict(sources.user.data),
            managed_toml_data=sources.managed.data,
        )
        assert resolved == "openai:gpt-4o-mini"
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


def test_managed_default_model_replaces_the_cli_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed launch model must displace an explicit `--model`."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\ndefault = "managed:model"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.model = "user:model"
    try:
        main._apply_managed_runtime_exceptions(args)
        assert args.model == "managed:model"
    finally:
        service.invalidate_config_sources()


def test_managed_interpreter_ptc_clears_the_cli_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed PTC policy must prevent raw CLI forwarding to the server."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[interpreter]\nptc = ["read_file"]\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    assert args.interpreter_tools == "all"
    try:
        main._apply_managed_runtime_exceptions(args)
        assert args.interpreter_tools is None
    finally:
        service.invalidate_config_sources()


def test_merge_provenance_reports_only_real_leaves() -> None:
    """Provenance must carry no empty-root key and no stale parent entry.

    Two regressions, both in the output an administrator reads to audit what
    policy enforces. An empty lower table at the root joined `()` into the key
    `""`, which every merge produced on a machine with no user `config.toml`. And
    a lower empty table that the higher table filled kept an entry for the table
    itself, claiming a table was a user-controlled leaf next to the managed
    leaves inside it.
    """
    from deepagents_code.configuration.resolver import merge_toml_tables

    _, empty = merge_toml_tables(
        {},
        {},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert empty == {}

    _, managed_only = merge_toml_tables(
        {},
        {"startup": {"mode": "manual"}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert managed_only == {"startup.mode": "managed config"}

    _, filled = merge_toml_tables(
        {"a": {"b": {}}},
        {"a": {"b": {"c": 1}}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert filled == {"a.b.c": "managed config"}

    # An empty table the higher layer does not fill is still a real leaf: it is
    # the only record that the user declared that section.
    _, untouched = merge_toml_tables(
        {"a": {"b": {}}},
        {},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert untouched == {"a.b": "config.toml"}


def test_structured_resolution_matches_the_effective_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranked resolution and `ConfigSources.merged` must not disagree.

    Regression: the structured resolution branch merged without
    `higher_leaf_is_valid`, and the merger gates the "managed scalar displaces a
    user table" rule on that argument being `None`. So a managed scalar
    colliding with a user table resolved one way for the runtime (which reads
    `merged`) and the other way for `dcode config`, whose row takes `value` from
    the resolver and `provenance` from the validated merge — a row could show
    the user's table as effective while claiming managed policy owned that leaf.
    """
    from deepagents_code import model_config
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        "[themes.mytheme.nested]\nprimary = '#ffffff'\n",
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text('[themes]\nmytheme = "pinned"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        option = get_option("display.themes")
        assert option is not None
        sources = service.get_config_sources()
        merged, _ = sources.merged()
        value, _source = _resolve(
            option,
            toml_data=dict(sources.user.data),
            managed_toml_data=sources.managed.data,
        )
        assert value == merged["themes"]
        assert value == {"mytheme": "pinned"}
    finally:
        service.invalidate_config_sources()


def test_diagnostics_report_an_unenforceable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file that parses but cannot be enforced must not read as `ok`.

    Regression: all three surfaces branched on `ProviderStatus.usable`, which is
    true for a file whose health is `OK`. So the `ManagedPolicyError` half of
    exit 78 produced a green `doctor` row, a clean `dcode config` table with no
    warning, and `ok` from `dcode config path` — telling a user whose launch was
    just refused that managed config was fine. `config` and `doctor` are exempt
    from the startup gate precisely so they can explain this.
    """
    from deepagents_code.client.commands.config import (
        _MANAGED_PATH_LABEL,
        _config_path_status,
        _managed_health_warning,
    )
    from deepagents_code.configuration import service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = _managed_only(tmp_path, monkeypatch, '[startup]\nmode = "YOLO"\n')
    try:
        assert service.managed_config_status(refresh=True).usable is True

        item = _managed_config_diagnostic()
        assert item.ok is False
        assert "startup.mode" in item.value
        assert str(managed) in item.value

        warning = _managed_health_warning()
        assert warning is not None
        assert "startup.mode" in warning

        assert _config_path_status(_MANAGED_PATH_LABEL, exists=True) == "rejected"
    finally:
        service.invalidate_config_sources()


def test_project_dotenv_path_reports_disabled_when_reading_off() -> None:
    """A skipped project `.env` must not read as a live `ok` config source.

    `dcode config path` lists the project `.env` whether or not it is loaded;
    with `startup.read_project_dotenv` off the file exists but is skipped at
    bootstrap, so the row says `disabled` instead of `ok`.
    """
    from deepagents_code.client.commands.config import _config_path_status

    assert (
        _config_path_status("project .env", exists=True, project_dotenv_enabled=False)
        == "disabled"
    )
    # Enabled and missing cases are unaffected.
    assert (
        _config_path_status("project .env", exists=True, project_dotenv_enabled=True)
        == "ok"
    )
    assert (
        _config_path_status("project .env", exists=False, project_dotenv_enabled=False)
        == "disabled"
    )
    assert _config_path_status("global .env", exists=True) == "ok"


def test_a_guessed_managed_path_is_not_a_clean_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed registry read must not look like "no policy deployed".

    The guessed path holds no file, which is the same state as a machine with
    no managed policy at all. Reporting that as `MISSING` made every managed
    setting silently inert on a host whose ProgramData is relocated: `MISSING`
    is usable, so the startup gate passed and every reader saw an empty managed
    table. The reason has to reach the status, and the status has to be
    unusable.
    """
    from deepagents_code.configuration import paths, service
    from deepagents_code.configuration.paths import ResolvedManagedPath

    guessed = ResolvedManagedPath(
        Path("/nonexistent/managed.toml"), "registry unreadable"
    )
    for module in (paths, service):
        monkeypatch.setattr(module, "resolve_managed_path", lambda **_k: guessed)
    service.invalidate_config_sources()
    try:
        status = service.managed_config_status(refresh=True)
        assert status.health is ProviderHealth.INDETERMINATE
        assert status.detail == "registry unreadable"
        assert status.usable is False
        # The gate must refuse the launch rather than run with no policy.
        with pytest.raises(service.ManagedConfigError) as excinfo:
            service.require_healthy_managed_config(refresh=True)
        assert "registry unreadable" in str(excinfo.value)
    finally:
        service.invalidate_config_sources()


def test_default_read_includes_policy_and_an_explicit_path_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a default read carries policy, and that is not a caller's choice.

    An excluded managed layer reports `MISSING` with an empty table, which is
    indistinguishable from a machine with no policy installed. Deriving it from
    `user_path` keeps that state out of reach of a keyword argument.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\ndefault = "managed:model"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        default_read = service.get_config_sources()
        assert default_read.managed.data != {}
        assert default_read.merged()[0]["models"]["default"] == "managed:model"

        isolated = service.get_config_sources(user_path=user)
        assert isolated.managed.data == {}
        assert "models" not in isolated.merged()[0]
    finally:
        service.invalidate_config_sources()


async def test_unreadable_managed_policy_disables_every_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deny decision must reach "which servers start", not just a predicate.

    `is_server_disabled` failing closed is not enough: this is the call that
    turns policy into running processes.
    """
    import json
    from unittest.mock import AsyncMock, MagicMock

    from deepagents_code import mcp_tools
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    explicit = tmp_path / "mcp.json"
    explicit.write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx", "args": []}}}),
        encoding="utf-8",
    )
    load = AsyncMock(return_value=([], None, []))
    monkeypatch.setattr(mcp_tools, "_load_tools_from_config", load)
    monkeypatch.setattr(
        mcp_tools,
        "discover_mcp_config_sources",
        MagicMock(return_value=[]),
    )
    try:
        tools, manager, infos = await mcp_tools.resolve_and_load_mcp_tools(
            explicit_config_path=str(explicit),
            trust_project_mcp=True,
        )
        assert tools == []
        assert manager is None
        assert [info.status for info in infos] == ["disabled"]
        # No server may reach the loader at all.
        load.assert_not_called()
    finally:
        service.invalidate_config_sources()


async def test_server_mode_refuses_to_build_a_graph_without_enforceable_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server gate is a second entry point with the same duty as the CLI."""
    from deepagents_code import server_graph
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    failures: list[BaseException] = []
    monkeypatch.setattr(server_graph, "emit_startup_failure", failures.append)
    try:
        factory = server_graph._build_graph_factory(builder=None)
        with pytest.raises(SystemExit) as excinfo:
            await factory()
        assert excinfo.value.code == 1
        assert isinstance(failures[0], ManagedConfigError)
    finally:
        service.invalidate_config_sources()


def test_corrupt_managed_policy_fails_the_mcp_trust_lists_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable managed file must set `read_error`, keyed on its health."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
    finally:
        service.invalidate_config_sources()


def test_managed_structured_table_displaces_a_valid_user_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented structured-table exception, pinned so it cannot drift.

    `is_valid_managed_scalar` accepts any value at a `STRUCTURED` path, so a
    wrong-typed managed table displaces the user's and the typed reader falls
    back to its default. `README.md` documents this; nothing tested it.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        '[models.providers.custom]\napi_key_env = "CUSTOM_KEY"\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\nproviders = "junk"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        assert model_config.ModelConfig.load().providers == {}
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_loaded_config_cannot_mutate_the_shared_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer must not be able to rewrite policy for the rest of the session.

    The managed snapshot is cached process-wide, so handing out a live sub-dict
    would let one caller's edit outlive its own read.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models.providers.corp]\napi_key_env = "CORP_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = model_config.ModelConfig.load()
        assert "corp" in config.providers
        config.providers["corp"]["api_key_env"] = "ATTACKER_KEY"
        snapshot = service.get_managed_snapshot()
        assert snapshot.data["models"]["providers"]["corp"]["api_key_env"] == "CORP_KEY"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_out_of_range_managed_recursion_limit_falls_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded resolver keeps the lower layer; the launch gate stops instead."""
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[runtime]\nrecursion_limit = 3\n")
    try:
        assert (
            resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 400}})
            == 400
        )
    finally:
        service.invalidate_config_sources()


def test_managed_theme_outranks_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[ui].theme` cannot be overridden by an exported variable."""
    from deepagents_code._env_vars import THEME
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[ui]\ntheme = "textual-dark"\n')
    monkeypatch.setenv(THEME, "nord")
    option = get_option("display.theme")
    assert option is not None
    try:
        value, source = _resolve(option, toml_data={})
        assert value == "textual-dark"
        assert source.startswith("managed config")
    finally:
        service.invalidate_config_sources()


def test_managed_yolo_switcher_removes_yolo_from_the_approval_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `yolo_switcher = false` must take YOLO out of Shift+Tab.

    The rejection half of this key was covered; the applied half was not. It
    reaches the runtime through the resolver's implicit managed tier rather
    than through `_apply_managed_runtime_exceptions`, so a reader switched to
    `managed_toml_data={}` — as two auto-classifier readers deliberately are —
    would make enforcement a silent no-op with the fail-closed test still green.
    """
    from deepagents_code import config, model_config
    from deepagents_code.approval_mode import ApprovalMode, next_approval_mode
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[startup]\nyolo_switcher = false\n")
    model_config.DEFAULT_CONFIG_PATH.write_text(
        "[startup]\nyolo_switcher = true\n", encoding="utf-8"
    )
    service.invalidate_config_sources()
    try:
        assert config.is_yolo_switcher_enabled() is False
        assert (
            next_approval_mode(
                ApprovalMode.AUTO,
                auto_eligible=True,
                yolo_switcher_enabled=config.is_yolo_switcher_enabled(),
            )
            is ApprovalMode.MANUAL
        )
    finally:
        service.invalidate_config_sources()


def test_managed_langsmith_redaction_outranks_a_user_opt_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `langsmith_redact = true` cannot be turned off locally."""
    from deepagents_code import config, model_config
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[tracing]\nlangsmith_redact = true\n")
    model_config.DEFAULT_CONFIG_PATH.write_text(
        "[tracing]\nlangsmith_redact = false\n", encoding="utf-8"
    )
    service.invalidate_config_sources()
    try:
        assert config.is_langsmith_redaction_enabled() is True
    finally:
        service.invalidate_config_sources()


def test_managed_ptc_acknowledgement_outranks_a_user_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed acknowledgement value wins over the user's own file.

    This key decides whether the interpreter may call every tool
    programmatically, so a user value must never outrank policy.
    """
    from deepagents_code import model_config
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
    )
    from deepagents_code.configuration import service

    _managed_only(
        tmp_path,
        monkeypatch,
        "[interpreter]\nptc_acknowledge_unsafe = false\n",
    )
    model_config.DEFAULT_CONFIG_PATH.write_text(
        "[interpreter]\nptc_acknowledge_unsafe = true\n", encoding="utf-8"
    )
    service.invalidate_config_sources()
    option = get_option("interpreter.ptc_acknowledge_unsafe")
    assert option is not None
    try:
        value, source = _resolve(option, toml_data=load_config_toml())
        assert value is False
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    "managed_toml",
    [
        pytest.param('[threads]\nrelative_time = "invalid"\n', id="wrong-typed-scalar"),
        pytest.param("[ui]\ncursor_style = 5\n", id="wrong-typed-ui-scalar"),
    ],
)
def test_a_benign_managed_typo_still_launches(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected non-enforced managed value must not stop a launch.

    The negative control for the fail-closed set. Only
    `ENFORCED_MANAGED_KEYS` and a malformed known section exit 78; every other
    rejected value falls through to the user tier by design. Without this,
    widening enforcement — adding a scalar key to the enforced tuple, or making
    the shape check reject wrong-typed leaves — would exit 78 on every machine
    whose administrator has a harmless typo, and the whole suite would stay
    green.

    The value is still *reported*, so the administrator can find it.
    """
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, managed_toml)
    try:
        # No exception: the launch gate accepts the file.
        service.require_healthy_managed_config(refresh=True)
        health = service.managed_health(refresh=True)
        assert health.ok is True
        assert health.violations == ()
        # But it must not be silent, or an administrator cannot learn the value
        # was dropped: the only other announcement is a `logger.warning` that
        # the package's in-memory handler keeps off stderr.
        assert health.rejections != ()
    finally:
        service.invalidate_config_sources()


def test_every_managed_table_path_is_enforced() -> None:
    """Each declared table path must really produce a shape violation.

    The analogue of `test_every_enforced_managed_key_resolves_to_a_manifest_option`
    for `MANAGED_TABLE_PATHS`. Two entries are not derivable from a manifest
    option's parents (`async_subagents` and `effort`), so a renamed section
    would silently stop being guarded — and a managed scalar would then replace
    the user's whole section instead of being rejected.
    """
    from deepagents_code.configuration.service import (
        MANAGED_TABLE_PATHS,
        managed_section_shape_violations,
    )

    unguarded = []
    for path in MANAGED_TABLE_PATHS:
        managed: dict[str, Any] = {}
        node: dict[str, Any] = managed
        for part in path[:-1]:
            child: dict[str, Any] = {}
            node[part] = child
            node = child
        # A scalar where the section belongs.
        node[path[-1]] = "not-a-table"
        if ".".join(path) not in managed_section_shape_violations(managed):
            unguarded.append(".".join(path))
    assert unguarded == []


def test_a_managed_scalar_cannot_replace_the_user_effort_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`[effort]` has no manifest option, so it needs its own shape guard.

    `README.md` promised that a scalar at a known section stops the launch, but
    the check derived its paths from manifest-backed options, and `[effort]` is
    read and written by `model_config` without one. A managed `effort = "bad"`
    was therefore accepted and replaced the user's entire table.
    """
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, 'effort = "bad"\n')
    try:
        with pytest.raises(service.ManagedPolicyError) as excinfo:
            service.require_healthy_managed_config(refresh=True)
        assert "effort" in str(excinfo.value)
    finally:
        service.invalidate_config_sources()


def test_a_managed_scalar_cannot_replace_the_user_effort_by_model_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nested `[effort.by_model]` table needs its own shape guard.

    Guarding `("effort",)` alone still passes when the managed file declares
    `[effort]` as a table, and no manifest option supplies a type for the
    `effort.by_model` path, so the merge validator accepted a managed
    `by_model = "bad"` and replaced the user's whole table with the scalar.
    `load_effort_for_model` then rejected the scalar and returned `None`,
    dropping the user's stored preference instead of leaving it effective.
    """
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[effort]\nby_model = "bad"\n')
    try:
        with pytest.raises(service.ManagedPolicyError) as excinfo:
            service.require_healthy_managed_config(refresh=True)
        assert "effort.by_model" in str(excinfo.value)
    finally:
        service.invalidate_config_sources()


def test_the_writer_refuses_the_managed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The managed tier is read-only by guard, not only by convention.

    `THREAT_MODEL.md` states that the CLI never writes the managed file. That
    held because no caller passed the path, which is not the same as being
    unable to.
    """
    from deepagents_code.configuration import writer

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)

    def mutate(data: dict[str, Any]) -> bool:
        data["startup"] = {"mode": "yolo"}
        return True

    result = writer.update_user_config(mutate, config_path=managed)
    assert result.ok is False
    assert result.changed is False
    assert result.error is not None
    assert "read-only" in result.error
    # The file on disk is untouched.
    assert 'mode = "manual"' in managed.read_text(encoding="utf-8")


def test_no_credential_option_reads_managed_policy() -> None:
    """Managed policy cannot supply a credential, so no reader may imply it.

    `_resolve` carried a managed branch ahead of the `auth.json` store that
    could never fire: resolution consults managed policy only for an
    option with `toml_keys`, and no credential option has them. If a credential
    option ever gains `toml_keys`, that decision should be deliberate — the
    managed file is world-readable by design.
    """
    from deepagents_code.config_manifest import get_config_options

    with_toml_keys = [
        option.key
        for option in get_config_options()
        if option.group == "Credentials" and option.toml_keys
    ]
    assert with_toml_keys == []


def test_failed_write_leaves_no_temporary_file_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write that fails mid-flight must not litter the config directory."""
    import tomli_w

    from deepagents_code.configuration import writer

    target = tmp_path / "config.toml"
    target.write_text('[ui]\ntheme = "dark"\n', encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(tomli_w, "dump", explode)
    result = writer.update_user_config(
        lambda data: bool(data.__setitem__("ui", {"theme": "light"})) or True,
        config_path=target,
    )
    assert result.ok is False
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_missing_writer_dependency_leaks_no_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An `ImportError` between `mkstemp` and `os.fdopen` must not leak an fd.

    Regression: `import tomli_w` sat inside the `try` that follows `mkstemp`, so
    on an install without the writer dependency the cleanup handler unlinked the
    temp path but never closed the descriptor — only `os.fdopen` takes ownership
    of it. Repeated failed writes exhausted the process fd limit.
    """
    import builtins

    from deepagents_code.configuration import writer

    target = tmp_path / "config.toml"
    target.write_text("", encoding="utf-8")

    real_import = builtins.__import__

    def refuse_tomli_w(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "tomli_w":
            msg = "simulated missing writer dependency"
            raise ImportError(msg)
        return real_import(name, globals_, locals_, fromlist, level)

    def open_descriptor_count() -> int:
        return len(list(Path("/dev/fd").iterdir()))

    monkeypatch.setattr(builtins, "__import__", refuse_tomli_w)
    before = open_descriptor_count()
    for _ in range(20):
        result = writer.update_user_config(
            lambda data: bool(data.__setitem__("ui", {"theme": "light"})) or True,
            config_path=target,
        )
        assert result.ok is False
    monkeypatch.undo()

    assert open_descriptor_count() == before
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("env_value", "user_toml", "expected_source"),
    [
        (None, '[shell]\nallow_list = ["ls"]\n', "config.toml"),
        ("git", '[shell]\nallow_list = ["ls"]\n', "env"),
    ],
    ids=["user-toml-honored", "env-outranks-user-toml"],
)
def test_shell_allow_list_reads_the_user_toml_below_the_environment(
    env_value: str | None,
    user_toml: str,
    expected_source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key gained `toml_keys`, so the TOML tier needs its own coverage.

    It was env-only before this feature, and it grants shell auto-approval, so
    the new tier is a user-writable permission surface.
    """
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option

    user = tmp_path / "config.toml"
    user.write_text(user_toml, encoding="utf-8")
    if env_value is None:
        monkeypatch.delenv(SHELL_ALLOW_LIST, raising=False)
    else:
        monkeypatch.setenv(SHELL_ALLOW_LIST, env_value)
    option = get_option("shell.allow_list")
    assert option is not None
    import tomllib

    with user.open("rb") as handle:
        toml_data = tomllib.load(handle)
    _, source = _resolve(option, toml_data=toml_data, managed_toml_data={})
    assert source.startswith(expected_source)


def test_managed_shell_allow_list_outranks_a_shell_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported allow list cannot defeat the managed one."""
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[shell]\nallow_list = ["ls"]\n')
    monkeypatch.setenv(SHELL_ALLOW_LIST, "all")
    option = get_option("shell.allow_list")
    assert option is not None
    try:
        value, source = _resolve(option, toml_data={})
        assert value == ["ls"]
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


def test_empty_managed_shell_allow_list_is_a_lockdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`allow_list = []` must remove every grant, not fall through to one."""
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[shell]\nallow_list = []\n")
    monkeypatch.setenv(SHELL_ALLOW_LIST, "all")
    option = get_option("shell.allow_list")
    assert option is not None
    try:
        value, source = _resolve(option, toml_data={})
        assert value is None
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


def test_saving_a_shadowed_ui_preference_says_policy_still_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save succeeds, and the user learns why nothing changed on screen.

    `README.md` advertises this notice for the theme, terminal-mapping,
    UI-toggle, and MCP-server screens; nothing tested any of them.
    """
    from deepagents_code import app, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[ui]\nshow_scrollbar = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        result = app._save_show_scrollbar_result(visible=False)
        assert result.ok is True
        assert result.message == (
            "Preference saved, but managed config remains effective."
        )
        # The preference is still written, so removing policy reveals it.
        assert "show_scrollbar = false" in user.read_text(encoding="utf-8")
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_saving_after_a_rejected_managed_reload_reports_retained_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed refresh must not inherit the retained snapshot's healthy status."""
    from deepagents_code import app, model_config
    from deepagents_code.configuration import resolver as resolver_module, service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[ui]\nshow_scrollbar = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        resolver_module.get_config_resolver()
        managed.write_text("[ui\n", encoding="utf-8")

        result = app._save_show_scrollbar_result(visible=False)

        assert result.ok is True
        assert result.message is not None
        assert "current managed config file was rejected" in result.message
        assert "last readable version remains effective" in result.message
        assert result.severity == "warning"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_saving_a_theme_reports_unreadable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable policy file must not read as "no policy" on theme writes.

    The theme writes cannot use `_managed_verdict` -- theme resolution is
    bespoke -- but they shared its bug: both read the empty table
    `load_managed_config_toml` returns for a file it could not read, so a
    broken policy file produced the plain success message.
    """
    from deepagents_code import app, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[ui\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        result = app._save_theme_preference_result("langchain")
        assert result.ok is True
        assert result.message is not None
        assert "unreadable" in result.message
        assert result.severity == "warning"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_saving_a_theme_after_a_rejected_reload_reports_retained_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Theme writes must consult shared provider health after a failed reload."""
    from deepagents_code import app, model_config
    from deepagents_code.configuration import resolver as resolver_module, service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[ui]\ntheme = "langchain"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        resolver_module.get_config_resolver()
        managed.write_text("[ui\n", encoding="utf-8")

        result = app._save_theme_preference_result("langchain")

        assert result.ok is True
        assert result.message is not None
        assert "current managed config file was rejected" in result.message
        assert "last readable version remains effective" in result.message
        assert result.severity == "warning"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_saving_past_a_malformed_ui_policy_says_it_was_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected managed entry is reported to whoever is at the keyboard.

    The administrator who wrote it is not here and never sees the log line, so
    the toast is the only signal their policy is inert. The `decided` sibling
    above was covered and this branch was not, which is how the missing
    diagnostics call on this path went unnoticed once already.
    """
    from deepagents_code import app, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[ui]\nshow_scrollbar = "yes"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        result = app._save_show_scrollbar_result(visible=False)
        assert result.ok is True
        assert result.message == (
            "Preference saved. A managed policy for this option was rejected "
            "as malformed and is not being applied."
        )
        assert result.severity == "warning"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def _reload_previous() -> dict[str, object]:
    """Return a `previous` mapping shaped like the reloadable settings."""
    from deepagents_code.config import _RELOADABLE_FIELDS

    previous: dict[str, object] = dict.fromkeys(_RELOADABLE_FIELDS)
    previous["shell_allow_list"] = ["ls"]
    previous["extra_skills_dirs"] = []
    return previous


@pytest.mark.parametrize(
    "managed_toml",
    ["[broken", "[shell]\nallow_list = 5\n"],
    ids=["unparseable", "unenforceable"],
)
def test_blocked_reload_keeps_policy_and_says_so(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload that cannot enforce policy keeps it and reports the block.

    Regression: an unenforceable managed value had no reload-time equivalent of
    the launch-time reject, so `/reload` silently downgraded the shell allow
    list to the user's env value. The empty change list also told the user
    nothing had happened.
    """
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config import Settings
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    previous = _reload_previous()
    try:
        refreshed, blocked = Settings._reload_values(
            start_path=tmp_path,
            env={SHELL_ALLOW_LIST: "all"},
            previous=previous,
        )
        assert refreshed == previous
        assert blocked is not None
        assert str(managed) in blocked
    finally:
        service.invalidate_config_sources()


def test_managed_startup_mode_revokes_a_user_yolo_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed policy outranks the `--yolo` flag it was written to forbid."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args).value == "manual"
    finally:
        service.invalidate_config_sources()

    assert args.yolo is True
    assert args.auto_approve is False


def test_managed_auto_classifier_clears_the_cli_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed classifier model outranks `--auto-classifier-model`.

    Regression: the hook left the flag in place, so `build_server_config` took
    it as an explicit user override and never reached the managed tier — the
    user-selected (weaker) classifier graded gated actions even though the key
    is listed as enforced. Clearing to `None` (not assigning the managed value)
    keeps ACP from synthesizing a `--auto-classifier-model` flag that exits 2
    without `--auto-approve`, and the fall-through resolver applies the managed
    value instead.
    """
    from deepagents_code import config as config_module, main, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\nauto_classifier = "anthropic:claude-opus-4-7"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.delenv("DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL", raising=False)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    args.auto_classifier_model = "openai:user-weaker-model"
    try:
        main._apply_managed_runtime_exceptions(args)
        assert args.auto_classifier_model is None
        # The managed value still reaches the runtime through the resolver the
        # flag normally defers to.
        assert config_module.resolve_auto_classifier_model_with_problem() == (
            "anthropic:claude-opus-4-7",
            None,
        )
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_shell_allow_list_masks_the_cli_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed allow list must displace `--shell-allow-list all`."""
    from deepagents_code import main
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        option = get_option("shell.allow_list")
        assert option is not None
        assert main._resolver_for_args(args).get(option).value == ["ls"]
    finally:
        service.invalidate_config_sources()

    assert args.shell_allow_list == "all"


def test_masked_cli_flag_warns_the_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag beaten by managed policy must say so on stderr.

    Regression: `masked_ranks` was computed and never read, so `--yolo` under
    a managed `manual` started in Manual, printed nothing and exited 0. The
    resolution loop cannot report it -- for a REPLACE option it breaks at the
    winning tier and never reaches the masked CLI entry.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args).value == "manual"
    finally:
        service.invalidate_config_sources()

    err = capsys.readouterr().err
    # Names the flag the user actually typed. `_managed_policy_args` sets
    # `--yolo`, so naming `--auto-approve` too would warn about a flag that
    # was never passed.
    assert "--yolo was ignored" in err
    assert "--auto-approve" not in err
    assert "managed config takes precedence" in err


@pytest.mark.parametrize(
    ("key", "managed_toml", "cli_args", "negative_flag", "positive_flag"),
    [
        (
            "interpreter.enable_interpreter",
            {"interpreter": {"enable_interpreter": True}},
            {"interpreter": False},
            "--no-interpreter",
            "--interpreter",
        ),
        (
            "threads.relative_time",
            {"threads": {"relative_time": True}},
            {"relative": False},
            "--no-relative",
            "--relative",
        ),
    ],
)
def test_masked_negative_boolean_flag_warns_with_the_negative_spelling(
    key: str,
    managed_toml: dict[str, Any],
    cli_args: dict[str, object],
    negative_flag: str,
    positive_flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Managed-policy warnings must name the Boolean flag the user supplied."""
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.provider import CliProvider
    from deepagents_code.configuration.resolver import resolver_from_snapshots
    from deepagents_code.configuration.types import TomlSnapshot

    option = get_option(key)
    assert option is not None
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot.from_table("managed config", managed_toml),
        user=TomlSnapshot.from_table("config.toml", {}),
        cli_provider=CliProvider(cli_args),
    )

    _emit_ranked_diagnostics(option, resolver.get(option))

    err = capsys.readouterr().err
    assert f"{negative_flag} was ignored" in err
    assert f"{positive_flag} was ignored" not in err


def test_config_command_reports_the_cli_tier() -> None:
    """`dcode config` must credit a flag the user passed in this argv.

    Regression: `resolver_from_snapshots` gained an optional `cli_provider`
    and every pre-existing call site kept the default, so the command that
    users open *because* a flag is not taking was the one reader that could
    not see the CLI tier. It reported `default` for an option the current
    argv was setting.
    """
    from deepagents_code.client.commands.config import _resolve
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.provider import CliProvider
    from deepagents_code.configuration.resolver import (
        install_cli_provider,
        reset_config_resolver,
    )

    option = get_option("interpreter.ptc")
    assert option is not None
    install_cli_provider(CliProvider({"interpreter_tools": "task"}))
    try:
        overridden, source, value = _resolve(option, toml_data={}, managed_toml_data={})
    finally:
        reset_config_resolver()

    assert overridden is True
    assert source == "CLI argument"
    assert value == ["task"]


def test_rejected_cli_value_warns_the_user(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag value the provider refused must say so on stderr.

    Regression: the rejection reason went only to `logger.warning`, and the
    always-on buffer handler on the package logger means that reaches no
    stream at all. `dcode --interpreter-tools '[/tmp],all' config` therefore
    exited 0 and reported the option as `default` -- actively confirming the
    wrong hypothesis for a user debugging why their flag did nothing. The
    bracket-shaped input must also stay literal instead of being parsed as
    Rich markup.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.provider import CliProvider
    from deepagents_code.configuration.providers import TomlSnapshot
    from deepagents_code.configuration.resolver import resolver_from_snapshots

    option = get_option("interpreter.ptc")
    assert option is not None
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot.from_table("managed config", {}),
        user=TomlSnapshot.from_table("config.toml", {}),
        cli_provider=CliProvider({"interpreter_tools": "[/tmp],all"}),
    )
    resolved = resolver.get(option)
    _emit_ranked_diagnostics(option, resolved)

    err = capsys.readouterr().err
    assert "Warning:" in err
    assert "--interpreter-tools" in err
    assert "[/tmp],all" in err


def test_unmasked_cli_flag_is_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag that wins must not warn."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args).value == "yolo"
    finally:
        service.invalidate_config_sources()

    assert "was ignored" not in capsys.readouterr().err


def test_managed_recursion_limit_masks_the_cli_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed limit wins even when `--recursion-limit` is explicit.

    The parent resolves the full managed → CLI → env → TOML → default chain
    before serializing an explicit flag to the child server process.
    """
    from deepagents_code import main
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[runtime]\nrecursion_limit = 500\n", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._resolver_for_args(args)
        # No CLI flag: the managed value wins at build time.
        assert main._resolved_recursion_limit(args) is None
        assert resolve_recursion_limit() == 500
        # An explicit flag is serialized as the effective managed value.
        args.recursion_limit = 75
        assert main._resolved_recursion_limit(args) == 500
        assert resolve_recursion_limit() == 500
    finally:
        service.invalidate_config_sources()


def test_rejected_managed_limit_does_not_warn_about_the_honoured_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag the fall-through ends up honouring must not be called ignored.

    Regression: `_emit_ranked_diagnostics` fired on every iteration of the
    fall-through loop. With an out-of-range managed limit masking an explicit
    `--recursion-limit`, iteration 1 warned "was ignored: managed config takes
    precedence. Ask your administrator" and iteration 2 then returned the CLI
    value. A false warning is worse than silence: it sends the user to their
    administrator about a non-problem.
    """
    from deepagents_code import main
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    # Below `RECURSION_LIMIT_FLOOR`, so the managed tier is rejected and the
    # loop falls through to the CLI value.
    managed.write_text("[runtime]\nrecursion_limit = 5\n", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.recursion_limit = 7
    try:
        main._resolver_for_args(args)
        assert resolve_recursion_limit() == 7
    finally:
        service.invalidate_config_sources()

    assert "was ignored" not in capsys.readouterr().err


def test_masked_recursion_limit_still_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Deferring the warning must not silence a flag that genuinely lost."""
    from deepagents_code import main
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[runtime]\nrecursion_limit = 500\n", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.recursion_limit = 7
    try:
        main._resolver_for_args(args)
        assert resolve_recursion_limit() == 500
    finally:
        service.invalidate_config_sources()

    err = capsys.readouterr().err
    assert "--recursion-limit was ignored" in err
    assert "managed config takes precedence" in err


def test_cli_recursion_limit_outranks_user_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmanaged `--recursion-limit` still masks the user config.

    The launcher resolves and forwards an explicit flag so it survives the
    server subprocess boundary; only the managed tier masks it.
    """
    from deepagents_code import main, model_config
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[runtime]\nrecursion_limit = 42\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    args.recursion_limit = 75
    try:
        main._resolver_for_args(args)
        assert main._resolved_recursion_limit(args) == 75
        assert resolve_recursion_limit() == 75
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


@pytest.mark.parametrize("limit", [1, 24, 100_001])
def test_positive_cli_recursion_limit_preserves_the_documented_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    """Every positive CLI limit wins over lower-ranked configuration."""
    from deepagents_code import _env_vars, main, model_config
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[runtime]\nrecursion_limit = 400\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "500")
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.recursion_limit = limit
    try:
        main._resolver_for_args(args)
        assert resolve_recursion_limit() == limit
        assert main._resolved_recursion_limit(args) == limit
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    ("registry_value", "expected_root"),
    [
        ("D:/SharedData", "D:/SharedData"),
        ("", "C:/ProgramData"),
        (None, "C:/ProgramData"),
    ],
)
def test_windows_program_data_comes_from_the_registry(
    monkeypatch: pytest.MonkeyPatch,
    registry_value: str | None,
    expected_root: str,
) -> None:
    """The Windows root is read from HKLM, never from a user-settable env var.

    A relocated ProgramData is a real enterprise configuration, and reading
    `%ProgramData%` would let any unprivileged user redirect the lookup.
    """
    import sys as _sys
    import types

    from deepagents_code.configuration import paths

    class _FakeKey:
        def __enter__(self) -> _FakeKey:  # noqa: PYI034 - local test double
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def query(_key: object, _name: str) -> tuple[object, int]:
        if registry_value is None:
            raise OSError
        return registry_value, 1

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=lambda *_a, **_k: _FakeKey(),
        QueryValueEx=query,
    )
    monkeypatch.setitem(_sys.modules, "winreg", fake)
    monkeypatch.setattr(paths.sys, "platform", "win32")

    # Target the helper directly: the autouse isolation fixture replaces both
    # public path entry points.
    root, fallback = paths._windows_program_data(None)
    assert root == expected_root
    # A guessed root must say so. Without the reason, an empty read at the
    # guessed path is indistinguishable from an administrator deploying no
    # policy at all, which is the fail-open this pairing exists to prevent.
    if registry_value:
        assert fallback is None
    else:
        assert fallback is not None


def test_disabled_server_write_recomputes_inside_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale pre-lock snapshot must not drop a concurrently disabled server.

    The writer recomputes the deny set from the table it parses inside the
    lock, so a disable that landed after the caller's read still survives.
    """
    from deepagents_code import mcp_disabled

    config_path = tmp_path / "config.toml"
    config_path.write_text('[mcp]\ndisabled_servers = ["a"]\n', encoding="utf-8")

    # Stand in for a concurrent writer: the caller's snapshot predates the
    # disable of "a" that is already on disk.
    monkeypatch.setattr(mcp_disabled, "_load_config", lambda _path: {})

    ok, _detail = mcp_disabled.set_server_disabled("b", True, config_path=config_path)
    assert ok

    import tomllib

    with config_path.open("rb") as handle:
        written = tomllib.load(handle)
    assert written["mcp"]["disabled_servers"] == ["a", "b"]


def test_startup_gate_exits_78_on_unusable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gated command must stop rather than run with policy unenforced."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(SystemExit) as excinfo:
            main._require_managed_config_or_exit()
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


def test_startup_gate_accepts_a_missing_managed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing file applies no policy and must not block any command."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    redirect_managed_config(monkeypatch, tmp_path / "absent.toml")
    service.invalidate_config_sources()
    try:
        main._require_managed_config_or_exit()
    finally:
        service.invalidate_config_sources()


def _remote_snapshot(
    response: _RemoteResponse,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TomlSnapshot:
    """Load one remote policy from a fixed fake response.

    Returns:
        The snapshot the remote provider produced.
    """
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    return RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()


@pytest.mark.parametrize("status", [204, 205, 206])
def test_remote_toml_provider_rejects_partial_success_status(
    status: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a 200 carries a whole policy, and `urllib` raises for neither.

    `HTTPErrorProcessor` raises only outside `200..299`, so a `206` arrives as
    a success. TOML cut at a line boundary still parses, so accepting one would
    enforce a policy with entries silently missing.
    """
    snapshot = _remote_snapshot(
        _RemoteResponse(b"[mcp]\n", status=status),
        tmp_path,
        monkeypatch,
    )

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert f"HTTP {status}" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_rejects_undelimited_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection-close framing cannot prove the policy arrived whole."""
    response = _RemoteResponse(b"[mcp]\n")
    del response.headers["Content-Length"]
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "did not delimit" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_accepts_chunked_framing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunked responses have no declared length and are still complete.

    `http.client` raises `IncompleteRead` when the terminating chunk is
    missing, so chunked framing detects its own truncation and does not need
    the `Content-Length` check.
    """
    snapshot = _remote_snapshot(
        _RemoteResponse(b'[startup]\nmode = "manual"\n', chunked=True),
        tmp_path,
        monkeypatch,
    )

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}


def test_remote_toml_provider_accepts_a_body_at_the_size_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit is inclusive, so a maximum-size policy still loads."""
    policy = b'[startup]\nmode = "manual"\n# '
    payload = policy + b"x" * (REMOTE_MANAGED_CONFIG_MAX_BYTES - len(policy))
    assert len(payload) == REMOTE_MANAGED_CONFIG_MAX_BYTES
    snapshot = _remote_snapshot(_RemoteResponse(payload), tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}


def test_remote_toml_provider_rejects_a_keyless_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document with no keys contradicts the descriptor that named it.

    The trust anchor asserts that the URL holds the complete managed policy, so
    an empty body is a failed publish, not an administrator enforcing nothing.
    Accepting it would let a bad deploy evict the last enforceable generation.
    """
    snapshot = _remote_snapshot(_RemoteResponse(b""), tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "declared no policy" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_rejects_an_over_declared_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body longer than its declared length is trailing garbage.

    The under-declared direction surfaces as `IncompleteRead`. This one is the
    smuggling shape: extra TOML appended past the length the framing promised.
    """
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n', content_length="5")
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "more than its declared" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_rejects_a_repeated_content_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lengths cannot both delimit the body, so neither is trusted.

    `Message.get` and `http.client` both take the *first* value, so a short
    first header would agree with itself and accept the prefix of a longer
    policy as a whole document -- silently dropping deny-list entries.
    """
    payload = b'[startup]\nmode = "manual"\n'
    response = _RemoteResponse(payload, content_length="5")
    response.headers["Content-Length"] = str(len(payload))
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "conflicting body lengths" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_rejects_conflicting_framing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Content-Length` with chunked framing disagree about where the body ends."""
    payload = b'[startup]\nmode = "manual"\n'
    response = _RemoteResponse(payload, content_length=str(len(payload)))
    response.headers["Transfer-Encoding"] = "chunked"
    response.chunked = True
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "conflicting body framing" in (snapshot.status.detail or "")
    assert not snapshot.data


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Content-Type": "text/html; charset=utf-8"}, "'text/html', not TOML"),
        ({"Content-Encoding": "gzip"}, "sent a compressed body"),
    ],
)
def test_remote_toml_provider_rejects_a_body_that_is_not_the_policy(
    headers: dict[str, str],
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 that is not TOML blames the server, not the published document.

    A captive portal, an SSO interstitial, and a gateway error page all answer
    200 with HTML. Reading them would fail in `tomllib` and report `CORRUPT`,
    which tells the administrator to repair a file that is byte-perfect.
    """
    response = _RemoteResponse(b"<html>Sign in</html>")
    for name, value in headers.items():
        response.headers[name] = value
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert expected in (snapshot.status.detail or "")
    assert not snapshot.data


@pytest.mark.parametrize(
    "content_type",
    ["application/toml", "text/plain; charset=utf-8", "application/octet-stream"],
)
def test_remote_toml_provider_accepts_declared_toml_media_types(
    content_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that labels the policy body is not penalized for doing so."""
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')
    response.headers["Content-Type"] = content_type
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}


def test_remote_toml_provider_reports_an_unenforceable_read_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response with no controllable socket fails closed, and says why.

    `fp.raw._sock` is a CPython internal. If it stops resolving, every remote
    fetch degrades -- so the detail must name the real cause instead of
    rendering as the generic read failure shared with a DNS miss.
    """
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')
    response.fp = SimpleNamespace(raw=SimpleNamespace())
    snapshot = _remote_snapshot(response, tmp_path, monkeypatch)

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "read timeout could not be enforced" in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_toml_provider_bounds_a_real_http_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private-attribute traversal must work on a genuine response.

    Every other test builds a double shaped to match `fp.raw._sock`, so none of
    them would notice a CPython layout change that breaks the real thing.
    """
    import socket as socket_module
    from http.client import HTTPResponse

    from deepagents_code.configuration import providers

    body = b'[startup]\nmode = "manual"\n'
    server, client = socket_module.socketpair()
    try:
        server.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body),
        )
        response = HTTPResponse(client)
        response.begin()

        class Opener:
            def open(self, _request: object, *, timeout: int) -> HTTPResponse:
                assert timeout > 0
                return response

        monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
        snapshot = RemoteTomlProvider(
            "managed config",
            "https://config.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
    finally:
        server.close()
        client.close()

    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {"startup": {"mode": "manual"}}


@pytest.mark.parametrize("transfer_encoding", ["gzip, chunked", "chunked "])
def test_remote_toml_provider_rejects_transfer_encoding_http_client_does_not_decode(
    transfer_encoding: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completeness must follow the real response's selected framing mode."""
    import socket as socket_module
    from http.client import HTTPResponse

    from deepagents_code.configuration import providers

    body = b'[startup]\nmode = "manual"\n'
    server, client = socket_module.socketpair()
    try:
        server.sendall(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: "
            + transfer_encoding.encode("ascii")
            + b"\r\n\r\n"
            + body
        )
        response = HTTPResponse(client)
        response.begin()
        assert response.chunked is False

        class Opener:
            def open(self, _request: object, *, timeout: int) -> HTTPResponse:
                assert timeout > 0
                return response

        monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
        snapshot = RemoteTomlProvider(
            "managed config",
            "https://config.example.com/policy.toml",
            tmp_path / "managed.toml",
        ).load()
    finally:
        server.close()
        client.close()

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert "unsupported transfer encoding" in (snapshot.status.detail or "")
    assert not snapshot.data


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "https://config.example.com:8443/policy.toml",
            "https://config.example.com:8443/policy.toml",
        ),
        ("https://[2001:db8::1]/policy.toml", "https://[2001:db8::1]/policy.toml"),
        (
            "https://CONFIG.Example.COM./policy.toml",
            "https://config.example.com/policy.toml",
        ),
    ],
)
def test_remote_url_normalization_preserves_the_destination(
    source: str,
    expected: str,
) -> None:
    """Normalizing must not repoint the fetch at a different service.

    Dropping the port rebuild would silently contact 443 instead of the port
    the administrator pinned; dropping the IPv6 re-bracketing would make every
    address-literal policy host unreachable.
    """
    from deepagents_code.configuration.providers import _validate_remote_url

    assert _validate_remote_url(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("https://config.example.com:99999/policy.toml", "invalid port"),
        ("https://config.example.com:abc/policy.toml", "invalid port"),
        (
            "https://:t0ken@config.example.com/policy.toml",
            "must not contain credentials",
        ),
        ("https://user:t0ken@ex℀ample.com/policy.toml", "is not a valid URL"),
    ],
)
def test_remote_url_rejections_name_their_cause_without_echoing_the_url(
    source: str,
    expected: str,
    tmp_path: Path,
) -> None:
    """Each rejection has its own message, and none forwards the URL.

    `urlsplit` puts the whole netloc -- `user:password@` included -- in its own
    `ValueError`, and this detail reaches exit-78 stderr and the `doctor` row.
    """
    snapshot = RemoteTomlProvider(
        "managed config", source, tmp_path / "managed.toml"
    ).load()

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert expected in (snapshot.status.detail or "")
    assert "t0ken" not in (snapshot.status.detail or "")
    assert snapshot.status.remote_source is None


def test_remote_policy_violation_names_the_url_not_the_trust_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unenforceable remote value blames the document that declares it.

    The sibling `ManagedConfigError` branch covers a failed fetch. This is the
    other half of exit 78: the fetch worked and the policy it returned cannot
    be applied. The local anchor holds only a URL, so telling the administrator
    to correct the value there names a file with no such key.
    """
    from deepagents_code.configuration import providers, service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b'[startup]\nmode = "not-a-real-mode"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()

    with pytest.raises(service.ManagedPolicyError) as caught:
        require_healthy_managed_config(refresh=True)

    message = str(caught.value)
    assert "https://config.example.com/policy.toml" in message
    assert "startup.mode" in message
    assert f"{managed} rejects" not in message


def test_managed_refresh_does_not_hold_the_snapshot_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow remote fetch cannot block an ordinary config read.

    Every config read reaches `get_managed_snapshot`, much of it from the
    Textual event loop. Offloading the fetch to a worker thread is pointless if
    the loop then waits on `_snapshot_lock` for the same five seconds.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    fetching = Event()
    release = Event()
    reader_returned = Event()

    def slow_load(_path: Path | None = None) -> TomlSnapshot:
        fetching.set()
        assert release.wait(timeout=5)
        return TomlSnapshot(
            {"startup": {"mode": "manual"}},
            ProviderStatus(
                "managed config",
                managed,
                ProviderHealth.OK,
                None,
                remote_source="https://config.example.com/policy.toml",
            ),
        )

    monkeypatch.setattr(service, "_load_managed", slow_load)

    def refresh() -> None:
        service.get_managed_snapshot(refresh=True)

    worker = Thread(target=refresh, name="managed-refresh", daemon=True)
    worker.start()
    assert fetching.wait(timeout=5)

    def read() -> None:
        with service._snapshot_lock:
            reader_returned.set()

    reader = Thread(target=read, name="config-reader", daemon=True)
    reader.start()
    # The lock must be free while the fetch is still in flight.
    assert reader_returned.wait(timeout=5)
    release.set()
    worker.join(timeout=5)
    reader.join(timeout=5)


def test_config_write_refresh_does_not_hold_the_resolver_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote refresh after a user write cannot block resolver reads."""
    from deepagents_code import model_config
    from deepagents_code.configuration import (
        resolver as resolver_module,
        service,
        writer,
    )

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("[feature]\nenabled = true\n", encoding="utf-8")
    managed.write_text("[feature]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    resolver = resolver_module.get_config_resolver()
    current = service.get_managed_snapshot()
    fetching = Event()
    release = Event()
    reader_returned = Event()

    def slow_load(_path: Path | None = None) -> TomlSnapshot:
        fetching.set()
        assert release.wait(timeout=5)
        return current

    monkeypatch.setattr(service, "_load_managed", slow_load)
    refresh = Thread(
        target=lambda: writer.refresh_shared_resolver(user),
        name="config-write-refresh",
        daemon=True,
    )
    refresh.start()
    assert fetching.wait(timeout=5)
    option = ConfigOption(
        key="feature.enabled",
        group="Test",
        summary="test",
        default=False,
        toml_keys=("feature", "enabled"),
        kind=OptionKind.BOOL,
    )

    def read() -> None:
        resolver.get(option)
        reader_returned.set()

    reader = Thread(target=read, name="config-reader", daemon=True)
    try:
        reader.start()
        assert reader_returned.wait(timeout=5)
    finally:
        release.set()
    refresh.join(timeout=5)
    reader.join(timeout=5)


def test_invalidation_bars_an_in_flight_load_from_republishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A load started before `invalidate_config_sources` must not resurrect it.

    The reset advances both tickets precisely so this cannot happen: without
    it, a load in flight across the invalidation would repopulate the cache
    with the generation the caller deliberately discarded, and the process
    would silently resume enforcing it.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    started = Event()
    release = Event()

    def slow_load(_path: Path | None = None) -> TomlSnapshot:
        started.set()
        assert release.wait(timeout=5)
        return TomlSnapshot(
            {"startup": {"mode": "discarded"}},
            ProviderStatus("managed config", managed, ProviderHealth.OK, None),
        )

    monkeypatch.setattr(service, "_load_managed", slow_load)
    loaded: list[TomlSnapshot] = []
    worker = Thread(
        target=lambda: loaded.append(service.get_managed_snapshot(refresh=True)),
        name="invalidated-refresh",
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=5)

    service.invalidate_config_sources()
    release.set()
    worker.join(timeout=5)

    # The load still returns its own result to the caller that asked for it,
    # but the cache stays empty so no later reader observes the cleared policy.
    assert loaded[0].data == {"startup": {"mode": "discarded"}}
    assert service._snapshot_state.managed is None


def test_earlier_load_publishes_when_a_later_one_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A good generation is not discarded because a later load started.

    Tickets order *publication*, not load start. Ordering on start would let
    any overlapping caller forfeit its right to publish, so a successful fetch
    racing a failing one would leave the cache empty -- every later reader
    re-fetching, and two readers moments apart disagreeing about whether
    policy is enforced at all.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    started = Event()
    release = Event()

    def good_load(_path: Path | None = None) -> TomlSnapshot:
        started.set()
        assert release.wait(timeout=5)
        return TomlSnapshot(
            {"startup": {"mode": "manual"}},
            ProviderStatus("managed config", managed, ProviderHealth.OK, None),
        )

    monkeypatch.setattr(service, "_load_managed", good_load)
    good_result: list[TomlSnapshot] = []
    worker = Thread(
        target=lambda: good_result.append(service.get_managed_snapshot(refresh=True)),
        name="good-refresh",
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=5)

    # A later refresh starts and fails while the good one is still in flight.
    monkeypatch.setattr(
        service,
        "_load_managed",
        lambda _path=None: TomlSnapshot(
            {},
            ProviderStatus(
                "managed config",
                managed,
                ProviderHealth.UNREADABLE,
                "remote source timed out",
            ),
        ),
    )
    assert service.get_managed_snapshot(refresh=True).status.health is (
        ProviderHealth.UNREADABLE
    )

    monkeypatch.setattr(service, "_load_managed", good_load)
    release.set()
    worker.join(timeout=5)

    assert good_result[0].data == {"startup": {"mode": "manual"}}
    assert service._snapshot_state.managed is not None
    assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}
    failure = service.managed_refresh_failure()
    assert failure is not None
    assert failure.health is ProviderHealth.UNREADABLE


def test_stale_refresh_cannot_overwrite_a_newer_published_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older in-flight load must not roll back a newer policy generation.

    Loads run outside `_snapshot_lock`, so two refreshes can overlap. If the
    managed file changes between their reads and the older read finishes last,
    publishing it would resume enforcing policy the administrator already
    replaced — including permissions the newer generation revoked.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    old_started = Event()
    finish_old = Event()

    def snapshot(mode: str) -> TomlSnapshot:
        return TomlSnapshot(
            {"startup": {"mode": mode}},
            ProviderStatus("managed config", managed, ProviderHealth.OK, None),
        )

    def old_load(_path: Path | None = None) -> TomlSnapshot:
        old_started.set()
        assert finish_old.wait(timeout=5)
        return snapshot("old")

    monkeypatch.setattr(service, "_load_managed", old_load)
    old_result: list[TomlSnapshot] = []

    def refresh_old() -> None:
        old_result.append(service.get_managed_snapshot(refresh=True))

    worker = Thread(target=refresh_old, name="stale-refresh", daemon=True)
    worker.start()
    assert old_started.wait(timeout=5)

    # A newer refresh publishes while the older load is still in flight.
    monkeypatch.setattr(service, "_load_managed", lambda _path=None: snapshot("new"))
    assert service.get_managed_snapshot(refresh=True).data == {
        "startup": {"mode": "new"}
    }

    finish_old.set()
    worker.join(timeout=5)

    # The stale load finishes last but must not overwrite the newer snapshot;
    # as a refresh caller it still receives the generation it loaded.
    assert old_result[0].data == {"startup": {"mode": "old"}}
    assert service.get_managed_snapshot().data == {"startup": {"mode": "new"}}


def test_newer_refresh_publishes_after_older_refresh_finishes_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each overlapping refresh receives its own publication ticket.

    The older refresh can publish while the newer one is still fetching. The
    newer result must remain eligible to replace it when that fetch finishes;
    otherwise revoked permissions stay active until another reload.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    old_started = Event()
    finish_old = Event()
    new_started = Event()
    finish_new = Event()
    load_number = 0

    def snapshot(mode: str) -> TomlSnapshot:
        return TomlSnapshot(
            {"startup": {"mode": mode}},
            ProviderStatus("managed config", managed, ProviderHealth.OK, None),
        )

    def overlapping_load(_path: Path | None = None) -> TomlSnapshot:
        nonlocal load_number
        load_number += 1
        if load_number == 1:
            old_started.set()
            assert finish_old.wait(timeout=5)
            return snapshot("old")
        new_started.set()
        assert finish_new.wait(timeout=5)
        return snapshot("new")

    monkeypatch.setattr(service, "_load_managed", overlapping_load)
    old_result: list[TomlSnapshot] = []
    new_result: list[TomlSnapshot] = []
    old_worker = Thread(
        target=lambda: old_result.append(service.get_managed_snapshot(refresh=True)),
        name="older-refresh",
        daemon=True,
    )
    new_worker = Thread(
        target=lambda: new_result.append(service.get_managed_snapshot(refresh=True)),
        name="newer-refresh",
        daemon=True,
    )

    old_worker.start()
    assert old_started.wait(timeout=5)
    new_worker.start()
    assert new_started.wait(timeout=5)
    finish_old.set()
    old_worker.join(timeout=5)
    assert old_result[0].data == {"startup": {"mode": "old"}}

    finish_new.set()
    new_worker.join(timeout=5)
    assert new_result[0].data == {"startup": {"mode": "new"}}
    assert service.get_managed_snapshot().data == {"startup": {"mode": "new"}}


def test_stale_refresh_failure_cannot_overwrite_newer_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure from an older load must not mark a healthy generation failed.

    Refresh A starts, refresh B publishes a healthy generation, then A
    finishes failing. Recording A's failure anyway would leave
    `managed_refresh_failure` and `doctor` reporting the current, enforceable
    policy as a stale refresh. Failure publication follows the same ticket
    ordering as snapshot publication.
    """
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import ProviderStatus, TomlSnapshot

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()

    old_started = Event()
    finish_old = Event()

    def old_load(_path: Path | None = None) -> TomlSnapshot:
        old_started.set()
        assert finish_old.wait(timeout=5)
        return TomlSnapshot(
            {},
            ProviderStatus(
                "managed config",
                managed,
                ProviderHealth.UNREADABLE,
                "remote source timed out",
            ),
        )

    monkeypatch.setattr(service, "_load_managed", old_load)
    old_result: list[TomlSnapshot] = []
    worker = Thread(
        target=lambda: old_result.append(service.get_managed_snapshot(refresh=True)),
        name="stale-failing-refresh",
        daemon=True,
    )
    worker.start()
    assert old_started.wait(timeout=5)

    # A newer refresh publishes healthy policy while the older load is in
    # flight. Advance the file so the read below does not take the cache
    # fast-path (`cached is not None`) and actually loads -- a refresh that
    # hits the cache never publishes, so it cannot participate in the race.
    # The published value must be enforceable: an unenforceable candidate is
    # never cached, so it would not advance the publication ticket either.
    managed.write_text('[startup]\nmode = "auto"\n', encoding="utf-8")
    monkeypatch.setattr(
        service,
        "_load_managed",
        lambda _path=None: TomlSnapshot(
            {"startup": {"mode": "auto"}},
            ProviderStatus("managed config", managed, ProviderHealth.OK, None),
        ),
    )
    assert service.get_managed_snapshot(refresh=True).data == {
        "startup": {"mode": "auto"}
    }

    finish_old.set()
    worker.join(timeout=5)

    # The refresh caller still receives its own failed generation, but the
    # recorded health stays with the published snapshot.
    assert old_result[0].status.health is ProviderHealth.UNREADABLE
    assert service.managed_refresh_failure() is None


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            URLError(ssl.SSLCertVerificationError("certificate has expired")),
            "SSLCertVerificationError",
        ),
        (URLError(socket.gaierror("name resolution failed")), "gaierror"),
        (URLError(ConnectionRefusedError("refused")), "ConnectionRefusedError"),
        (BadStatusLine("garbage"), "BadStatusLine"),
    ],
)
def test_remote_read_failure_names_its_cause(
    failure: Exception,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One fixed string cannot separate a bad certificate from a DNS miss.

    An expired or untrusted certificate on the policy host is the most
    security-relevant failure on this boundary, and the administrator has to be
    able to tell it apart from the network being down. The class name is a
    type, not server output, so it carries no untrusted text.
    """
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            raise failure

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert expected in (snapshot.status.detail or "")
    assert not snapshot.data


def test_remote_connect_stall_reports_a_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`urllib` wraps a connect-phase stall as `URLError`, not `TimeoutError`.

    A host that blackholes packets would otherwise report the generic read
    failure, hiding the one cause the five-second bound exists to produce.
    """
    from deepagents_code.configuration import providers

    class Opener:
        def open(self, _request: object, *, timeout: int) -> _RemoteResponse:
            assert timeout > 0
            raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert "timed out" in (snapshot.status.detail or "")


def test_agreeing_cli_flag_is_not_reported_as_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag that agrees with managed policy must not be called ignored.

    Masking is structural: every non-durable tier below a durable one lands in
    `masked_ranks`, so a managed `[startup] mode` masks `--yolo` even when both
    select YOLO. Warning there told the user their flag was dropped and sent
    them to their administrator about a policy that did exactly what they
    asked -- and the run started in YOLO regardless, so the message was simply
    false.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "yolo"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        assert main._resolve_approval_mode(args).value == "yolo"
    finally:
        service.invalidate_config_sources()

    assert "was ignored" not in capsys.readouterr().err


def test_blank_auto_classifier_flag_overrides_as_explicit_inherit() -> None:
    """`--flag ""` must not win its rank as a plain empty string.

    A CLI value is a shell string, not a TOML literal. `coerce_toml_value`
    accepts `""` as a legitimate value, so a blank flag resolved as `Found('')`
    at the CLI rank and suppressed every real value below it -- reporting an
    empty classifier sourced from `CLI argument` on the `dcode config` surface.

    For `models.auto_classifier` the blank flag is not absent either: it is the
    explicit "inherit the main agent model" instruction, which the launch path
    maps to `INHERIT_CLASSIFIER_MODEL` ahead of env and `config.toml`. The CLI
    tier resolves the same sentinel so introspection agrees with the launch.
    """
    from deepagents_code._cli_context import INHERIT_CLASSIFIER_MODEL
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.provider import CliProvider
    from deepagents_code.configuration.resolver import (
        CLI_RANK,
        resolver_from_snapshots,
    )
    from deepagents_code.configuration.types import TomlSnapshot

    option = get_option("models.auto_classifier")
    assert option is not None
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot.from_table("managed config", {}),
        user=TomlSnapshot.from_table(
            "config.toml", {"models": {"auto_classifier": "openai:gpt-5"}}
        ),
        cli_provider=CliProvider({"auto_classifier_model": ""}),
    )
    resolved = resolver.get(option)

    assert resolved.value == INHERIT_CLASSIFIER_MODEL
    assert CLI_RANK in resolved.ranks


_NO_THREAD = "can't start new thread"


class _FakeClock:
    """Monotonic clock a test advances explicitly.

    Stateful rather than an `iter([...])` of readings: a scripted iterator
    fails with `StopIteration` the moment production reads the clock one extra
    time, which reports a refactor as an unrelated crash instead of an
    assertion. Every reading here is still ordered and reproducible.
    """

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current reading.

        Returns:
            Seconds since this clock started.
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds: How far to advance.
        """
        self.now += seconds


def test_remote_toml_provider_stops_a_body_dripped_past_the_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body fed a chunk at a time cannot outlive the end-to-end budget.

    Every per-read timeout is recomputed from what is left of the deadline, so
    a server that answers just inside each shrinking window is bounded only by
    the cumulative check after each chunk. Without that check this response
    reads to completion well past five seconds, and the partial policy it
    delivers parses.
    """
    from deepagents_code.configuration import providers

    clock = _FakeClock()

    class DrippingResponse(_RemoteResponse):
        """Answers a little at a time, spending real budget on each chunk."""

        def read1(self, size: int = -1) -> bytes:
            del size
            clock.advance(2.0)
            return b'mode = "manual"\n'

    response = DrippingResponse(b"", content_length="64")

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers.time, "monotonic", clock)
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert snapshot.status.detail == "remote source timed out"
    # No partial body may reach `tomllib`: a prefix of a policy parses.
    assert snapshot.data == {}
    assert clock.now >= providers.REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS
    assert response.closed.is_set()


def test_remote_toml_provider_closes_a_response_that_lands_at_the_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect that returns exactly as the budget runs out reads no body.

    Distinct from the deadline expiring *before* the open is attempted, which
    fails earlier while computing the connect timeout. Here the response
    exists, so it must also be released rather than leaked.
    """
    from deepagents_code.configuration import providers

    clock = _FakeClock()
    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            # A connect that consumes the entire budget and then succeeds.
            assert timeout > 0
            clock.advance(providers.REMOTE_MANAGED_CONFIG_TIMEOUT_SECONDS + 1.0)
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    monkeypatch.setattr(providers.time, "monotonic", clock)
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    assert snapshot.status.detail == "remote source timed out"
    assert snapshot.data == {}
    assert response.closed.is_set()
    assert response.read_timeouts == []


@pytest.mark.parametrize(
    "media_type",
    [
        "text/html\x1b[2J\x1b[H",
        "text/html\r\nX-Injected: 1",
        "text/" + "h" * 4096,
        "text/html \u200b",
    ],
    ids=["ansi-escape", "header-injection", "overlong", "unicode-separator"],
)
def test_remote_toml_provider_sanitizes_a_hostile_media_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    media_type: str,
) -> None:
    """A server-chosen media type is bounded before it reaches an operator.

    This detail lands in exit-78 stderr and the `doctor` row, and it is the one
    piece of server-controlled text reproduced on this boundary. An escape
    sequence could rewrite the surrounding report and an unbounded value could
    bury it.
    """
    from deepagents_code.configuration import providers

    response = _RemoteResponse(b'[startup]\nmode = "manual"\n')
    response.headers["Content-Type"] = media_type

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return response

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    snapshot = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.UNREADABLE
    detail = snapshot.status.detail or ""
    assert "not TOML" in detail
    assert all(char.isascii() and char.isprintable() for char in detail)
    assert len(detail) < 128


def test_remote_toml_provider_releases_its_slot_when_no_worker_can_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thread exhaustion fails closed and leaves the destination reusable.

    The open slot is released by a done callback on the future, so a start
    failure that left the future pending would occupy that URL for the rest of
    the process: every later fetch would be refused even after threads freed
    up.
    """
    from deepagents_code.configuration import providers

    real_start = Thread.start
    fail_start = True

    def start(self: Thread) -> None:
        if fail_start and self.name == "managed-config-open":
            raise RuntimeError(_NO_THREAD)
        real_start(self)

    monkeypatch.setattr(Thread, "start", start)

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    provider = RemoteTomlProvider(
        "managed config",
        "https://config.example.com/policy.toml",
        tmp_path / "managed.toml",
    )
    failed = provider.load()

    assert failed.status.health is ProviderHealth.UNREADABLE
    assert "could not be read" in (failed.status.detail or "")

    fail_start = False
    recovered = provider.load()

    # The slot was released, so the same URL is fetchable again rather than
    # permanently refused as still in progress.
    assert recovered.status.health is ProviderHealth.OK
    assert recovered.data == {"startup": {"mode": "manual"}}


def test_doctor_keeps_a_rejected_remote_url_out_of_its_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source that fails validation is never echoed, token and all.

    The descriptor is administrator-authored, but a mistyped source can still
    carry a secret in its query string, and `doctor` output is pasted into
    tickets. `remote_source` stays unset for a rejected URL, so the row falls
    back to naming the descriptor file.
    """
    from deepagents_code.configuration import service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/p?token=s3cret"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    try:
        item = _managed_config_diagnostic()
    finally:
        service.invalidate_config_sources()

    assert item.ok is False
    assert "s3cret" not in item.value
    assert "config.example.com" not in item.value
    assert str(managed) in item.value
    assert "query string" in item.value


def test_doctor_reports_a_failed_refresh_behind_a_served_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy host that stopped answering cannot be silent.

    Fail-closed means the last enforceable generation keeps resolving, so every
    other field on this row stays green and correct. That is exactly why the
    failed refresh needs its own clause: `logger.warning` cannot reach a
    terminal here, because the package installs its own handler at import time.
    """
    from deepagents_code.configuration import providers, service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed)
    healthy = True
    failure = URLError("dns failed")

    class Opener:
        def open(self, _request: object, *, timeout: float) -> _RemoteResponse:
            assert timeout > 0
            if not healthy:
                raise failure
            return _RemoteResponse(b'[startup]\nmode = "manual"\n')

    monkeypatch.setattr(providers, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()
    try:
        assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}
        healthy = False

        item = _managed_config_diagnostic()

        # Still enforcing the generation it could read. A non-refresh read is
        # what the process resolves from; a `refresh=True` caller deliberately
        # receives the failure it asked to see.
        assert service.get_managed_snapshot().data == {"startup": {"mode": "manual"}}
        # ...and the row says both halves. Reporting only the read failure
        # reads as "no policy is in force", which is the opposite of what
        # fail-closed retention just did.
        assert item.ok is False
        assert "could not be read" in item.value
        assert "still enforcing the last generation" in item.value
        assert "not unmanaged" in item.value
        assert "dns failed" not in item.value
    finally:
        service.invalidate_config_sources()


def test_provider_status_rejects_an_unvalidated_remote_source() -> None:
    """The "only a validated URL is retained" rule is enforced, not documented.

    Every surface interpolates this field straight into operator text, so a
    construction site that skipped `_validate_remote_url` would put a rejected
    source -- credentials and all -- into a `doctor` row.
    """
    from deepagents_code.configuration.types import (
        REMOTE_SOURCE_MAX_CHARS,
        ProviderStatus,
    )

    for source in (
        "http://config.example.com/policy.toml",
        "https://",
        "https:///policy.toml",
        "https://config.example.com:invalid/policy.toml",
        "https://user:pw@config.example.com/policy.toml",
        "https://config.example.com/policy.toml?token=s3cret",
        "https://config.example.com/policy.toml#frag",
        "https://config.example.com/pol\u200bicy.toml",
        "https://config.example.com/policy.toml\x1b[2J",
        "https://config.example.com/" + "a" * REMOTE_SOURCE_MAX_CHARS,
        "https://CONFIG.example.com./policy.toml",
    ):
        with pytest.raises(ValueError, match="validated absolute HTTPS URL"):
            ProviderStatus(
                "managed config",
                None,
                ProviderHealth.UNREADABLE,
                "boom",
                remote_source=source,
            )

    # The shape production actually produces still constructs.
    ProviderStatus(
        "managed config",
        None,
        ProviderHealth.UNREADABLE,
        "boom",
        remote_source="https://config.example.com/policy.toml",
    )


def test_provider_status_accepts_at_sign_in_the_url_path() -> None:
    """A valid URL path character must not crash status construction.

    `@` is a legal path character and `_validate_remote_url` accepts it there,
    rejecting only authority userinfo. Banning the character outright would
    raise here for a descriptor that validated and fetched, turning startup
    into a traceback instead of a health report.
    """
    from deepagents_code.configuration.types import ProviderStatus

    source = "https://config.example.com/policy@v1.toml"
    status = ProviderStatus(
        "managed config",
        None,
        ProviderHealth.OK,
        None,
        remote_source=source,
    )
    assert status.remote_source == source


def test_remote_source_longer_than_the_bound_is_rejected(
    tmp_path: Path,
) -> None:
    """An unbounded source would be the one unbounded string on this path."""
    from deepagents_code.configuration.types import REMOTE_SOURCE_MAX_CHARS

    source = "https://config.example.com/" + "a" * REMOTE_SOURCE_MAX_CHARS
    snapshot = RemoteTomlProvider(
        "managed config",
        source,
        tmp_path / "managed.toml",
    ).load()

    assert snapshot.status.health is ProviderHealth.CORRUPT
    assert snapshot.status.detail == "remote source is too long"
    assert snapshot.status.remote_source is None
