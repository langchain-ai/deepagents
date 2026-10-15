"""Tests for MCP OAuth helpers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx2
import pytest
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthMetadata, OAuthToken

from deepagents_code._mcp_oauth import MCPOAuthProvider
from deepagents_code.mcp_auth import (
    FileTokenStorage,
    MCPReauthRequiredError,
    find_oauth_challenge,
    find_reauth_required,
    format_login_failure,
    resolve_headers,
)

if TYPE_CHECKING:
    from types import ModuleType

    import httpx
    from fastmcp.client.transports import StreamableHttpTransport

_RESOURCE_METADATA_URL = "https://mcp.example.com/.well-known/oauth-protected-resource"
_BEARER_CHALLENGE = f'Bearer resource_metadata="{_RESOURCE_METADATA_URL}"'
"""A minimal RFC 9728 Bearer challenge pointing at the resource metadata."""


class TestResolveHeaders:
    """Compatibility coverage for the public header resolver."""

    def test_delegates_to_shared_interpolation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The legacy helper remains importable and supports shared syntax."""
        monkeypatch.delenv("MCP_HEADER_TOKEN", raising=False)

        resolved = resolve_headers(
            {"Authorization": "Bearer ${MCP_HEADER_TOKEN:-fallback}"},
            server_name="remote",
        )

        assert resolved == {"Authorization": "Bearer fallback"}

    def test_preserves_original_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Compatibility resolution returns a copy without mutating its input."""
        monkeypatch.setenv("MCP_HEADER_TOKEN", "resolved")
        headers = {"Authorization": "Bearer ${MCP_HEADER_TOKEN}"}

        resolved = resolve_headers(headers)

        assert resolved == {"Authorization": "Bearer resolved"}
        assert headers == {"Authorization": "Bearer ${MCP_HEADER_TOKEN}"}


def _http_status_error(
    status_code: int,
    *,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
    client: ModuleType = httpx2,
) -> httpx.HTTPStatusError | httpx2.HTTPStatusError:
    """Build an `HTTPStatusError` with a canned response.

    Defaults to `httpx2`, which is what the MCP SDK's client raises in
    production; `httpx` is still reachable because dcode makes its own calls
    (device flow, remote preflight) with that client.

    Args:
        status_code: Status code for the canned response.
        headers: Response headers to attach.
        client: Which HTTP client library to build the error from.

    Returns:
        An `HTTPStatusError` from the requested client library.
    """
    request = client.Request("GET", "https://mcp.example.com/")
    response = client.Response(status_code, headers=headers or {}, request=request)
    return client.HTTPStatusError("boom", request=request, response=response)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `Path.home()` and `DEFAULT_STATE_DIR` into a temp directory.

    `Path.home` is patched for code that resolves it at call time;
    `DEFAULT_STATE_DIR` is patched for code (like `mcp_auth.token_store_dir`)
    that pulls from the import-time-frozen constant in `model_config`.
    """
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_STATE_DIR",
        fake / ".deepagents" / ".state",
    )
    return fake


def _make_tokens(access_token: str = "at"):
    return OAuthToken(
        access_token=access_token,
        token_type="Bearer",
        refresh_token="rt",
        expires_in=3600,
    )


def _make_client_info():
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def _make_oauth_metadata(token_endpoint: str = "https://auth.example/token"):
    from mcp.shared.auth import AnyHttpUrl, OAuthMetadata

    return OAuthMetadata(
        issuer=AnyHttpUrl("https://auth.example"),
        authorization_endpoint=AnyHttpUrl("https://auth.example/authorize"),
        token_endpoint=AnyHttpUrl(token_endpoint),
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
    )


def _make_client_info_with_secret(
    auth_method: Literal["client_secret_basic", "client_secret_post", "none"],
):
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    # Public clients (`none`) carry no secret; confidential clients do.
    client_secret = None if auth_method == "none" else "client-secret"
    return OAuthClientInformationFull(
        client_id="client-id",
        client_secret=client_secret,
        token_endpoint_auth_method=auth_method,
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


def _make_client_info_with_loopback(port: int):
    from mcp.shared.auth import AnyUrl, OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=[AnyUrl(f"http://localhost:{port}/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorage:
    """Tests for the file-backed OAuth token store."""

    async def test_missing_file_returns_none(self) -> None:
        """Missing token files return `None` for both tokens and client info."""
        storage = FileTokenStorage("notion")
        assert await storage.get_tokens() is None
        assert await storage.get_client_info() is None

    async def test_round_trip_tokens_and_client_info(self) -> None:
        """Tokens and client info round-trip through disk storage."""
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        got_ci = await storage.get_client_info()
        got_tok = await storage.get_tokens()

        assert got_ci is not None
        assert got_tok is not None
        assert got_ci.client_id == "client-id"
        assert got_tok.access_token == "at"

    async def test_concurrent_instances_preserve_distinct_updates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-file read-modify-write operations retain both updates."""
        token_storage = FileTokenStorage("notion")
        client_storage = FileTokenStorage("notion")
        write_barrier = threading.Barrier(2)
        physical_write_lock = threading.Lock()
        loop_thread = threading.get_ident()
        filesystem_threads: list[int] = []
        real_write = FileTokenStorage._write

        def _coordinated_write(storage: FileTokenStorage, data: dict[str, Any]) -> None:
            filesystem_threads.append(threading.get_ident())
            with contextlib.suppress(threading.BrokenBarrierError):
                write_barrier.wait(timeout=1)
            # If the per-file lock regressed and both writers ran concurrently,
            # serialize the physical writes so the test observes a clean
            # last-writer-wins lost update rather than a shared-`.tmp` collision.
            with physical_write_lock:
                real_write(storage, data)

        monkeypatch.setattr(FileTokenStorage, "_write", _coordinated_write)

        await asyncio.gather(
            token_storage.set_tokens(_make_tokens()),
            client_storage.set_client_info(_make_client_info()),
        )

        stored_tokens = await token_storage.get_tokens()
        stored_client = await client_storage.get_client_info()
        assert stored_tokens is not None
        assert stored_client is not None
        assert filesystem_threads
        assert all(thread_id != loop_thread for thread_id in filesystem_threads)

    async def test_concurrent_instances_do_not_overlap_tmp_writes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same-file mutations cannot collide while writing the shared `.tmp`."""
        first = FileTokenStorage("notion")
        second = FileTokenStorage("notion")
        write_barrier = threading.Barrier(2)
        real_write = FileTokenStorage._write

        def _collision_detecting_write(
            storage: FileTokenStorage, data: dict[str, Any]
        ) -> None:
            try:
                write_barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                real_write(storage, data)
                return
            msg = "concurrent writers collided on the shared temporary file"
            raise FileExistsError(msg)

        monkeypatch.setattr(
            FileTokenStorage,
            "_write",
            _collision_detecting_write,
        )

        await asyncio.gather(
            first.set_tokens(_make_tokens("first")),
            second.set_tokens(_make_tokens("second")),
        )

        # Both serialized writes landed rather than colliding on the `.tmp`.
        assert await first.get_tokens() is not None

    async def test_different_token_files_can_write_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-file synchronization does not serialize unrelated servers."""
        write_barrier = threading.Barrier(2)
        real_write = FileTokenStorage._write

        def _coordinated_write(storage: FileTokenStorage, data: dict[str, Any]) -> None:
            write_barrier.wait(timeout=1)
            real_write(storage, data)

        monkeypatch.setattr(FileTokenStorage, "_write", _coordinated_write)

        await asyncio.gather(
            FileTokenStorage("alpha").set_tokens(_make_tokens("alpha")),
            FileTokenStorage("beta").set_tokens(_make_tokens("beta")),
        )

        assert not write_barrier.broken

    async def test_sets_file_permissions_on_posix(self, fake_home: Path) -> None:
        """Token files are created with private user-only permissions."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())

        token_path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        assert token_path.exists()
        if hasattr(token_path, "stat"):
            assert token_path.stat().st_mode & 0o777 == 0o600

    async def test_corrupt_file_raises(self, fake_home: Path) -> None:
        """Corrupt files fail with a remediation hint."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        storage = FileTokenStorage("notion")

        with pytest.raises(RuntimeError, match="Delete the file"):
            await storage.get_tokens()

    async def test_server_names_are_isolated(self) -> None:
        """Different servers use different token files."""
        alpha = FileTokenStorage("alpha")
        beta = FileTokenStorage("beta")
        await alpha.set_tokens(_make_tokens())
        await beta.set_tokens(_make_tokens())

        got_alpha = await alpha.get_tokens()
        got_beta = await beta.get_tokens()

        assert got_alpha is not None
        assert got_beta is not None

    async def test_same_server_name_with_different_urls_isolated(self) -> None:
        """Same-named servers on different endpoints use separate files."""
        alpha = FileTokenStorage("github", server_url="https://alpha.example/mcp")
        beta = FileTokenStorage("github", server_url="https://beta.example/mcp")
        await alpha.set_tokens(_make_tokens("alpha-token"))
        await beta.set_tokens(_make_tokens("beta-token"))

        got_alpha = await alpha.get_tokens()
        got_beta = await beta.get_tokens()

        assert alpha.path != beta.path
        assert got_alpha is not None
        assert got_alpha.access_token == "alpha-token"
        assert got_beta is not None
        assert got_beta.access_token == "beta-token"

    @pytest.mark.parametrize(
        "name",
        [
            "../escape",
            "../../etc/cron.d/evil",
            "name/with/slashes",
            "name\\with\\backslashes",
            "name with spaces",
            "name\x00null",
            "..",
            ".",
            "",
        ],
    )
    def test_unsafe_server_name_rejected(self, name: str) -> None:
        """Names that could traverse out of the tokens dir are rejected.

        Guards against path traversal via attacker-controlled `mcpServers`
        keys (Corridor finding d5d5b0c1).
        """
        with pytest.raises(ValueError, match="Invalid MCP server name"):
            FileTokenStorage(name)

    async def test_set_tokens_records_absolute_expiry(self) -> None:
        """`set_tokens` writes an `expires_at` sidecar derived from `expires_in`."""
        storage = FileTokenStorage("notion")
        before = time.time()
        await storage.set_tokens(_make_tokens())
        after = time.time()

        got = await storage.get_expires_at()
        assert got is not None
        # 3600 from `_make_tokens`; widen the wall-clock window to absorb
        # GC pauses on busy CI runners.
        assert before + 3600 <= got <= after + 3600 + 1.0

    async def test_set_tokens_and_client_info_records_expiry(self) -> None:
        """The combined writer also persists `expires_at`."""
        storage = FileTokenStorage("notion")
        before = time.time()
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())
        after = time.time()

        got = await storage.get_expires_at()
        assert got is not None
        assert before + 3600 <= got <= after + 3600 + 1.0

    @pytest.mark.parametrize("include_client_info", [False, True])
    async def test_expiry_is_captured_before_worker_delay(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        include_client_info: bool,
    ) -> None:
        """Executor delays do not extend the persisted token lifetime."""
        storage = FileTokenStorage("notion")
        clock = {"now": 1_000.0}
        real_read = storage._read

        def _delayed_read() -> dict[str, Any] | None:
            # Model time spent waiting for an executor worker or the mutation
            # lock before the worker begins its read-modify-write operation.
            clock["now"] = 2_000.0
            return real_read()

        monkeypatch.setattr(time, "time", lambda: clock["now"])
        monkeypatch.setattr(storage, "_read", _delayed_read)
        tokens = OAuthToken(
            access_token="at",
            token_type="Bearer",
            refresh_token="rt",
            expires_in=60,
        )

        if include_client_info:
            await storage.set_tokens_and_client_info(tokens, _make_client_info())
        else:
            await storage.set_tokens(tokens)

        data = json.loads(storage.path.read_text())
        assert data["expires_at"] == pytest.approx(1_060.0)

    async def test_get_expires_at_returns_none_for_legacy_file(
        self, fake_home: Path
    ) -> None:
        """Token files written before this field existed return `None`."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "tokens": {"access_token": "x"}}))
        storage = FileTokenStorage("notion")

        assert await storage.get_expires_at() is None

    async def test_get_expires_at_rejects_non_numeric(self, fake_home: Path) -> None:
        """A garbage sidecar value falls back to `None` rather than raising."""
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "tokens": {"access_token": "x"},
                    "expires_at": "soon",
                }
            )
        )
        storage = FileTokenStorage("notion")

        assert await storage.get_expires_at() is None

    async def test_set_tokens_clears_stale_expiry_when_expires_in_absent(self) -> None:
        """Writing a token without `expires_in` removes any prior `expires_at`."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())
        assert await storage.get_expires_at() is not None

        # Some providers omit `expires_in` on refresh; the sidecar must not
        # linger from the prior write or the next cold start will use a
        # bogus expiry.
        await storage.set_tokens(
            OAuthToken(access_token="x2", token_type="Bearer", refresh_token="rt2")
        )
        assert await storage.get_expires_at() is None

    async def test_credentials_round_trip_in_version_1_envelope(self) -> None:
        """`save`/`load` round-trip every field and keep the on-disk format."""
        from mcp.shared.auth import AnyHttpUrl, ProtectedResourceMetadata

        from deepagents_code._mcp_oauth import OAuthCredentials

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_text(json.dumps({"version": 1, "unrelated": "kept"}))
        credentials = OAuthCredentials(
            tokens=_make_tokens(),
            expires_at=1234.5,
            client_info=_make_client_info(),
            oauth_metadata=_make_oauth_metadata(),
            protected_resource_metadata=ProtectedResourceMetadata(
                resource=AnyHttpUrl("https://mcp.notion.com/mcp"),
                authorization_servers=[AnyHttpUrl("https://auth.example")],
            ),
            auth_server_url="https://auth.example",
        )

        await storage.save(credentials)

        assert await storage.load() == credentials
        data = json.loads(storage.path.read_text())
        assert data["version"] == 1
        assert data["unrelated"] == "kept"
        assert data["expires_at"] == pytest.approx(1234.5)
        # Readers that predate `load` still find each field.
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "at"
        assert await storage.get_client_info() == _make_client_info()
        assert await storage.get_expires_at() == pytest.approx(1234.5)

    async def test_save_without_tokens_keeps_stored_token(self) -> None:
        """Registering a client mid-login never deletes the working token."""
        from deepagents_code._mcp_oauth import OAuthCredentials

        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens("old"))
        expires_at = await storage.get_expires_at()

        await storage.save(OAuthCredentials(client_info=_make_client_info()))

        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "old"
        assert await storage.get_expires_at() == expires_at
        assert await storage.get_client_info() == _make_client_info()

    async def test_save_clears_expiry_for_token_without_expires_in(self) -> None:
        """A new token without an expiry must not inherit the previous one."""
        from deepagents_code._mcp_oauth import OAuthCredentials

        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())

        await storage.save(
            OAuthCredentials(tokens=OAuthToken(access_token="x2", token_type="Bearer"))
        )

        assert await storage.get_expires_at() is None

    async def test_set_tokens_persists_off_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refreshed-token persistence must not block the event-loop thread.

        `set_tokens` is awaited from the MCP SDK's async OAuth refresh flow.
        Its synchronous read-modify-write (including `path.parent.mkdir`)
        must run in a worker thread; otherwise BlockBuster under
        `langgraph dev` raises `BlockingError`, which cancels the transport
        task group and crashes the tool node. Regression for the MCP OAuth
        refresh failure.
        """
        storage = FileTokenStorage("notion")
        loop_thread = threading.get_ident()
        write_threads: list[int] = []

        real_write = storage._write

        def _spy_write(data: dict) -> None:
            write_threads.append(threading.get_ident())
            real_write(data)

        monkeypatch.setattr(storage, "_write", _spy_write)

        await storage.set_tokens(_make_tokens())

        assert write_threads, "_write should have run"
        assert all(thread_id != loop_thread for thread_id in write_threads), (
            "token persistence must run off the event-loop thread"
        )
        # The expected state is still written despite the offload.
        got = await storage.get_tokens()
        assert got is not None
        assert got.access_token == "at"
        assert await storage.get_expires_at() is not None

    async def test_set_tokens_joins_write_despite_repeated_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated cancellation waits for a rotated refresh token to reach disk."""
        storage = FileTokenStorage("notion")
        write_started = threading.Event()
        write_finished = threading.Event()
        allow_write = threading.Event()
        real_write = storage._write

        def _blocking_write(data: dict[str, Any]) -> None:
            write_started.set()
            if not allow_write.wait(timeout=5):
                pytest.fail("timed out waiting to finish the token write")
            try:
                real_write(data)
            finally:
                write_finished.set()

        monkeypatch.setattr(storage, "_write", _blocking_write)
        task = asyncio.create_task(storage.set_tokens(_make_tokens()))

        try:
            assert await asyncio.to_thread(write_started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), (
                "cancellation must stay joined to the in-flight token write"
            )

            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), (
                    "repeated cancellation must not detach the token write"
                )

            allow_write.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert write_finished.is_set()
            got = await storage.get_tokens()
            assert got is not None
            assert got.access_token == "at"
        finally:
            allow_write.set()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_set_tokens_joins_write_under_anyio_level_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AnyIO cancelled scope cannot detach in-flight token persistence."""
        storage = FileTokenStorage("notion")
        write_started = threading.Event()
        allow_write = threading.Event()
        scopes: list[anyio.CancelScope] = []

        real_write = storage._write

        def _patched_write(data: dict[str, Any]) -> None:
            write_started.set()
            if not allow_write.wait(timeout=5):
                pytest.fail("timed out waiting to finish the token write")
            real_write(data)

        monkeypatch.setattr(storage, "_write", _patched_write)

        async def _persist_in_cancel_scope() -> bool:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await storage.set_tokens(_make_tokens())
            return scope.cancelled_caught

        task = asyncio.create_task(_persist_in_cancel_scope())
        try:
            assert await asyncio.to_thread(write_started.wait, 5)
            scopes[0].cancel()
            await asyncio.sleep(0)

            assert not task.done(), (
                "level cancellation must stay joined to the in-flight write"
            )
            allow_write.set()

            assert await task is True
            got = await storage.get_tokens()
            assert got is not None
            assert got.access_token == "at"
        finally:
            allow_write.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_set_tokens_reports_write_failure_while_cancelled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A persistence failure takes precedence without leaking to the loop."""
        storage = FileTokenStorage("notion")
        write_started = threading.Event()
        allow_write = threading.Event()
        loop = asyncio.get_running_loop()
        loop_contexts: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_contexts.append(context))

        def _failing_write(data: dict[str, Any]) -> None:
            del data
            write_started.set()
            if not allow_write.wait(timeout=5):
                pytest.fail("timed out waiting to fail the token write")
            msg = "token persistence failed"
            raise OSError(msg)

        monkeypatch.setattr(storage, "_write", _failing_write)
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        task = asyncio.create_task(storage.set_tokens(_make_tokens()))
        try:
            assert await asyncio.to_thread(write_started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_write.set()
            with pytest.raises(OSError, match="token persistence failed"):
                await task
            await asyncio.sleep(0)
        finally:
            allow_write.set()
            await asyncio.gather(task, return_exceptions=True)
            loop.set_exception_handler(previous_handler)

        assert not loop_contexts
        # The write error supersedes the deferred cancellation; that loss must
        # be logged rather than dropped silently.
        assert any(
            "a deferred cancellation is superseded by the write error"
            in record.getMessage()
            for record in caplog.records
        )

    async def test_reads_run_off_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every async read offloads its blocking `_read` off the loop thread.

        `BlockBuster` under `langgraph dev` trips on any synchronous file read
        left on the loop, so this pins all of the read accessors — not just
        `get_tokens` — including `load`, which sits on the hot
        refresh-decision path.
        """
        storage = FileTokenStorage("notion")
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())

        loop_thread = threading.get_ident()
        read_threads: list[int] = []
        real_read = storage._read

        def _spy_read() -> dict | None:
            read_threads.append(threading.get_ident())
            return real_read()

        monkeypatch.setattr(storage, "_read", _spy_read)

        assert await storage.get_tokens() is not None
        assert await storage.get_client_info() is not None
        assert await storage.get_expires_at() is not None
        assert await storage.load() is not None

        assert len(read_threads) == 4, "each read accessor should have run once"
        assert all(thread_id != loop_thread for thread_id in read_threads), (
            "token reads must run off the event-loop thread"
        )

    async def test_all_setters_persist_off_event_loop_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every setter offloads its blocking `_write` off the loop thread.

        `set_tokens` is covered above; this pins the remaining setters —
        notably `set_tokens_and_client_info`, the atomic combined write on the
        primary OAuth-completion path this PR targets.
        """
        storage = FileTokenStorage("notion")
        loop_thread = threading.get_ident()
        write_threads: list[int] = []
        real_write = storage._write

        def _spy_write(data: dict) -> None:
            write_threads.append(threading.get_ident())
            real_write(data)

        monkeypatch.setattr(storage, "_write", _spy_write)

        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())
        await storage.set_client_info(_make_client_info())
        await storage.save(await storage.load() or pytest.fail("no credentials"))

        assert len(write_threads) == 3, "each setter should have written once"
        assert all(thread_id != loop_thread for thread_id in write_threads), (
            "token persistence must run off the event-loop thread"
        )

    async def test_read_not_blocked_by_in_flight_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read runs lock-free instead of serializing behind a live mutation.

        A mutation holding the per-file lock during its blocking write must not
        stall a concurrent read, which never takes that lock.
        """
        storage = FileTokenStorage("notion")
        await storage.set_tokens(_make_tokens())

        write_holding_lock = threading.Event()
        release_write = threading.Event()
        real_write = storage._write

        def _blocking_write(data: dict) -> None:
            # Called while `_set_tokens_sync` holds the per-file lock.
            write_holding_lock.set()
            if not release_write.wait(timeout=5):
                pytest.fail("timed out holding the per-file mutation lock")
            real_write(data)

        monkeypatch.setattr(storage, "_write", _blocking_write)
        write_task = asyncio.create_task(storage.set_tokens(_make_tokens("new")))
        try:
            assert await asyncio.to_thread(write_holding_lock.wait, 5)
            # The mutation is parked inside `_write` still holding the lock; a
            # read must complete instead of serializing behind it.
            got = await asyncio.wait_for(storage.get_tokens(), timeout=1)
            assert got is not None
        finally:
            release_write.set()
            await write_task


_MCP_URL = "https://mcp.notion.com/mcp"
_ISSUER = "https://auth.example"
_TOKEN_URL = f"{_ISSUER}/token"


class _FakeOAuthServers:
    """An MCP resource server and its authorization server, in memory."""

    def __init__(
        self,
        *,
        mcp_url: str = _MCP_URL,
        token_url: str = _TOKEN_URL,
        refresh_status: int = 200,
    ) -> None:
        parsed = urlsplit(mcp_url)
        self.mcp_url = mcp_url
        self.token_url = token_url
        self.prm_url = (
            f"{parsed.scheme}://{parsed.netloc}"
            f"/.well-known/oauth-protected-resource{parsed.path}"
        )
        self.refresh_status = refresh_status
        self.valid_tokens: set[str] = set()
        self.issued = 0
        self.auth_requests: list[httpx2.Request] = []
        self.resource_requests: list[httpx2.Request] = []

    def resource(self, request: httpx2.Request) -> httpx2.Response:
        self.resource_requests.append(request)
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if bearer in self.valid_tokens:
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        challenge = f'Bearer resource_metadata="{self.prm_url}"'
        return httpx2.Response(401, headers={"WWW-Authenticate": challenge})

    async def send(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        self.auth_requests.append(request)
        url = str(request.url)
        if url == self.prm_url:
            return httpx2.Response(
                200, json={"resource": self.mcp_url, "authorization_servers": [_ISSUER]}
            )
        if url == f"{_ISSUER}/.well-known/oauth-authorization-server":
            return httpx2.Response(200, json=self.metadata_json())
        if url == f"{_ISSUER}/register":
            body = json.loads(request.content)
            return httpx2.Response(201, json={**body, "client_id": "registered"})
        if url == self.token_url:
            return self._token()
        return httpx2.Response(404)

    def metadata_json(self) -> dict[str, Any]:
        return {
            "issuer": _ISSUER,
            "authorization_endpoint": f"{_ISSUER}/authorize",
            "token_endpoint": self.token_url,
            "registration_endpoint": f"{_ISSUER}/register",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        }

    def _token(self) -> httpx2.Response:
        if self.refresh_status != 200:
            return httpx2.Response(self.refresh_status, json={"error": "invalid_grant"})
        self.issued += 1
        access = f"at-rotated-{self.issued}"
        self.valid_tokens.add(access)
        return httpx2.Response(
            200,
            json={
                "access_token": access,
                "token_type": "Bearer",
                "refresh_token": f"rt-rotated-{self.issued}",
                "expires_in": 3600,
            },
        )

    def token_requests(self) -> list[httpx2.Request]:
        return [r for r in self.auth_requests if str(r.url) == self.token_url]

    async def post(self, provider: MCPOAuthProvider) -> httpx2.Response:
        """Send one MCP request through `provider` to the resource server."""
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(self.resource), auth=provider
        ) as client:
            return await client.post(self.mcp_url, json={})


@pytest.fixture
def oauth_servers(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch
) -> _FakeOAuthServers:
    """Route the provider's authorization-server requests to fake servers."""
    del fake_home
    servers = _FakeOAuthServers()
    monkeypatch.setattr(
        "deepagents_code._mcp_oauth._send_with_default_client", servers.send
    )
    return servers


async def _store_credentials(
    storage: FileTokenStorage,
    *,
    tokens: OAuthToken | None = None,
    expires_at: float | None,
    servers: _FakeOAuthServers | None = None,
) -> None:
    """Seed `storage` with client info, `tokens`, and optionally metadata."""
    from mcp.shared.auth import OAuthMetadata

    from deepagents_code._mcp_oauth import OAuthCredentials

    metadata = (
        OAuthMetadata.model_validate(servers.metadata_json()) if servers else None
    )
    await storage.save(
        OAuthCredentials(
            tokens=tokens or _make_tokens("stale"),
            expires_at=expires_at,
            client_info=_make_client_info(),
            oauth_metadata=metadata,
        )
    )


def _non_interactive(
    storage: FileTokenStorage, url: str = _MCP_URL
) -> MCPOAuthProvider:
    from deepagents_code.mcp_auth import build_oauth_provider

    return build_oauth_provider(
        server_name=storage._server_name,
        server_url=url,
        storage=storage,
        interactive=False,
    )


def _write_legacy_token_file(storage: FileTokenStorage, tokens: dict) -> None:
    """Write a token file from before the `expires_at` sidecar existed."""
    storage.path.parent.mkdir(parents=True, exist_ok=True)
    storage.path.write_text(
        json.dumps(
            {
                "version": 1,
                "client_info": json.loads(
                    _make_client_info().model_dump_json(exclude_none=True)
                ),
                "tokens": tokens,
            }
        )
    )


class TestRestartExpiry:
    """A stored token's expiry survives a restart and drives the refresh grant."""

    async def test_live_token_is_sent_without_refresh(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, tokens=_make_tokens("live"), expires_at=time.time() + 3600
        )
        oauth_servers.valid_tokens.add("live")

        response = await oauth_servers.post(_non_interactive(storage))

        assert response.status_code == 200
        assert oauth_servers.auth_requests == []

    async def test_expired_token_refreshes_at_stored_endpoint(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """The refresh goes straight to the stored token endpoint and persists."""
        from filelock import FileLock

        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )

        response = await oauth_servers.post(_non_interactive(storage))

        assert response.status_code == 200
        assert [str(r.url) for r in oauth_servers.auth_requests] == [_TOKEN_URL]
        body = parse_qs(oauth_servers.auth_requests[0].content.decode())
        assert body["grant_type"] == ["refresh_token"]
        assert body["refresh_token"] == ["rt"]
        assert oauth_servers.resource_requests[-1].headers["authorization"] == (
            "Bearer at-rotated-1"
        )
        persisted = await storage.get_tokens()
        assert persisted is not None
        assert persisted.refresh_token == "rt-rotated-1"
        expires_at = await storage.get_expires_at()
        assert expires_at is not None
        assert expires_at > time.time()
        # The refresh lock is released for the next process.
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        await asyncio.to_thread(probe.acquire, timeout=0)
        probe.release()

    async def test_refresh_without_stored_metadata_discovers_endpoint(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """Older token files discover and persist metadata instead of guessing."""
        oauth_servers.token_url = f"{_ISSUER}/oauth/token"
        storage = FileTokenStorage("notion")
        await _store_credentials(storage, expires_at=time.time() - 60)

        response = await oauth_servers.post(_non_interactive(storage))

        assert response.status_code == 200
        assert len(oauth_servers.token_requests()) == 1
        credentials = await storage.load()
        assert credentials is not None
        assert credentials.oauth_metadata is not None
        assert str(credentials.oauth_metadata.token_endpoint) == (
            oauth_servers.token_url
        )

    async def test_preseeded_slack_client_refreshes(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slack's preseeded public client refreshes at its token endpoint."""
        del fake_home
        from deepagents_code.mcp_providers.slack import (
            _SLACK_MCP_CLIENT_ID,
            _preseed_slack_client_info,
        )

        url = "https://mcp.slack.com/mcp"
        servers = _FakeOAuthServers(
            mcp_url=url, token_url="https://slack.com/api/oauth.v2.user.access"
        )
        monkeypatch.setattr(
            "deepagents_code._mcp_oauth._send_with_default_client", servers.send
        )
        storage = FileTokenStorage("slack", server_url=url)
        await _preseed_slack_client_info(storage)
        credentials = await storage.load()
        assert credentials is not None
        await storage.save(
            credentials.model_copy(
                update={
                    "tokens": _make_tokens("stale"),
                    "expires_at": time.time() - 60,
                    "oauth_metadata": OAuthMetadata.model_validate(
                        servers.metadata_json()
                    ),
                }
            )
        )

        response = await servers.post(_non_interactive(storage, url))

        assert response.status_code == 200
        (refresh,) = servers.auth_requests
        assert str(refresh.url) == servers.token_url
        assert parse_qs(refresh.content.decode())["client_id"] == [_SLACK_MCP_CLIENT_ID]

    async def test_legacy_file_with_refresh_token_refreshes(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """A token file without `expires_at` is treated as already expired."""
        storage = FileTokenStorage("notion")
        _write_legacy_token_file(
            storage, json.loads(_make_tokens("stale").model_dump_json())
        )

        response = await oauth_servers.post(_non_interactive(storage))

        assert response.status_code == 200
        assert len(oauth_servers.token_requests()) == 1

    async def test_legacy_file_without_refresh_token_requires_reauth(
        self,
        oauth_servers: _FakeOAuthServers,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Nothing can pre-empt expiry, so the 401 surfaces a reauth error."""
        caplog.set_level(logging.INFO, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        _write_legacy_token_file(
            storage, {"access_token": "stale", "token_type": "Bearer"}
        )

        with pytest.raises(MCPReauthRequiredError, match="notion"):
            await oauth_servers.post(_non_interactive(storage))

        assert oauth_servers.resource_requests[0].headers["authorization"] == (
            "Bearer stale"
        )
        assert oauth_servers.token_requests() == []
        assert "has no refresh_token" in caplog.text

    async def test_rejected_refresh_requires_reauth_quietly(
        self,
        oauth_servers: _FakeOAuthServers,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A revoked grant raises the login hint without noisy SDK error logs."""
        caplog.set_level(logging.WARNING, logger="mcp.client.auth.oauth2")
        oauth_servers.refresh_status = 400
        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )

        with pytest.raises(MCPReauthRequiredError) as exc_info:
            await oauth_servers.post(_non_interactive(storage))

        assert find_reauth_required(exc_info.value) is exc_info.value
        assert "/mcp login notion" in format_login_failure(exc_info.value)
        assert len(oauth_servers.token_requests()) == 1
        assert not [r for r in caplog.records if r.name.startswith("mcp.client")]

    async def test_refresh_outage_keeps_stored_credentials(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """A 5xx is not relabeled as reauth and leaves the grant on disk."""
        from deepagents_code._mcp_oauth import MCPOAuthUnavailableError

        oauth_servers.refresh_status = 503
        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )

        with pytest.raises(MCPOAuthUnavailableError) as exc_info:
            await oauth_servers.post(_non_interactive(storage))

        assert find_reauth_required(exc_info.value) is None
        assert "HTTP 503" in format_login_failure(exc_info.value)
        persisted = await storage.get_tokens()
        assert persisted is not None
        assert persisted.refresh_token == "rt"

    async def test_refresh_read_timeout_propagates_and_releases_lock(
        self,
        oauth_servers: _FakeOAuthServers,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ambiguous timeout is not retried and does not orphan the lock."""
        from filelock import FileLock

        async def _timeout(request: httpx2.Request) -> httpx2.Response:
            msg = "timed out"
            raise httpx2.ReadTimeout(msg, request=request)

        monkeypatch.setattr(
            "deepagents_code._mcp_oauth._send_with_default_client", _timeout
        )
        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )

        with pytest.raises(httpx2.ReadTimeout):
            await oauth_servers.post(_non_interactive(storage))

        persisted = await storage.get_tokens()
        assert persisted is not None
        assert persisted.refresh_token == "rt"
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        await asyncio.to_thread(probe.acquire, timeout=0)
        probe.release()

    async def test_basic_auth_refresh_omits_body_client_id(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """Servers like Pylon reject a duplicate `client_id` under HTTP Basic."""
        from mcp.shared.auth import OAuthMetadata

        from deepagents_code._mcp_oauth import OAuthCredentials

        storage = FileTokenStorage("pylon")
        await storage.save(
            OAuthCredentials(
                tokens=_make_tokens("stale"),
                expires_at=time.time() - 60,
                client_info=_make_client_info_with_secret("client_secret_basic"),
                oauth_metadata=OAuthMetadata.model_validate(
                    oauth_servers.metadata_json()
                ),
            )
        )

        await oauth_servers.post(_non_interactive(storage))

        (refresh,) = oauth_servers.token_requests()
        assert refresh.headers["authorization"].startswith("Basic ")
        assert "client_id" not in parse_qs(refresh.content.decode())

    async def test_unauthenticated_flow_persists_registration_and_metadata(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """Discovery and registration from a 401 are saved for later refreshes."""
        storage = FileTokenStorage("notion")

        with pytest.raises(MCPReauthRequiredError):
            await oauth_servers.post(_non_interactive(storage))

        credentials = await storage.load()
        assert credentials is not None
        assert credentials.tokens is None
        assert credentials.client_info is not None
        assert credentials.client_info.client_id == "registered"
        assert credentials.oauth_metadata is not None
        assert str(credentials.oauth_metadata.token_endpoint) == _TOKEN_URL


@pytest.mark.usefixtures("fake_home")
class TestRefreshTokenSerialization:
    """Cross-process-safe refresh serialization to avoid refresh-token reuse.

    The LangSmith OAuth server rotates refresh tokens and revokes the entire
    identity+client token family when an already-rotated token is replayed, so
    the provider must reload the on-disk token under a lock before refreshing.
    """

    async def test_refresh_lock_path_is_sibling_of_token_file(self) -> None:
        """The lock lives beside the token file and never replaces it."""
        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        assert storage.refresh_lock_path == storage.path.with_name(
            f"{storage.path.name}.lock"
        )
        assert storage.refresh_lock_path.parent == storage.path.parent
        assert storage.refresh_lock_path != storage.path

    async def test_waits_for_peer_refresh_and_reuses_its_token(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """A peer's rotation, made while this process waits, is reused.

        Replaying this process's stale refresh token after a peer rotated it
        would make rotating-token servers revoke the whole token family.
        """
        from filelock import FileLock

        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )
        provider = _non_interactive(storage)
        peer = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer.acquire()
        task = asyncio.ensure_future(oauth_servers.post(provider))
        try:
            await asyncio.sleep(0.2)
            assert not task.done()
            await storage.set_tokens(_make_tokens("peer-rotated"))
            oauth_servers.valid_tokens.add("peer-rotated")
            peer.release()

            response = await asyncio.wait_for(task, timeout=5)
        finally:
            if peer.is_locked:
                peer.release()
            await asyncio.gather(task, return_exceptions=True)

        assert response.status_code == 200
        assert oauth_servers.token_requests() == []
        assert oauth_servers.resource_requests[-1].headers["authorization"] == (
            "Bearer peer-rotated"
        )

    async def test_refresh_waits_for_peer_holding_lock_then_proceeds(
        self, oauth_servers: _FakeOAuthServers
    ) -> None:
        """No refresh is sent until a peer releases the lock."""
        from filelock import FileLock

        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )
        peer = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer.acquire()
        task = asyncio.ensure_future(oauth_servers.post(_non_interactive(storage)))
        try:
            await asyncio.sleep(0.2)
            assert not task.done()
            assert oauth_servers.token_requests() == []
            peer.release()

            response = await asyncio.wait_for(task, timeout=5)
        finally:
            if peer.is_locked:
                peer.release()
            await asyncio.gather(task, return_exceptions=True)

        assert response.status_code == 200
        assert len(oauth_servers.token_requests()) == 1

    async def test_lock_timeout_skips_refresh_token_reuse(
        self,
        oauth_servers: _FakeOAuthServers,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A peer-held lock timeout never replays the refresh token."""
        from filelock import FileLock

        from deepagents_code import mcp_auth

        monkeypatch.setattr(mcp_auth, "_REFRESH_LOCK_TIMEOUT_SECONDS", 0.1)
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        await _store_credentials(
            storage, expires_at=time.time() - 60, servers=oauth_servers
        )
        peer = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer.acquire()
        try:
            with pytest.raises(MCPReauthRequiredError):
                await oauth_servers.post(_non_interactive(storage))
        finally:
            peer.release()

        assert oauth_servers.token_requests() == []
        assert "Authorization" not in oauth_servers.resource_requests[0].headers
        assert "skipping refresh to avoid refresh-token reuse" in caplog.text

    async def test_refresh_lock_held_until_cancelled_write_finishes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancellation cannot release the refresh lock ahead of persistence."""
        from filelock import FileLock, Timeout

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        write_started = threading.Event()
        allow_write = threading.Event()
        real_write = storage._write

        def _blocking_write(data: dict[str, Any]) -> None:
            write_started.set()
            if not allow_write.wait(timeout=5):
                pytest.fail("timed out waiting to finish the token write")
            real_write(data)

        monkeypatch.setattr(storage, "_write", _blocking_write)

        async def _persist_under_refresh_lock() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired
                await storage.set_tokens(_make_tokens())

        task = asyncio.create_task(_persist_under_refresh_lock())
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        try:
            assert await asyncio.to_thread(write_started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            with pytest.raises(Timeout):
                await asyncio.to_thread(probe.acquire, timeout=0)

            allow_write.set()
            with pytest.raises(asyncio.CancelledError):
                await task

            await asyncio.to_thread(probe.acquire, timeout=1)
        finally:
            allow_write.set()
            await asyncio.gather(task, return_exceptions=True)
            if probe.is_locked:
                probe.release()

    async def test_refresh_lock_write_failure_wins_over_level_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled refresh reports persistence failure after releasing its lock."""
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        loop = asyncio.get_running_loop()
        write_started = asyncio.Event()
        allow_write = threading.Event()
        release_started = asyncio.Event()
        allow_release = threading.Event()
        scopes: list[anyio.CancelScope] = []
        real_release = FileLock.release

        def _failing_write(data: dict[str, Any]) -> None:
            del data
            loop.call_soon_threadsafe(write_started.set)
            if not allow_write.wait(timeout=5):
                pytest.fail("timed out waiting to fail the token write")
            msg = "token persistence failed"
            raise OSError(msg)

        def _blocking_release(lock: FileLock, *, force: bool = False) -> None:
            if force:
                real_release(lock, force=True)
                return
            loop.call_soon_threadsafe(release_started.set)
            if not allow_release.wait(timeout=5):
                pytest.fail("timed out waiting to release the refresh lock")
            real_release(lock)

        monkeypatch.setattr(storage, "_write", _failing_write)
        monkeypatch.setattr(FileLock, "release", _blocking_release)

        async def _persist_in_cancel_scope() -> bool:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async with mcp_auth._refresh_lock_guard(
                    storage.refresh_lock_path, _MCP_URL
                ) as acquired:
                    assert acquired
                    await storage.set_tokens(_make_tokens())
            return scope.cancelled_caught

        task = asyncio.create_task(_persist_in_cancel_scope())
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        try:
            await asyncio.wait_for(write_started.wait(), timeout=5)
            scopes[0].cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_write.set()
            await asyncio.wait_for(release_started.wait(), timeout=5)
            assert not task.done()

            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_release.set()
            with pytest.raises(OSError, match="token persistence failed"):
                await task

            await asyncio.to_thread(probe.acquire, timeout=1)
        finally:
            allow_write.set()
            allow_release.set()
            await asyncio.gather(task, return_exceptions=True)
            if probe.is_locked:
                probe.release()

    async def test_refresh_lock_acquisition_joins_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancellation waits for a lock-acquire worker and releases its result."""
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        acquire_started = asyncio.Event()
        real_acquire = mcp_auth._acquire_refresh_lock

        async def _tracked_acquire(lock: FileLock, server_url: str) -> bool:
            acquire_started.set()
            return await real_acquire(lock, server_url)

        monkeypatch.setattr(mcp_auth, "_acquire_refresh_lock", _tracked_acquire)
        peer = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer.acquire()

        async def _enter_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired

        task = asyncio.create_task(_enter_guard())
        probe = FileLock(str(storage.refresh_lock_path), thread_local=False)
        try:
            await acquire_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            peer.release()
            with pytest.raises(asyncio.CancelledError):
                await task

            await asyncio.to_thread(probe.acquire, timeout=1)
        finally:
            if peer.is_locked:
                peer.release()
            await asyncio.gather(task, return_exceptions=True)
            if probe.is_locked:
                probe.release()

    async def test_refresh_lock_acquire_timeout_yields_to_cancellation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Cancellation deferred through an acquire wins over the acquire timeout.

        When a cancel lands while the acquire worker is still waiting on a
        peer-held lock that ultimately times out, the guard must surface the
        `CancelledError` — not swallow it into the "skip refresh" (`False`)
        path the timeout takes on its own.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        monkeypatch.setattr(mcp_auth, "_REFRESH_LOCK_TIMEOUT_SECONDS", 0.3)
        acquire_started = asyncio.Event()
        real_acquire = mcp_auth._acquire_refresh_lock

        async def _tracked_acquire(lock: FileLock, server_url: str) -> bool:
            acquire_started.set()
            return await real_acquire(lock, server_url)

        monkeypatch.setattr(mcp_auth, "_acquire_refresh_lock", _tracked_acquire)
        # A peer holds the lock for the whole flow, so the provider's acquire
        # can only ever time out.
        peer = FileLock(str(storage.refresh_lock_path), thread_local=False)
        peer.acquire()

        async def _enter_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ):
                pass

        task = asyncio.create_task(_enter_guard())
        try:
            caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
            await acquire_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            with pytest.raises(asyncio.CancelledError):
                await task

            # The timeout's "skip refresh" warning must not fire: cancellation
            # took precedence over the timed-out acquire.
            assert not any(
                "skipping refresh to avoid refresh-token reuse" in record.getMessage()
                for record in caplog.records
            )
        finally:
            if peer.is_locked:
                peer.release()
            await asyncio.gather(task, return_exceptions=True)

    async def test_refresh_lock_release_failure_preserves_body_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A cleanup failure is reported without replacing the guarded error."""
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        real_release = FileLock.release

        def _release_then_fail(lock: FileLock, *, force: bool = False) -> None:
            real_release(lock, force=force)
            if not force:
                msg = "refresh lock release failed"
                raise OSError(msg)

        async def _raise_in_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired
                msg = "guarded operation failed"
                raise ValueError(msg)

        monkeypatch.setattr(FileLock, "release", _release_then_fail)
        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.mcp_auth"),
            pytest.raises(ValueError, match="guarded operation failed") as exc_info,
        ):
            await _raise_in_guard()

        assert any(
            "release also failed with OSError" in note
            for note in exc_info.value.__notes__
        )
        assert any(
            "Failed to release the MCP token refresh lock" in record.getMessage()
            for record in caplog.records
        )

    async def test_refresh_lock_release_failure_preserves_cancellation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A release failure must not mask a cancelled guarded body.

        This is the counterpart to the ``ValueError`` body above: when the
        guarded operation is cancelled and the lock release then fails, the
        ``CancelledError`` must still propagate (structured cancellation
        depends on it) with the release failure attached as a note.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        real_release = FileLock.release

        def _release_then_fail(lock: FileLock, *, force: bool = False) -> None:
            real_release(lock, force=force)
            if not force:
                msg = "refresh lock release failed"
                raise OSError(msg)

        async def _cancel_in_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired
                raise asyncio.CancelledError

        monkeypatch.setattr(FileLock, "release", _release_then_fail)
        with (
            caplog.at_level(logging.WARNING, logger="deepagents_code.mcp_auth"),
            pytest.raises(asyncio.CancelledError) as exc_info,
        ):
            await _cancel_in_guard()

        assert any(
            "release also failed with OSError" in note
            for note in getattr(exc_info.value, "__notes__", [])
        )
        assert any(
            "Failed to release the MCP token refresh lock" in record.getMessage()
            for record in caplog.records
        )

    async def test_refresh_lock_clean_body_release_failure_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A release failure on a clean refresh surfaces instead of vanishing.

        With no guarded error and no cancellation, the release `OSError` is the
        only failure in play, so it must propagate rather than be swallowed.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        real_release = FileLock.release

        def _release_then_fail(lock: FileLock, *, force: bool = False) -> None:
            real_release(lock, force=force)
            if not force:
                msg = "refresh lock release failed"
                raise OSError(msg)

        async def _clean_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired

        monkeypatch.setattr(FileLock, "release", _release_then_fail)
        with pytest.raises(OSError, match="refresh lock release failed"):
            await _clean_guard()

    async def test_refresh_lock_clean_body_release_failure_supersedes_cancellation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A clean-body release failure supersedes a deferred cancellation.

        When the guarded body succeeds but the lock release then fails, the
        release error is the primary failure. A cancellation deferred during the
        release join is dropped in its favor — but that loss must be logged, not
        silent.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        real_release = FileLock.release
        release_started = threading.Event()
        allow_release = threading.Event()

        def _blocking_release_then_fail(lock: FileLock, *, force: bool = False) -> None:
            if force:
                # Best-effort GC-time release must not block or raise.
                real_release(lock, force=force)
                return
            release_started.set()
            if not allow_release.wait(timeout=5):
                pytest.fail("timed out waiting to fail the lock release")
            real_release(lock)
            msg = "refresh lock release failed"
            raise OSError(msg)

        async def _clean_guard() -> None:
            async with mcp_auth._refresh_lock_guard(
                storage.refresh_lock_path, _MCP_URL
            ) as acquired:
                assert acquired

        monkeypatch.setattr(FileLock, "release", _blocking_release_then_fail)
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        task = asyncio.create_task(_clean_guard())
        try:
            # The guarded body has exited; cancel while the release worker is
            # parked so the cancellation is deferred through the release join.
            assert await asyncio.to_thread(release_started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()

            allow_release.set()
            with pytest.raises(OSError, match="refresh lock release failed"):
                await task
        finally:
            allow_release.set()
            await asyncio.gather(task, return_exceptions=True)

        assert any(
            "a deferred cancellation is superseded by the release error"
            in record.getMessage()
            for record in caplog.records
        )

    async def test_acquire_refresh_lock_oserror_skips_refresh(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An un-creatable sidecar lock skips the refresh rather than racing it.

        On a hardened host the `.lock` file may be un-openable (read-only or
        missing tokens dir). `_acquire_refresh_lock` must return `False` — never
        refresh unlocked — and say why, so an operator can see the refresh was
        skipped to avoid refresh-token reuse.
        """
        from filelock import FileLock

        from deepagents_code import mcp_auth

        storage = FileTokenStorage("notion")
        lock = FileLock(str(storage.refresh_lock_path), thread_local=False)

        def _raise_oserror(*_args: Any, **_kwargs: Any) -> None:
            msg = "cannot open lock file"
            raise OSError(msg)

        monkeypatch.setattr(FileLock, "acquire", _raise_oserror)
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")

        assert await mcp_auth._acquire_refresh_lock(lock, _MCP_URL) is False
        assert any(
            "skipping refresh to avoid refresh-token reuse" in record.getMessage()
            for record in caplog.records
        )


@pytest.mark.usefixtures("fake_home")
class TestJoinTaskDeferringCancellation:
    """Tests for the cancellation-deferring task join primitive."""

    async def test_defers_caller_cancellation_without_cancelling_task(self) -> None:
        """A cancelled caller neither cancels the wrapped task nor loses its result.

        The join must let the wrapped task run to completion and hand the
        deferred `CancelledError` back to the caller to re-raise, so an
        in-flight write/lock operation is never abandoned mid-flight.
        """
        from deepagents_code.mcp_auth import _join_task_deferring_cancellation

        started = threading.Event()
        allow_finish = threading.Event()
        captured: dict[str, Any] = {}

        def _blocking() -> str:
            started.set()
            if not allow_finish.wait(timeout=5):
                pytest.fail("timed out waiting to finish the wrapped task")
            return "done"

        async def _caller() -> None:
            task = asyncio.create_task(asyncio.to_thread(_blocking))
            deferred = await _join_task_deferring_cancellation(task)
            captured["deferred"] = deferred
            captured["value"] = task.result()
            if deferred is not None:
                raise deferred

        caller = asyncio.create_task(_caller())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            caller.cancel()
            await asyncio.sleep(0)
            assert not caller.done(), "the join must defer the caller's cancellation"

            allow_finish.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
        finally:
            allow_finish.set()
            await asyncio.gather(caller, return_exceptions=True)

        # The wrapped task ran to completion despite the caller being cancelled,
        # and the deferred cancellation was returned (not swallowed) for re-raise.
        assert captured["value"] == "done"
        assert isinstance(captured["deferred"], asyncio.CancelledError)


@pytest.mark.usefixtures("fake_home")
class TestFindReauthRequired:
    """Tests for unwrapping nested re-auth errors."""

    def test_returns_direct_error(self) -> None:
        """Direct `MCPReauthRequiredError` instances are returned unchanged."""
        exc = MCPReauthRequiredError("srv")
        assert find_reauth_required(exc) is exc

    def test_finds_error_inside_exception_group(self) -> None:
        """Nested exception groups are searched recursively."""
        exc = ExceptionGroup(
            "outer", [RuntimeError("x"), MCPReauthRequiredError("srv")]
        )
        found = find_reauth_required(exc)
        assert isinstance(found, MCPReauthRequiredError)
        assert found.server_name == "srv"

    def test_finds_error_via_cause_chain(self) -> None:
        """`raise X from MCPReauthRequiredError(...)` is unwrapped."""
        reauth = MCPReauthRequiredError("srv")
        outer_msg = "outer"
        try:
            try:
                raise reauth
            except MCPReauthRequiredError as inner:
                raise RuntimeError(outer_msg) from inner
        except RuntimeError as exc:
            found = find_reauth_required(exc)
        assert found is reauth

    def test_finds_error_via_context(self) -> None:
        """Implicit `__context__` chains are searched."""
        reauth = MCPReauthRequiredError("srv")
        outer_msg = "outer"
        try:
            try:
                raise reauth
            except MCPReauthRequiredError:
                raise RuntimeError(outer_msg)  # noqa: B904
        except RuntimeError as exc:
            found = find_reauth_required(exc)
        assert found is reauth

    def test_returns_none_when_absent(self) -> None:
        """Pure exception trees without reauth errors yield `None`."""
        exc = ExceptionGroup("outer", [RuntimeError("x"), ValueError("y")])
        assert find_reauth_required(exc) is None

    def test_handles_cyclic_chain(self) -> None:
        """Self-referencing `__context__` cycles terminate without recursion."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        assert find_reauth_required(a) is None


class TestFindOauthChallenge:
    """Tests for detecting a 401 OAuth challenge in an exception tree."""

    def test_direct_401_with_challenge(self) -> None:
        """A 401 carrying an RFC 9728 Bearer challenge yields its URL."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_header_match_is_case_insensitive(self) -> None:
        """The scheme and parameter matching ignore casing."""
        exc = _http_status_error(
            401,
            headers={
                "www-authenticate": (
                    f'bearer resource_METADATA="{_RESOURCE_METADATA_URL}"'
                )
            },
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_multiparam_bearer_challenge(self) -> None:
        """`resource_metadata` is found after other Bearer auth-params."""
        exc = _http_status_error(
            401,
            headers={
                "WWW-Authenticate": (
                    'Bearer error="invalid_token", '
                    'error_description="The access token expired", '
                    f'resource_metadata="{_RESOURCE_METADATA_URL}"'
                )
            },
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_bearer_not_first_in_multischeme_line(self) -> None:
        """A Bearer challenge behind another scheme on one line is detected."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": f'Basic realm="mcp", {_BEARER_CHALLENGE}'},
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_bearer_across_repeated_headers(self) -> None:
        """A Bearer challenge on a second `WWW-Authenticate` line is detected."""
        exc = _http_status_error(
            401,
            headers=[
                ("WWW-Authenticate", 'Basic realm="mcp"'),
                (
                    "WWW-Authenticate",
                    _BEARER_CHALLENGE,
                ),
            ],
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_401_without_challenge_header_ignored(self) -> None:
        """A 401 lacking `WWW-Authenticate` is not an OAuth challenge."""
        exc = _http_status_error(401)
        assert find_oauth_challenge(exc) is None

    def test_401_basic_challenge_ignored(self) -> None:
        """A non-OAuth auth challenge is not treated as an MCP login prompt."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Basic realm="mcp"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_401_bearer_without_resource_metadata_ignored(self) -> None:
        """A Bearer challenge with params but no `resource_metadata` is ignored."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_401_resource_metadata_substring_not_matched(self) -> None:
        """`resource_metadata` embedded in another token is not a match."""
        exc = _http_status_error(
            401,
            headers={"WWW-Authenticate": 'Bearer error="x_resource_metadata_y"'},
        )
        assert find_oauth_challenge(exc) is None

    def test_non_401_status_ignored(self) -> None:
        """Other status codes never count as a challenge."""
        exc = _http_status_error(
            403,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        assert find_oauth_challenge(exc) is None

    def test_found_inside_exception_group(self) -> None:
        """Nested exception groups are searched recursively."""
        exc = ExceptionGroup(
            "outer",
            [
                RuntimeError("x"),
                _http_status_error(
                    401,
                    headers={"WWW-Authenticate": (_BEARER_CHALLENGE)},
                ),
            ],
        )
        assert find_oauth_challenge(exc) == _RESOURCE_METADATA_URL

    def test_found_via_cause_chain(self) -> None:
        """`raise X from HTTPStatusError(...)` is unwrapped."""
        challenge = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        wrapped = RuntimeError("wrapped")
        wrapped.__cause__ = challenge
        assert find_oauth_challenge(wrapped) == _RESOURCE_METADATA_URL

    def test_found_via_context_chain(self) -> None:
        """Implicit chaining (`__context__`) is unwrapped, not only `__cause__`."""
        challenge = _http_status_error(
            401,
            headers={"WWW-Authenticate": _BEARER_CHALLENGE},
        )
        wrapped = RuntimeError("wrapped")
        wrapped.__context__ = challenge
        assert find_oauth_challenge(wrapped) == _RESOURCE_METADATA_URL

    def test_returns_none_when_absent(self) -> None:
        """Trees without a 401 challenge yield `None`."""
        exc = ExceptionGroup("outer", [RuntimeError("x"), ValueError("y")])
        assert find_oauth_challenge(exc) is None

    def test_handles_cyclic_chain(self) -> None:
        """Self-referencing `__context__` cycles terminate without recursion."""
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__context__ = b
        b.__context__ = a
        assert find_oauth_challenge(a) is None


class TestFormatLoginFailure:
    """Tests for the token-safe summary helper used in app + CLI logs."""

    def test_returns_reauth_message_for_nested_reauth_error(self) -> None:
        """ExceptionGroup wrapping `MCPReauthRequiredError` surfaces its message."""
        exc = ExceptionGroup(
            "anyio task group",
            [RuntimeError("upstream"), MCPReauthRequiredError("notion")],
        )
        summary = format_login_failure(exc)
        assert "notion" in summary
        assert "Run `/mcp login notion`" in summary

    def test_omits_message_for_unknown_exception_types(self) -> None:
        """Unrecognized exceptions degrade to a class-name chain — no `str()`.

        Tokens can hide in `args`/`repr` of unfamiliar MCP-SDK error types;
        the helper must never include those payloads.
        """

        class FakeMcpError(RuntimeError):
            pass

        sentinel = "TOKEN_PAYLOAD_DO_NOT_LEAK"
        exc = FakeMcpError(sentinel)
        summary = format_login_failure(exc)
        assert sentinel not in summary
        assert "FakeMcpError" in summary

    def test_preserves_message_for_config_errors(self) -> None:
        """Config errors are pre-handshake and token-free, so keep the message.

        These carry the actionable field path (e.g. which var is unset);
        collapsing them to a bare class name would strip the only guidance
        the user has for fixing their `.mcp.json`.
        """
        from deepagents_code.mcp_tools import MCPConfigError

        message = "mcpServers.notion.url references unset env var MCP_GATEWAY_HOST."
        summary = format_login_failure(MCPConfigError(message))
        assert summary == message

    def test_includes_message_for_known_loopback_errors(self) -> None:
        """Loopback-internal exceptions are token-free and may include their message."""
        from deepagents_code.mcp_auth import _LoopbackCallbackTimeoutError

        exc = _LoopbackCallbackTimeoutError("Callback timed out")
        summary = format_login_failure(exc)
        assert "Callback timed out" in summary
        assert "_LoopbackCallbackTimeoutError" in summary

    def test_walks_cause_chain_into_class_names(self) -> None:
        """A chained unknown exception still surfaces every link's class name."""

        class OuterError(RuntimeError):
            pass

        class InnerError(RuntimeError):
            pass

        inner_msg = "inner-payload"
        outer_msg = "outer-payload"
        try:
            try:
                raise InnerError(inner_msg)  # noqa: TRY301
            except InnerError as inner:
                raise OuterError(outer_msg) from inner
        except OuterError as exc:
            summary = format_login_failure(exc)
        assert "OuterError" in summary
        assert "InnerError" in summary
        assert inner_msg not in summary
        assert outer_msg not in summary


class TestAppendQueryParams:
    """Tests for `_append_query_params` URL manipulation."""

    def test_adds_params_to_url_without_query(self) -> None:
        """Params are appended when the URL has no query string."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "T123"})
        assert "team=T123" in result

    def test_overwrites_existing_same_key(self) -> None:
        """Existing same-key query params are replaced, not merged."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x?team=OLD", {"team": "NEW"})
        assert "team=NEW" in result
        assert "team=OLD" not in result

    def test_url_encodes_special_characters(self) -> None:
        """Special characters in values are properly URL-encoded."""
        from deepagents_code.mcp_auth import _append_query_params

        result = _append_query_params("https://example.com/x", {"team": "a b&c"})
        assert "team=a+b%26c" in result


class TestPasteBackHandlers:
    """Tests for the interactive OAuth paste-back callback handler."""

    async def test_callback_parses_code_and_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Callback URL with `code` and `state` yields both values."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr(
            "builtins.input", lambda _: "https://localhost/?code=abc&state=xyz"
        )
        result = await callback()
        assert result.code == "abc"
        assert result.state == "xyz"

    async def test_callback_returns_the_sdk_authorization_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The handler returns what the MCP SDK reads, not a bare tuple.

        `OAuthClientProvider._perform_authorization` accesses `.state`, `.iss`
        and `.code` on this value, so returning a `(code, state)` tuple made
        every OAuth login fail with `AttributeError` and dropped the RFC 9207
        issuer the SDK validates.
        """
        from mcp.shared.auth import AuthorizationCodeResult

        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr(
            "builtins.input",
            lambda _: (
                "https://localhost/?code=abc&state=xyz&iss=https%3A%2F%2Fslack.com"
            ),
        )
        result = await callback()

        assert isinstance(result, AuthorizationCodeResult)
        assert (result.code, result.state) == ("abc", "xyz")
        assert result.iss == "https://slack.com"

    async def test_callback_leaves_absent_state_and_iss_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A callback URL carrying only `code` leaves the optionals `None`."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr("builtins.input", lambda _: "https://localhost/?code=abc")
        result = await callback()

        assert result.code == "abc"
        assert result.state is None
        assert result.iss is None

    async def test_callback_missing_code_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """URL without `code` raises a clear error."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr("builtins.input", lambda _: "https://localhost/?other=1")
        with pytest.raises(RuntimeError, match="missing the 'code' parameter"):
            await callback()

    async def test_callback_surfaces_provider_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`error=` in the callback URL surfaces provider-side denials."""
        from deepagents_code.mcp_auth import _make_paste_back_handlers

        _, callback = _make_paste_back_handlers()
        monkeypatch.setattr(
            "builtins.input",
            lambda _: (
                "https://localhost/?error=access_denied"
                "&error_description=User%20declined"
            ),
        )
        with pytest.raises(RuntimeError, match="access_denied"):
            await callback()


class TestBuildOAuthProvider:
    """Tests for `build_oauth_provider` branching."""

    def test_slack_url_is_detected(self) -> None:
        """The Slack URL detector treats slack.com subdomains as Slack."""
        from deepagents_code.mcp_providers.slack import _is_slack_mcp_url

        assert _is_slack_mcp_url("https://slack.com/mcp")
        assert _is_slack_mcp_url("https://deep.slack.com/mcp")
        assert not _is_slack_mcp_url("https://mcp.notion.com/mcp")

    def test_slack_provider_uses_fixed_loopback_port(self) -> None:
        """SlackProvider uses a fixed port matching the Slack app registration."""
        from deepagents_code.mcp_providers.slack import (
            _SLACK_LOOPBACK_PORT,
            SlackProvider,
        )

        provider = SlackProvider()
        assert provider.supports_loopback_callback() is True
        assert provider.loopback_port() == _SLACK_LOOPBACK_PORT

    def test_slack_branch_sets_public_client_metadata(self) -> None:
        """Slack branch configures a public OAuth client using the loopback URI."""
        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _SLACK_REDIRECT_URI

        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://slack.com/mcp",
            storage=FileTokenStorage("slack"),
        )
        metadata = provider.context.client_metadata
        assert metadata.token_endpoint_auth_method == "none"
        assert metadata.redirect_uris is not None
        assert [str(uri) for uri in metadata.redirect_uris] == [_SLACK_REDIRECT_URI]

    def test_generic_branch_uses_loopback_callback(self) -> None:
        """Non-Slack URLs (including Notion) use a local callback server redirect."""
        from deepagents_code.mcp_auth import build_oauth_provider

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        assert re.fullmatch(r"http://localhost:\d+/callback", redirect_uri)
        # Generic (non-Slack) providers default to client-secret auth, so the
        # Slack-only `token_endpoint_auth_method="none"` override must not
        # leak into this branch.
        assert metadata.token_endpoint_auth_method != "none"

    def test_generic_branch_reuses_stored_loopback_port(self, fake_home: Path) -> None:
        """A persisted DCR redirect URI pins the callback port across launches."""
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(51208)))
        first = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        second = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        first_metadata = first.context.client_metadata
        second_metadata = second.context.client_metadata
        assert first_metadata.redirect_uris is not None
        assert second_metadata.redirect_uris is not None
        assert str(first_metadata.redirect_uris[0]) == "http://localhost:51208/callback"
        assert (
            str(second_metadata.redirect_uris[0]) == "http://localhost:51208/callback"
        )

    def test_fixed_loopback_port_wins_over_stored_port(self, fake_home: Path) -> None:
        """Provider-fixed callback ports take precedence over stored DCR ports."""
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider
        from deepagents_code.mcp_providers.slack import _SLACK_REDIRECT_URI

        storage = FileTokenStorage("slack")
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(51208)))
        provider = build_oauth_provider(
            server_name="slack",
            server_url="https://slack.com/mcp",
            storage=storage,
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == _SLACK_REDIRECT_URI

    def test_generic_branch_random_port_when_stored_uri_non_loopback(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-loopback stored URI falls back to a fresh random port.

        A token is seeded so the stale-registration self-heal is skipped
        (`discard_client_info_if_loopback_unusable` only fires when no token is
        persisted); that keeps this test focused on the random-port fallback,
        distinct from `test_build_oauth_provider_clears_stale_portless_registration`.
        """
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        monkeypatch.setattr(
            "deepagents_code.mcp_auth._choose_loopback_port", lambda: 60001
        )
        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info()))  # localhost, no port
        asyncio.run(storage.set_tokens(_make_tokens()))  # blocks self-heal discard
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == "http://localhost:60001/callback"
        assert "http://localhost/callback" in caplog.text
        assert "not a reusable loopback callback URI" in caplog.text

    def test_stored_loopback_port(self, fake_home: Path) -> None:
        """The storage helper extracts ports only from valid loopback URIs."""
        del fake_home

        storage = FileTokenStorage("notion")
        # No token file on disk yet.
        assert storage.stored_loopback_port() is None
        # Loopback URI with explicit port — reused.
        asyncio.run(storage.set_client_info(_make_client_info_with_loopback(54321)))
        assert storage.stored_loopback_port() == 54321

    @pytest.mark.parametrize(
        "uri",
        [
            "https://localhost:5000/callback",
            "http://127.0.0.1:5000/callback",
            "http://localhost:5000/cb",
            "http://localhost:notaport/callback",
        ],
    )
    def test_stored_loopback_port_rejects_non_reusable_uris(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture, uri: str
    ) -> None:
        """Stored ports are reused only for the exact loopback callback shape."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "client_info": {
                        "client_id": "client-id",
                        "redirect_uris": [uri],
                        "grant_types": ["authorization_code", "refresh_token"],
                        "response_types": ["code"],
                    },
                }
            ),
            encoding="utf-8",
        )

        assert storage.stored_loopback_port() is None
        assert uri in caplog.text
        assert "not a reusable loopback callback URI" in caplog.text

    def test_stored_loopback_port_warns_when_token_file_unreadable(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unreadable token files fall back with a warning breadcrumb."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_bytes(b"{not json")

        assert storage.stored_loopback_port() is None
        assert "unreadable during loopback port lookup" in caplog.text
        assert "Failed to read MCP token file" in caplog.text


class TestLoopbackHandlers:
    """Tests for the local OAuth callback server."""

    async def test_escape_aborts_browser_callback(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """Esc closes the callback server without entering paste-back fallback."""
        from http.server import ThreadingHTTPServer

        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from deepagents_code.mcp_auth import (
            _LoopbackOAuthCallbackServer,
            _make_loopback_handlers,
        )
        from deepagents_code.mcp_oauth_ui import (
            CliOAuthInteraction,
            MCPLoginAbortedError,
        )

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stderr.isatty", lambda: True)
        monkeypatch.setattr(
            "builtins.input", lambda _: pytest.fail("Esc must not prompt for a URL")
        )
        output = DummyOutput()
        monkeypatch.setattr(
            "prompt_toolkit.output.defaults.create_output", lambda **_: output
        )
        server = _LoopbackOAuthCallbackServer(port=0)
        redirect, callback = _make_loopback_handlers(
            callback_server=server, ui=CliOAuthInteraction()
        )
        try:
            await redirect("https://auth.example/authorize")
            assert isinstance(server._server, ThreadingHTTPServer)
            assert server._server.socket.fileno() != -1
            with (
                create_pipe_input() as pipe,
                create_app_session(input=pipe, output=output),
            ):
                pipe.send_text("\x1b")
                with pytest.raises(MCPLoginAbortedError):
                    await asyncio.wait_for(callback(), timeout=3)
            assert server._server.socket.fileno() == -1
        finally:
            server.close()

    async def test_loopback_callback_returns_code_and_state(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A browser callback to the loopback URI completes the handler."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{redirect_uri}?code=abc&state=xyz")

        assert response.status_code == 200
        result = await callback_handler()
        assert result.code == "abc"
        assert result.state == "xyz"

    async def test_loopback_callback_surfaces_provider_error(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """Provider-side callback errors propagate with a useful message."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{redirect_uri}?error=access_denied&error_description=User%20declined"
            )

        assert response.status_code == 400
        with pytest.raises(RuntimeError, match="access_denied"):
            await callback_handler()

    async def test_loopback_callback_missing_code_raises(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A callback URL missing the `code` parameter sends 400 and raises."""
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{redirect_uri}?state=xyz")

        assert response.status_code == 400
        with pytest.raises(RuntimeError, match="missing the 'code' parameter"):
            await callback_handler()

    async def test_loopback_falls_back_when_browser_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the browser cannot open, callback() falls back to paste-back at once."""
        from deepagents_code.mcp_auth import build_oauth_provider

        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: False)
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        result = await callback_handler()
        assert result.code == "fallback"
        assert result.state == "s"

    async def test_loopback_falls_back_on_bind_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bind failure in redirect() causes callback() to fall back to paste-back."""
        from deepagents_code.mcp_auth import (
            _LoopbackOAuthCallbackServer,
            build_oauth_provider,
        )

        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        monkeypatch.setattr(
            _LoopbackOAuthCallbackServer,
            "start",
            lambda _self: (_ for _ in ()).throw(OSError("Address already in use")),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        result = await callback_handler()
        assert result.code == "fallback"
        assert result.state == "s"

    async def test_loopback_falls_back_when_webbrowser_get_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No-browser environments (headless, SSH) skip the 300s loopback wait.

        `webbrowser.open` can return `True` in some headless setups even
        when nothing actually launches. `webbrowser.get` raising
        `webbrowser.Error` is the reliable signal that no browser is
        available — the redirect handler must trip the paste-back path
        without binding a socket or burning the timeout.
        """
        import webbrowser

        from deepagents_code.mcp_auth import build_oauth_provider

        def _raise_no_browser(*_a: object, **_kw: object) -> object:
            no_browser_msg = "no browser"
            raise webbrowser.Error(no_browser_msg)

        monkeypatch.setattr("webbrowser.get", _raise_no_browser)
        # Sanity: webbrowser.open intentionally returns True to prove we
        # never call it when get() fails first.
        monkeypatch.setattr(
            "webbrowser.open",
            lambda _url: pytest.fail("webbrowser.open should not be called"),
        )
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=fallback&state=s",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        result = await callback_handler()
        assert result.code == "fallback"
        assert result.state == "s"

    async def test_loopback_falls_back_on_callback_timeout(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A loopback callback that never arrives falls through to paste-back.

        Regression guard for `_LoopbackCallbackTimeoutError`. Without this
        path, a user whose browser opens but never redirects would hang
        for the full `_LOOPBACK_CALLBACK_TIMEOUT` (300s).
        """
        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("deepagents_code.mcp_auth._LOOPBACK_CALLBACK_TIMEOUT", 0.05)
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        monkeypatch.setattr(
            "builtins.input",
            lambda _: "https://localhost/?code=after_timeout",
        )
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None

        await redirect_handler("https://auth.example/authorize")
        result = await callback_handler()
        assert result.code == "after_timeout"

    async def test_loopback_repeat_request_after_error_shows_error_page(
        self, monkeypatch: pytest.MonkeyPatch, socket_enabled: object
    ) -> None:
        """A duplicate request after a failed callback must not show success HTML.

        Regression guard: previously `_handle_get` early-returned success
        whenever the future was done, even if the future resolved with an
        exception. A second browser hit (prefetch, favicon) would render
        "You're signed in" while the worker was actually surfacing the
        underlying error.
        """
        import httpx

        from deepagents_code.mcp_auth import build_oauth_provider

        del socket_enabled
        monkeypatch.setattr("webbrowser.get", lambda *_a, **_kw: object())
        monkeypatch.setattr("webbrowser.open", lambda _url: True)
        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=FileTokenStorage("notion"),
        )
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        redirect_uri = str(metadata.redirect_uris[0])
        redirect_handler = provider.context.redirect_handler
        assert redirect_handler is not None

        await redirect_handler("https://auth.example/authorize")
        async with httpx.AsyncClient(timeout=5.0) as client:
            first = await client.get(f"{redirect_uri}?error=access_denied")
            second = await client.get(f"{redirect_uri}?code=late")

        assert first.status_code == 400
        # Second request must surface the prior error state, not success.
        assert second.status_code == 400
        assert "You're signed in" not in second.text
        assert "did not complete" in second.text


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorageExtras:
    """Extended storage tests (migration, atomic writes)."""

    async def test_version_mismatch_raises(self, fake_home: Path) -> None:
        """Token files with an unknown version fail with a remediation hint."""
        storage = FileTokenStorage("notion")
        path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 999, "tokens": {}}))

        with pytest.raises(RuntimeError, match="unsupported version"):
            await storage.get_tokens()

    async def test_set_tokens_and_client_info_atomic(self, fake_home: Path) -> None:
        """Atomic setter writes both fields in a single on-disk payload."""
        storage = FileTokenStorage("notion")
        await storage.set_tokens_and_client_info(_make_tokens(), _make_client_info())

        token_path = fake_home / ".deepagents" / ".state" / "mcp-tokens" / "notion.json"
        raw = token_path.read_text()
        data = json.loads(raw)
        assert "tokens" in data
        assert "client_info" in data
        assert data["tokens"]["access_token"] == "at"
        assert data["client_info"]["client_id"] == "client-id"

    async def test_discard_removes_portless_registration_without_tokens(
        self, fake_home: Path
    ) -> None:
        """A portless loopback registration with no tokens is removed."""
        del fake_home
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())  # localhost, no port

        assert storage.discard_client_info_if_loopback_unusable() is True
        assert await storage.get_client_info() is None

    async def test_discard_keeps_ported_loopback_registration(
        self, fake_home: Path
    ) -> None:
        """A reusable ported loopback registration is left intact."""
        del fake_home
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info_with_loopback(51208))

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert await storage.get_client_info() is not None

    async def test_discard_keeps_registration_when_tokens_present(
        self, fake_home: Path
    ) -> None:
        """A still-usable token blocks discard so refresh isn't downgraded."""
        del fake_home
        storage = FileTokenStorage("notion")
        # Portless registration, but a persisted token can still authenticate.
        await storage.set_client_info(_make_client_info())
        await storage.set_tokens(_make_tokens())

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert await storage.get_client_info() is not None

    async def test_discard_and_token_write_are_serialized_across_instances(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The synchronous self-heal cannot race an async token update."""
        del fake_home
        token_storage = FileTokenStorage("notion")
        heal_storage = FileTokenStorage("notion")
        await token_storage.set_client_info(_make_client_info())

        write_barrier = threading.Barrier(2)
        real_write = FileTokenStorage._write

        def _collision_detecting_write(
            storage: FileTokenStorage, data: dict[str, Any]
        ) -> None:
            try:
                write_barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                real_write(storage, data)
                return
            msg = "self-heal overlapped another shared-envelope mutation"
            raise FileExistsError(msg)

        monkeypatch.setattr(
            FileTokenStorage,
            "_write",
            _collision_detecting_write,
        )

        await asyncio.gather(
            token_storage.set_tokens(_make_tokens()),
            asyncio.to_thread(heal_storage.discard_client_info_if_loopback_unusable),
        )

        assert await token_storage.get_tokens() is not None

    async def test_discard_noop_without_client_info(self, fake_home: Path) -> None:
        """No persisted registration means nothing to discard."""
        del fake_home
        storage = FileTokenStorage("notion")

        assert storage.discard_client_info_if_loopback_unusable() is False

    def test_discard_returns_false_and_warns_on_unreadable_file(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt token file is surfaced (not silently swallowed)."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        storage.path.parent.mkdir(parents=True)
        storage.path.write_bytes(b"{not json")

        assert storage.discard_client_info_if_loopback_unusable() is False
        assert "unreadable while checking for a stale client registration" in (
            caplog.text
        )

    async def test_discard_returns_false_and_keeps_file_when_write_fails(
        self, fake_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed atomic write leaves the registration intact and warns."""
        del fake_home
        caplog.set_level(logging.WARNING, logger="deepagents_code.mcp_auth")
        storage = FileTokenStorage("notion")
        await storage.set_client_info(_make_client_info())  # portless, no tokens
        # Occupy the temp path with a directory so the real atomic write fails
        # with an OSError instead of replacing the token file — no mocks needed.
        tmp = storage.path.with_suffix(storage.path.suffix + ".tmp")
        tmp.mkdir()

        assert storage.discard_client_info_if_loopback_unusable() is False
        # The original registration must still be on disk.
        assert await storage.get_client_info() is not None
        assert "Could not remove stale MCP client registration" in caplog.text

    def test_build_oauth_provider_clears_stale_portless_registration(
        self,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Interactive loopback login drops a stale portless registration.

        Regression: a portless `http://localhost/callback` registration (left
        by an earlier non-loopback login) was reused with a fresh random port,
        so the authorize request sent the stale `client_id` with a
        redirect_uri it was never registered for and the server rejected it
        with "invalid or missing redirect_uri". The build must instead discard
        the registration so the handshake re-runs DCR with a matching URI.
        """
        del fake_home
        from deepagents_code.mcp_auth import build_oauth_provider

        monkeypatch.setattr(
            "deepagents_code.mcp_auth._choose_loopback_port", lambda: 60001
        )
        storage = FileTokenStorage("notion")
        asyncio.run(storage.set_client_info(_make_client_info()))  # localhost, no port

        provider = build_oauth_provider(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            storage=storage,
        )

        # Stale registration gone, so the SDK will re-register via DCR.
        assert asyncio.run(storage.get_client_info()) is None
        # The authorize request will carry the freshly bound loopback URI.
        metadata = provider.context.client_metadata
        assert metadata.redirect_uris is not None
        assert str(metadata.redirect_uris[0]) == "http://localhost:60001/callback"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `asyncio.sleep` with a yield so device-flow tests stay fast."""
    real_sleep = asyncio.sleep

    async def _fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


@pytest.mark.usefixtures("no_sleep")
class TestDeviceFlow:
    """Tests for the OAuth 2.0 Device Authorization Grant helper."""

    async def test_happy_path_returns_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful poll returns the issued `OAuthToken`."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(200, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_slow_down_increases_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`slow_down` errors bump the poll interval and continue polling."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(200, json={"error": "slow_down"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_pending_on_http_400_still_polls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Providers returning HTTP 400 for `authorization_pending` still poll."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        state = {"polls": 0}

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            state["polls"] += 1
            if state["polls"] == 1:
                return httpx.Response(400, json={"error": "authorization_pending"})
            return httpx.Response(
                200,
                json={"access_token": "tok", "token_type": "Bearer"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        token = await _run_device_flow(
            device_code_url="https://example/device",
            token_url="https://example/token",
            client_id="cid",
        )
        assert token.access_token == "tok"

    async def test_error_surfaces_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-recoverable errors surface the provider's description."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        "expires_in": 30,
                        "interval": 0,
                    },
                )
            return httpx.Response(
                200,
                json={"error": "access_denied", "error_description": "nope"},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="access_denied"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )

    async def test_device_code_request_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 4xx on the initial device-code request raises `RuntimeError`."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={})

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="Device code request failed"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )


@pytest.mark.usefixtures("fake_home")
class TestLogin:
    """Tests for the interactive OAuth login entrypoint."""

    @pytest.mark.parametrize(
        ("declaration", "endpoint", "expected"),
        [
            *[
                ({field: alias}, "mcp", "sse" if alias == "sse" else "http")
                for field in ("type", "transport")
                for alias in ("streamable_http", "streamable-http", "http", "sse")
            ],
            ({"type": "sse", "transport": "streamable_http"}, "mcp", "sse"),
            ({"type": "streamable-http", "transport": "sse"}, "sse", "http"),
            ({}, "mcp", "http"),
            ({}, "sse", "sse"),
        ],
    )
    async def test_login_transport_selection(
        self, declaration: dict[str, str], endpoint: str, expected: str
    ) -> None:
        """Login validates aliases and retains transport precedence and inference."""
        from fastmcp.client.transports import SSETransport, StreamableHttpTransport

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        config = {**declaration, "url": f"https://example.com/{endpoint}"}
        original = config.copy()
        with patch("deepagents_code.mcp_auth._drive_handshake") as handshake:
            await login(
                server_name="api", server_config=config, ui=CliOAuthInteraction()
            )
        transport = handshake.call_args.args[0]
        transport_class = SSETransport if expected == "sse" else StreamableHttpTransport
        assert isinstance(transport, transport_class)
        assert isinstance(transport.auth, OAuthClientProvider)
        assert config == original

    async def test_login_persists_tokens(self) -> None:
        """Successful login persists tokens to the server-specific file."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login

        async def _fake_handshake(transport: StreamableHttpTransport) -> None:
            storage = FileTokenStorage("notion", server_url=transport.url)
            await storage.set_tokens(
                OAuthToken(access_token="new", token_type="Bearer")
            )
            await storage.set_client_info(_make_client_info())

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                },
                ui=CliOAuthInteraction(),
            )

        storage = FileTokenStorage(
            "notion",
            server_url="https://mcp.notion.com/mcp",
        )
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_forces_fresh_auth_when_stored_token_is_valid(self) -> None:
        """Re-auth must re-authorize, not ride the still-valid stored token.

        A healthy server has a valid token on disk. If the provider can see
        it, the handshake succeeds without ever prompting and the user is
        told they logged in again when nothing happened.
        """
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        url = "https://mcp.notion.com/mcp"
        storage = FileTokenStorage("notion", server_url=url)
        await storage.set_tokens(OAuthToken(access_token="old", token_type="Bearer"))
        await storage.set_client_info(_make_client_info())

        seen: dict[str, Any] = {}

        async def _fake_handshake(transport: StreamableHttpTransport) -> None:
            server_name, connection = "notion", transport
            provider = connection.auth
            assert isinstance(provider, MCPOAuthProvider)
            provider_storage = provider._store
            assert isinstance(provider_storage, FileTokenStorage)
            seen["credentials"] = await provider_storage.load()
            fresh = FileTokenStorage(server_name, server_url=connection.url)
            await fresh.set_tokens(OAuthToken(access_token="new", token_type="Bearer"))

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={"transport": "http", "url": url, "auth": "oauth"},
                ui=CliOAuthInteraction(),
            )

        # The provider saw no token, so it had to run the full flow.
        credentials = seen["credentials"]
        assert credentials.tokens is None
        assert credentials.expires_at is None
        # Client registration still resolves, so the handshake reuses it
        # rather than re-running DCR against a new redirect URI.
        assert credentials.client_info is not None
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_keeps_stored_token_when_handshake_fails(self) -> None:
        """An aborted re-auth leaves the working credential on disk."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        url = "https://mcp.notion.com/mcp"
        storage = FileTokenStorage("notion", server_url=url)
        await storage.set_tokens(OAuthToken(access_token="old", token_type="Bearer"))
        await storage.set_client_info(_make_client_info())

        async def _failing_handshake(transport: StreamableHttpTransport) -> None:
            msg = "user aborted"
            raise RuntimeError(msg)

        with (
            patch("deepagents_code.mcp_auth._drive_handshake", _failing_handshake),
            pytest.raises(RuntimeError, match="user aborted"),
        ):
            await login(
                server_name="notion",
                server_config={"transport": "http", "url": url, "auth": "oauth"},
                ui=CliOAuthInteraction(),
            )

        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "old"

    async def test_login_allows_http_server_without_explicit_oauth(self) -> None:
        """Auto-detected servers (no `auth: oauth`) can still run OAuth login."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        async def _fake_handshake(transport: StreamableHttpTransport) -> None:
            storage = FileTokenStorage("notion", server_url=transport.url)
            await storage.set_tokens(
                OAuthToken(access_token="new", token_type="Bearer")
            )
            await storage.set_client_info(_make_client_info())

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                },
                ui=CliOAuthInteraction(),
            )

        storage = FileTokenStorage("notion", server_url="https://mcp.notion.com/mcp")
        tokens = await storage.get_tokens()
        assert tokens is not None
        assert tokens.access_token == "new"

    async def test_login_rejects_stdio_server(self) -> None:
        """OAuth login is limited to HTTP/SSE transports."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with pytest.raises(ValueError, match="only valid for http/sse"):
            await login(
                server_name="srv",
                server_config={"command": "echo", "auth": "oauth"},
                ui=CliOAuthInteraction(),
            )

    async def test_login_propagates_static_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured static headers flow into the OAuth handshake connection."""
        from deepagents_code.mcp_auth import login

        monkeypatch.setenv("MCP_GATEWAY_HOST", "mcp.notion.com")
        monkeypatch.setenv("MCP_GATEWAY_TOKEN", "gw-token")
        captured: dict[str, Any] = {}

        async def _fake_handshake(transport: StreamableHttpTransport) -> None:
            await asyncio.sleep(0)
            captured["url"] = transport.url
            captured["headers"] = dict(transport.headers or {})

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://${MCP_GATEWAY_HOST}/mcp",
                    "auth": "oauth",
                    "headers": {
                        "X-Tenant": "acme",
                        "Authorization": "Bearer ${MCP_GATEWAY_TOKEN}",
                    },
                },
                ui=CliOAuthInteraction(),
            )

        assert captured["url"] == "https://mcp.notion.com/mcp"
        assert captured["headers"] == {
            "X-Tenant": "acme",
            "Authorization": "Bearer gw-token",
        }

    async def test_login_unset_env_var_in_headers_raises(self) -> None:
        """Unset env vars in static headers fail before the handshake."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction
        from deepagents_code.mcp_tools import MCPConfigError

        with pytest.raises(MCPConfigError, match="unset env var"):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": "oauth",
                    "headers": {"Authorization": "Bearer ${MISSING_VAR}"},
                },
                ui=CliOAuthInteraction(),
            )

    async def test_login_unset_env_var_in_url_raises_config_error(self) -> None:
        """An unset var in a non-header field fails with its field path."""
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction
        from deepagents_code.mcp_tools import MCPConfigError

        with pytest.raises(MCPConfigError, match=r"mcpServers\.notion\.url"):
            await login(
                server_name="notion",
                server_config={
                    "transport": "http",
                    "url": "https://${MISSING_HOST}/mcp",
                    "auth": "oauth",
                },
                ui=CliOAuthInteraction(),
            )

    async def test_login_non_string_field_raises_config_error(self) -> None:
        """A non-string supported field is wrapped as `MCPConfigError` too.

        Exercises the `TypeError` arm of `login()`'s resolution wrapper (the
        unset-var tests only cover the `RuntimeError` arm).
        """
        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction
        from deepagents_code.mcp_tools import MCPConfigError

        # Deliberately malformed (non-string header value) to hit the
        # `TypeError` arm; typed separately so the intent is explicit.
        bad_config: dict[str, Any] = {
            "transport": "http",
            "url": "https://mcp.example.com/mcp",
            "auth": "oauth",
            "headers": {"X-Bad": 1},
        }

        with pytest.raises(MCPConfigError, match=r"mcpServers\.notion\.headers\.X-Bad"):
            await login(
                server_name="notion",
                server_config=bad_config,  # ty: ignore
                ui=CliOAuthInteraction(),
            )

    async def test_github_login_runs_device_flow_and_seeds_client(self) -> None:
        """GitHub URLs short-circuit to device flow and persist client info."""
        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_providers.github import _GITHUB_MCP_CLIENT_ID

        async def _fake_device_flow(
            *,
            device_code_url: str,
            token_url: str,
            client_id: str,
            scope: str | None = None,
            ui: object | None = None,
        ) -> OAuthToken:
            del device_code_url, token_url, client_id, scope, ui
            return OAuthToken(access_token="gh-tok", token_type="Bearer")

        handshake_called = False

        async def _handshake_should_not_run(connections: dict) -> None:
            del connections
            nonlocal handshake_called
            handshake_called = True

        from deepagents_code.mcp_oauth_ui import CliOAuthInteraction

        with (
            patch(
                "deepagents_code.mcp_providers.github._run_device_flow",
                _fake_device_flow,
            ),
            patch(
                "deepagents_code.mcp_auth._drive_handshake",
                _handshake_should_not_run,
            ),
        ):
            await login(
                server_name="github",
                server_config={
                    "type": "http",
                    "url": "https://api.githubcopilot.com/mcp/",
                    "auth": "oauth",
                },
                ui=CliOAuthInteraction(),
            )

        assert handshake_called is False, (
            "GitHub login must use device flow, not the authorization-code handshake."
        )
        storage = FileTokenStorage(
            "github",
            server_url="https://api.githubcopilot.com/mcp/",
        )
        tokens = await storage.get_tokens()
        client_info = await storage.get_client_info()
        assert tokens is not None
        assert tokens.access_token == "gh-tok"
        assert client_info is not None
        assert client_info.client_id == _GITHUB_MCP_CLIENT_ID

    async def test_slack_login_routes_team_into_redirect_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slack login threads the entered team id into the interactive URL."""
        monkeypatch.setattr("webbrowser.open", lambda _url: False)

        from mcp.shared.auth import OAuthToken

        from deepagents_code.mcp_auth import login
        from deepagents_code.mcp_oauth_ui import OAuthInteraction

        class _CapturingUI:
            def __init__(self) -> None:
                self.authorize_urls: list[tuple[str, bool]] = []

            async def show_authorize_url(
                self, url: str, *, opened_in_browser: bool
            ) -> None:
                self.authorize_urls.append((url, opened_in_browser))

            async def request_callback_url(self) -> str:
                msg = "not expected in this test"
                raise AssertionError(msg)

            async def show_device_code(
                self, *, verification_uri: str, user_code: str, expires_in: int
            ) -> None: ...

            async def prompt_slack_team_id(self) -> str | None:
                return "T01234567"

            async def show_success(self, message: str) -> None: ...

            async def show_notice(self, message: str) -> None: ...

            async def show_error(self, message: str) -> None: ...

        # Structural check: all required protocol methods are present.
        protocol_methods = [
            "show_authorize_url",
            "request_callback_url",
            "show_device_code",
            "prompt_slack_team_id",
            "show_success",
            "show_notice",
            "show_error",
        ]
        ui_instance = _CapturingUI()
        assert all(callable(getattr(ui_instance, m, None)) for m in protocol_methods)

        ui = _CapturingUI()

        async def _fake_handshake(transport: StreamableHttpTransport) -> None:
            server_name, connection = "notion", transport
            provider = connection.auth
            assert isinstance(provider, OAuthClientProvider)
            redirect = provider.context.redirect_handler
            assert redirect is not None
            await redirect("https://slack.com/oauth/v2/authorize?client_id=x")
            storage = FileTokenStorage(server_name, server_url=connection.url)
            await storage.set_tokens(OAuthToken(access_token="t", token_type="Bearer"))

        with patch("deepagents_code.mcp_auth._drive_handshake", _fake_handshake):
            await login(
                server_name="slack",
                server_config={
                    "type": "http",
                    "url": "https://slack.com/mcp",
                    "auth": "oauth",
                },
                ui=ui,
            )

        assert ui.authorize_urls, "authorize URL must be shown"
        shown_url, _opened = ui.authorize_urls[0]
        assert "team=T01234567" in shown_url

    async def test_slack_preseed_is_idempotent(self) -> None:
        """Preseeding Slack client info a second time reads rather than writes."""
        from deepagents_code.mcp_providers.slack import (
            _SLACK_MCP_CLIENT_ID,
            _preseed_slack_client_info,
        )

        storage = FileTokenStorage(
            "slack",
            server_url="https://slack.com/mcp",
        )
        await _preseed_slack_client_info(storage)
        first = await storage.get_client_info()
        assert first is not None
        first_mtime = storage.path.stat().st_mtime_ns

        # Calling a second time must not rewrite the token file.
        await _preseed_slack_client_info(storage)
        second = await storage.get_client_info()
        assert second is not None
        assert second.client_id == _SLACK_MCP_CLIENT_ID
        assert storage.path.stat().st_mtime_ns == first_mtime


@pytest.mark.usefixtures("fake_home", "no_sleep")
class TestDeviceFlowTimeout:
    """Timeout-path coverage for `_run_device_flow`."""

    async def test_device_flow_times_out_when_pending_forever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The device-code deadline expires when polling never resolves."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(request: httpx.Request) -> httpx.Response:
            if "device" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "device_code": "d",
                        "user_code": "U-1",
                        "verification_uri": "https://example/d",
                        # expires_in=0 means the deadline fires on the
                        # first loop iteration after sleep returns.
                        "expires_in": 0,
                        "interval": 0,
                    },
                )
            return httpx.Response(200, json={"error": "authorization_pending"})

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="Device flow timed out"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )

    async def test_device_code_response_missing_required_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider response missing `verification_uri` surfaces as RuntimeError."""
        import httpx

        from deepagents_code.mcp_auth import _run_device_flow

        def _handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"device_code": "d", "user_code": "U", "expires_in": 30},
            )

        transport = httpx.MockTransport(_handler)
        real_client = httpx.AsyncClient

        def _patched(**kw: Any) -> httpx.AsyncClient:
            kw.pop("transport", None)
            return real_client(transport=transport, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", _patched)

        with pytest.raises(RuntimeError, match="missing required fields"):
            await _run_device_flow(
                device_code_url="https://example/device",
                token_url="https://example/token",
                client_id="cid",
            )


@pytest.mark.usefixtures("fake_home")
class TestFileTokenStorageWriteFailures:
    """Partial-write failure cleanup for `FileTokenStorage._write`."""

    async def test_replace_failure_removes_tmp_and_leaves_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup when `tmp.replace` fails mid-write.

        The `.tmp` file must be unlinked and any existing primary token
        file must remain untouched so a failed write never clobbers
        existing credentials.
        """
        storage = FileTokenStorage("acme")
        await storage.set_client_info(_make_client_info())
        original_bytes = storage.path.read_bytes()

        real_replace = Path.replace

        def _failing_replace(self: Path, target: Path | str) -> None:
            if self.suffix == ".tmp":
                msg = "simulated"
                raise OSError(msg)
            real_replace(self, target)

        monkeypatch.setattr(Path, "replace", _failing_replace)

        with pytest.raises(OSError, match="simulated"):
            await storage.set_tokens(_make_tokens("new"))

        tmp = storage.path.with_suffix(storage.path.suffix + ".tmp")
        assert not tmp.exists(), ".tmp must be cleaned up after replace failure"
        assert storage.path.read_bytes() == original_bytes, (
            "primary token file must not be clobbered when write fails"
        )
