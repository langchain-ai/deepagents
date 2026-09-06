from __future__ import annotations

import asyncio
import threading
import traceback
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents_talon import history_backends
from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.history import ArchiveScope
from deepagents_talon.history_store import HistoryStorage
from deepagents_talon.store_archive import StoreConversationArchive

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

URI_KEY = "DEEPAGENTS_TALON_HISTORY_URI"
SCOPE = ArchiveScope(talon_history_channel="test", talon_history_chat="one")


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv", "postgres", "postgresql"])
def test_history_uri_accepts_supported_databases_without_exposing_config(tmp_path, scheme):
    uri = f"{scheme}://user:example-password@database.example/talon"
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    assert config.history_uri == uri
    assert uri not in repr(config)
    assert "example-password" not in repr(config)


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "sqlite:///tmp/history.sqlite",
        "https://example/talon",
        "postgresql://host",
        "mongodb://host/",
        "mongodb://[invalid/talon",
        "mongodb://host/talon#fragment",
        "postgresql://bad host/talon",
    ],
)
def test_invalid_history_uri_fails_without_echoing_input(tmp_path, uri):
    with pytest.raises(TalonConfigError, match=URI_KEY) as error:
        TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    if uri:
        assert uri not in str(error.value)


def postgres_driver(shared, state, failing):
    class Postgres:
        @classmethod
        @asynccontextmanager
        async def from_conn_string(cls, uri) -> AsyncIterator[InMemoryStore]:
            class ReadyStore(InMemoryStore):
                async def setup(self):
                    if failing:
                        raise RuntimeError(uri)

            store = ReadyStore()
            store._data = shared._data
            store._task = asyncio.create_task(asyncio.Event().wait())
            state.dispatcher = store._task
            try:
                yield store
            finally:
                state.closed = True

    return Postgres


def fake_backend(monkeypatch, *, failing=False, started=None, release=None):
    store = InMemoryStore()
    state = SimpleNamespace(closed=False, dispatcher=None)

    class Client:
        def __init__(self, uri, **_kwargs: object) -> None:
            self.uri = uri

        def get_default_database(self):
            return {"talon_history": self.uri}

        def close(self):
            state.closed = True

    def mongo_store(uri):
        if started is not None:
            started.set()
            release.wait(timeout=5)
        if failing:
            raise RuntimeError(uri)
        return store

    def driver(module, _extra):
        if module == "pymongo":
            return SimpleNamespace(MongoClient=Client)
        return SimpleNamespace(
            AsyncPostgresStore=postgres_driver(store, state, failing), MongoDBStore=mongo_store
        )

    monkeypatch.setattr(history_backends, "_driver", driver)
    return state


@pytest.mark.parametrize("scheme", ["mongodb", "postgresql"])
@pytest.mark.parametrize("fail_host", [False, True])
async def test_env_backend_persists_isolates_and_closes(tmp_path, monkeypatch, scheme, fail_host):
    state = fake_backend(monkeypatch)
    config = TalonConfig.from_env({URI_KEY: f"{scheme}://localhost/talon"}, base_home=tmp_path)
    async with HistoryStorage().open(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained", id="message")])
    assert state.closed
    state.closed = False
    with pytest.raises(RuntimeError, match="host failed") if fail_host else nullcontext():
        async with HistoryStorage().open(config) as archive:
            assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["retained"]
            other = TalonConfig.from_env(
                {**config.env, "DEEPAGENTS_TALON_ASSISTANT_ID": "other"}, base_home=tmp_path
            )
            async with HistoryStorage().open(other) as isolated:
                assert await isolated.entries(SCOPE) == []
            await archive.delete_session("session")
            assert await archive.entries(SCOPE) == []
            if fail_host:
                msg = "host failed"
                raise RuntimeError(msg)
    assert state.closed
    if state.dispatcher is not None:
        assert state.dispatcher.done()
    assert not config.checkpoint_path.exists()


@pytest.mark.parametrize("scheme", ["mongodb", "postgresql"])
async def test_startup_errors_are_redacted_and_resources_closed(tmp_path, monkeypatch, scheme):
    state = fake_backend(monkeypatch, failing=True)
    uri = f"{scheme}://user:example-password@localhost/talon"
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    with pytest.raises(TalonConfigError) as error:
        async with HistoryStorage().open(config):
            pytest.fail("startup must fail")
    rendered = "".join(traceback.format_exception(error.value))
    assert uri not in rendered
    assert "example-password" not in rendered
    assert state.closed
    if state.dispatcher is not None:
        assert state.dispatcher.done()


async def test_cancelled_mongo_setup_finishes_before_closing_client(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    state = fake_backend(monkeypatch, started=started, release=release)
    config = TalonConfig.from_env({URI_KEY: "mongodb://localhost/talon"}, base_home=tmp_path)

    async def open_archive():
        async with HistoryStorage().open(config):
            pytest.fail("cancelled startup must not open the archive")

    task = asyncio.create_task(open_archive())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not state.closed
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.closed


@pytest.mark.parametrize(("scheme", "extra"), [("mongodb", "mongodb"), ("postgresql", "postgres")])
async def test_missing_backend_extra_has_install_guidance(tmp_path, monkeypatch, scheme, extra):
    def missing(_module):
        msg = "missing optional driver"
        raise ModuleNotFoundError(msg)

    monkeypatch.setattr(history_backends.importlib, "import_module", missing)
    config = TalonConfig.from_env({URI_KEY: f"{scheme}://localhost/talon"}, base_home=tmp_path)
    with pytest.raises(ImportError, match=f"uv sync --extra {extra}"):
        async with HistoryStorage().open(config):
            pytest.fail("missing backend must not fall back to SQLite")


async def test_explicit_factory_overrides_env_backend(tmp_path):
    config = TalonConfig.from_env({URI_KEY: "mongodb://unused/talon"}, base_home=tmp_path)

    @asynccontextmanager
    async def factory(_config):
        async with StoreConversationArchive(
            InMemoryStore(), namespace=("custom",)
        ).open() as archive:
            yield archive

    async with HistoryStorage(archive_factory=factory).open(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("custom")])
        assert await archive.entries(SCOPE)
    assert not config.checkpoint_path.exists()
