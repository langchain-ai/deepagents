from __future__ import annotations

import asyncio
import threading
import traceback
from contextlib import asynccontextmanager
from importlib.metadata import EntryPoint, EntryPoints
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from deepagents_talon import history_backends
from deepagents_talon.archive import ArchiveScope
from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.history_backends import open_history

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

URI_KEY = "DEEPAGENTS_TALON_HISTORY_URI"
SCOPE = ArchiveScope(talon_history_channel="test", talon_history_chat="one")


@pytest.mark.parametrize("scheme", ["mongodb", "mongodb+srv", "postgres", "postgresql", "mysql"])
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
        "history.sqlite",
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


@pytest.mark.parametrize("uri", ["postgresql://host", "mongodb://host/", "sqlite://host/file"])
async def test_backend_validates_its_own_uri(tmp_path, uri):
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    with pytest.raises(TalonConfigError, match=URI_KEY):
        async with open_history(config):
            pytest.fail("invalid backend URI must fail at startup")


@pytest.mark.parametrize("scheme", [None, "sqlite", "file"])
async def test_sqlite_uri_persists(tmp_path, scheme):
    path = tmp_path / "history archive.sqlite"
    uri = path.as_uri().replace("file:", f"{scheme}:", 1) + "?mode=rwc"
    config = TalonConfig.from_env({URI_KEY: uri} if scheme else {}, base_home=tmp_path)
    config.ensure_home()
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
    if scheme is None:
        config = TalonConfig.from_env(
            {URI_KEY: config.checkpoint_path.as_uri()}, base_home=tmp_path
        )
    async with open_history(config) as reopened:
        assert [entry["text"] for entry in await reopened.entries(SCOPE)] == ["retained"]


async def test_sqlite_uri_preserves_connection_options(tmp_path):
    path = tmp_path / "missing.sqlite"
    config = TalonConfig.from_env({URI_KEY: path.as_uri() + "?mode=rw"}, base_home=tmp_path)
    with pytest.raises(TalonConfigError, match="SQLite history"):
        async with open_history(config):
            pytest.fail("mode=rw must not create a missing database")
    assert not path.exists()


def install_plugin(monkeypatch, factory, *, count=1):
    plugin = EntryPoint(
        name="mysql", value="example_history:open_store", group="deepagents_talon.history_backends"
    )
    monkeypatch.setattr(history_backends, "entry_points", EntryPoints([plugin] * count).select)
    monkeypatch.setattr(EntryPoint, "load", lambda _self: factory)


async def test_installed_backend_accepts_custom_uri_and_closes(tmp_path, monkeypatch):
    uri = "mysql://user:example-password@localhost/talon?custom=option"
    store = InMemoryStore()
    closed = []

    @asynccontextmanager
    async def factory(connection):
        assert connection == uri
        try:
            yield store
        finally:
            closed.append(True)

    install_plugin(monkeypatch, factory)
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["retained"]
    assert closed == [True]


async def test_plugin_startup_errors_redact_uri(tmp_path, monkeypatch):
    uri = "mysql://user:example-password@localhost/talon"

    def factory(connection):
        raise RuntimeError(connection)

    install_plugin(monkeypatch, factory)
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    with pytest.raises(TalonConfigError) as error:
        async with open_history(config):
            pytest.fail("startup must fail")
    assert "example-password" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("count", [0, 2])
async def test_unavailable_or_ambiguous_backend_fails_closed(tmp_path, monkeypatch, count):
    install_plugin(monkeypatch, None, count=count)
    config = TalonConfig.from_env({URI_KEY: "mysql://localhost/talon"}, base_home=tmp_path)
    with pytest.raises(TalonConfigError, match="exactly one"):
        async with open_history(config):
            pytest.fail("backend must not fall back to SQLite or PostgreSQL")


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
async def test_env_backend_isolates_assistants_and_closes(tmp_path, monkeypatch, scheme):
    state = fake_backend(monkeypatch)
    config = TalonConfig.from_env({URI_KEY: f"{scheme}://localhost/talon"}, base_home=tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        other = TalonConfig.from_env(
            {**config.env, "DEEPAGENTS_TALON_ASSISTANT_ID": "other"}, base_home=tmp_path
        )
        async with open_history(other) as isolated:
            assert await isolated.entries(SCOPE) == []
    async with open_history(config) as reopened:
        assert [entry["text"] for entry in await reopened.entries(SCOPE)] == ["retained"]
    assert state.closed
    if state.dispatcher is not None:
        assert state.dispatcher.done()
    assert not config.checkpoint_path.exists()


@pytest.mark.parametrize("scheme", ["mongodb", "postgresql"])
@pytest.mark.parametrize("failure", ["setup", "write"])
async def test_startup_errors_are_redacted_and_resources_closed(
    tmp_path, monkeypatch, scheme, failure
):
    state = fake_backend(monkeypatch, failing=failure == "setup")
    uri = f"{scheme}://user:example-password@localhost/talon"
    if failure == "write":

        async def deny_write(*_args: object, **_kwargs: object):
            raise PermissionError(uri)

        monkeypatch.setattr(InMemoryStore, "aput", deny_write)
    config = TalonConfig.from_env({URI_KEY: uri}, base_home=tmp_path)
    with pytest.raises(TalonConfigError) as error:
        async with open_history(config):
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
        async with open_history(config):
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
        async with open_history(config):
            pytest.fail("missing backend must not fall back to SQLite")
