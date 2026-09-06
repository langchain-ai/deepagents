from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from deepagents_talon import history_adapters
from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.history_adapters import BoundedEmbeddings, open_profile
from deepagents_talon.history_backends import open_history
from deepagents_talon.history_vector_backends import vector_backend
from tests.unit_tests.test_history_vectors import SCOPE, settled

PREFIX = "DEEPAGENTS_TALON_HISTORY_EMBED_"


def configuration(tmp_path, **settings: str):
    env = {
        "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "1",
        PREFIX + "ADAPTER": "openai-compatible",
        PREFIX + "MODEL": "test-model",
        PREFIX + "DIMS": "2",
        PREFIX + "MAX_INPUT_TOKENS": "512",
    }
    env.update({PREFIX + key: value for key, value in settings.items()})
    config = TalonConfig.from_env(env, base_home=tmp_path)
    config.ensure_home()
    return config


@pytest.mark.parametrize(
    "settings",
    [
        {"ADAPTER": "untrusted.module:factory"},
        {"DIMS": "0"},
        {"DIMS": "invalid"},
        {"CONCURRENCY": "100"},
        {"BATCH_SIZE": "10000"},
        {"MAX_INPUT_TOKENS": "64"},
        {"BASE_URL": "http://example.org"},
        {"BASE_URL": "https://key:password@example.org"},
        {"BASE_URL": "https://example.org?key=secret"},
        {"BASE_URL": "https://example.org:invalid"},
        {"ADAPTER": "atlas", "DIMS": "1024", "MAX_INPUT_TOKENS": "512"},
    ],
)
def test_invalid_profile_fails_during_configuration(tmp_path, settings):
    with pytest.raises(TalonConfigError) as error:
        configuration(tmp_path, **settings)
    assert "password" not in str(error.value)
    assert "key=secret" not in str(error.value)


def test_fingerprint_covers_vector_space_but_not_keys_or_throughput(tmp_path):
    config = configuration(tmp_path, API_KEY="test-key")
    profile = config.history_embedding_profile
    assert profile.fingerprint == replace(profile, batch_size=8, concurrency=2).fingerprint
    for change in (
        {"dims": 3},
        {"model": "other"},
        {"query_prompt": "query: "},
        {"max_input_tokens": 1024},
        {"base_url": "https://other.example/v1"},
    ):
        assert profile.fingerprint != replace(profile, **change).fingerprint
    assert "test-key" not in repr(profile)
    assert "test-key" not in repr(config)


async def test_openrouter_sends_text_and_dimensions_without_loading_local_models(
    tmp_path, monkeypatch
):
    requests = []
    sync_client, async_client = httpx.Client, httpx.AsyncClient

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0], "object": "embedding"}
                    for i, _ in enumerate(body["input"])
                ]
            },
        )

    transport = httpx.MockTransport(respond)

    class Client(sync_client):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(transport=transport, **kwargs)

    class AsyncClient(async_client):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    config = configuration(tmp_path, BASE_URL="https://openrouter.ai/api/v1", API_KEY="test")
    async with open_profile(config) as profile:
        assert await profile.embed.aembed_documents(["document"]) == [[1, 0]]
        assert await profile.embed.aembed_query("question") == [1, 0]
    assert [body["input"] for body in requests] == [["document"], ["question"]]
    assert all(body["dimensions"] == 2 for body in requests)


async def test_voyage_preserves_document_and_query_input_types(tmp_path, monkeypatch):
    voyageai = pytest.importorskip("voyageai")

    calls = []

    async def embed(_self, texts, **kwargs: object):
        calls.append((texts, kwargs))
        return SimpleNamespace(embeddings=[[1.0] * 256 for _ in texts])

    monkeypatch.setattr(voyageai.AsyncClient, "embed", embed)
    config = configuration(
        tmp_path, ADAPTER="voyage", MODEL="voyage-3-large", DIMS="256", API_KEY="test"
    )
    async with open_profile(config) as profile:
        await profile.embed.aembed_documents(["document"])
        await profile.embed.aembed_query("question")
    assert [options["input_type"] for _, options in calls] == ["document", "query"]
    assert all(options["truncation"] is False for _, options in calls)
    assert [texts for texts, _ in calls] == [["document"], ["question"]]


class RecordingEmbeddings(Embeddings):
    def __init__(self, dims=2) -> None:
        self.dims = dims
        self.documents = []
        self.queries = []
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def embed_documents(self, _texts):
        pytest.fail("remote inference must use the native async implementation")

    def embed_query(self, _text):
        pytest.fail("remote inference must use the native async implementation")

    async def aembed_documents(self, texts):
        self.documents.extend(texts)
        self.started.set()
        if self.block:
            self.started.set()
            await self.release.wait()
        return [[1.0, *([0.0] * (self.dims - 1))] for _ in texts]

    async def aembed_query(self, text):
        self.queries.append(text)
        return [1.0, *([0.0] * (self.dims - 1))]


def fake_adapter(monkeypatch, embed):
    async def adapter(*_args: object):
        return embed

    monkeypatch.setattr(history_adapters, "_adapter", adapter)


async def test_unicode_documents_are_embedded_completely_with_bounded_inputs(tmp_path):
    raw = RecordingEmbeddings()
    profile = configuration(tmp_path).history_embedding_profile
    bounded = BoundedEmbeddings(raw, profile)
    text = "日本語😀" * 400
    vectors = await bounded.aembed_documents([text])
    assert len(vectors) == 1
    assert "".join(raw.documents) == text
    assert all(len(chunk.encode()) <= 384 for chunk in raw.documents)
    with pytest.raises(ValueError, match="input budget"):
        await bounded.aembed_query(text)
    assert raw.queries == []


async def test_sqlite_remote_search_can_complete_during_indexing(tmp_path, monkeypatch):
    raw = RecordingEmbeddings()
    fake_adapter(monkeypatch, raw)
    config = configuration(tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "first", "time", [HumanMessage("car")])
        await settled(archive)
        raw.started.clear()
        raw.block = True
        await archive.append(SCOPE, "second", "time", [HumanMessage("indexing")])
        try:
            await asyncio.wait_for(raw.started.wait(), 1)
            page = await archive.search_page(SCOPE, query="automobile")
            assert page["semantic_status"] == "completed"
            assert any(hit["text"] == "car" for hit in page["results"])
            assert raw.queries == ["automobile"]
            assert "automobile" not in raw.documents
        finally:
            raw.release.set()


async def test_model_change_requires_explicit_rebuild_and_preserves_transcripts(
    tmp_path, monkeypatch
):
    raw = RecordingEmbeddings()
    fake_adapter(monkeypatch, raw)
    config = configuration(tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        await settled(archive)
        namespace = archive.vectors.namespace("whatsapp", "one")
    changed = configuration(tmp_path, MODEL="other", DIMS="3")
    with pytest.raises(TalonConfigError, match="HISTORY_REINDEX"):
        async with open_history(changed):
            pytest.fail("mixed vector spaces must be refused")
    assert raw.documents == ["retained"]
    raw.dims = 3
    rebuild = replace(changed, env={**changed.env, "DEEPAGENTS_TALON_HISTORY_REINDEX": "1"})
    async with open_history(rebuild) as archive:
        await settled(archive)
        assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["retained"]
        assert (await archive.search_page(SCOPE, query="recall"))["semantic_status"] == "completed"
    async with vector_backend(config, None, config.history_embedding_profile.fingerprint) as old:
        assert await old.asearch(namespace) == []
    before = len(raw.documents)
    async with open_history(rebuild) as archive:
        await settled(archive)
    assert len(raw.documents) == before
    disabled = replace(changed, env={**changed.env, "DEEPAGENTS_TALON_HISTORY_VECTOR_SEARCH": "0"})
    async with open_history(disabled) as archive:
        await archive.delete_session("session")
    async with open_history(changed) as archive:
        assert await archive.entries(SCOPE) == []


async def test_failed_provider_write_remains_pending_until_retry(tmp_path, monkeypatch):
    raw = RecordingEmbeddings(dims=3)
    fake_adapter(monkeypatch, raw)
    config = configuration(tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        await asyncio.wait_for(raw.started.wait(), 2)
        assert await archive.vectors.archive.pending(SCOPE)
    raw.dims = 2
    async with open_history(config) as archive:
        await settled(archive)
        assert not await archive.vectors.archive.pending(SCOPE)
        assert (await archive.search_page(SCOPE, query="recall"))["results"]


async def test_atlas_profile_never_constructs_a_client_embedding_model(tmp_path):
    pytest.importorskip("langchain_mongodb")
    config = configuration(
        tmp_path, ADAPTER="atlas", MODEL="voyage-3-large", DIMS="1024", MAX_INPUT_TOKENS="32000"
    )
    async with open_profile(config) as profile:
        assert not profile.client_side
        assert type(profile.embed).__name__ == "AutoEmbeddings"
        with pytest.raises(TalonConfigError, match="MongoDB"):
            async with vector_backend(config, profile, profile.fingerprint):
                pytest.fail("SQLite cannot execute Atlas embeddings")


async def test_rebuild_failure_retains_transcripts_and_old_fingerprint(tmp_path, monkeypatch):
    from deepagents_talon import history_vector_backends  # noqa: PLC0415

    raw = RecordingEmbeddings()
    fake_adapter(monkeypatch, raw)
    config = configuration(tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "session", "time", [HumanMessage("retained")])
        await settled(archive)
    original = history_vector_backends._erase_vectors

    async def interrupted(archive, store):
        await original(archive, store)
        msg = "interrupted before fingerprint publication"
        raise RuntimeError(msg)

    changed = configuration(tmp_path, MODEL="other")
    rebuild = replace(changed, env={**changed.env, "DEEPAGENTS_TALON_HISTORY_REINDEX": "1"})
    monkeypatch.setattr(history_vector_backends, "_erase_vectors", interrupted)
    with pytest.raises(TalonConfigError):
        async with open_history(rebuild):
            pytest.fail("interrupted rebuild must not expose a mixed index")
    with pytest.raises(TalonConfigError, match="HISTORY_REINDEX"):
        async with open_history(changed):
            pytest.fail("unfinished rebuild must remain explicit")
    with pytest.raises(TalonConfigError, match="HISTORY_REINDEX"):
        async with open_history(config):
            pytest.fail("the old profile cannot reuse partially erased vectors")
    monkeypatch.setattr(history_vector_backends, "_erase_vectors", original)
    async with open_history(rebuild) as archive:
        await settled(archive)
        assert [entry["text"] for entry in await archive.entries(SCOPE)] == ["retained"]
        assert (await archive.search_page(SCOPE, query="recall"))["results"]


def test_vector_generation_cannot_escape_assistant_home(tmp_path):
    config = configuration(tmp_path)
    path = config.home / f"history-vectors-{config.history_embedding_profile.fingerprint}.sqlite"
    path.symlink_to(tmp_path / "outside.sqlite")
    with pytest.raises(TalonConfigError, match="inside the assistant home"):
        config.history_generation_path(config.history_embedding_profile.fingerprint)


async def test_atlas_backend_stores_text_without_calling_embeddings(tmp_path, monkeypatch):
    from langgraph.store.memory import InMemoryStore  # noqa: PLC0415

    from deepagents_talon import history_backends  # noqa: PLC0415

    pytest.importorskip("langchain_mongodb")
    captured = {}

    class Client:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_default_database(self):
            return {captured["collection"]: InMemoryStore()}

        def close(self):
            pass

    def make_store(target, **kwargs: object):
        captured.update(kwargs)
        return target

    def driver(module, _extra):
        return (
            SimpleNamespace(MongoClient=Client)
            if module == "pymongo"
            else SimpleNamespace(
                MongoDBStore=make_store, create_vector_index_config=lambda **kwargs: kwargs
            )
        )

    config = configuration(
        tmp_path, ADAPTER="atlas", MODEL="voyage-3-large", DIMS="1024", MAX_INPUT_TOKENS="32000"
    )
    config = replace(
        config, env={**config.env, "DEEPAGENTS_TALON_HISTORY_URI": "mongodb://localhost/test"}
    )
    captured["collection"] = "talon_history_vectors_" + config.history_embedding_profile.fingerprint
    monkeypatch.setattr(history_backends, "_driver", driver)
    async with (
        open_profile(config) as profile,
        vector_backend(config, profile, profile.fingerprint) as store,
    ):
        await store.aput(("test",), "document", {"text": "retained"})
        assert (await store.aget(("test",), "document")).value == {"text": "retained"}
    index = captured["index_config"]
    assert index["dims"] == -1
    assert index["relevance_score_fn"] is None
    assert type(index["embed"]).__name__ == "AutoEmbeddings"


async def test_rate_limit_backoff_survives_new_appends(tmp_path, monkeypatch):
    from deepagents_talon import history_vectors  # noqa: PLC0415

    class RateLimited(RecordingEmbeddings):
        attempts = 0
        failed = asyncio.Event()
        succeeded = asyncio.Event()

        async def aembed_documents(self, texts):
            self.attempts += 1
            if self.attempts == 1:
                self.failed.set()
                response = httpx.Response(429, headers={"retry-after": "0.1"})
                msg = "rate limited"
                raise httpx.HTTPStatusError(
                    msg,
                    request=httpx.Request("POST", "https://test.invalid"),
                    response=response,
                )
            self.succeeded.set()
            return await super().aembed_documents(texts)

    raw = RateLimited()
    fake_adapter(monkeypatch, raw)
    monkeypatch.setattr(history_vectors, "_RETRY_SECONDS", 0.001)
    monkeypatch.setattr(history_vectors.secrets, "randbelow", lambda _bound: 0)
    config = configuration(tmp_path)
    async with open_history(config) as archive:
        await archive.append(SCOPE, "first", "time", [HumanMessage("first")])
        await asyncio.wait_for(raw.failed.wait(), 1)
        for i in range(3):
            await archive.append(SCOPE, f"later-{i}", "time", [HumanMessage("later")])
        assert not raw.succeeded.is_set()
        assert await archive.vectors.archive.pending(SCOPE)
        await asyncio.wait_for(raw.succeeded.wait(), 1)
        await settled(archive)
        assert not await archive.vectors.archive.pending(SCOPE)


async def test_threaded_store_uses_native_async_embeddings(tmp_path):
    from langgraph.store.memory import InMemoryStore  # noqa: PLC0415

    from deepagents_talon.history_prepared_store import PreparedVectorStore  # noqa: PLC0415

    class ThreadedStore(InMemoryStore):
        async def abatch(self, ops):
            return await asyncio.to_thread(self.batch, list(ops))

    raw = RecordingEmbeddings()
    profile = configuration(tmp_path).history_embedding_profile
    embed = BoundedEmbeddings(raw, profile)
    store = PreparedVectorStore(
        ThreadedStore(index={"dims": 2, "embed": embed, "fields": ["text"]}), embed
    )
    await store.aput(("test",), "one", {"text": "document"})
    result = await store.asearch(("test",), query="question")
    assert result[0].key == "one"
    assert raw.documents == ["document"]
    assert set(raw.queries) == {"question"}


async def test_vector_plugin_receives_generation_and_deletion_mode(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager  # noqa: PLC0415

    from langgraph.store.memory import InMemoryStore  # noqa: PLC0415

    from deepagents_talon import history_vector_backends  # noqa: PLC0415

    calls, closed = [], []

    @asynccontextmanager
    async def factory(uri, *, index, generation):
        calls.append((uri, index, generation))
        try:
            yield InMemoryStore(index=index)
        finally:
            closed.append(True)

    monkeypatch.setattr(
        history_vector_backends,
        "entry_points",
        lambda **_kwargs: [SimpleNamespace(load=lambda: factory)],
    )
    raw = RecordingEmbeddings()
    fake_adapter(monkeypatch, raw)
    config = configuration(tmp_path)
    config = replace(
        config, env={**config.env, "DEEPAGENTS_TALON_HISTORY_URI": "mysql://localhost/test"}
    )
    async with (
        open_profile(config) as profile,
        vector_backend(config, profile, profile.fingerprint) as store,
    ):
        await store.aput(("test",), "one", {"text": "document"})
        assert (await store.asearch(("test",), query="question"))[0].key == "one"
    async with vector_backend(config, None, profile.fingerprint) as store:
        await store.adelete(("test",), "one")
    assert calls[0][0] == "mysql://localhost/test"
    assert calls[0][2] == calls[1][2] == profile.fingerprint
    assert calls[1][1] is None
    assert closed == [True, True]
