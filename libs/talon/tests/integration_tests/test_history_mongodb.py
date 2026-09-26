"""Optional real Atlas Local contract test; see README for local setup."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from deepagents_talon import history_adapters
from tests.store_archive_contract import StaticEmbeddings, assert_store_archive_contract

pytestmark = [
    pytest.mark.skipif(os.environ.get("TALON_TEST_MONGODB") != "1", reason="Atlas Local opt-in"),
    pytest.mark.timeout(90),
]


async def test_mongodb_history_contract(tmp_path, monkeypatch):
    embeddings = StaticEmbeddings()

    async def adapter(*_args: object):
        return embeddings

    monkeypatch.setattr(history_adapters, "_adapter", adapter)
    mongodb = pytest.importorskip("langgraph.store.mongodb")
    pymongo = pytest.importorskip("pymongo")
    port = int(os.environ.get("TALON_TEST_MONGODB_PORT", "27028"))
    assert 1024 <= port <= 65535
    client = pymongo.MongoClient(
        f"mongodb://127.0.0.1:{port}/?directConnection=true",
        serverSelectionTimeoutMS=5000,
        socketTimeoutMS=5000,
    )
    database = "talon_test_" + uuid4().hex
    try:
        config = mongodb.create_vector_index_config(
            embed=StaticEmbeddings(), dims=2, fields=["text"]
        )
        store = await asyncio.to_thread(
            mongodb.MongoDBStore,
            client[database]["talon_history_vectors"],
            index_config=config,
            auto_index_timeout=60,
        )
        metadata = await asyncio.to_thread(mongodb.MongoDBStore, client[database]["talon_history"])
        await assert_store_archive_contract(
            metadata,
            store,
            tmp_path,
            history_uri=f"mongodb://127.0.0.1:{port}/{database}?directConnection=true",
        )
    finally:
        try:
            await asyncio.to_thread(client.drop_database, database)
        finally:
            client.close()
