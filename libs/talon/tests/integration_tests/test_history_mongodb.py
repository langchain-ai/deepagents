"""Optional MongoDB metadata contract test; see README for local setup."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from tests.store_archive_contract import assert_store_archive_contract

pytestmark = [
    pytest.mark.skipif(os.environ.get("TALON_TEST_MONGODB") != "1", reason="MongoDB opt-in"),
    pytest.mark.timeout(90),
]


async def test_mongodb_history_contract(tmp_path):
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
        metadata = await asyncio.to_thread(mongodb.MongoDBStore, client[database]["talon_history"])
        await assert_store_archive_contract(
            metadata,
            tmp_path,
            history_uri=f"mongodb://127.0.0.1:{port}/{database}?directConnection=true",
        )
    finally:
        try:
            await asyncio.to_thread(client.drop_database, database)
        finally:
            client.close()
