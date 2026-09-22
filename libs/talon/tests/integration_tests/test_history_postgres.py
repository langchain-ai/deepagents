"""Optional PostgreSQL/pgvector contract test using only a local disposable database."""

from __future__ import annotations

import asyncio
import hashlib
import os
from uuid import uuid4

import pytest

from deepagents_talon import history_adapters
from deepagents_talon.history_backends import _postgres_store
from tests.store_archive_contract import StaticEmbeddings, assert_store_archive_contract

pytestmark = [
    pytest.mark.skipif(os.environ.get("TALON_TEST_POSTGRES") != "1", reason="PostgreSQL opt-in"),
    pytest.mark.timeout(90),
]


async def test_postgres_archive_contract(tmp_path, monkeypatch):
    embeddings = StaticEmbeddings()

    async def adapter(*_args: object):
        return embeddings

    monkeypatch.setattr(history_adapters, "_adapter", adapter)
    postgres = pytest.importorskip("langgraph.store.postgres.aio")
    psycopg = pytest.importorskip("psycopg")
    port = int(os.environ.get("TALON_TEST_POSTGRES_PORT", "5440"))
    assert 1024 <= port <= 65535
    connection = f"host=127.0.0.1 port={port} user=postgres connect_timeout=5"
    database = "talon_test_" + uuid4().hex
    async with await psycopg.AsyncConnection.connect(connection, autocommit=True) as admin:
        await admin.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(database))
        )
        try:
            async with (
                postgres.AsyncPostgresStore.from_conn_string(
                    connection + " dbname=" + database
                ) as metadata,
                postgres.AsyncPostgresStore.from_conn_string(
                    connection + " dbname=" + database,
                    index={"dims": 2, "embed": StaticEmbeddings(), "fields": ["text"]},
                ) as vectors,
            ):
                try:
                    await metadata.setup()
                    await vectors.setup()
                    await assert_store_archive_contract(
                        metadata,
                        vectors,
                        tmp_path,
                        history_uri=f"postgresql://postgres@127.0.0.1:{port}/{database}?connect_timeout=5",
                    )
                finally:
                    for store in (metadata, vectors):
                        if (task := getattr(store, "_task", None)) is not None:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)

            class WideEmbeddings(StaticEmbeddings):
                def __init__(self, dims: int) -> None:
                    self.dims = dims

                def embed_documents(self, texts):
                    return [[1.0, *([0.0] * (self.dims - 1))] for _ in texts]

            for dims in (3, 4096):
                async with _postgres_store(
                    f"postgresql://postgres@127.0.0.1:{port}/{database}?connect_timeout=5",
                    index={"dims": dims, "embed": WideEmbeddings(dims), "fields": ["text"]},
                    generation=hashlib.sha256(str(dims).encode()).hexdigest(),
                ) as store:
                    await store.aput(("generation-test",), "one", {"text": "retained"})
                    hits = await store.asearch(("generation-test",), query="retained")
                    assert len(hits) == 1
                    assert hits[0].key == "one"
        finally:
            await admin.execute(
                psycopg.sql.SQL("DROP DATABASE {}").format(psycopg.sql.Identifier(database))
            )
