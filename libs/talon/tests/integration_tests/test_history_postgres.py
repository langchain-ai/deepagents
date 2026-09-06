"""Optional PostgreSQL metadata contract test using only a local disposable database."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

from tests.store_archive_contract import assert_store_archive_contract

pytestmark = [
    pytest.mark.skipif(os.environ.get("TALON_TEST_POSTGRES") != "1", reason="PostgreSQL opt-in"),
    pytest.mark.timeout(90),
]


async def test_postgres_archive_contract(tmp_path):
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
            async with postgres.AsyncPostgresStore.from_conn_string(
                connection + " dbname=" + database
            ) as metadata:
                try:
                    await metadata.setup()
                    await assert_store_archive_contract(
                        metadata,
                        tmp_path,
                        history_uri=f"postgresql://postgres@127.0.0.1:{port}/{database}?connect_timeout=5",
                    )
                finally:
                    if (task := getattr(metadata, "_task", None)) is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
        finally:
            await admin.execute(
                psycopg.sql.SQL("DROP DATABASE {}").format(psycopg.sql.Identifier(database))
            )
