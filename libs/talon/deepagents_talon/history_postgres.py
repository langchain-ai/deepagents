"""Isolate pgvector dimensions and migrations by embedding generation.

This module drives internals of `langgraph-checkpoint-postgres` that are not public
API: `store.VECTOR_MIGRATIONS`, each migration's `condition` and `_replace`, and the
`dict_row` cursor factory behind the cast in `_vector_table`. A rename inside 3.x
would not fail loudly - it would leave the migration rewrite below a silent no-op -
so the `postgres` extra pins that dependency to a single minor version. Widening the
pin means re-checking those four couplings first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from psycopg import AsyncConnection, sql

if TYPE_CHECKING:
    from langgraph.store.postgres.aio import AsyncPostgresStore


async def setup_generation(store: AsyncPostgresStore, generation: str) -> None:
    """Create owned tables before adding the extension schema to the search path."""
    if not isinstance(store.conn, AsyncConnection):
        msg = "History vector generations require a dedicated PostgreSQL connection"
        raise TypeError(msg)
    schema = sql.Identifier("talon_history_" + generation[:40])
    await store.conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
    await store.conn.execute(sql.SQL("SET search_path TO {}").format(schema))
    index = store.index_config
    store.index_config = None
    try:
        await store.setup()
    finally:
        store.index_config = index
    if index is not None:
        await _vector_table(store, schema, index["dims"])
        # The pinned async Store ignores migration conditions, unlike its sync implementation.
        store.VECTOR_MIGRATIONS = [
            migration
            if migration.condition is None or migration.condition(store)
            else migration._replace(sql="SELECT 1", params=None)
            for migration in store.VECTOR_MIGRATIONS
        ]


async def _vector_table(store: AsyncPostgresStore, schema: sql.Identifier, dims: int) -> None:
    if not isinstance(store.conn, AsyncConnection):
        msg = "History vector generations require a dedicated PostgreSQL connection"
        raise TypeError(msg)
    await store.conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    cursor = await store.conn.execute(
        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace "
        "WHERE e.extname='vector'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        msg = "PostgreSQL vector extension is unavailable"
        raise RuntimeError(msg)
    extension = sql.Identifier(cast("dict[str, str]", row)["nspname"])
    await store.conn.execute("CREATE TABLE IF NOT EXISTS vector_migrations (v INTEGER PRIMARY KEY)")
    await store.conn.execute(
        sql.SQL(
            "CREATE TABLE IF NOT EXISTS store_vectors (prefix TEXT NOT NULL, key TEXT NOT NULL, "
            "field_name TEXT NOT NULL, embedding {}.vector({}), "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY (prefix,key,field_name), "
            "FOREIGN KEY (prefix,key) REFERENCES store(prefix,key) ON DELETE CASCADE)"
        ).format(extension, sql.Literal(dims))
    )
    await store.conn.execute(sql.SQL("SET search_path TO {}, {}").format(schema, extension))
