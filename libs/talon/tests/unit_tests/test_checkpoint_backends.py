"""Checkpoint selection and persistence."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon import checkpoint_backends
from deepagents_talon.checkpoint_backends import open_checkpointer
from deepagents_talon.config import TalonConfig, TalonConfigError


@pytest.mark.parametrize("configured", [False, True])
async def test_sqlite_persists(tmp_path, configured):
    path = tmp_path / "custom checkpoints.sqlite"
    env = {"DEEPAGENTS_TALON_CHECKPOINT_URI": path.as_uri()} if configured else {}
    config = TalonConfig(assistant_id="test", home=tmp_path, env=env)
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"value": "retained"}
    async with open_checkpointer(config) as saver:
        saved = await saver.aput(
            {"configurable": {"thread_id": "test", "checkpoint_ns": ""}}, checkpoint, {}, {}
        )
    async with open_checkpointer(config) as saver:
        assert (await saver.aget(saved))["channel_values"] == {"value": "retained"}


async def test_custom_backend_cleanup(tmp_path, monkeypatch):
    closed = []
    saver = InMemorySaver()

    @asynccontextmanager
    async def factory(uri):
        assert uri == "custom://host/database"
        try:
            yield saver
        finally:
            closed.append(True)

    monkeypatch.setattr(
        checkpoint_backends,
        "entry_points",
        lambda **_kwargs: [SimpleNamespace(load=lambda: factory)],
    )
    config = TalonConfig(
        assistant_id="test",
        home=tmp_path,
        env={"DEEPAGENTS_TALON_CHECKPOINT_URI": "custom://host/database"},
    )

    async def run():
        async with open_checkpointer(config) as opened:
            assert opened is saver
            raise RuntimeError

    with pytest.raises(RuntimeError):
        await run()
    assert closed == [True]


@pytest.mark.parametrize(
    "uri", ["unknown://host/db", "sqlite://host/db", "postgresql://host", "mongodb://host"]
)
async def test_invalid_configuration(tmp_path, uri):
    config = TalonConfig(
        assistant_id="test", home=tmp_path, env={"DEEPAGENTS_TALON_CHECKPOINT_URI": uri}
    )
    with pytest.raises(TalonConfigError):
        async with open_checkpointer(config):
            pytest.fail("invalid configuration accepted")


async def test_startup_error_sanitized(tmp_path, monkeypatch):
    @asynccontextmanager
    async def factory(uri):
        raise ValueError(uri)
        yield

    monkeypatch.setattr(
        checkpoint_backends,
        "entry_points",
        lambda **_kwargs: [SimpleNamespace(load=lambda: factory)],
    )
    config = TalonConfig(
        assistant_id="test",
        home=tmp_path,
        env={"DEEPAGENTS_TALON_CHECKPOINT_URI": "custom://secret@host/database"},
    )
    with pytest.raises(TalonConfigError, match="Could not initialize") as error:
        async with open_checkpointer(config):
            pytest.fail("failed startup accepted")
    assert "secret" not in str(error.value)
