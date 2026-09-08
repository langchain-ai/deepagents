from __future__ import annotations

import argparse
import logging
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deepagents_talon.__main__ import (
    _channel_log_level,
    _configure_logging,
    _run_host,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import CronJobStore


async def test_run_host_uses_configured_checkpointer(tmp_path, monkeypatch) -> None:
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant-1", "AGENT_MODEL": "test:model"},
        base_home=tmp_path,
    )
    cron_store = CronJobStore(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    configured_checkpointer = InMemorySaver()
    captured: dict[str, object] = {}

    async def fake_agent_runtime(_config, cron_store=None, checkpointer=None):
        captured["cron_store"] = cron_store
        captured["checkpointer"] = checkpointer
        return object()

    async def fake_run_host_with_agent(*_args: object) -> None:
        return None

    monkeypatch.setattr("deepagents_talon.__main__._agent_runtime", fake_agent_runtime)
    monkeypatch.setattr("deepagents_talon.__main__._run_host_with_agent", fake_run_host_with_agent)

    await _run_host(
        argparse.Namespace(once=True),
        config,
        cron_store,
        (),
        checkpointer=configured_checkpointer,
    )

    assert captured == {
        "cron_store": cron_store,
        "checkpointer": configured_checkpointer,
    }
    assert not config.checkpoint_path.exists()


async def test_run_host_persists_langgraph_checkpoints(tmp_path, monkeypatch) -> None:
    config = TalonConfig.from_env(
        {"AGENT_ASSISTANT_ID": "assistant-1", "AGENT_MODEL": "test:model"},
        base_home=tmp_path,
    )
    config.ensure_home()
    cron_store = CronJobStore(assistant_id=config.assistant_id, cron_dir=config.cron_dir)
    captured: dict[str, Any] = {}

    async def fake_agent_runtime(_config, cron_store=None, checkpointer=None):
        captured["cron_store"] = cron_store
        captured["checkpointer"] = checkpointer
        return object()

    async def fake_run_host_with_agent(*_args: object) -> None:
        await captured["checkpointer"].aput(
            {"configurable": {"thread_id": "conversation", "checkpoint_ns": ""}},
            {"id": "checkpoint", "ts": "2026-09-04T00:00:00Z", "channel_values": {}},
            {},
            {},
        )

    monkeypatch.setattr("deepagents_talon.__main__._agent_runtime", fake_agent_runtime)
    monkeypatch.setattr("deepagents_talon.__main__._run_host_with_agent", fake_run_host_with_agent)

    await _run_host(argparse.Namespace(once=True), config, cron_store, ())

    assert config.checkpoint_path.is_file()
    async with AsyncSqliteSaver.from_conn_string(str(config.checkpoint_path)) as checkpointer:
        checkpoint = await checkpointer.aget(
            {"configurable": {"thread_id": "conversation", "checkpoint_ns": ""}}
        )
    assert checkpoint is not None
    assert checkpoint["id"] == "checkpoint"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, logging.INFO),
        ({"DEEPAGENTS_CODE_DEBUG": "1"}, logging.DEBUG),
        ({"DEEPAGENTS_CODE_DEBUG": " TrUe "}, logging.DEBUG),
        ({"DEEPAGENTS_CODE_DEBUG": "on"}, logging.DEBUG),
        ({"DEEPAGENTS_CODE_DEBUG": "false"}, logging.INFO),
        ({"DEEPAGENTS_CODE_LOG_LEVEL": "debug"}, logging.DEBUG),
        ({"DEEPAGENTS_CODE_LOG_LEVEL": " WARNING "}, logging.WARNING),
        (
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_LOG_LEVEL": "INFO",
            },
            logging.INFO,
        ),
        (
            {
                "DEEPAGENTS_CODE_DEBUG": "1",
                "DEEPAGENTS_CODE_LOG_LEVEL": "invalid",
            },
            logging.DEBUG,
        ),
        ({"DEEPAGENTS_CODE_LOG_LEVEL": "invalid"}, logging.INFO),
    ],
)
def test_channel_log_level_matches_dcode_environment(
    env: dict[str, str],
    expected: int,
) -> None:
    assert _channel_log_level(env) == expected


def test_configure_logging_enables_only_channel_debug_logs(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))
    channel_logger = logging.getLogger("deepagents_talon.channels")
    runtime_logger = logging.getLogger("deepagents_talon.runtime")
    previous_channel_level = channel_logger.level
    previous_runtime_level = runtime_logger.level

    try:
        _configure_logging({"DEEPAGENTS_CODE_DEBUG": "1"})

        assert channel_logger.level == logging.DEBUG
        assert runtime_logger.level == previous_runtime_level
        assert calls == [
            {
                "level": logging.INFO,
                "format": "%(levelname)s:%(name)s:%(message)s",
            }
        ]
    finally:
        channel_logger.setLevel(previous_channel_level)


def test_channel_log_level_reports_invalid_value_without_echoing_it(caplog) -> None:
    invalid_value = "private-invalid-value"

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.__main__"):
        level = _channel_log_level({"DEEPAGENTS_CODE_LOG_LEVEL": invalid_value})

    assert level == logging.INFO
    assert "DEEPAGENTS_CODE_LOG_LEVEL" in caplog.text
    assert invalid_value not in caplog.text
