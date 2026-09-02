from __future__ import annotations

import logging

import pytest

from deepagents_talon.__main__ import _channel_log_level, _configure_logging


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
