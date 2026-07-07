from __future__ import annotations

import logging

from deepagents_talon.async_subagents import load_async_subagents


def test_load_async_subagents_reads_deepagents_config(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[async_subagents.researcher]
description = "Research agent"
graph_id = "agent"
url = "https://deployment.example"

[async_subagents.reviewer]
description = "Review agent"
graph_id = "review"

[async_subagents.reviewer.headers]
Authorization = "Bearer test"
""".strip(),
        encoding="utf-8",
    )

    agents = load_async_subagents(config_path)

    assert agents == [
        {
            "name": "researcher",
            "description": "Research agent",
            "graph_id": "agent",
            "url": "https://deployment.example",
        },
        {
            "name": "reviewer",
            "description": "Review agent",
            "graph_id": "review",
            "headers": {"Authorization": "Bearer test"},
        },
    ]


def test_load_async_subagents_skips_invalid_specs(tmp_path, caplog) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[async_subagents.valid]
description = "Valid agent"
graph_id = "agent"

[async_subagents.missing]
description = "Missing graph"

[async_subagents.wrong_type]
description = 123
graph_id = "agent"
""".strip(),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        agents = load_async_subagents(config_path)

    assert agents == [{"name": "valid", "description": "Valid agent", "graph_id": "agent"}]
    assert "missing fields" in caplog.text
    assert "description and graph_id must be strings" in caplog.text


def test_load_async_subagents_returns_empty_for_absent_config(tmp_path) -> None:
    assert load_async_subagents(tmp_path / "missing.toml") == []
