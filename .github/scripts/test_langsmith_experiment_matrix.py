"""Tests for the LangSmith experiment-matrix helper used by Harbor evals."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "langsmith_experiment_matrix.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gha_langsmith_experiments", SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_experiment_ids_creates_one_session_per_model() -> None:
    script = _load_script()
    created: list[tuple[str, dict[str, object]]] = []

    def create_session(name: str, metadata: dict[str, object]) -> str:
        created.append((name, metadata))
        return f"experiment-{len(created)}"

    matrix = {
        "include": [
            {"model": "openai:gpt-5.4", "shard": 0},
            {"model": "openai:gpt-5.4", "shard": 1},
            {"model": "anthropic:claude-opus", "shard": 0},
        ]
    }

    result = script.add_experiment_ids(
        matrix,
        agent_impl="dcode",
        run_id="12345",
        run_attempt="2",
        create_session=create_session,
    )

    assert len(created) == 2
    assert result["include"] == [
        {
            "model": "openai:gpt-5.4",
            "shard": 0,
            "langsmith_experiment_name": "deepagents-harbor-dcode-openai-gpt-5.4-12345-2",
            "langsmith_experiment_id": "experiment-1",
        },
        {
            "model": "openai:gpt-5.4",
            "shard": 1,
            "langsmith_experiment_name": "deepagents-harbor-dcode-openai-gpt-5.4-12345-2",
            "langsmith_experiment_id": "experiment-1",
        },
        {
            "model": "anthropic:claude-opus",
            "shard": 0,
            "langsmith_experiment_name": "deepagents-harbor-dcode-anthropic-claude-opus-12345-2",
            "langsmith_experiment_id": "experiment-2",
        },
    ]
