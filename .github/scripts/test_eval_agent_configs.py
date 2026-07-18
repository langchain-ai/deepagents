import os
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_agent_configs as configs  # noqa: E402


def test_parse_code_configs_defaults_dedupes_and_preserves_order() -> None:
    assert configs.parse_code_configs("") == ["bare"]
    assert configs.parse_code_configs("dcode,bare,dcode") == ["dcode", "bare"]


def test_parse_code_configs_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown"):
        configs.parse_code_configs("bare,new_config")


def test_default_conversation_runtime_is_shared() -> None:
    assert configs.conversation_runtime_for("bare") == "tau3"
    assert configs.conversation_runtime_for("dcode") == "tau3"


def test_runtime_packages_are_minimal() -> None:
    assert configs.runtime_config("bare")["packages"] == ("deepagents",)
    assert configs.runtime_config("tau3")["packages"] == ("deepagents",)
    assert configs.runtime_config("dcode")["packages"] == (
        "deepagents",
        "langchain-quickjs",
        "deepagents-code",
    )
    assert configs.required_packages(["tau3", "dcode", "bare"]) == [
        "deepagents",
        "langchain-quickjs",
        "deepagents-code",
    ]


def test_main_prints_runtime_graph_and_label(capsys) -> None:
    assert configs.main(["dcode"]) == 0
    assert capsys.readouterr().out.splitlines() == ["deepagent", "dcode harness"]


def test_registered_runtime_graphs_exist_in_fixed_controller() -> None:
    root = Path(__file__).resolve().parents[2]
    langgraph = json.loads(
        (
            root / "libs/evals/deepagents_harbor/langgraph_project/langgraph.json"
        ).read_text()
    )
    graphs = set(langgraph["graphs"])
    assert {runtime["graph"] for runtime in configs.RUNTIME_CONFIGS.values()} <= graphs
