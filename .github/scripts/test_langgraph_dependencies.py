"""Unit tests for the provider-specific LangGraph dependency generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "langgraph_dependencies.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gha_langgraph_dependencies", SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec for {SCRIPT}"
        raise AssertionError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("provider", "provider_dependency"),
    [
        ("anthropic", "langchain-anthropic>=1.4.6,<1.5.0"),
        ("baseten", "langchain-baseten>=0.2.0,<0.3.0"),
        ("fireworks", "langchain-fireworks>=1.4.2,<1.5.0"),
        ("google_genai", "langchain-google-genai>=4.2.4,<4.3.0"),
        ("groq", "langchain-groq>=1.1.3,<1.2.0"),
        ("nvidia", "langchain-nvidia-ai-endpoints>=1.4.1,<1.5.0"),
        ("ollama", "langchain-ollama>=1.1.0,<1.2.0"),
        ("openai", "langchain-openai>=1.3.0,<1.4.0"),
        ("openrouter", "langchain-openrouter>=0.2.3,<0.3.0"),
        ("xai", "langchain-xai>=1.2.2,<1.3.0"),
    ],
)
def test_config_for_provider_includes_only_the_selected_adapter(
    provider: str, provider_dependency: str
) -> None:
    script = _load_script()

    config = script.config_for_provider(provider)

    assert config["dependencies"] == [
        "./.local_deps/deepagents",
        "./.local_deps/deepagents-code",
        "langchain>=1.3.9,<2.0.0",
        provider_dependency,
        "langchain-mcp-adapters>=0.3.0,<0.4.0",
        "aiohttp>=3.14.0,<4.0.0",
        "toml>=0.10.2,<1.0.0",
    ]
    assert config["graphs"] == {
        "deepagent": "./langgraph_agent.py:make_graph",
        "bare_deepagent": "./langgraph_agent.py:make_bare_graph",
        "tau3_deepagent": "./langgraph_agent.py:make_tau3_graph",
    }


def test_config_for_provider_rejects_unknown_provider() -> None:
    script = _load_script()

    with pytest.raises(ValueError, match="Unsupported model provider: unknown"):
        script.config_for_provider("unknown")


def test_write_config_writes_pretty_json(tmp_path: Path) -> None:
    script = _load_script()
    path = tmp_path / "langgraph.json"

    script.write_config("openai", path)

    assert json.loads(path.read_text()) == script.config_for_provider("openai")
    assert path.read_text().endswith("\n")
