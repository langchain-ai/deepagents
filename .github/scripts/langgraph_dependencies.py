"""Generate provider-specific LangGraph dependency configurations for Harbor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROVIDER_DEPENDENCIES = {
    "anthropic": "langchain-anthropic>=1.4.6,<1.5.0",
    "baseten": "langchain-baseten>=0.2.0,<0.3.0",
    "fireworks": "langchain-fireworks>=1.4.2,<1.5.0",
    "google_genai": "langchain-google-genai>=4.2.4,<4.3.0",
    "groq": "langchain-groq>=1.1.3,<1.2.0",
    "nvidia": "langchain-nvidia-ai-endpoints>=1.4.1,<1.5.0",
    "ollama": "langchain-ollama>=1.1.0,<1.2.0",
    "openai": "langchain-openai>=1.3.0,<1.4.0",
    "openrouter": "langchain-openrouter>=0.2.3,<0.3.0",
    "xai": "langchain-xai>=1.2.2,<1.3.0",
}

BASE_DEPENDENCIES = [
    "./.local_deps/deepagents",
    "./.local_deps/deepagents-code",
    "langchain>=1.3.9,<2.0.0",
]

SHARED_DEPENDENCIES = [
    "langchain-mcp-adapters>=0.3.0,<0.4.0",
    "aiohttp>=3.14.0,<4.0.0",
    "toml>=0.10.2,<1.0.0",
]

GRAPHS = {
    "deepagent": "./langgraph_agent.py:make_graph",
    "bare_deepagent": "./langgraph_agent.py:make_bare_graph",
    "tau3_deepagent": "./langgraph_agent.py:make_tau3_graph",
}


def config_for_provider(provider: str) -> dict[str, Any]:
    """Build the LangGraph config required for one allowlisted model provider."""
    try:
        provider_dependency = PROVIDER_DEPENDENCIES[provider]
    except KeyError as exc:
        msg = f"Unsupported model provider: {provider}"
        raise ValueError(msg) from exc
    return {
        "dependencies": [*BASE_DEPENDENCIES, provider_dependency, *SHARED_DEPENDENCIES],
        "graphs": GRAPHS,
    }


def write_config(provider: str, path: Path) -> None:
    """Write a provider-specific LangGraph configuration."""
    path.write_text(json.dumps(config_for_provider(provider), indent=2) + "\n")


def main() -> None:
    """Generate a provider-specific LangGraph configuration from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(PROVIDER_DEPENDENCIES))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_config(args.provider, args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc
