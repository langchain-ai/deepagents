"""Output the eval matrix JSON for the GitHub Actions evals workflow.

Prints a single line: matrix={"model":["provider:model-name", ...]}
suitable for appending to $GITHUB_OUTPUT.

Reads the EVAL_MODELS env var to determine which models to include:
  - "all" (default): every model in MODELS
  - "set1": a curated subset of flagship models
  - any other value: treated as a single "provider:model" spec
"""

from __future__ import annotations

import json
import os

MODELS: list[str] = [
    # Anthropic
    "anthropic:claude-haiku-4-5-20251001",
    "anthropic:claude-sonnet-4-20250514",
    "anthropic:claude-sonnet-4-5-20250929",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-1",
    "anthropic:claude-opus-4-5-20251101",
    "anthropic:claude-opus-4-6",
    # OpenAI
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
    "openai:gpt-4.1",
    "openai:o3",
    "openai:o4-mini",
    "openai:gpt-5",
    "openai:gpt-5.1-codex",
    "openai:gpt-5.2-codex",
    # Google
    "google_genai:gemini-2.5-flash",
    "google_genai:gemini-2.5-pro",
    "google_genai:gemini-3-flash-preview",
    "google_genai:gemini-3.1-pro-preview",
    # xAI
    "xai:grok-4",
    "xai:grok-3-mini-fast",
    # Groq
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
    # Ollama Cloud
    "ollama:glm-5",
    "ollama:minimax-m2.5",
    "ollama:nemotron-3-nano:30b",
    "ollama:cogito-2.1:671b",
    "ollama:devstral-2:123b",
    "ollama:ministral-3:14b",
    "ollama:qwen3-next:80b",
    "ollama:qwen3-coder:480b-cloud",
    "ollama:qwen3.5:397b-cloud",
    "ollama:deepseek-v3.2:cloud",
]

SET1: list[str] = [
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-6",
    "openai:gpt-4.1",
    "openai:o3",
    "openai:gpt-5",
    "google_genai:gemini-2.5-pro",
    "xai:grok-4",
]


def _resolve_models(selection: str) -> list[str]:
    """Return the list of models for the given selection string.

    Accepts "all", "set1", a single model spec, or comma-separated model specs.
    """
    selection = selection.strip()
    if selection == "all":
        return MODELS
    if selection == "set1":
        return SET1
    specs = [s.strip() for s in selection.split(",") if s.strip()]
    unknown = [s for s in specs if s not in MODELS]
    if unknown:
        msg = f"Unknown model(s): {', '.join(repr(s) for s in unknown)}"
        raise ValueError(msg)
    return specs


def main() -> None:
    selection = os.environ.get("EVAL_MODELS", "all")
    models = _resolve_models(selection)
    matrix = {"model": models}
    github_output = os.environ.get("GITHUB_OUTPUT")
    line = f"matrix={json.dumps(matrix, separators=(',', ':'))}"
    if github_output:
        with open(github_output, "a") as f:
            f.write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
