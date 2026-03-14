"""Output the Harbor eval matrix JSON for the GitHub Actions workflow.

Prints a single line: matrix={"model":["provider:model-name", ...]}
suitable for appending to $GITHUB_OUTPUT.

Reads the HARBOR_MODELS env var to determine which models to include:
  - "all" (default): every supported Harbor workflow model
  - "anthropic": Anthropic-hosted models
  - "openai": OpenAI-hosted models
  - "baseten": Baseten-hosted models
  - any other value: treated as a single "provider:model" spec or comma-separated specs
"""

from __future__ import annotations

import json
import os

ANTHROPIC_MODELS: list[str] = [
    "anthropic:claude-sonnet-4-20250514",
    "anthropic:claude-sonnet-4-5-20250929",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-1",
    "anthropic:claude-opus-4-5-20251101",
    "anthropic:claude-opus-4-6",
]

OPENAI_MODELS: list[str] = [
    "openai:gpt-4.1",
    "openai:o3",
    "openai:o4-mini",
    "openai:gpt-5.4",
]

BASETEN_MODELS: list[str] = [
    "baseten:zai-org/GLM-5",
    "baseten:MiniMaxAI/MiniMax-M2.5",
    "baseten:moonshotai/Kimi-K2.5",
    "baseten:deepseek-ai/DeepSeek-V3.2",
    "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct",
]


def _dedupe(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        if model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _resolve_models(selection: str) -> list[str]:
    """Return the list of models for the given selection string."""
    normalized = selection.strip()
    if normalized == "all":
        return _dedupe(ANTHROPIC_MODELS + OPENAI_MODELS + BASETEN_MODELS)
    if normalized == "anthropic":
        return ANTHROPIC_MODELS
    if normalized == "openai":
        return OPENAI_MODELS
    if normalized == "baseten":
        return BASETEN_MODELS

    specs = [spec.strip() for spec in normalized.split(",") if spec.strip()]
    invalid = [spec for spec in specs if ":" not in spec]
    if invalid:
        msg = f"Invalid model spec(s) (expected 'provider:model'): {', '.join(repr(spec) for spec in invalid)}"
        raise ValueError(msg)
    return specs


def main() -> None:
    selection = os.environ.get("HARBOR_MODELS", "all")
    models = _resolve_models(selection)
    matrix = {"model": models}
    github_output = os.environ.get("GITHUB_OUTPUT")
    line = f"matrix={json.dumps(matrix, separators=(',', ':'))}"
    if github_output:
        with open(github_output, "a") as file_handle:
            file_handle.write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
