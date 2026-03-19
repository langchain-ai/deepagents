"""Output the eval matrix JSON for the GitHub Actions evals workflow.

Prints a single line: matrix={"model":["provider:model-name", ...]}
suitable for appending to $GITHUB_OUTPUT.

Reads the EVAL_MODELS env var to determine which models to include:
  - "all" (default): every model across all sets (deduplicated)
  - "set0": first-party API providers (Anthropic, OpenAI, Google)
  - "set1": a curated subset of flagship models
  - "set2": slower third-party hosted models
  - "open": open-weight models (Nemotron, GLM-5, MiniMax M2.7)
  - any other value: treated as a single "provider:model" spec
"""

from __future__ import annotations

import json
import os

SET0: list[str] = [
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
    "openai:gpt-5.1-codex",
    "openai:gpt-5.2-codex",
    "openai:gpt-5.4",
    # Google
    "google_genai:gemini-2.5-flash",
    "google_genai:gemini-2.5-pro",
    "google_genai:gemini-3-flash-preview",
    "google_genai:gemini-3.1-pro-preview",
    # OpenRouter
    "openrouter:minimax/minimax-m2.7",
    # Baseten
    "baseten:zai-org/GLM-5",
    "baseten:MiniMaxAI/MiniMax-M2.5",
    "baseten:moonshotai/Kimi-K2.5",
    "baseten:deepseek-ai/DeepSeek-V3.2",
    "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct",
    # Fireworks
    "fireworks:fireworks/qwen3-vl-235b-a22b-thinking",
    "fireworks:fireworks/deepseek-v3-0324",
    "fireworks:fireworks/minimax-m2p1",
    "fireworks:fireworks/kimi-k2p5",
    "fireworks:fireworks/glm-5",
    "fireworks:fireworks/minimax-m2p5",
]

SET1: list[str] = [
    "anthropic:claude-haiku-4-5-20251001",
    "anthropic:claude-sonnet-4-6",
    "anthropic:claude-opus-4-6",
    "openai:gpt-4.1",
    "openai:gpt-5.2-codex",
    "openai:gpt-5.4",
    "google_genai:gemini-3.1-pro-preview",
    "google_genai:gemini-2.5-pro",
    "ollama:glm-5",
    "ollama:minimax-m2.5",
    "ollama:qwen3.5:397b-cloud",
    "baseten:zai-org/GLM-5",
    "baseten:MiniMaxAI/MiniMax-M2.5",
    "fireworks:fireworks/qwen3-vl-235b-a22b-thinking",
]

# These are a bit slower
SET2: list[str] = [
    # Groq
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
    # xAI
    "xai:grok-4",
    "xai:grok-3-mini-fast",
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

# Open-weight models
OPEN: list[str] = [
    "nvidia:nvidia/nemotron-3-super-120b-a12b",
    "baseten:zai-org/GLM-5",
    "openrouter:minimax/minimax-m2.7",
]


def _resolve_models(selection: str) -> list[str]:
    """Return the list of models for the given selection string.

    Accepts "all", "set0", "set1", "set2", "open", a single model spec,
    or comma-separated model specs.
    """
    selection = selection.strip()
    if selection == "all":
        seen: set[str] = set()
        result: list[str] = []
        for model in SET0 + SET1 + SET2 + OPEN:
            if model not in seen:
                seen.add(model)
                result.append(model)
        return result
    if selection == "set0":
        return SET0
    if selection == "set1":
        return SET1
    if selection == "set2":
        return SET2
    if selection == "open":
        return OPEN
    specs = [s.strip() for s in selection.split(",") if s.strip()]
    invalid = [s for s in specs if ":" not in s]
    if invalid:
        msg = f"Invalid model spec(s) (expected 'provider:model'): {', '.join(repr(s) for s in invalid)}"
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
