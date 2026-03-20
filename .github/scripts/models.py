"""Unified model registry for eval and harbor GitHub Actions workflows.

Single source of truth for all model definitions. Each model is declared once
with tags encoding workflow and group membership.

Usage:
    python .github/scripts/models.py eval    # reads EVAL_MODELS env var
    python .github/scripts/models.py harbor  # reads HARBOR_MODELS env var

Env var values: a preset name (e.g. "all", "set0", "anthropic"), or
comma-separated "provider:model" specs.
"""

from __future__ import annotations

import json
import os
import sys
from typing import NamedTuple


class Model(NamedTuple):
    """A model spec with group tags."""

    spec: str
    groups: frozenset[str]


# ---------------------------------------------------------------------------
# Registry — canonical order determines output order within each preset.
# Tags follow the convention {workflow}:{group}.
# ---------------------------------------------------------------------------
REGISTRY: tuple[Model, ...] = (
    # -- Anthropic --
    Model(
        "anthropic:claude-haiku-4-5-20251001",
        frozenset({"eval:set0", "eval:set1"}),
    ),
    Model(
        "anthropic:claude-sonnet-4-20250514",
        frozenset({"eval:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-sonnet-4-5-20250929",
        frozenset({"eval:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-sonnet-4-6",
        frozenset({"eval:set0", "eval:set1", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-opus-4-1",
        frozenset({"eval:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-opus-4-5-20251101",
        frozenset({"eval:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-opus-4-6",
        frozenset({"eval:set0", "eval:set1", "harbor:anthropic"}),
    ),
    # -- OpenAI --
    Model("openai:gpt-4o", frozenset({"eval:set0"})),
    Model("openai:gpt-4o-mini", frozenset({"eval:set0"})),
    Model(
        "openai:gpt-4.1",
        frozenset({"eval:set0", "eval:set1", "harbor:openai"}),
    ),
    Model("openai:o3", frozenset({"eval:set0", "harbor:openai"})),
    Model("openai:o4-mini", frozenset({"eval:set0", "harbor:openai"})),
    Model("openai:gpt-5.1-codex", frozenset({"eval:set0"})),
    Model("openai:gpt-5.2-codex", frozenset({"eval:set0", "eval:set1"})),
    Model(
        "openai:gpt-5.4",
        frozenset({"eval:set0", "eval:set1", "harbor:openai"}),
    ),
    # -- Google --
    Model("google_genai:gemini-2.5-flash", frozenset({"eval:set0"})),
    Model("google_genai:gemini-2.5-pro", frozenset({"eval:set0", "eval:set1"})),
    Model("google_genai:gemini-3-flash-preview", frozenset({"eval:set0"})),
    Model(
        "google_genai:gemini-3.1-pro-preview",
        frozenset({"eval:set0", "eval:set1"}),
    ),
    # -- OpenRouter --
    Model(
        "openrouter:minimax/minimax-m2.7",
        frozenset({"eval:set0", "eval:open"}),
    ),
    # -- Baseten --
    Model(
        "baseten:zai-org/GLM-5",
        frozenset({"eval:set0", "eval:set1", "eval:open", "harbor:baseten"}),
    ),
    Model(
        "baseten:MiniMaxAI/MiniMax-M2.5",
        frozenset({"eval:set0", "eval:set1", "harbor:baseten"}),
    ),
    Model(
        "baseten:moonshotai/Kimi-K2.5",
        frozenset({"eval:set0", "harbor:baseten"}),
    ),
    Model(
        "baseten:deepseek-ai/DeepSeek-V3.2",
        frozenset({"eval:set0", "harbor:baseten"}),
    ),
    Model(
        "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct",
        frozenset({"eval:set0", "harbor:baseten"}),
    ),
    # -- Fireworks --
    Model(
        "fireworks:fireworks/qwen3-vl-235b-a22b-thinking",
        frozenset({"eval:set0", "eval:set1"}),
    ),
    Model("fireworks:fireworks/deepseek-v3-0324", frozenset({"eval:set0"})),
    Model("fireworks:fireworks/minimax-m2p1", frozenset({"eval:set0"})),
    Model("fireworks:fireworks/kimi-k2p5", frozenset({"eval:set0"})),
    Model("fireworks:fireworks/glm-5", frozenset({"eval:set0"})),
    Model("fireworks:fireworks/minimax-m2p5", frozenset({"eval:set0"})),
    # -- Ollama (SET1 + SET2) --
    Model("ollama:glm-5", frozenset({"eval:set1", "eval:set2"})),
    Model("ollama:minimax-m2.5", frozenset({"eval:set1", "eval:set2"})),
    Model("ollama:qwen3.5:397b-cloud", frozenset({"eval:set1", "eval:set2"})),
    # -- Groq (SET2) --
    Model("groq:openai/gpt-oss-120b", frozenset({"eval:set2"})),
    Model("groq:qwen/qwen3-32b", frozenset({"eval:set2"})),
    Model("groq:moonshotai/kimi-k2-instruct", frozenset({"eval:set2"})),
    # -- xAI (SET2) --
    Model("xai:grok-4", frozenset({"eval:set2"})),
    Model("xai:grok-3-mini-fast", frozenset({"eval:set2"})),
    # -- Ollama (SET2 only) --
    Model("ollama:nemotron-3-nano:30b", frozenset({"eval:set2"})),
    Model("ollama:cogito-2.1:671b", frozenset({"eval:set2"})),
    Model("ollama:devstral-2:123b", frozenset({"eval:set2"})),
    Model("ollama:ministral-3:14b", frozenset({"eval:set2"})),
    Model("ollama:qwen3-next:80b", frozenset({"eval:set2"})),
    Model("ollama:qwen3-coder:480b-cloud", frozenset({"eval:set2"})),
    Model("ollama:deepseek-v3.2:cloud", frozenset({"eval:set2"})),
    # -- NVIDIA (OPEN) --
    Model(
        "nvidia:nvidia/nemotron-3-super-120b-a12b",
        frozenset({"eval:open"}),
    ),
)

# ---------------------------------------------------------------------------
# Preset definitions — map preset names to tag filters per workflow.
# None means "any tag with the workflow prefix" (i.e. the "all" preset).
# ---------------------------------------------------------------------------
_EVAL_PRESETS: dict[str, str | None] = {
    "all": None,
    "set0": "eval:set0",
    "set1": "eval:set1",
    "set2": "eval:set2",
    "open": "eval:open",
}

_HARBOR_PRESETS: dict[str, str | None] = {
    "all": None,
    "anthropic": "harbor:anthropic",
    "openai": "harbor:openai",
    "baseten": "harbor:baseten",
}

_WORKFLOW_CONFIG: dict[str, tuple[str, dict[str, str | None]]] = {
    "eval": ("EVAL_MODELS", _EVAL_PRESETS),
    "harbor": ("HARBOR_MODELS", _HARBOR_PRESETS),
}


def _filter_by_tag(prefix: str, tag: str | None) -> list[str]:
    """Return model specs matching a tag filter, in REGISTRY order."""
    if tag is not None:
        return [m.spec for m in REGISTRY if tag in m.groups]
    return [
        m.spec for m in REGISTRY if any(g.startswith(prefix) for g in m.groups)
    ]


def _resolve_models(workflow: str, selection: str) -> list[str]:
    """Resolve a selection string to a list of model specs.

    Args:
        workflow: "eval" or "harbor".
        selection: A preset name, or comma-separated "provider:model" specs.

    Returns:
        Ordered list of model spec strings.

    Raises:
        ValueError: If the selection is empty or contains invalid specs.
    """
    env_var, presets = _WORKFLOW_CONFIG[workflow]
    normalized = selection.strip()

    if normalized in presets:
        return _filter_by_tag(f"{workflow}:", presets[normalized])

    specs = [s.strip() for s in normalized.split(",") if s.strip()]
    if not specs:
        msg = f"No models resolved from {env_var} (got empty or whitespace-only input)"
        raise ValueError(msg)
    invalid = [s for s in specs if ":" not in s]
    if invalid:
        msg = f"Invalid model spec(s) (expected 'provider:model'): {', '.join(repr(s) for s in invalid)}"
        raise ValueError(msg)
    return specs


def main() -> None:
    """Entry point — reads workflow arg and env var, writes matrix JSON."""
    if len(sys.argv) != 2 or sys.argv[1] not in _WORKFLOW_CONFIG:  # noqa: PLR2004
        msg = f"Usage: {sys.argv[0]} {{{' | '.join(_WORKFLOW_CONFIG)}}}"
        raise SystemExit(msg)

    workflow = sys.argv[1]
    env_var, _ = _WORKFLOW_CONFIG[workflow]
    selection = os.environ.get(env_var, "all")
    models = _resolve_models(workflow, selection)
    matrix = {"model": models}

    github_output = os.environ.get("GITHUB_OUTPUT")
    line = f"matrix={json.dumps(matrix, separators=(',', ':'))}"
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            f.write(line + "\n")
    else:
        print(line)  # noqa: T201


if __name__ == "__main__":
    main()
