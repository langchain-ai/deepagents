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
import re
import sys
from typing import NamedTuple

_SAFE_SPEC_RE = re.compile(r"^[a-zA-Z0-9:_\-./]+$")
"""Allowed characters in model specs: alphanumeric, colon, hyphen, underscore,
dot, slash.

Rejects shell metacharacters ($, `, ;, |, &, (, ), etc.).
"""


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
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:anthropic",
                "harbor:set0",
                "harbor:set1",
                "harbor:anthropic",
            }
        ),
    ),
    Model(
        "anthropic:claude-sonnet-4-5-20250929",
        frozenset({"eval:set0", "eval:anthropic", "harbor:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-sonnet-4-6",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:fast",
                "eval:anthropic",
                "harbor:set0",
                "harbor:set1",
                "harbor:fast",
                "harbor:anthropic",
            }
        ),
    ),
    Model(
        "anthropic:claude-opus-4-1",
        frozenset({"eval:set0", "eval:anthropic", "harbor:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-opus-4-5-20251101",
        frozenset({"eval:set0", "eval:anthropic", "harbor:set0", "harbor:anthropic"}),
    ),
    Model(
        "anthropic:claude-opus-4-6",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:anthropic",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:anthropic",
            }
        ),
    ),
    # -- Baseten --
    Model(
        "baseten:zai-org/GLM-5",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:open",
                "eval:baseten",
                "harbor:set0",
                "harbor:set1",
                "harbor:open",
                "harbor:baseten",
            }
        ),
    ),
    Model(
        "baseten:MiniMaxAI/MiniMax-M2.5",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:baseten",
                "harbor:set0",
                "harbor:set1",
                "harbor:baseten",
            }
        ),
    ),
    Model(
        "baseten:moonshotai/Kimi-K2.5",
        frozenset({"eval:set0", "eval:baseten", "harbor:set0", "harbor:baseten"}),
    ),
    Model(
        "baseten:nvidia/Nemotron-120B-A12B",
        frozenset(
            {
                "eval:set0",
                "eval:baseten",
                "harbor:set0",
                "harbor:baseten",
            }
        ),
    ),
    Model(
        "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct",
        frozenset({"eval:set0", "eval:baseten", "harbor:set0", "harbor:baseten"}),
    ),
    # -- Fireworks --
    Model(
        "fireworks:fireworks/deepseek-v3p2",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    Model(
        "fireworks:fireworks/deepseek-v3-0324",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    Model(
        "fireworks:fireworks/qwen3-vl-235b-a22b-thinking",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:fireworks",
                "harbor:set0",
                "harbor:set1",
                "harbor:fireworks",
            }
        ),
    ),
    Model(
        "fireworks:fireworks/minimax-m2p1",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    Model(
        "fireworks:fireworks/kimi-k2p5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    Model(
        "fireworks:fireworks/glm-5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    Model(
        "fireworks:fireworks/minimax-m2p5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
    ),
    # -- Google --
    Model(
        "google_genai:gemini-2.5-flash",
        frozenset(
            {"eval:set0", "eval:google_genai", "harbor:set0", "harbor:google_genai"}
        ),
    ),
    Model(
        "google_genai:gemini-2.5-pro",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:google_genai",
                "harbor:set0",
                "harbor:set1",
                "harbor:google_genai",
            }
        ),
    ),
    Model(
        "google_genai:gemini-3-flash-preview",
        frozenset(
            {
                "eval:set0",
                "eval:fast",
                "eval:google_genai",
                "harbor:set0",
                "harbor:fast",
                "harbor:google_genai",
            }
        ),
    ),
    Model(
        "google_genai:gemini-3.1-pro-preview",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:google_genai",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:google_genai",
            }
        ),
    ),
    # -- Groq --
    Model(
        "groq:openai/gpt-oss-120b",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
    ),
    Model(
        "groq:qwen/qwen3-32b",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
    ),
    Model(
        "groq:moonshotai/kimi-k2-instruct",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
    ),
    # -- NVIDIA --
    Model(
        "nvidia:nvidia/nemotron-3-super-120b-a12b",
        frozenset({"eval:open", "eval:nvidia", "harbor:open", "harbor:nvidia"}),
    ),
    # -- Ollama --
    Model(
        "ollama:glm-5",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:glm-5.1",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:minimax-m2.5",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:minimax-m2.7:cloud",
        frozenset(
            {
                "eval:set0",
                "eval:open",
                "eval:ollama",
                "harbor:set0",
                "harbor:open",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:qwen3.5:397b-cloud",
        frozenset(
            {
                "eval:set1",
                "eval:set2",
                "eval:ollama",
                "harbor:set1",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:nemotron-3-nano:30b",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:nemotron-3-super",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
    ),
    Model(
        "ollama:cogito-2.1:671b",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:devstral-2:123b",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:ministral-3:14b",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:qwen3-next:80b",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:qwen3-coder:480b-cloud",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    Model(
        "ollama:deepseek-v3.2:cloud",
        frozenset({"eval:set2", "eval:ollama", "harbor:set2", "harbor:ollama"}),
    ),
    # -- OpenAI --
    Model(
        "openai:gpt-4o",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
    ),
    Model(
        "openai:gpt-4o-mini",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
    ),
    Model(
        "openai:gpt-4.1",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:openai",
            }
        ),
    ),
    Model(
        "openai:o3",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
    ),
    Model(
        "openai:o4-mini",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
    ),
    Model(
        "openai:gpt-5.1-codex",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
    ),
    Model(
        "openai:gpt-5.2-codex",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:openai",
            }
        ),
    ),
    Model(
        "openai:gpt-5.3-codex",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:openai",
            }
        ),
    ),
    Model(
        "openai:gpt-5.4",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:openai",
            }
        ),
    ),
    Model(
        "openai:gpt-5.5",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:openai",
            }
        ),
    ),
    Model(
        "openai:gpt-5.4-mini",
        frozenset(
            {
                "eval:set0",
                "eval:fast",
                "eval:openai",
                "harbor:set0",
                "harbor:fast",
                "harbor:openai",
            }
        ),
    ),
    # -- OpenRouter --
    Model(
        "openrouter:minimax/minimax-m2.7",
        frozenset(
            {
                "eval:openrouter",
                "harbor:openrouter",
            }
        ),
    ),
    Model(
        "openrouter:z-ai/glm-5.1",
        frozenset(
            {
                "eval:open",
                "eval:openrouter",
                "harbor:open",
                "harbor:openrouter",
            }
        ),
    ),
    Model(
        "openrouter:nvidia/nemotron-3-super-120b-a12b",
        frozenset(
            {
                "eval:openrouter",
                "harbor:openrouter",
            }
        ),
    ),
    # -- xAI --
    Model(
        "xai:grok-4",
        frozenset({"eval:set2", "eval:xai", "harbor:set2", "harbor:xai"}),
    ),
    Model(
        "xai:grok-3-mini-fast",
        frozenset({"eval:set2", "eval:xai", "harbor:set2", "harbor:xai"}),
    ),
)

# ---------------------------------------------------------------------------
# Preset definitions — map preset names to tag filters per workflow.
#
# _PRESET_SECTIONS is the single source of truth for preset names, doc
# ordering, and section grouping.
# Each entry is (section_name, [(preset_name, tag_suffix | None), ...]).
#   - section_name = None  → no heading is emitted for that group.
#   - tag_suffix = None    → matches any tag with the workflow prefix
#                            (i.e. the "all" preset).
# ---------------------------------------------------------------------------
_PRESET_SECTIONS: list[tuple[str | None, list[tuple[str, str | None]]]] = [
    (
        "Model groups",
        [
            ("set0", "set0"),
            ("set1", "set1"),
            ("set2", "set2"),
            ("frontier", "frontier"),
            ("fast", "fast"),
            ("open", "open"),
        ],
    ),
    (
        "Provider groups",
        [
            ("anthropic", "anthropic"),
            ("baseten", "baseten"),
            ("fireworks", "fireworks"),
            ("google_genai", "google_genai"),
            ("groq", "groq"),
            ("nvidia", "nvidia"),
            ("ollama", "ollama"),
            ("openai", "openai"),
            ("openrouter", "openrouter"),
            ("xai", "xai"),
        ],
    ),
    (
        None,
        [
            ("all", None),
        ],
    ),
]


def _build_presets(prefix: str) -> dict[str, str | None]:
    """Derive a flat preset lookup dict from `_PRESET_SECTIONS`."""
    return {
        name: f"{prefix}:{suffix}" if suffix is not None else None
        for _, presets in _PRESET_SECTIONS
        for name, suffix in presets
    }


_EVAL_PRESETS: dict[str, str | None] = _build_presets("eval")
"""Flat preset name → `eval:{tag}` mapping for the evals workflow."""

_HARBOR_PRESETS: dict[str, str | None] = _build_presets("harbor")
"""Flat preset name → `harbor:{tag}` mapping for the Harbor workflow."""

_WORKFLOW_CONFIG: dict[str, tuple[str, dict[str, str | None]]] = {
    "eval": ("EVAL_MODELS", _EVAL_PRESETS),
    "harbor": ("HARBOR_MODELS", _HARBOR_PRESETS),
}


def _filter_by_tag(prefix: str, tag: str | None) -> list[str]:
    """Return model specs matching a tag filter, in REGISTRY order."""
    if tag is not None:
        return [m.spec for m in REGISTRY if tag in m.groups]
    return [m.spec for m in REGISTRY if any(g.startswith(prefix) for g in m.groups)]


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
    unsafe = [s for s in specs if not _SAFE_SPEC_RE.match(s)]
    if unsafe:
        msg = f"Model spec(s) contain disallowed characters: {', '.join(repr(s) for s in unsafe)}"
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
    matrix = {
        "include": [{"model": m, "provider": m.split(":")[0]} for m in models],
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    line = f"matrix={json.dumps(matrix, separators=(',', ':'))}"
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            f.write(line + "\n")
    else:
        print(line)  # noqa: T201


if __name__ == "__main__":
    main()
