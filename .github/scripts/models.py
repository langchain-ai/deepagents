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
    """A model spec with group tags and display labels."""

    spec: str
    """Canonical `{provider}:{model}` identifier passed to the eval/harbor runner.

    The portion before the colon must match a key resolvable by langchain's
    `init_chat_model` (e.g. `anthropic`, `google_genai`); the suffix is the
    provider-native model name. Used as the routing key for matrix entries
    and artifact names — must be unique across the registry.
    """

    groups: frozenset[str]
    """Workflow + group tags applied to this model.

    Each entry follows the `{workflow}:{group}` convention (e.g. `eval:set0`,
    `harbor:anthropic`). Membership in a group includes the model in the
    matching preset for that workflow.
    """

    display_name: str
    """Human-readable model label for chart legends and tables.

    Curated per-model (e.g. `Claude Sonnet 4.6`, `GPT-5.4 mini`) so consumers
    don't have to render raw spec slugs.

    Not used for matrix routing or workflow gating.
    """

    provider_label: str
    """Human-readable provider label for section headings and grouping.

    Uniform within a `provider:` prefix (e.g. all `google_genai:*` entries
    use `"Google"`). When this label lowercases to the prefix itself
    (e.g. `Anthropic` -> `anthropic`), `MODEL_GROUPS.md` collapses to the
    compact `## anthropic` form; otherwise it renders `## Google (google_genai)`.

    Not used for matrix routing or workflow gating.
    """


# ---------------------------------------------------------------------------
# Registry — canonical order determines output order within each preset.
# Tags follow the convention {workflow}:{group}.
# ---------------------------------------------------------------------------
REGISTRY: tuple[Model, ...] = (
    # -- Anthropic --
    Model(
        "anthropic:claude-haiku-4-5",
        frozenset({"eval:anthropic", "harbor:anthropic"}),
        "Claude Haiku 4.5",
        "Anthropic",
    ),
    Model(
        "anthropic:claude-sonnet-4-5-20250929",
        frozenset({"eval:set0", "eval:anthropic", "harbor:set0", "harbor:anthropic"}),
        "Claude Sonnet 4.5",
        "Anthropic",
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
        "Claude Sonnet 4.6",
        "Anthropic",
    ),
    Model(
        "anthropic:claude-opus-4-5-20251101",
        frozenset({"eval:set0", "eval:anthropic", "harbor:set0", "harbor:anthropic"}),
        "Claude Opus 4.5",
        "Anthropic",
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
        "Claude Opus 4.6",
        "Anthropic",
    ),
    Model(
        "anthropic:claude-opus-4-7",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:docs",
                "eval:anthropic",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:docs",
                "harbor:anthropic",
            }
        ),
        "Claude Opus 4.7",
        "Anthropic",
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
        "GLM-5",
        "Baseten",
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
        "MiniMax M2.5",
        "Baseten",
    ),
    Model(
        "baseten:moonshotai/Kimi-K2.5",
        frozenset({"eval:set0", "eval:baseten", "harbor:set0", "harbor:baseten"}),
        "Kimi K2.5",
        "Baseten",
    ),
    Model(
        "baseten:moonshotai/Kimi-K2.6",
        frozenset(
            {
                "eval:set0",
                "eval:open",
                "eval:docs",
                "eval:baseten",
                "harbor:set0",
                "harbor:open",
                "harbor:docs",
                "harbor:baseten",
            }
        ),
        "Kimi K2.6",
        "Baseten",
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
        "Nemotron 120B A12B",
        "Baseten",
    ),
    Model(
        "baseten:Qwen/Qwen3-Coder-480B-A35B-Instruct",
        frozenset({"eval:set0", "eval:baseten", "harbor:set0", "harbor:baseten"}),
        "Qwen3 Coder 480B A35B",
        "Baseten",
    ),
    # -- Fireworks --
    Model(
        "fireworks:accounts/fireworks/models/deepseek-v3p2",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
        "DeepSeek V3.2",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/deepseek-v3-0324",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
        "DeepSeek V3 (0324)",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/deepseek-v4-pro",
        frozenset(
            {
                "eval:open-fireworks",
                "eval:fireworks",
                "harbor:open-fireworks",
                "harbor:fireworks",
            }
        ),
        "DeepSeek V4 Pro",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/kimi-k2p5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
        "Kimi K2.5",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/kimi-k2p6",
        frozenset(
            {
                "eval:open-fireworks",
                "eval:fireworks",
                "harbor:open-fireworks",
                "harbor:fireworks",
            }
        ),
        "Kimi K2.6",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/glm-5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
        "GLM-5",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/glm-5p1",
        frozenset(
            {
                "eval:open-fireworks",
                "eval:fireworks",
                "harbor:open-fireworks",
                "harbor:fireworks",
            }
        ),
        "GLM-5.1",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/minimax-m2p5",
        frozenset({"eval:set0", "eval:fireworks", "harbor:set0", "harbor:fireworks"}),
        "MiniMax M2.5",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/minimax-m2p7",
        frozenset(
            {
                "eval:open-fireworks",
                "eval:fireworks",
                "harbor:open-fireworks",
                "harbor:fireworks",
            }
        ),
        "MiniMax M2.7",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/nvidia-nemotron-3-super-120b-a12b-fp8",
        frozenset(
            {
                "eval:open-fireworks",
                "eval:fireworks",
                "harbor:open-fireworks",
                "harbor:fireworks",
            }
        ),
        "Nemotron 3 Super 120B A12B FP8",
        "Fireworks",
    ),
    Model(
        "fireworks:accounts/fireworks/models/qwen3-vl-235b-a22b-thinking",
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
        "Qwen3 VL 235B A22B Thinking",
        "Fireworks",
    ),
    # -- Google --
    Model(
        "google_genai:gemini-2.5-flash",
        frozenset(
            {"eval:set0", "eval:google_genai", "harbor:set0", "harbor:google_genai"}
        ),
        "Gemini 2.5 Flash",
        "Google",
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
        "Gemini 2.5 Pro",
        "Google",
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
        "Gemini 3 Flash (preview)",
        "Google",
    ),
    Model(
        "google_genai:gemini-3.1-pro-preview",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:docs",
                "eval:google_genai",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:docs",
                "harbor:google_genai",
            }
        ),
        "Gemini 3.1 Pro (preview)",
        "Google",
    ),
    # -- Groq --
    Model(
        "groq:openai/gpt-oss-120b",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
        "GPT-OSS 120B",
        "Groq",
    ),
    Model(
        "groq:qwen/qwen3-32b",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
        "Qwen3 32B",
        "Groq",
    ),
    Model(
        "groq:moonshotai/kimi-k2-instruct",
        frozenset({"eval:set2", "eval:groq", "harbor:set2", "harbor:groq"}),
        "Kimi K2 Instruct",
        "Groq",
    ),
    # -- NVIDIA --
    Model(
        "nvidia:nvidia/nemotron-3-super-120b-a12b",
        frozenset({"eval:nvidia", "harbor:nvidia"}),
        "Nemotron 3 Super 120B A12B",
        "NVIDIA",
    ),
    # -- Ollama --
    Model(
        "ollama:glm-5:cloud",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
        "GLM-5 (cloud)",
        "Ollama",
    ),
    Model(
        "ollama:glm-5.1:cloud",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
        "GLM-5.1 (cloud)",
        "Ollama",
    ),
    Model(
        "ollama:minimax-m2.5:cloud",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
        "MiniMax M2.5 (cloud)",
        "Ollama",
    ),
    Model(
        "ollama:minimax-m2.7:cloud",
        frozenset(
            {
                "eval:set0",
                "eval:ollama",
                "harbor:set0",
                "harbor:ollama",
            }
        ),
        "MiniMax M2.7 (cloud)",
        "Ollama",
    ),
    Model(
        "ollama:qwen3.5:cloud",
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
        "Qwen3.5 (cloud)",
        "Ollama",
    ),
    Model(
        "ollama:nemotron-3-super:cloud",
        frozenset(
            {
                "eval:set2",
                "eval:ollama",
                "harbor:set2",
                "harbor:ollama",
            }
        ),
        "Nemotron 3 Super (cloud)",
        "Ollama",
    ),
    # -- OpenAI --
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
        "GPT-4.1",
        "OpenAI",
    ),
    Model(
        "openai:gpt-5.1-codex",
        frozenset({"eval:set0", "eval:openai", "harbor:set0", "harbor:openai"}),
        "GPT-5.1 Codex",
        "OpenAI",
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
        "GPT-5.2 Codex",
        "OpenAI",
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
        "GPT-5.3 Codex",
        "OpenAI",
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
        "GPT-5.4",
        "OpenAI",
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
        "GPT-5.4 mini",
        "OpenAI",
    ),
    Model(
        "openai:gpt-5.5",
        frozenset(
            {
                "eval:set0",
                "eval:set1",
                "eval:frontier",
                "eval:docs",
                "eval:openai",
                "harbor:set0",
                "harbor:set1",
                "harbor:frontier",
                "harbor:docs",
                "harbor:openai",
            }
        ),
        "GPT-5.5",
        "OpenAI",
    ),
    Model(
        "openai:gpt-5.5-pro",
        frozenset(
            {
                "eval:mega",
                "harbor:mega",
            }
        ),
        "GPT-5.5 Pro",
        "OpenAI",
    ),
    # -- OpenRouter --
    Model(
        "openrouter:minimax/minimax-m2.7",
        frozenset(
            {
                "eval:open",
                "eval:docs",
                "eval:openrouter",
                "harbor:open",
                "harbor:docs",
                "harbor:openrouter",
            }
        ),
        "MiniMax M2.7",
        "OpenRouter",
    ),
    Model(
        "openrouter:moonshotai/kimi-k2.5",
        frozenset(
            {
                "eval:openrouter",
                "harbor:openrouter",
            }
        ),
        "Kimi K2.5",
        "OpenRouter",
    ),
    Model(
        "openrouter:moonshotai/kimi-k2.6",
        frozenset(
            {
                "eval:openrouter",
                "harbor:openrouter",
            }
        ),
        "Kimi K2.6",
        "OpenRouter",
    ),
    Model(
        "openrouter:z-ai/glm-5.1",
        frozenset(
            {
                "eval:open",
                "eval:docs",
                "eval:openrouter",
                "harbor:open",
                "harbor:docs",
                "harbor:openrouter",
            }
        ),
        "GLM-5.1",
        "OpenRouter",
    ),
    Model(
        "openrouter:nvidia/nemotron-3-super-120b-a12b",
        frozenset(
            {
                "eval:open",
                "eval:openrouter",
                "harbor:open",
                "harbor:openrouter",
            }
        ),
        "Nemotron 3 Super 120B A12B",
        "OpenRouter",
    ),
    Model(
        "openrouter:deepseek/deepseek-v4-pro",
        frozenset(
            {
                "eval:open",
                "eval:docs",
                "eval:openrouter",
                "harbor:open",
                "harbor:docs",
                "harbor:openrouter",
            }
        ),
        "DeepSeek V4 Pro",
        "OpenRouter",
    ),
    # -- xAI --
    Model(
        "xai:grok-4",
        frozenset({"eval:set2", "eval:xai", "harbor:set2", "harbor:xai"}),
        "Grok 4",
        "xAI",
    ),
    Model(
        "xai:grok-3-mini-fast",
        frozenset({"eval:set2", "eval:xai", "harbor:set2", "harbor:xai"}),
        "Grok 3 mini fast",
        "xAI",
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
            ("mega", "mega"),
            ("fast", "fast"),
            ("open", "open"),
            ("open-fireworks", "open-fireworks"),
            ("docs", "docs"),
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

_EVAL_PROVIDER_OUTPUTS: tuple[str, ...] = (
    "anthropic",
    "baseten",
    "fireworks",
    "google_genai",
    "groq",
    "nvidia",
    "ollama",
    "openai",
    "openrouter",
    "xai",
    "other",
)
"""Names of the per-provider matrix outputs emitted for the evals workflow.

All entries except `"other"` are real provider prefixes matched against
`_provider(model_spec)`. The `"other"` bucket is a catch-all for model specs
whose provider is not enumerated here, so they still flow into a real eval
job (see `_matrix_outputs`). Adding a new provider here also requires adding
a matching `eval-<provider>` job in `evals.yml`; the test suite enforces this
contract.
"""

_ARTIFACT_KEY_RE = re.compile(r"[^a-zA-Z0-9._-]+")
"""Characters disallowed in GHA artifact names (model specs use `:` and `/`)."""


def _filter_by_tag(prefix: str, tag: str | None) -> list[str]:
    """Return model specs matching a tag filter, in REGISTRY order."""
    if tag is not None:
        return [m.spec for m in REGISTRY if tag in m.groups]
    return [m.spec for m in REGISTRY if any(g.startswith(prefix) for g in m.groups)]


def _provider(model_spec: str) -> str:
    """Return the provider prefix from a model spec."""
    return model_spec.split(":", 1)[0]


_BY_SPEC: dict[str, Model] = {m.spec: m for m in REGISTRY}
"""Spec → `Model` lookup. Built once; safe to read concurrently."""


def display_name(model_spec: str) -> str:
    """Return the human-readable display name for a model spec.

    Falls back to the bare model name (after stripping the `provider:` prefix)
    when the spec is not in `REGISTRY`. Consumers (radar charts, doc
    generators) use this for legends and headings.
    """
    entry = _BY_SPEC.get(model_spec)
    if entry is not None:
        return entry.display_name
    return model_spec.split(":", 1)[1] if ":" in model_spec else model_spec


def provider_label(model_spec: str) -> str:
    """Return the human-readable provider label for a model spec.

    Falls back to the raw provider prefix (e.g. `openai`, `xai`) when the
    spec is not in `REGISTRY`. The fallback is intentionally lowercase so
    drift between the registry and an ad-hoc spec is visually obvious.
    """
    entry = _BY_SPEC.get(model_spec)
    if entry is not None:
        return entry.provider_label
    return _provider(model_spec)


def _artifact_key(model_spec: str) -> str:
    """Build an artifact-safe key for one model matrix entry.

    GitHub Actions artifact names disallow several characters that appear in
    model specs (e.g., `:` and `/`), so the regex replaces every disallowed
    character with `-`. Uniqueness across a workflow run is enforced by
    `_resolve_models` (dedupe) and `_matrix_outputs` (assertion); see those
    callers.
    """
    return _ARTIFACT_KEY_RE.sub("-", model_spec).strip("-")


def _matrix_entry(model_spec: str) -> dict[str, str]:
    """Build one GitHub Actions matrix entry for a model."""
    return {
        "model": model_spec,
        "provider": _provider(model_spec),
        "artifact_key": _artifact_key(model_spec),
    }


def _matrix_outputs(workflow: str, models: list[str]) -> dict[str, object]:
    """Build matrix outputs consumed by GitHub Actions workflows.

    The evals workflow needs one matrix per provider so each provider can use
    `strategy.max-parallel: 1` as a real per-provider queue. The catch-all
    `other` matrix runs any models whose provider is not enumerated in
    `_EVAL_PROVIDER_OUTPUTS`, so newly added providers (or one-off
    `models_override` entries) still execute even before a dedicated
    `eval-<provider>` job is wired up in `evals.yml`.

    Args:
        workflow: "eval" or "harbor".
        models: Ordered model specs selected for the workflow.

    Returns:
        Mapping of GitHub output names to JSON-serializable values.
    """
    entries = [_matrix_entry(model) for model in models]
    # Tripwire: `_resolve_models` dedupes raw specs, and our `provider:model`
    # convention plus the registry's canonical specs make it effectively
    # impossible for two distinct specs to collapse to the same slug. If this
    # ever fires, reintroduce a disambiguating prefix in `_artifact_key`.
    by_key: dict[str, list[str]] = {}
    for entry in entries:
        by_key.setdefault(entry["artifact_key"], []).append(entry["model"])
    collisions = {key: specs for key, specs in by_key.items() if len(specs) > 1}
    if collisions:
        details = "; ".join(f"{key!r} <- {specs}" for key, specs in collisions.items())
        msg = f"Duplicate artifact_key(s) in matrix: {details}"
        raise ValueError(msg)
    outputs: dict[str, object] = {"matrix": {"include": entries}}

    if workflow != "eval":
        return outputs

    provider_entries: dict[str, list[dict[str, str]]] = {
        provider: [] for provider in _EVAL_PROVIDER_OUTPUTS
    }
    for entry in entries:
        provider = entry["provider"]
        output_provider = provider if provider in provider_entries else "other"
        provider_entries[output_provider].append(entry)

    # Empty includes are emitted intentionally and *must* be guarded by
    # `<provider>_has_models == 'true'` in evals.yml — GitHub Actions fails
    # the workflow when `matrix.include == []`. See the per-provider job
    # `if:` clauses in evals.yml.
    for provider, include in provider_entries.items():
        outputs[f"{provider}_matrix"] = {"include": include}
        outputs[f"{provider}_has_models"] = bool(include)

    return outputs


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
        specs = _filter_by_tag(f"{workflow}:", presets[normalized])
    else:
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
    # Order-preserving dedupe so duplicate entries (typo'd `models_override`,
    # or a future REGISTRY edit that accidentally repeats a spec) cannot
    # collide on `artifact_key` downstream.
    return list(dict.fromkeys(specs))


def main() -> None:
    """Entry point — reads workflow arg and env var, writes matrix JSON."""
    if len(sys.argv) != 2 or sys.argv[1] not in _WORKFLOW_CONFIG:  # noqa: PLR2004
        msg = f"Usage: {sys.argv[0]} {{{' | '.join(_WORKFLOW_CONFIG)}}}"
        raise SystemExit(msg)

    workflow = sys.argv[1]
    env_var, _ = _WORKFLOW_CONFIG[workflow]
    selection = os.environ.get(env_var, "all")
    models = _resolve_models(workflow, selection)
    outputs = _matrix_outputs(workflow, models)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            for key, value in outputs.items():
                payload = json.dumps(value, separators=(",", ":"))
                f.write(f"{key}={payload}\n")
    else:
        payload = json.dumps(outputs["matrix"], separators=(",", ":"))
        print(f"matrix={payload}")  # noqa: T201


if __name__ == "__main__":
    main()
