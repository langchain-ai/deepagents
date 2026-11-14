"""Audited route-specific metadata for the bundled stable model lineup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


type _ProfileValue = str | int | bool | list[str] | None

_REASONING_LEVELS = ["low", "medium", "high", "xhigh", "max"]
_OPTIONAL_REASONING_LEVELS = ["none", *_REASONING_LEVELS]
_META_REASONING_LEVELS = ["minimal", *_REASONING_LEVELS]
_GOOGLE_REASONING_LEVELS = ["low", "medium", "high"]


def _profile(
    name: str, source: str, note: str, *, inputs: str = "", **fields: _ProfileValue
) -> dict[str, _ProfileValue]:
    """Return text/tool metadata, explicitly clearing unaudited capabilities."""
    modalities = inputs.split()
    return {
        "name": name,
        "source": source,
        "profile_note": note,
        "context_window": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "text_inputs": True,
        "image_inputs": "image" in modalities,
        "audio_inputs": "audio" in modalities,
        "pdf_inputs": None if "file" in modalities else "pdf" in modalities,
        "video_inputs": "video" in modalities,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "tool_choice": None,
        "structured_output": None,
        "reasoning_effort_levels": None,
        "reasoning_effort_default": None,
        **{
            key: value.copy() if isinstance(value, list) else value
            for key, value in fields.items()
        },
    }


def _codex_profile(
    name: str, *, context: int | None = None
) -> dict[str, _ProfileValue]:
    """Return ChatGPT OAuth metadata independent of OpenAI API profiles."""
    return _profile(
        f"{name} (ChatGPT OAuth)",
        "https://developers.openai.com/codex/models",
        "ChatGPT OAuth account/plan limits apply, not OpenAI API-key limits.",
        context_window=context,
        max_input_tokens=context,
        image_inputs=None,
        audio_inputs=None,
        pdf_inputs=None,
        video_inputs=None,
        image_outputs=None,
        audio_outputs=None,
        video_outputs=None,
    )


BUNDLED_MODEL_PROFILES: dict[str, dict[str, _ProfileValue]] = {}

for _model, _name, _page, _default, _note in (
    (
        "claude-fable-5-2",
        "Claude Fable 5.2",
        "fable-5-2/whats-new-fable-5-2",
        "high",
        "30-day retention; auto tools; strict schema except CMEK; append-only history.",
    ),
    (
        "claude-opus-5-5",
        "Claude Opus 5.5",
        "opus-5-5/overview",
        "medium",
        "Adaptive thinking always on; 128K ordinary output, not Message Batches beta.",
    ),
    (
        "claude-sonnet-5-5",
        "Claude Sonnet 5.5",
        "sonnet-5-5/overview",
        "high",
        "Lowest thinking mode is between_tools, not off; omit sampling parameters.",
    ),
    (
        "claude-haiku-5-5",
        "Claude Haiku 5.5",
        "haiku-5-5/overview",
        "medium",
        "Auxiliary-only curation; no automatic role-default changes.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"anthropic:{_model}"] = _profile(
        _name,
        f"https://platform.claude.com/docs/en/models/{_page}",
        _note,
        max_input_tokens=1_000_000,
        max_output_tokens=128_000,
        image_inputs=True,
        pdf_inputs=None if _model == "claude-fable-5-2" else True,
        tool_choice=_model != "claude-fable-5-2",
        structured_output=True,
        reasoning_effort_levels=_REASONING_LEVELS,
        reasoning_effort_default=_default,
    )

for _model, _name, _levels, _default, _note in (
    (
        "gemini-4-carbon",
        "Gemini 4 Carbon",
        _GOOGLE_REASONING_LEVELS,
        "medium",
        "Unified input/thinking/output; implicit cache; 65,536 SDK/stream output.",
    ),
    (
        "gemini-4-hydrogen",
        "Gemini 4 Hydrogen",
        _GOOGLE_REASONING_LEVELS,
        "high",
        "Unified budget; provider xhigh unavailable in adapter; preserve signatures.",
    ),
    (
        "gemini-4-argon",
        "Gemini 4 Argon",
        _GOOGLE_REASONING_LEVELS,
        "medium",
        "Unified input/thinking/output; implicit cache; continuation not assumed.",
    ),
    (
        "gemini-3.8-flash",
        "Gemini 3.8 Flash",
        _GOOGLE_REASONING_LEVELS,
        "medium",
        "Previous generation; minimal effort unsupported; no shutdown announced.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"google_genai:{_model}"] = _profile(
        _name,
        f"https://aistudio.google.com/docs/models/{_model}",
        _note,
        context_window=1_048_576 if _model != "gemini-3.8-flash" else None,
        max_input_tokens=983_040 if _model != "gemini-3.8-flash" else 1_048_576,
        max_output_tokens=65_536,
        image_inputs=True,
        audio_inputs=True,
        pdf_inputs=True,
        video_inputs=True,
        structured_output=True,
        reasoning_effort_levels=_levels,
        reasoning_effort_default=_default,
    )

for _model, _name, _levels, _default, _note in (
    (
        "gpt-6-astra",
        "GPT-6 Astra",
        _REASONING_LEVELS,
        None,
        "Responses API required; no none/minimal effort; shared context/output.",
    ),
    (
        "gpt-6.1-sol",
        "GPT-6.1 Sol",
        _REASONING_LEVELS,
        "medium",
        "Responses required; no none/minimal; long-context pricing above 272K.",
    ),
    (
        "gpt-6.1-terra",
        "GPT-6.1 Terra",
        _OPTIONAL_REASONING_LEVELS,
        "medium",
        "Responses required; native computer use is a separate integration.",
    ),
    (
        "gpt-6-luna",
        "GPT-6 Luna",
        _OPTIONAL_REASONING_LEVELS,
        "medium",
        "Reasoning + tools need Responses; Chat Completions tools need effort none.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"openai:{_model}"] = _profile(
        _name,
        f"https://developers.openai.com/api/docs/models/{_model}",
        _note,
        context_window=1_050_000,
        max_input_tokens=922_000,
        max_output_tokens=128_000,
        image_inputs=True,
        pdf_inputs=True,
        tool_choice=True,
        structured_output=True,
        reasoning_effort_levels=_levels,
        reasoning_effort_default=_default,
    )

for _model, _name in (
    ("gpt-6-astra", "GPT-6 Astra"),
    ("gpt-6.1-sol", "GPT-6.1 Sol"),
    ("gpt-6.1-terra", "GPT-6.1 Terra"),
    ("gpt-6-luna", "GPT-6 Luna"),
):
    BUNDLED_MODEL_PROFILES[f"openai_codex:{_model}"] = _codex_profile(
        _name, context=500_000 if _model == "gpt-6.1-sol" else None
    )

BUNDLED_MODEL_PROFILES["openai_codex:gpt-6-sol"] = _codex_profile("GPT-6 Sol")
BUNDLED_MODEL_PROFILES["openai_codex:gpt-6-sol"]["profile_note"] = (
    "Unverified compatibility ID; ChatGPT OAuth access/limits depend on account/plan."
)

for _model, _name, _audio, _output, _default, _note in (
    (
        "muse-spark-1.3",
        "Muse Spark 1.3",
        False,
        None,
        "high",
        "Audio input degraded/not fully supported; Responses preserves reasoning.",
    ),
    (
        "muse-ember-1.0",
        "Muse Ember 1.0",
        True,
        524_288,
        "max",
        "Standard mode; reasoning shares output budget; no role-default changes.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"meta:{_model}"] = _profile(
        _name,
        f"https://dev.meta.ai/docs/{_model.rsplit('-', 1)[0]}",
        _note,
        context_window=1_048_576,
        max_output_tokens=_output,
        image_inputs=True,
        audio_inputs=_audio,
        pdf_inputs=True,
        video_inputs=True,
        structured_output=True,
        reasoning_effort_levels=_META_REASONING_LEVELS,
        reasoning_effort_default=_default,
    )

BUNDLED_MODEL_PROFILES["xai:grok-4.7"] = _profile(
    "Grok 4.7",
    "https://docs.x.ai/developers/models/grok-4.7.md",
    "Cannot disable reasoning; output/PDF unverified; retain Responses reasoning.",
    context_window=500_000,
    image_inputs=True,
    pdf_inputs=None,
    structured_output=True,
    reasoning_effort_levels=["low", "medium", "high", "xhigh"],
    reasoning_effort_default="high",
)

for _model, _name, _image, _video, _levels, _default, _note in (
    (
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "DeepSeek V4.1 Flash 1017",
        True,
        False,
        ["none", "low", "medium", "high", "max"],
        "high",
        "Host labels 1048k context / 262k output are not verified exact integer caps.",
    ),
    (
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "DeepSeek V4 Pro 0813",
        False,
        False,
        ["none", "low", "high", "max"],
        None,
        "1048k context/262k output labels; thinking on; default level unknown.",
    ),
    (
        "zai-org/GLM-5.3",
        "GLM 5.3",
        True,
        False,
        ["none", "low", "high", "max"],
        "high",
        "1048k context/262k output labels; none effort maps to low, not off.",
    ),
    (
        "zai-org/GLM-5.3-Flash",
        "GLM 5.3 Flash",
        True,
        True,
        ["none", "low", "high", "max"],
        "high",
        "1048k context/262k output labels; none effort maps to low, not off.",
    ),
    (
        "meta-models/muse-spark-open",
        "Muse Spark Open",
        True,
        True,
        ["low", "medium", "high", "xhigh"],
        "xhigh",
        "1048k context/output labels; no JSON/schema/PDF; none/minimal map to low.",
    ),
    (
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
        "Nemotron 3.5 Super VL",
        True,
        True,
        None,
        None,
        "1048k context/output labels; no audio; thinking off, enabled maps high.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"baseten:{_model}"] = _profile(
        _name,
        "https://docs.baseten.co/inference/model-apis/overview.md",
        _note,
        image_inputs=_image,
        audio_inputs=_model == "meta-models/muse-spark-open",
        video_inputs=_video,
        structured_output=_model != "meta-models/muse-spark-open",
        reasoning_effort_levels=_levels,
        reasoning_effort_default=_default,
    )

for _model, _name, _image, _note in (
    (
        "deepseek-v4p1-flash-1017",
        "DeepSeek V4.1 Flash 1017",
        True,
        "Serverless; output/effort unverified; per-route schema test pending.",
    ),
    (
        "glm-5p3",
        "GLM 5.3",
        False,
        "Serverless; output/effort unverified; per-route schema test pending.",
    ),
    (
        "glm-5p3-flash",
        "GLM 5.3 Flash",
        True,
        "Serverless; output/effort unverified; per-route schema test pending.",
    ),
    (
        "minimax-m3p1-flash",
        "MiniMax M3.1 Flash",
        True,
        "Always thinking; exact effort/output unknown; schema smoke test pending.",
    ),
    (
        "muse-spark-open",
        "Muse Spark Open",
        True,
        "US-only standard capacity; output unknown; schema smoke test pending.",
    ),
    (
        "qwen3p8-max",
        "Qwen 3.8 Max (hosted)",
        True,
        "Hosted Max, not released checkpoint; caps/effort unknown; schema pending.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"fireworks:accounts/fireworks/models/{_model}"] = _profile(
        _name,
        f"https://fireworks.ai/models/fireworks/{_model}",
        _note,
        context_window=None if _model == "qwen3p8-max" else 1_048_576,
        image_inputs=_image,
        structured_output=True,
    )

_ROUTER_NOTE = (
    "Catalog context; output/schema/effort require compatible backend routing."
)

for _model, _name, _inputs, _context, _note in (
    (
        "deepseek/deepseek-v4.1-flash-1017",
        "DeepSeek V4.1 Flash 1017",
        "image",
        1_048_576,
        _ROUTER_NOTE,
    ),
    (
        "deepseek/deepseek-v4-pro-0813",
        "DeepSeek V4 Pro 0813",
        "",
        1_048_576,
        _ROUTER_NOTE,
    ),
    ("z-ai/glm-5.3", "GLM 5.3", "", 1_048_576, _ROUTER_NOTE),
    ("z-ai/glm-5.3-flash", "GLM 5.3 Flash", "image video", 1_048_576, _ROUTER_NOTE),
    (
        "minimax/minimax-m3.1-flash",
        "MiniMax M3.1 Flash",
        "image video",
        1_048_576,
        _ROUTER_NOTE,
    ),
    (
        "meta/muse-spark-open",
        "Muse Spark Open",
        "image audio video file",
        1_048_576,
        "Backend-dependent output/effort; schema not universal (Baseten lacks it).",
    ),
    (
        "nvidia/nemotron-3.5-super-vl",
        "Nemotron 3.5 Super VL",
        "image video",
        1_048_576,
        _ROUTER_NOTE,
    ),
    (
        "mistralai/mistral-large-4-0",
        "Mistral Large 4",
        "image",
        1_048_576,
        "Native effort does not verify host mapping; output/schema vary by backend.",
    ),
    (
        "qwen/qwen3.8-max-0902",
        "Qwen 3.8 Max (hosted)",
        "image video",
        1_000_000,
        "Hosted Max, not released checkpoint; output/schema/effort vary by backend.",
    ),
    (
        "anthropic/claude-fable-5.2",
        "Claude Fable 5.2 (OpenRouter)",
        "image file",
        1_000_000,
        "Backend-dependent; review Fable retention, forced-tool and history limits.",
    ),
    (
        "anthropic/claude-opus-5.5",
        "Claude Opus 5.5 (OpenRouter)",
        "image file",
        1_000_000,
        _ROUTER_NOTE,
    ),
    (
        "anthropic/claude-sonnet-5.5",
        "Claude Sonnet 5.5 (OpenRouter)",
        "image file",
        1_000_000,
        _ROUTER_NOTE,
    ),
    (
        "anthropic/claude-haiku-5.5",
        "Claude Haiku 5.5 (OpenRouter)",
        "image file",
        1_000_000,
        _ROUTER_NOTE,
    ),
    (
        "google/gemini-4-carbon",
        "Gemini 4 Carbon (OpenRouter)",
        "image audio video file",
        1_048_576,
        "Catalog output is not a verified response cap; schema/effort vary by host.",
    ),
    (
        "google/gemini-4-hydrogen",
        "Gemini 4 Hydrogen (OpenRouter)",
        "image audio video file",
        1_048_576,
        "Catalog output is not a verified response cap; schema/effort vary by host.",
    ),
    (
        "google/gemini-4-argon",
        "Gemini 4 Argon (OpenRouter)",
        "image audio video file",
        1_048_576,
        "Catalog output is not a verified response cap; schema/effort vary by host.",
    ),
    (
        "google/gemini-3.8-flash",
        "Gemini 3.8 Flash (OpenRouter)",
        "image audio video file",
        1_048_576,
        "Supported previous generation; output/schema/effort depend on backend.",
    ),
):
    BUNDLED_MODEL_PROFILES[f"openrouter:{_model}"] = _profile(
        _name,
        f"https://openrouter.ai/{_model}",
        _note,
        inputs=_inputs,
        context_window=_context,
    )

for _model, _name, _image, _context in (
    ("deepseek-v4.1-flash:1017-cloud", "DeepSeek V4.1 Flash 1017", True, "1M"),
    ("deepseek-v4-pro:0813-cloud", "DeepSeek V4 Pro 0813", False, "1M"),
    ("glm-5.3:cloud", "GLM 5.3", False, "1M"),
    ("glm-5.3-flash:cloud", "GLM 5.3 Flash", True, "1M"),
    ("minimax-m3.1-flash:cloud", "MiniMax M3.1 Flash", True, "976K"),
    ("muse-spark-open:cloud", "Muse Spark Open", True, "1M"),
    ("mistral-large-4:cloud", "Mistral Large 4", True, "1M"),
):
    BUNDLED_MODEL_PROFILES[f"ollama:{_model}"] = _profile(
        _name,
        f"https://ollama.com/library/{_model}",
        f"Cloud {_context} label; caps/efforts unknown; no schema; access required.",
        image_inputs=_image,
        structured_output=False,
    )


def merge_model_profile(
    spec: str, upstream: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Overlay audited route fields without importing OpenAI API metadata into OAuth.

    Args:
        spec: Exact provider-qualified model identifier.
        upstream: Installed provider metadata, when available.

    Returns:
        Independent profile mapping with bundled fields taking precedence.
    """
    bundled = BUNDLED_MODEL_PROFILES.get(spec, {})
    profile = (
        dict(bundled)
        if spec.startswith("openai_codex:")
        else {**(upstream or {}), **bundled}
    )
    for key in ("reasoning_effort_levels", "reasoning_effort_default"):
        if key in bundled and profile[key] is None:
            profile.pop(key)
    return profile
