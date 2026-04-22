"""Beta APIs for configuring deep agent runtime behavior.

!!! beta

    `deepagents.profiles` exposes beta APIs that may receive minor changes in
    future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
    for more details.

Harness profiles declare how `create_deep_agent` should shape the agent's
runtime behavior for a given provider or specific model spec. They tune
prompt assembly, tool visibility, middleware, and default subagent behavior
*after* the chat model has been constructed — orthogonal to
`ProviderProfile`, which controls the model-construction phase.

Users may register additional profiles via `register_harness_profile`. Built-in
profiles are registered as side effects of importing partner modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import AgentMiddleware


@dataclass(frozen=True)
class GeneralPurposeSubagentProfile:
    """Edits applied to the auto-added `general-purpose` subagent.

    !!! beta

        `GeneralPurposeSubagentProfile` is a beta API. It is safe for
        production use, but may receive minor changes in future releases.

    These settings only affect the default subagent that `create_deep_agent`
    inserts when the caller does not explicitly provide a subagent named
    `general-purpose`.
    """

    enabled: bool | None = None
    """Whether to auto-add the default general-purpose subagent.

    `None` means inherit the parent/default behavior of including the
    subagent. `False` disables the auto-added subagent entirely.

    !!! note

        If the default subagent is disabled and no other synchronous subagents are
        configured, the main agent will not expose the `task` tool.
    """

    description: str | None = None
    """Override for the default subagent description."""

    system_prompt: str | None = None
    """Override for the default subagent system prompt."""


@dataclass(frozen=True)
class HarnessProfile:
    """Declarative configuration for deep agent runtime behavior.

    !!! beta

        `HarnessProfile` is a beta API. It is safe for production use, but may
        receive minor changes in future releases.

    A `HarnessProfile` describes prompt-assembly, tool-visibility, middleware,
    and default-subagent adjustments applied by `create_deep_agent` once a
    chat model has been constructed. Profiles are registered via
    `register_harness_profile` under a provider key (`"openai"`) or a full
    `provider:model` key (`"openai:gpt-5.4"`).

    This complements `ProviderProfile`, which controls the model-construction
    phase (e.g. `init_chat_model` kwargs, pre-init side effects). Concerns
    that shape *how the model is built* belong in `ProviderProfile`; concerns
    that shape *how the agent runs* belong here.

    Example:
        Append a model-specific system-prompt suffix:

        ```python
        from deepagents import HarnessProfile, register_harness_profile

        register_harness_profile(
            "openai:gpt-5.4",
            HarnessProfile(system_prompt_suffix="Think step by step."),
        )
        ```
    """

    base_system_prompt: str | None = None
    """When set, completely replaces `BASE_AGENT_PROMPT` as the base prompt.

    `None` means use `BASE_AGENT_PROMPT` unchanged.

    If both `base_system_prompt` and `system_prompt_suffix` are set, the
    suffix is appended to this custom base.
    """

    system_prompt_suffix: str | None = None
    """Text appended to the assembled base system prompt.

    The suffix is appended to either `BASE_AGENT_PROMPT` or the profile's
    `base_system_prompt` when set. `None` means no suffix.
    """

    tool_description_overrides: dict[str, str] = field(default_factory=dict)
    """Per-tool description replacements keyed by tool name.

    Applied only where Deep Agents has a stable description hook: built-in
    filesystem tools, the `task` tool, and user-supplied `BaseTool` or dict
    tools. Plain callable tools are left unchanged.

    !!! warning

        Keys are matched by tool name string. If a built-in tool is renamed
        or removed, stale keys silently become no-ops with no error. Keep
        overrides minimal and verify against the current tool names.
    """

    excluded_tools: frozenset[str] = frozenset()
    """Tool names to remove from the tool set for this profile.

    Applied via a tool-exclusion middleware after tool-injecting middleware
    has run, so it can remove both user-supplied tools and tools added by
    Deep Agents middleware from the visible tool set.

    When profiles are merged, exclusions are additive rather than replacing
    each other. For example, if a provider profile excludes `execute` and an
    exact-model profile excludes `grep`, the resolved profile excludes both
    tools.
    """

    extra_middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]] = ()
    """Middleware appended to every runtime middleware stack.

    Applied to the main agent, the auto-added `general-purpose` subagent, and
    declarative synchronous subagents created from `SubAgent` specs.

    May be a static sequence or a zero-arg factory that returns one. Use a
    factory when middleware instances should not be shared across stacks.
    """

    general_purpose_subagent: GeneralPurposeSubagentProfile | None = None
    """Edits for the auto-added general-purpose subagent.

    Set `enabled=False` to remove the default `general-purpose` subagent
    entirely.
    """


_HARNESS_PROFILES: dict[str, HarnessProfile] = {}
"""Internal registry mapping harness-profile keys to `HarnessProfile` instances.

Keys are either a full `provider:model` spec for per-model overrides or a
bare provider name for provider-wide defaults. Lookup order is exact spec,
then provider prefix, then an empty default profile.
"""


def register_harness_profile(key: str, profile: HarnessProfile) -> None:
    """Register a `HarnessProfile` for a provider or specific model.

    !!! beta

        `register_harness_profile` is a beta API. It is safe for production
        use, but may receive minor changes in future releases.

    Registrations are **additive**: if a profile is already registered under
    `key` (including a built-in profile loaded at import time), the new
    profile is merged on top via `_merge_profiles` rather than replacing it.
    The incoming profile's fields win on conflicts; unspecified fields inherit
    from the existing profile. Excluded-tool sets union, middleware sequences
    merge by type, and `general_purpose_subagent` settings merge field-wise.

    To layer onto a built-in, register under the same key:

    ```python
    from deepagents import HarnessProfile, register_harness_profile

    # Adds a system-prompt suffix alongside any built-in harness defaults.
    register_harness_profile(
        "openai:gpt-5.4",
        HarnessProfile(system_prompt_suffix="Respond in under 100 words."),
    )
    ```

    Args:
        key: A provider name like `"openai"` for provider-wide defaults, or a
            full `provider:model` spec like `"openai:gpt-5.4"` for a
            per-model override.
        profile: The harness profile to register.
    """
    existing = _HARNESS_PROFILES.get(key)
    if existing is not None:
        profile = _merge_profiles(existing, profile)
    _HARNESS_PROFILES[key] = profile


def _get_harness_profile(spec: str) -> HarnessProfile:
    """Look up the `HarnessProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec`.
    2. Provider prefix (everything before the first `:`).
    3. A default empty `HarnessProfile`.

    When both an exact-model profile and a provider-level profile exist, they
    are merged field-by-field. Unset model-level fields inherit provider
    defaults, while explicit model-level overrides still replace or augment
    provider settings according to each field's merge semantics.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

    Returns:
        The matching `HarnessProfile`, or an empty default.
    """
    exact = _HARNESS_PROFILES.get(spec)

    provider, sep, _ = spec.partition(":")
    base = _HARNESS_PROFILES.get(provider) if sep else None

    if exact is not None and base is not None:
        return _merge_profiles(base, exact)
    if exact is not None:
        return exact
    if base is not None:
        return base
    return HarnessProfile()


def _resolve_middleware_seq(
    middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
) -> Sequence[AgentMiddleware]:
    """Resolve middleware to a concrete sequence, calling the factory if needed."""
    if callable(middleware):
        return middleware()  # ty: ignore[call-top-callable]  # Callable & Sequence union confuses ty
    return middleware


def _merge_middleware(
    base_middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
    override_middleware: Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]],
) -> Sequence[AgentMiddleware] | Callable[[], Sequence[AgentMiddleware]]:
    """Merge two middleware sequences by type.

    If the override supplies middleware whose type already exists in the base,
    the override instance replaces it in place and preserves the original
    position. Novel override middleware is appended.

    Args:
        base_middleware: Base middleware sequence with lower priority.
        override_middleware: Override middleware sequence with higher priority.

    Returns:
        A merged middleware sequence or factory.
    """
    if not base_middleware or not override_middleware:
        return override_middleware or base_middleware

    def factory() -> Sequence[AgentMiddleware]:
        base_seq = _resolve_middleware_seq(base_middleware)
        override_seq = _resolve_middleware_seq(override_middleware)
        override_by_type: dict[type, AgentMiddleware] = {type(m): m for m in override_seq}
        merged: list[AgentMiddleware] = []
        seen: set[type] = set()
        for middleware in base_seq:
            middleware_type = type(middleware)
            if middleware_type in override_by_type:
                merged.append(override_by_type[middleware_type])
                seen.add(middleware_type)
            else:
                merged.append(middleware)
        merged.extend(m for m in override_seq if type(m) not in seen)
        return merged

    return factory


def _merge_general_purpose_subagent_profiles(
    base: GeneralPurposeSubagentProfile | None,
    override: GeneralPurposeSubagentProfile | None,
) -> GeneralPurposeSubagentProfile | None:
    """Merge two general-purpose subagent profiles."""
    if base is None:
        return override
    if override is None:
        return base
    return GeneralPurposeSubagentProfile(
        enabled=override.enabled if override.enabled is not None else base.enabled,
        description=override.description if override.description is not None else base.description,
        system_prompt=override.system_prompt if override.system_prompt is not None else base.system_prompt,
    )


def _merge_profiles(base: HarnessProfile, override: HarnessProfile) -> HarnessProfile:
    """Merge two harness profiles, layering `override` on top of `base`.

    Scalar fields such as prompts use the override value when set, otherwise
    fall back to the base. For example, if the provider sets
    `system_prompt_suffix="Use tools when helpful"` and the exact-model
    profile leaves `system_prompt_suffix=None`, the merged profile keeps the
    provider suffix.

    Tool-description mappings merge with the override winning per key. For
    example, a provider profile can override `"task"` while an exact-model
    profile overrides `"ls"`, and the merged profile keeps both overrides; if
    both define `"task"`, the exact-model value wins.

    Excluded-tool sets are unioned. For example, `{"execute"}` plus
    `{"grep"}` becomes `{"execute", "grep"}` in the merged profile.

    Middleware sequences are merged by type via `_merge_middleware`. For
    example, if both profiles provide a middleware of the same class, the
    override instance replaces the base instance in the same position, while
    novel middleware classes from the override are appended.

    General-purpose subagent settings are merged fieldwise so model-level
    tweaks can inherit provider defaults. For example, a provider profile can
    set the default subagent description while an exact-model profile only
    overrides its system prompt; the merged profile keeps both.

    Args:
        base: Lower-priority profile, typically from the provider.
        override: Higher-priority profile, typically from the exact model.

    Returns:
        A merged `HarnessProfile`.
    """
    return HarnessProfile(
        base_system_prompt=(override.base_system_prompt if override.base_system_prompt is not None else base.base_system_prompt),
        system_prompt_suffix=(override.system_prompt_suffix if override.system_prompt_suffix is not None else base.system_prompt_suffix),
        tool_description_overrides={
            **base.tool_description_overrides,
            **override.tool_description_overrides,
        },
        excluded_tools=base.excluded_tools | override.excluded_tools,
        extra_middleware=_merge_middleware(base.extra_middleware, override.extra_middleware),
        general_purpose_subagent=_merge_general_purpose_subagent_profiles(
            base.general_purpose_subagent,
            override.general_purpose_subagent,
        ),
    )
