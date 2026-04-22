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

Users may register profiles via `register_harness_profile`. Deep Agents
ships no built-in harness profiles; the registry is empty until a caller
registers one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from deepagents.profiles._keys import validate_profile_key

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneralPurposeSubagentProfile:
    """Edits applied to the auto-added `general-purpose` subagent.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    These settings only affect the default subagent that `create_deep_agent`
    inserts when the caller does not explicitly provide a subagent named
    `general-purpose`.
    """

    enabled: bool | None = None
    """Whether to auto-add the default general-purpose subagent.

    `None` means inherit from a base profile when merging, or fall back to
    the default of including the subagent. `True` forces inclusion and is
    what a model-level profile can use to re-enable a subagent that a
    provider-level profile disabled. `False` disables the auto-added
    subagent entirely.

    !!! note

        If the default subagent is disabled and no other synchronous subagents are
        configured, the main agent will not expose the `task` tool.
    """

    description: str | None = None
    """Override for the default subagent description.

    `None` means keep the default description.
    """

    system_prompt: str | None = None
    """Override for the default subagent system prompt.

    `None` means keep the default system prompt.
    """


@dataclass(frozen=True)
class HarnessProfile:
    """Declarative configuration for deep agent runtime behavior.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    A `HarnessProfile` describes prompt-assembly, tool-visibility, middleware,
    and default-subagent adjustments applied by `create_deep_agent` once a
    chat model has been constructed. Profiles are registered via
    `register_harness_profile` under a provider key (`"openai"`) or a full
    `provider:model` key (`"openai:gpt-5.4"`).

    This complements `ProviderProfile`, which controls the model-construction
    phase (e.g. `init_chat_model` kwargs, pre-init side effects). Concerns
    that shape *how the model is built* belong in `ProviderProfile`; concerns
    that shape *how the agent runs* belong here.

    The `extra_middleware` field expects
    `langchain.agents.middleware.types.AgentMiddleware` instances or a
    factory returning a sequence of them.

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

    tool_description_overrides: Mapping[str, str] = field(default_factory=dict)
    """Per-tool description replacements keyed by tool name.

    Applied only where Deep Agents has a stable description hook: built-in
    filesystem tools, the `task` tool, and user-supplied `BaseTool` or dict
    tools. Plain callable tools are left unchanged.

    Once a profile is constructed, its overrides can be read but not
    rewritten — for example, `profile.tool_description_overrides["ls"] =
    "new"` raises `TypeError`. The registry stores its own defensive copy,
    so mutating the dict you passed into the constructor after the fact
    won't affect the registered profile either. To change a registered
    profile's overrides, re-register (which merges on top) or construct a
    new profile.

    !!! warning

        Keys are matched by tool name string. If a built-in tool is renamed
        or removed, stale keys silently become no-ops with no error. Keep
        overrides minimal and verify against the current tool names.

    !!! warning "Overriding task tool description"

        The `task` tool's default description contains an `{available_agents}`
        format placeholder that `SubAgentMiddleware` replaces at build time
        with the registered subagent name/description list. If your
        override string does not include `{available_agents}`, the final
        description is used as-is and the model will not see which
        subagents exist — making the tool much less useful. Include the
        placeholder in any `"task"` override, e.g.
        `"My custom instructions.\\n\\n{available_agents}"`.
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
    declarative synchronous subagents created from `SubAgent` specs —
    i.e., the stacks that `create_deep_agent` assembles itself.

    *Not* applied to `CompiledSubAgent` runnables or `AsyncSubAgent` entries.
    A `CompiledSubAgent` is passed in pre-built (its `runnable` is already a
    compiled graph with its own middleware chain), so `create_deep_agent` has
    nothing to append to. An `AsyncSubAgent` runs out-of-process against a
    remote deployment and its middleware is configured on that remote graph,
    not here. In both cases, injecting local middleware would either fail
    silently or violate the caller's explicit configuration.

    May be a static sequence or a zero-arg factory that returns one. Use a
    factory when middleware instances should not be shared across stacks.
    """

    general_purpose_subagent: GeneralPurposeSubagentProfile | None = None
    """Edits for the auto-added general-purpose subagent.

    Set `enabled=False` to remove the default `general-purpose` subagent
    entirely.
    """

    def __post_init__(self) -> None:
        """Freeze mutable container fields to prevent post-construction mutation.

        `@dataclass(frozen=True)` only prevents rebinding attributes; it does
        not prevent mutating the contents of a mutable value. Without this
        hook, both of the following would silently alter a registered
        profile after the fact:

        ```python
        shared = {"ls": "original"}
        profile = HarnessProfile(tool_description_overrides=shared)
        register_harness_profile("openai", profile)

        shared["ls"] = "mutated"  # via external alias
        profile.tool_description_overrides["ls"] = "x"  # via direct write
        ```

        This method defensively copies `tool_description_overrides` into a
        fresh dict wrapped in `MappingProxyType` — a read-only view — so both
        scenarios become errors: the first because the registry holds its own
        copy independent of `shared`, and the second because item assignment
        on a `MappingProxyType` raises `TypeError`.

        `extra_middleware` receives the same treatment when supplied as a
        sequence: the contents are copied into a tuple so a caller who retains
        a reference to the original list cannot extend the registered profile
        after the fact. A callable factory is stored as-is since its output is
        resolved at each lookup.
        """
        if not isinstance(self.tool_description_overrides, MappingProxyType):
            object.__setattr__(
                self,
                "tool_description_overrides",
                MappingProxyType(dict(self.tool_description_overrides)),
            )
        extra = self.extra_middleware
        if not callable(extra) and not isinstance(extra, tuple):
            object.__setattr__(self, "extra_middleware", tuple(extra))


_HARNESS_PROFILES: dict[str, HarnessProfile] = {}
"""Internal registry mapping harness-profile keys to `HarnessProfile` instances.

Keys are either a full `provider:model` spec for per-model overrides or a
bare provider name for provider-wide defaults. Lookup order is exact spec,
then provider prefix, then no match (returns `None`).
"""


def register_harness_profile(key: str, profile: HarnessProfile) -> None:
    """Register a `HarnessProfile` for a provider or specific model.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    Deep Agents ships no built-in harness profiles, so the first call under
    `key` acts as a fresh registration. Subsequent calls are **additive**: the
    new profile is merged on top of the existing registration rather than
    replacing it. The incoming profile's fields win on conflicts; unspecified
    fields inherit from the existing profile. Excluded-tool sets union,
    middleware sequences merge by type, and `general_purpose_subagent`
    settings merge field-wise.

    To extend an existing registration, call `register_harness_profile` again
    under the same key:

    ```python
    from deepagents import HarnessProfile, register_harness_profile

    # Layer a system-prompt suffix on top of the previous registration.
    register_harness_profile(
        "openai:gpt-5.4",
        HarnessProfile(system_prompt_suffix="Respond in under 100 words."),
    )
    ```

    Args:
        key: Either a provider name (no colon) for provider-wide defaults,
            or a full `provider:model` spec for a per-model override. Valid
            shapes:

            - `"openai"` — provider-wide
            - `"openai:gpt-5.4"` — specific model
        profile: The harness profile to register.

    Raises:
        ValueError: If `key` is empty, contains more than one `:`, or has an
            empty provider/model half.
    """
    validate_profile_key(key)
    existing = _HARNESS_PROFILES.get(key)
    if existing is not None:
        profile = _merge_profiles(existing, profile)
    _HARNESS_PROFILES[key] = profile


def _has_any_harness_profile() -> bool:
    """Return `True` when a user has registered any harness profile.

    Narrow helper for modules (e.g. `graph.py`) that need to adjust logging
    verbosity based on whether the user has registered any harness profile.
    Built-in registrations loaded by `_ensure_builtin_profiles_loaded` are
    excluded — with only built-ins in play, a "no match" miss against a
    non-matching provider is unsurprising and should stay at debug.

    Exists so callers do not have to import the private `_HARNESS_PROFILES`
    registry directly.
    """
    from deepagents.profiles._builtin_profiles import _BUILTIN_HARNESS_KEYS  # noqa: PLC0415

    return bool(_HARNESS_PROFILES.keys() - _BUILTIN_HARNESS_KEYS)


def _get_harness_profile(spec: str) -> HarnessProfile | None:
    """Look up the `HarnessProfile` for a model spec.

    Resolution order:

    1. Exact match on `spec`.
    2. Provider prefix (everything before the first `:`), when `spec`
        contains a colon and both halves are non-empty.
    3. `None` when neither matches.

    When both an exact-model profile and a provider-level profile exist, they
    are merged field-by-field. Unset model-level fields inherit provider
    defaults, while explicit model-level overrides still replace or augment
    provider settings according to each field's merge semantics.

    When only the provider-level profile matches, a debug breadcrumb is
    emitted so registrations layered on an exact key can be traced when they
    don't apply (e.g. typo'd specs falling through to the provider default).

    Malformed specs (empty string, more than one `:`, or a `:` with an empty
    provider/model half) return `None` without consulting the registry. This
    prevents a spec like `"openai:"` from silently matching the provider-wide
    `"openai"` registration.

    Args:
        spec: Model spec in `provider:model` format, or a bare provider/model
            identifier.

    Returns:
        The matching `HarnessProfile`, or `None` when no registered profile matches.
    """
    if not spec or spec.count(":") > 1:
        return None

    provider, sep, model = spec.partition(":")
    if sep and (not provider or not model):
        return None

    exact = _HARNESS_PROFILES.get(spec)
    base = _HARNESS_PROFILES.get(provider) if sep else None

    if exact is not None and base is not None:
        return _merge_profiles(base, exact)
    if exact is not None:
        return exact
    if base is not None:
        logger.debug(
            "No exact HarnessProfile for %r; using provider %r profile.",
            spec,
            provider,
        )
        return base
    return None


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

    Middleware stacks have at most one instance of each concrete class, so
    the merge treats the class as the identity. When the override has an
    instance whose class already appears in the base, the override instance
    replaces the base instance *at the same position*; the rest of the
    base ordering is preserved. Classes that appear only in the override
    are appended at the end in override order.

    Example:
        Given base `[A, B]` and override `[A_new, C]` where `A_new` is a
        second instance of the same class as `A`:

        - `A_new` replaces `A` at position 0.
        - `B` is kept at position 1.
        - `C` is appended at the end.

        Merged result: `[A_new, B, C]`.

    Edge case — duplicates within the base:
        If the base somehow contains more than one instance of the same
        class (an unusual configuration), only the first occurrence is
        replaced; later duplicates are dropped. For example, base
        `[A1, A2]` + override `[A_new]` merges to `[A_new]`, not
        `[A_new, A_new]`. This mirrors the intent of "replace in place"
        rather than "insert once per base match".

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
        replaced: set[type] = set()
        for entry in base_seq:
            entry_type = type(entry)
            if entry_type in override_by_type:
                if entry_type not in replaced:
                    merged.append(override_by_type[entry_type])
                    replaced.add(entry_type)
                # Drop subsequent base duplicates so the override isn't inserted twice.
            else:
                merged.append(entry)
        merged.extend(m for m in override_seq if type(m) not in replaced)
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

    Single-value fields such as `base_system_prompt` and `system_prompt_suffix`
    use the override value when the override has set it, otherwise fall back
    to the base. For example, if the provider sets
    `system_prompt_suffix="Use tools when helpful"` and the exact-model
    profile leaves `system_prompt_suffix=None`, the merged profile keeps the
    provider suffix.

    Tool-description mappings merge with the override winning per key. For
    example, a provider profile can override `"task"` while an exact-model
    profile overrides `"ls"`, and the merged profile keeps both overrides; if
    both define `"task"`, the exact-model value wins.

    Excluded-tool sets are unioned. For example, `{"execute"}` plus
    `{"grep"}` becomes `{"execute", "grep"}` in the merged profile.

    Middleware sequences are merged by type (see `_merge_middleware`). For
    example, if both profiles provide a middleware of the same class, the
    override instance replaces the base instance in the same position, while
    novel middleware classes from the override are appended.

    `general_purpose_subagent` fields merge one at a time so model-level
    tweaks can inherit provider defaults: whichever side explicitly sets a
    field wins, and unset fields (left as `None`) fall back to the other
    side. This means a model-level `enabled=True` can re-enable a subagent
    that a provider-level profile disabled with `enabled=False`, and vice
    versa.

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
