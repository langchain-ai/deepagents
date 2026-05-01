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
ships built-in harness profiles for several frontier model specs.
They may be layered on top of via additive merge semantics.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from deepagents.profiles._keys import validate_profile_key

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def _scaffolding_violation_label(entry: object) -> str | None:
    """Return a violation label for `entry` when it names required scaffolding.

    Class entries are matched on `__name__`; string entries are matched
    directly. Returns `None` when `entry` does not name scaffolding so
    callers can collect violations across an entire `excluded_middleware`
    set and surface them in one error.

    The required-scaffolding set is owned by `deepagents.graph` and imported
    lazily to avoid a top-level cycle (`graph` already imports this module).
    By the time any `HarnessProfile`/`HarnessProfileConfig` is constructed,
    `graph` is loadable: the package `__init__` imports `graph` before
    `profiles`, and built-in profile registration runs lazily — well after
    module load.
    """
    from deepagents.graph import _REQUIRED_MIDDLEWARE_NAMES  # noqa: PLC0415

    if isinstance(entry, str):
        if entry in _REQUIRED_MIDDLEWARE_NAMES:
            return f"{entry!r} (string)"
        return None
    if isinstance(entry, type) and entry.__name__ in _REQUIRED_MIDDLEWARE_NAMES:
        return entry.__name__
    return None


def _format_scaffolding_rejection(violations: list[str]) -> str:
    """Format the construction-time scaffolding-rejection error message.

    Mirrors the assembly-time message from
    `_validate_excluded_middleware_config` so users see the same wording
    regardless of where the rejection fires.
    """
    labels = sorted(set(violations))
    return (
        "HarnessProfile.excluded_middleware is invalid:\n  - "
        f"required scaffolding cannot be excluded: {', '.join(labels)} "
        "(back filesystem tools, subagent dispatch, and permission "
        "enforcement — use excluded_tools for per-tool visibility or "
        "adjust profile settings instead of stripping scaffolding)"
    )


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
    """Whether to auto-add the default general-purpose subagent (three-state:
    `None` inherits / defaults on, `True` forces inclusion, `False` disables).

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
    """Override for the default general-purpose subagent system prompt.

    `None` means keep the default system prompt.

    !!! note "Precedence vs `HarnessProfile.base_system_prompt`"

        When a profile sets *both* this field and
        `HarnessProfile.base_system_prompt`, this field wins for the
        general-purpose subagent.

        The reasoning: `general_purpose_subagent.system_prompt` is GP-specific
        configuration, while `base_system_prompt` is a global override that
        applies to the main agent. The more-specific intent wins on the GP
        subagent so a user setting both never sees their GP override
        silently dropped. The profile's `system_prompt_suffix` still
        layers on top.
    """

    def to_dict(self) -> dict[str, Any]:
        """Dump this sub-profile to a plain dict.

        Only fields with non-`None` values are emitted so the serialized form
        round-trips cleanly without forcing `None` defaults into the config.

        Returns:
            A plain dict with at most `enabled`, `description`, and
                `system_prompt` keys.
        """
        out: dict[str, Any] = {}
        if self.enabled is not None:
            out["enabled"] = self.enabled
        if self.description is not None:
            out["description"] = self.description
        if self.system_prompt is not None:
            out["system_prompt"] = self.system_prompt
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneralPurposeSubagentProfile:
        """Construct a sub-profile from a plain dict.

        Args:
            data: Mapping with any subset of `enabled`, `description`, and
                `system_prompt` keys.

        Returns:
            A new `GeneralPurposeSubagentProfile`.

        Raises:
            TypeError: If `data` contains unknown keys, or if any value has
                the wrong type.
        """
        unknown = set(data.keys()) - _GENERAL_PURPOSE_SUBAGENT_KEYS
        if unknown:
            msg = f"Unknown keys in GeneralPurposeSubagentProfile dict: {sorted(unknown)}"
            raise TypeError(msg)
        enabled = data.get("enabled")
        description = data.get("description")
        system_prompt = data.get("system_prompt")
        if enabled is not None and not isinstance(enabled, bool):
            msg = f"`enabled` must be bool or None, got {type(enabled).__name__}"
            raise TypeError(msg)
        if description is not None and not isinstance(description, str):
            msg = f"`description` must be str or None, got {type(description).__name__}"
            raise TypeError(msg)
        if system_prompt is not None and not isinstance(system_prompt, str):
            msg = f"`system_prompt` must be str or None, got {type(system_prompt).__name__}"
            raise TypeError(msg)
        return cls(enabled=enabled, description=description, system_prompt=system_prompt)


@dataclass(frozen=True)
class HarnessProfileConfig:
    """Declarative harness-profile config for YAML/JSON-backed profiles.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    A `HarnessProfileConfig` contains the file-friendly subset of harness
    settings: plain strings, bools, lists, and nested dicts that can be loaded
    from YAML or JSON. For in-code/runtime-only adjustments such as
    `extra_middleware` or class-form `excluded_middleware`, use
    `HarnessProfile` instead.

    !!! note "Class-path serialization"

        `excluded_middleware` in config files currently only accepts plain
        middleware-name strings matched against `AgentMiddleware.name`. A
        future revision may add explicit class-path (`module:Class`) entries
        for excluding middleware whose class isn't part of the public import
        surface; until then, exclude such middleware via its `.name` (using
        `serialized_name` for stable public aliases) or stay on the runtime
        `HarnessProfile` and pass the class directly.

    Config objects may be passed directly to `register_harness_profile`; the
    helper converts them to runtime `HarnessProfile` objects automatically.

    Example:
        Construct a config object directly in Python:

        ```python
        from deepagents import HarnessProfileConfig, register_harness_profile

        register_harness_profile(
            "openai:gpt-5.4",
            HarnessProfileConfig(
                system_prompt_suffix="Think step by step.",
                excluded_middleware={"SummarizationMiddleware"},
            ),
        )
        ```

        Or load the equivalent declarative form from YAML:

        ```yaml
        # openai-gpt-5.4.yaml
        system_prompt_suffix: Think step by step.
        excluded_middleware:
          - SummarizationMiddleware
        ```

        ```python
        import yaml
        from deepagents import HarnessProfileConfig, register_harness_profile

        with open("openai-gpt-5.4.yaml") as f:
            register_harness_profile(
                "openai:gpt-5.4",
                HarnessProfileConfig.from_dict(yaml.safe_load(f)),
            )
        ```
    """

    base_system_prompt: str | None = None
    """`CUSTOM` slot in the prompt assembly order — completely replaces
    `BASE_AGENT_PROMPT` as the base prompt when set.

    `None` (the default) means use `BASE_AGENT_PROMPT` unchanged.

    If both `base_system_prompt` and `system_prompt_suffix` are set, the
    suffix is appended to this custom base. A caller-supplied
    `system_prompt=` is still placed before this base — see
    `create_deep_agent`'s `system_prompt` parameter or
    [Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)
    for the full assembly order.
    """

    system_prompt_suffix: str | None = None
    """`SUFFIX` slot in the prompt assembly order — text appended to
    the assembled base system prompt.

    Always sits last (after `BASE` or `CUSTOM`) so model-tuning guidance
    lands closest to the conversation history. `None` (the default)
    means no suffix.
    """

    tool_description_overrides: Mapping[str, str] = field(default_factory=dict)
    """Per-tool description replacements keyed by tool name."""

    excluded_tools: frozenset[str] = frozenset()
    """Tool names to remove from the tool set for this profile."""

    excluded_middleware: frozenset[str] = frozenset()
    """Middleware names to strip from every stack this profile applies to.

    Strings match `AgentMiddleware.name` exactly. Entries are grammar-checked
    at construction: empty/whitespace strings, colon-containing strings, and
    underscore-prefixed names all raise `ValueError` immediately.

    Class-path (`module:Class`) entries are not currently supported and may
    be added in a future revision. Until then, expose a stable public alias
    via `serialized_name` on the middleware class and exclude by that alias.

    This is the canonical on-disk representation used by `to_dict` /
    `from_dict`.
    """

    general_purpose_subagent: GeneralPurposeSubagentProfile | None = None
    """Edits for the auto-added `general-purpose` subagent.

    Unset is equivalent to passing a default-constructed
    `GeneralPurposeSubagentProfile()` — the auto-added subagent runs with
    its stock description and prompt.
    """

    def __post_init__(self) -> None:
        """Freeze mutable mappings and validate grammar of string entries."""
        if not isinstance(self.tool_description_overrides, MappingProxyType):
            object.__setattr__(
                self,
                "tool_description_overrides",
                MappingProxyType(dict(self.tool_description_overrides)),
            )
        scaffolding_violations: list[str] = []
        for entry in self.excluded_middleware:
            _validate_config_middleware_string(entry, "excluded_middleware")
            label = _scaffolding_violation_label(entry)
            if label is not None:
                scaffolding_violations.append(label)
        if scaffolding_violations:
            raise ValueError(_format_scaffolding_rejection(scaffolding_violations))

    def to_dict(self) -> dict[str, Any]:
        """Dump this config to plain dict/list/scalar values.

        Suitable for `json.dumps` or `yaml.safe_dump`. Fields at their
        default are omitted so the output stays minimal and round-trips
        cleanly through `from_dict`.

        Returns:
            A plain dict containing only the fields set on this config.

        Raises:
            TypeError: If `tool_description_overrides` contains a non-string
                key or value.
        """
        out: dict[str, Any] = {}
        if self.base_system_prompt is not None:
            out["base_system_prompt"] = self.base_system_prompt
        if self.system_prompt_suffix is not None:
            out["system_prompt_suffix"] = self.system_prompt_suffix
        if self.tool_description_overrides:
            out["tool_description_overrides"] = _coerce_str_mapping(
                dict(self.tool_description_overrides),
                "tool_description_overrides",
            )
        if self.excluded_tools:
            out["excluded_tools"] = sorted(self.excluded_tools)
        if self.excluded_middleware:
            out["excluded_middleware"] = sorted(self.excluded_middleware)
        if self.general_purpose_subagent is not None:
            # Emit the key even when the sub-profile has no fields set so
            # `from_dict(to_dict(c))` preserves the "explicit empty sub-profile"
            # vs. "no sub-profile" distinction.
            out["general_purpose_subagent"] = self.general_purpose_subagent.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessProfileConfig:
        """Construct a config object from a plain dict.

        Args:
            data: A mapping with any subset of the serializable
                `HarnessProfileConfig` fields. Unknown keys raise `TypeError`.

        Returns:
            A new `HarnessProfileConfig` populated from `data`.

        Raises:
            TypeError: If `data` contains unknown keys or fields of the wrong
                shape.
            ValueError: If any `excluded_middleware` entry violates the
                grammar rules enforced in `__post_init__` (empty/whitespace
                strings, class-path `module:Class` entries, or
                underscore-prefixed names), or names required scaffolding
                middleware.
        """
        unknown = set(data.keys()) - _HARNESS_PROFILE_CONFIG_KEYS
        if unknown:
            msg = f"Unknown keys in HarnessProfileConfig dict: {sorted(unknown)}"
            raise TypeError(msg)
        return cls(
            base_system_prompt=_coerce_str_or_none(data.get("base_system_prompt"), "base_system_prompt"),
            system_prompt_suffix=_coerce_str_or_none(data.get("system_prompt_suffix"), "system_prompt_suffix"),
            tool_description_overrides=_coerce_str_mapping(data.get("tool_description_overrides"), "tool_description_overrides"),
            excluded_tools=_coerce_frozen_strset(data.get("excluded_tools"), "excluded_tools"),
            excluded_middleware=_coerce_frozen_strset(data.get("excluded_middleware"), "excluded_middleware"),
            general_purpose_subagent=_coerce_general_purpose_subagent(data.get("general_purpose_subagent")),
        )

    def to_harness_profile(self) -> HarnessProfile:
        """Convert this declarative config into a runtime `HarnessProfile`.

        `excluded_middleware` entries are passed through as name-based
        exclusions matched against `AgentMiddleware.name`.

        !!! note "Intentional asymmetry with `from_harness_profile`"

            This direction is currently lossless because `HarnessProfileConfig`
            carries only the file-friendly subset of fields by design —
            `extra_middleware` instances and factories cannot be
            represented in YAML/JSON, so they are absent from this type
            and the resulting `HarnessProfile` will not have them
            populated. The reverse direction (`from_harness_profile`)
            *raises* when a runtime profile contains runtime-only state,
            rather than silently dropping it. Round-tripping a runtime
            profile that uses `extra_middleware` through config form is
            therefore not supported by design at this moment — keep
            such profiles in `HarnessProfile` form.

        Returns:
            A runtime `HarnessProfile` with equivalent declarative settings.
        """
        return HarnessProfile(
            base_system_prompt=self.base_system_prompt,
            system_prompt_suffix=self.system_prompt_suffix,
            tool_description_overrides=self.tool_description_overrides,
            excluded_tools=self.excluded_tools,
            excluded_middleware=frozenset(self.excluded_middleware),
            general_purpose_subagent=self.general_purpose_subagent,
        )

    @classmethod
    def from_harness_profile(cls, profile: HarnessProfile) -> HarnessProfileConfig:
        """Export a runtime `HarnessProfile` back to declarative config.

        String-form `excluded_middleware` entries are preserved as-is.
        Class-form entries are only serializable when the class advertises a
        public `serialized_name` alias; arbitrary class-path serialization is
        not currently supported.

        Args:
            profile: Runtime profile to export.

        Returns:
            A declarative `HarnessProfileConfig`.

        Raises:
            ValueError: If `profile` contains runtime-only state such as
                non-empty `extra_middleware`, or if a class-form
                `excluded_middleware` entry has no `serialized_name` alias.
            TypeError: If `tool_description_overrides` contains a non-string
                key or value.
        """
        extra = profile.extra_middleware
        if callable(extra) or (isinstance(extra, tuple) and extra):
            msg = (
                "HarnessProfileConfig.from_harness_profile() cannot export "
                "`extra_middleware`. Middleware instances and factories are "
                "runtime-only; keep them in `HarnessProfile`."
            )
            raise ValueError(msg)

        return cls(
            base_system_prompt=profile.base_system_prompt,
            system_prompt_suffix=profile.system_prompt_suffix,
            tool_description_overrides=_coerce_str_mapping(
                dict(profile.tool_description_overrides),
                "tool_description_overrides",
            ),
            excluded_tools=profile.excluded_tools,
            excluded_middleware=frozenset(_serialize_runtime_excluded_middleware_entry(entry) for entry in profile.excluded_middleware),
            general_purpose_subagent=profile.general_purpose_subagent,
        )


@dataclass(frozen=True)
class HarnessProfile:
    """Runtime configuration for deep agent behavior.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    A `HarnessProfile` describes prompt-assembly, tool visibility, middleware,
    and default-subagent adjustments applied by `create_deep_agent` once a
    chat model has been constructed. Profiles are registered via
    `register_harness_profile` under a provider key (`"openai"`) or a full
    `provider:model` key (`"openai:gpt-5.4"`).

    This complements `ProviderProfile`, which controls the model-construction
    phase (e.g. `init_chat_model` kwargs, pre-init side effects). Concerns
    that shape *how the model is built* belong in `ProviderProfile`; concerns
    that shape *how the agent runs* belong here.

    For YAML/JSON-backed profiles, use `HarnessProfileConfig`, which contains
    only the declarative subset and can be passed directly to
    `register_harness_profile`.

    The `extra_middleware` field expects
    `langchain.agents.middleware.types.AgentMiddleware` instances or a
    factory returning a sequence of them.

    Example:
        Minimal — append a model-specific system-prompt suffix:

        ```python
        from deepagents import HarnessProfile, register_harness_profile

        register_harness_profile(
            "openai:gpt-5.4",
            HarnessProfile(system_prompt_suffix="Think step by step."),
        )
        ```

        Richer — combine prompt tuning, tool exclusion, and a tweak to the
        auto-added general-purpose subagent:

        ```python
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )

        register_harness_profile(
            "openai:gpt-5.4",
            HarnessProfile(
                system_prompt_suffix="Respond in under 100 words.",
                excluded_tools=frozenset({"execute"}),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )
        ```
    """

    base_system_prompt: str | None = None
    """`CUSTOM` slot in the prompt assembly order — completely replaces
    `BASE_AGENT_PROMPT` as the base prompt when set.

    `None` (the default) means use `BASE_AGENT_PROMPT` unchanged.

    If both `base_system_prompt` and `system_prompt_suffix` are set, the
    suffix is appended to this custom base. A caller-supplied
    `system_prompt=` is still placed before this base — see
    `create_deep_agent`'s `system_prompt` parameter or
    [Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)
    for the full assembly order.

    Most profiles only set `system_prompt_suffix` to layer model-tuning
    guidance on top of the SDK base.
    """

    system_prompt_suffix: str | None = None
    """`SUFFIX` slot in the prompt assembly order — text appended to
    the assembled base system prompt.

    Always sits last (after `BASE` or `CUSTOM`) so model-tuning guidance
    lands closest to the conversation history. `None` (the default)
    means no suffix.

    Applied uniformly to every assembled stack that consults this
    profile: the main agent, declarative subagents whose model resolves
    to this profile, and the auto-added general-purpose subagent. Each
    stack receives the suffix on top of its own base prompt
    (`BASE_AGENT_PROMPT`, the subagent's authored prompt, and the GP
    base respectively).

    See `create_deep_agent`'s `system_prompt` parameter or
    [Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)
    for how `SUFFIX` composes with caller-supplied prompts and
    `base_system_prompt`.
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

    excluded_middleware: frozenset[type[AgentMiddleware] | str] = frozenset()
    """Middleware to strip from every stack this profile applies to.

    Entries may be a middleware *class* (matched by exact type, not subclass —
    consistent with `extra_middleware` slot merging) or a *string* matching
    `AgentMiddleware.name` exactly. `.name` defaults to the class's
    `__name__` but is overridable, so `{TodoListMiddleware}` (class form) and
    `{"TodoListMiddleware"}` (name form) behave identically for stock
    middleware.

    Prefer class form when the class is importable: typos surface at import
    time rather than at agent construction. Reserve string form for
    YAML/JSON-loaded profiles and for middleware whose class isn't part of
    the public import surface (e.g. `"SummarizationMiddleware"` drops the
    private `_DeepAgentsSummarizationMiddleware` via its public alias).

    The filter runs over the fully assembled stack, so exclusions remove
    middleware regardless of which layer added it — including instances
    passed via `create_deep_agent(middleware=[...])`. Merged profiles union
    their exclusion sets; mixed class/string sets are allowed. For config-file
    usage, `HarnessProfileConfig` stores the same exclusions using string names
    only.

    !!! tip "Stable string aliases for private impl classes"

        Middleware whose concrete class name differs from the public alias
        users would type (e.g. `_DeepAgentsSummarizationMiddleware` vs.
        `"SummarizationMiddleware"`) can expose a `serialized_name: ClassVar[str]`
        for stable config-file round-trips. `.name` on the instance returns
        the alias so string-form exclusion matches, and
        `HarnessProfileConfig.from_harness_profile` serializes the class back
        to that alias.

    !!! warning "Restrictions"

        - String grammar is checked at construction: empty, colon-containing,
            and underscore-prefixed names raise `ValueError` immediately.
            Class-path (`module:Class`) entries are not currently supported
            and may be added in a future revision; pass the class itself
            through the runtime `HarnessProfile` instead.
        - Scaffolding classes (`FilesystemMiddleware`, `SubAgentMiddleware`)
            cannot be excluded as class or as their `.name` string. The check
            fires at `HarnessProfile` construction,
            so register-site typos fail fast rather than waiting until
            `create_deep_agent` resolves the profile.
        - Entries that match no middleware in the assembled stack are
            rejected as likely typos or stale profiles.
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
    factory when middleware instances should not be shared across stacks. This
    field is runtime-only and intentionally absent from `HarnessProfileConfig`.
    """

    general_purpose_subagent: GeneralPurposeSubagentProfile | None = None
    """Edits for the auto-added general-purpose subagent.

    Unset (default `None`) is equivalent to passing
    `GeneralPurposeSubagentProfile()` — the auto-added subagent runs with
    its stock description and prompt. Set `enabled=False` on a populated
    sub-profile to remove the default `general-purpose` subagent entirely.
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
        scaffolding_violations: list[str] = []
        for entry in self.excluded_middleware:
            if isinstance(entry, str):
                _validate_config_middleware_string(entry, "excluded_middleware")
            label = _scaffolding_violation_label(entry)
            if label is not None:
                scaffolding_violations.append(label)
        if scaffolding_violations:
            raise ValueError(_format_scaffolding_rejection(scaffolding_violations))

    def materialize_extra_middleware(self) -> list[AgentMiddleware]:
        """Return a fresh list of `extra_middleware`, invoking factory if supplied.

        Each call returns a new list so consumers may mutate freely.
        """
        return list(_resolve_middleware_seq(self.extra_middleware))


def _apply_profile_prompt(profile: HarnessProfile, base_prompt: str) -> str:
    """Apply `profile`'s prompt overlay to `base_prompt`.

    `base_system_prompt` (when set) replaces `base_prompt` outright;
    `system_prompt_suffix` (when set) is appended with a blank-line
    separator. Both are independently optional, mirroring the field
    semantics — a profile that sets only the suffix layers it on top of
    whatever base the caller passes in.

    Used uniformly across the main agent (`base_prompt=BASE_AGENT_PROMPT`),
    declarative subagents (`base_prompt=spec["system_prompt"]`), and the
    auto-added general-purpose subagent (`base_prompt=GP base prompt`), so a
    profile registered under a model spec applies the same overlay regardless
    of which stack the model lands in.
    """
    prompt = profile.base_system_prompt if profile.base_system_prompt is not None else base_prompt
    if profile.system_prompt_suffix is not None:
        prompt = prompt + "\n\n" + profile.system_prompt_suffix
    return prompt


_HARNESS_PROFILE_CONFIG_KEYS: frozenset[str] = frozenset(f.name for f in fields(HarnessProfileConfig))
"""Top-level keys accepted by `HarnessProfileConfig.from_dict`.

Derived from `HarnessProfileConfig`'s dataclass fields so the set stays in
sync automatically. Runtime-only fields such as `extra_middleware` are
absent because they don't exist on `HarnessProfileConfig`.
"""

_GENERAL_PURPOSE_SUBAGENT_KEYS: frozenset[str] = frozenset(f.name for f in fields(GeneralPurposeSubagentProfile))
"""Keys accepted by `GeneralPurposeSubagentProfile.from_dict`.

Derived from the dataclass fields for drift-free parity with the class.
"""


def _coerce_str_or_none(value: object, field_name: str) -> str | None:
    """Validate that `value` is a string or `None` for dict-loaded string fields."""
    if value is None or isinstance(value, str):
        return value
    msg = f"`{field_name}` must be str or None, got {type(value).__name__}"
    raise TypeError(msg)


def _coerce_str_mapping(value: object, field_name: str) -> dict[str, str]:
    """Validate that `value` is a `str -> str` mapping (or `None`) and return a plain dict."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"`{field_name}` must be a mapping, got {type(value).__name__}"
        raise TypeError(msg)
    out: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not isinstance(val, str):
            msg = f"`{field_name}` keys and values must be strings"
            raise TypeError(msg)
        out[key] = val
    return out


def _coerce_frozen_strset(value: object, field_name: str) -> frozenset[str]:
    """Validate that `value` is an iterable of strings (or `None`)."""
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set, frozenset)):
        msg = f"`{field_name}` must be a list/set of strings, got {type(value).__name__}"
        raise TypeError(msg)
    entries: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            msg = f"`{field_name}` entries must be strings, got {type(entry).__name__} ({entry!r})"
            raise TypeError(msg)
        entries.append(entry)
    return frozenset(entries)


def _coerce_general_purpose_subagent(value: object) -> GeneralPurposeSubagentProfile | None:
    """Validate and construct a `GeneralPurposeSubagentProfile` from a dict value."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return GeneralPurposeSubagentProfile.from_dict(cast("Mapping[str, Any]", value))
    msg = f"`general_purpose_subagent` must be a mapping, got {type(value).__name__}"
    raise TypeError(msg)


def _validate_config_middleware_string(entry: object, field_name: str) -> None:
    """Validate grammar of a string `excluded_middleware` entry.

    Runs at `HarnessProfile` / `HarnessProfileConfig` construction so malformed
    entries fail immediately rather than at assembly time. Checks:

    - Entry is a non-empty string.
    - Entry does not contain `:`. Class-path (`module:Class`) entries are
        reserved for a future revision and rejected upfront so config files
        don't accumulate ambiguous shapes.
    - Entry does not start with `_`. Private middleware classes live outside
        the public exclusion surface.

    Scaffolding-class/name rejection and matched-something coverage are
    deliberately NOT checked here — those need the fully assembled middleware
    stack.

    Args:
        entry: Candidate entry. Must be a string to pass.
        field_name: Field being validated, used for error messages.

    Raises:
        TypeError: If `entry` is not a string.
        ValueError: If `entry` violates any of the grammar rules above.
    """
    if not isinstance(entry, str):
        msg = f"`{field_name}` entries must be strings, got {type(entry).__name__} ({entry!r})"
        raise TypeError(msg)
    if not entry or entry.isspace():
        msg = f"`{field_name}` entries must be non-empty, non-whitespace strings"
        raise ValueError(msg)
    if ":" in entry:
        msg = (
            f"`{field_name}` entries must be plain middleware names; class-path (`module:Class`) entries are not currently supported, got {entry!r}."
        )
        raise ValueError(msg)
    if entry.startswith("_"):
        msg = (
            f"`{field_name}` entry {entry!r} cannot start with '_' "
            f"(underscore-prefixed names refer to private middleware classes "
            f"not part of the public exclusion surface)."
        )
        raise ValueError(msg)


def _serialize_runtime_excluded_middleware_entry(
    entry: type[AgentMiddleware] | str,
) -> str:
    """Serialize a runtime `excluded_middleware` entry back to config form.

    Class entries are only serializable when the class advertises a public
    `serialized_name` alias; arbitrary class-path serialization is not
    currently supported.
    """
    if isinstance(entry, str):
        return entry

    alias = getattr(entry, "serialized_name", None)
    if isinstance(alias, str) and alias:
        return alias

    msg = (
        "HarnessProfileConfig.from_harness_profile() cannot serialize "
        f"`excluded_middleware` class {entry.__name__!r}: it has no public "
        "`serialized_name` alias, and arbitrary class-path serialization is "
        "not currently supported. Either add a `serialized_name: ClassVar[str]` "
        "to the class for stable round-trips, or exclude it by `.name` instead."
    )
    raise ValueError(msg)


_HARNESS_PROFILES: dict[str, HarnessProfile] = {}
"""Internal registry mapping harness-profile keys to `HarnessProfile` instances.

Keys are either a full `provider:model` spec for per-model overrides or a
bare provider name for provider-wide defaults. Lookup order is exact spec,
then provider prefix, then no match (returns `None`).
"""


def _ensure_harness_profiles_loaded() -> None:
    """Ensure the lazy built-in/profile-plugin bootstrap has completed."""
    from deepagents.profiles._builtin_profiles import _ensure_builtin_profiles_loaded  # noqa: PLC0415

    _ensure_builtin_profiles_loaded()


def _coerce_runtime_harness_profile(profile: HarnessProfile | HarnessProfileConfig) -> HarnessProfile:
    """Convert declarative config objects to runtime `HarnessProfile` objects."""
    if isinstance(profile, HarnessProfileConfig):
        return profile.to_harness_profile()
    return profile


def _register_harness_profile_impl(key: str, profile: HarnessProfile | HarnessProfileConfig) -> None:
    """Core implementation behind `register_harness_profile`.

    Callers are responsible for any lazy-bootstrap coordination.
    """
    validate_profile_key(key)
    profile = _coerce_runtime_harness_profile(profile)
    existing = _HARNESS_PROFILES.get(key)
    if existing is not None:
        logger.info(
            "Merging HarnessProfile under %r on top of existing registration; set and middleware fields union, scalar fields prefer the new value.",
            key,
        )
        profile = _merge_profiles(existing, profile)
    _HARNESS_PROFILES[key] = profile


def register_harness_profile(key: str, profile: HarnessProfile | HarnessProfileConfig) -> None:
    """Register a harness profile for a provider or specific model.

    !!! beta

        `deepagents.profiles` exposes beta APIs that may receive minor changes in
        future releases. Refer to the [versioning documentation](https://docs.langchain.com/oss/python/versioning)
        for more details.

    Accepts either a runtime `HarnessProfile` or a declarative
    `HarnessProfileConfig`. Config objects are converted to runtime profiles
    at registration time so YAML/JSON-backed callers do not need a separate
    manual conversion step.

    Registrations are **additive**: if a profile is already registered under
    `key` (including a built-in profile loaded during lazy bootstrap), the new
    profile is merged on top rather than replacing it. The incoming profile's
    fields win on conflicts; unspecified fields inherit from the existing
    profile. Excluded-tool sets union, middleware sequences merge by type, and
    `general_purpose_subagent` settings merge field-wise.

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
        profile: The runtime harness profile or declarative config to register.

    Raises:
        ValueError: If `key` is empty, contains more than one `:`, or has an
            empty provider/model half.
    """
    _ensure_harness_profiles_loaded()
    _register_harness_profile_impl(key, profile)


def _has_any_harness_profile() -> bool:
    """Return `True` when a user has registered any harness profile.

    Narrow helper for modules (e.g. `graph.py`) that need to adjust logging
    verbosity based on whether the user has registered any harness profile.
    Registrations made by plugins during the `_ensure_builtin_profiles_loaded`
    bootstrap are excluded — with only bootstrap-provided defaults in play, a
    "no match" miss against a non-matching provider is unsurprising and should
    stay at debug.

    Exists so callers do not have to import the private `_HARNESS_PROFILES`
    registry directly.
    """
    from deepagents.profiles import _builtin_profiles  # noqa: PLC0415

    _ensure_harness_profiles_loaded()
    return bool(_HARNESS_PROFILES.keys() - _builtin_profiles._BOOTSTRAP_HARNESS_KEYS)


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

    _ensure_harness_profiles_loaded()
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

    Excluded-middleware sets are unioned the same way as excluded tools. For
    example, `{SummarizationMiddleware}` plus `{AnthropicPromptCachingMiddleware}`
    becomes both classes in the merged profile.

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
        excluded_middleware=base.excluded_middleware | override.excluded_middleware,
        extra_middleware=_merge_middleware(base.extra_middleware, override.extra_middleware),
        general_purpose_subagent=_merge_general_purpose_subagent_profiles(
            base.general_purpose_subagent,
            override.general_purpose_subagent,
        ),
    )


def _harness_profile_for_model(model: BaseChatModel, spec: str | None) -> HarnessProfile:
    """Look up the `HarnessProfile` for an already-resolved model.

    If `spec` is provided (the original string the caller passed), it is used
    for registry lookup. Otherwise both the model identifier (via `model_dump`)
    and provider (via `_get_ls_params`) are extracted from the model instance
    and combined into a `provider:identifier` key so that model-level profiles
    registered under the canonical `provider:model` shape still resolve when
    the caller hands in a pre-built model. The combined lookup is followed by
    an identifier-only lookup (when the identifier is already in
    `provider:model` shape) and a provider-only fallback.

    A *bare* identifier (no `:`) is deliberately not consulted against the
    registry. If it were, a pre-built model whose `model_name` happened to
    coincide with a registered provider key (e.g. an in-house proxy whose
    identifier is `"openai"`) would silently pick up that provider's profile.
    Registering under a bare key is supported via the `spec` path, not
    inferred from a model's identifier.

    Args:
        model: Resolved chat model instance.
        spec: Original model spec string, or `None` for pre-built instances.

    Returns:
        The matching `HarnessProfile`, or an empty default (null object) when
            nothing resolves.
    """
    # Local import: `_models` indirectly triggers `profiles/__init__`, so a
    # top-level import here would cycle through `harness_profiles` itself.
    from deepagents._models import get_model_identifier, get_model_provider  # noqa: PLC0415

    if spec is not None:
        return _get_harness_profile(spec) or HarnessProfile()
    identifier = get_model_identifier(model)
    provider = get_model_provider(model)
    # Try the canonical `provider:model` key first so user registrations under
    # that shape match. `_get_harness_profile` internally falls back from the
    # exact key to the provider prefix, which also subsumes the pure
    # provider-only case below when both pieces are known. Skip when the
    # identifier already contains a colon to avoid producing a malformed
    # double-colon key.
    if provider and identifier and ":" not in identifier:
        profile = _get_harness_profile(f"{provider}:{identifier}")
        if profile is not None:
            return profile
    # Only consult identifier-only lookup when the identifier itself is in
    # `provider:model` shape — otherwise a bare identifier could accidentally
    # match a provider-wide registration (see docstring).
    if identifier is not None and ":" in identifier:
        profile = _get_harness_profile(identifier)
        if profile is not None:
            return profile
    if provider is not None:
        profile = _get_harness_profile(provider)
        if profile is not None:
            return profile
    # Surface at warning when the user has registered profiles but none
    # matched — a common "my profile isn't applying" failure mode where the
    # pre-built model's identifier/provider couldn't be derived. With an
    # empty registry, no profile was ever going to apply, so the miss is
    # unsurprising and stays at debug.
    level = logging.WARNING if _has_any_harness_profile() else logging.DEBUG
    logger.log(
        level,
        "No harness profile matched pre-built model %s (identifier=%r, provider=%r); using defaults. "
        "If you registered a profile for this model, ensure the key matches the model's resolved provider and identifier.",
        type(model).__name__,
        identifier,
        provider,
    )
    return HarnessProfile()
