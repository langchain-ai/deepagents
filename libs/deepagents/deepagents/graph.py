"""Primary graph assembly module for Deep Agents.

Provides `create_deep_agent`, the main entry point for constructing a fully
configured deep agent with planning, filesystem, subagent, and summarization
middleware.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ResponseT, _InputAgentState, _OutputAgentState
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT

from deepagents._api.deprecation import deprecated, warn_deprecated
from deepagents._excluded_middleware import (
    _apply_excluded_middleware,
    _validate_excluded_middleware_config,
    _verify_excluded_middleware_coverage,
)
from deepagents._models import resolve_model
from deepagents._tools import _apply_tool_description_overrides
from deepagents._version import __version__
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.profiles import GeneralPurposeSubagentProfile
from deepagents.profiles.harness.harness_profiles import _apply_profile_prompt, _harness_profile_for_model

logger = logging.getLogger(__name__)

BASE_AGENT_PROMPT = """You are a deep agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
- Don't say \"I'll now do X\" — just do it.
- If the request is underspecified, ask only the minimum followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**
- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Clarifying Requests

- Do not ask for details the user already supplied.
- Use reasonable defaults when the request clearly implies them.
- Prioritize missing semantics like content, delivery, detail level, or alert criteria.
- Avoid opening with a long explanation of tool, scheduling, or integration limitations when a concise blocking followup question would move the task forward.
- Ask domain-defining questions before implementation questions.
- For monitoring or alerting requests, ask what signals, thresholds, or conditions should trigger an alert.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""  # noqa: E501
"""Default base system prompt for every deep agent (`BASE`).

The final system prompt sent to the model is composed from up to four
named parts:

- `USER` — the `system_prompt=` argument to `create_deep_agent` (`str` or
    `SystemMessage`); when unset, no `USER` segment is included.
- `BASE` — this constant.
- `CUSTOM` — `HarnessProfile.base_system_prompt`. When set on a matching
    profile, replaces `BASE` outright; when unset, `BASE` is used.
- `SUFFIX` — `HarnessProfile.system_prompt_suffix`. When set on a
    matching profile, appended last; when unset, no `SUFFIX` segment is
    included.

The order is always `USER` -> (`BASE` or `CUSTOM`) -> `SUFFIX`, joined by
blank lines (`\\n\\n`). Two invariants follow:

1. `USER` is always at the front, so caller instructions take precedence
    over SDK and profile content regardless of which model is selected.
2. `SUFFIX` is always at the end, so model-tuning guidance sits closest
    to the conversation history (where the model attends most).

When `USER` is a `SystemMessage`, the right-hand assembly is appended as
an additional text content block onto the message's existing
`content_blocks` list, preserving any `cache_control` markers the caller
set.

See `create_deep_agent`'s `system_prompt` parameter or
[Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)
for the full assembly order.
"""


def _build_default_model() -> ChatAnthropic:
    """Construct the default model without emitting a deprecation warning.

    Internal helper used by `create_deep_agent` so the parameter-level
    `model=None` warning isn't paired with a separate function-level warning
    from `get_default_model`. Direct user calls go through `get_default_model`,
    which keeps its decorator and warns once per process.
    """
    return ChatAnthropic(model_name="claude-sonnet-4-6")


@deprecated(
    since="0.5.3",
    removal="1.0.0",
    message=(
        "Relying on the default model is deprecated and will be removed in "
        "deepagents==1.0.0 alongside support for `model=None` in "
        "`create_deep_agent`. Construct your model explicitly "
        "(e.g., `ChatAnthropic(model_name=...)`). See "
        "https://docs.langchain.com/oss/python/deepagents/models"
    ),
    package="deepagents",
)
def get_default_model() -> ChatAnthropic:
    """Get the default model for Deep Agents.

    !!! deprecated

        Deprecated since `0.5.3`; will be removed in `deepagents==1.0.0`.
        Construct your model explicitly (e.g.,
        `ChatAnthropic(model_name="claude-sonnet-4-6")`).

    Used as a fallback when `model=None` is passed to `create_deep_agent`.

    Requires `ANTHROPIC_API_KEY` to be set in the environment.

    Returns:
        `ChatAnthropic` instance configured with `claude-sonnet-4-6`.
    """
    return _build_default_model()


_REQUIRED_MIDDLEWARE: tuple[tuple[type[AgentMiddleware[Any, Any, Any]], tuple[str, ...]], ...] = (
    (FilesystemMiddleware, ()),
    (SubAgentMiddleware, ()),
)
"""Scaffolding middleware that core deep agent features depend on.

Each entry pairs a class with any extra string aliases its `.name` may take
beyond `__name__`. Removing any of these silently breaks core features:
`FilesystemMiddleware` backs every built-in file tool and now also enforces
`permissions` rules (a security guarantee), while `SubAgentMiddleware` backs
the `task` tool handler.

Tracked here so `HarnessProfile.excluded_middleware` cannot strip them:
`_apply_excluded_middleware` raises `ValueError` rather than proceeding with
a silently degraded agent.
"""

_REQUIRED_MIDDLEWARE_CLASSES: frozenset[type[AgentMiddleware[Any, Any, Any]]] = frozenset(cls for cls, _ in _REQUIRED_MIDDLEWARE)
"""Set of all class types that cannot be excluded from the middleware stack.

Derived from `_REQUIRED_MIDDLEWARE` and used for quick membership testing.
"""

_REQUIRED_MIDDLEWARE_NAMES: frozenset[str] = frozenset(name for cls, aliases in _REQUIRED_MIDDLEWARE for name in (cls.__name__, *aliases))
"""Set of all `.name` values that cannot be excluded from the middleware stack.

Derived from `_REQUIRED_MIDDLEWARE` and used for quick membership testing.
"""


def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]:  # ty: ignore[invalid-type-arguments]  # ty can't verify generic TypedDicts satisfy StateLike bound
    r"""Create a deep agent.

    !!! warning "Deep agents require a LLM that supports tool calling!"

    By default, this agent has access to the following tools:

    - `write_todos`: manage a todo list
    - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: file operations
    - `execute`: run shell commands
    - `task`: call subagents

    The `execute` tool allows running shell commands if the backend implements `SandboxBackendProtocol`.
    For non-sandbox backends, the `execute` tool will return an error message.

    Args:
        model: The model to use.

            Defaults to `claude-sonnet-4-6`.

            Accepts a `provider:model` string (e.g., `openai:gpt-5`); see
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)]
            for supported values. You can also pass a pre-initialized
            [`BaseChatModel`][langchain.chat_models.BaseChatModel] instance directly.

            !!! note "OpenAI Models and Data Retention"

                If an `openai:` model is used, the agent will use the OpenAI
                Responses API by default. To use OpenAI chat completions
                instead, initialize the model with
                `init_chat_model("openai:...", use_responses_api=False)` and
                pass the initialized model instance here.

                To disable data retention with the Responses API, use
                `init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
                and pass the initialized model instance here.
        tools: Additional tools the agent should have access to.

            These are merged with the built-in tool suite listed above
            (`write_todos`, filesystem tools, `execute`, and `task`).
        system_prompt: Custom system instructions placed at the front of
            the system prompt sent to the model.

            Whatever you pass here always sits before the SDK's default
            deep-agent prompt and any model-tuning suffix from a
            registered `HarnessProfile`. With `system_prompt=None`, the
            SDK default is used on its own (plus the profile suffix
            when one applies). Sections are joined by a blank line.

            Passing a `SystemMessage` instead of a string preserves any
            `cache_control` markers on the message's content blocks —
            useful for placing explicit Anthropic prompt-cache
            breakpoints. The same ordering applies (caller's blocks
            first, SDK content appended as an additional text block).

            See [Prompt assembly](https://docs.langchain.com/oss/deepagents/customization#prompt-assembly)
            for the full case-by-case breakdown.
        middleware: Additional middleware to apply after the base stack
            but before the tail middleware. The full ordering is:

            Base stack:

            - `TodoListMiddleware`
            - `SkillsMiddleware` (if `skills` is provided)
            - `FilesystemMiddleware`
            - `SubAgentMiddleware` (if any inline subagents — declarative `SubAgent` or `CompiledSubAgent` — are available)
            - `SummarizationMiddleware`
            - `PatchToolCallsMiddleware`
            - `AsyncSubAgentMiddleware` (if async `subagents` are provided)

            *User middleware is inserted here.*

            Tail stack:

            - Harness profile `extra_middleware` (if any)
            - `_ToolExclusionMiddleware` (if profile has `excluded_tools`)
            - `AnthropicPromptCachingMiddleware` (unconditional; no-ops for
                non-Anthropic models)
            - `MemoryMiddleware` (if `memory` is provided)
            - `HumanInTheLoopMiddleware` (if `interrupt_on` is provided)

            After assembly, any entries in the profile's
            `excluded_middleware` are filtered from the final stack. Class
            entries match exact type; string entries match
            `AgentMiddleware.name` exactly (e.g. `"SummarizationMiddleware"`
            drops the summarization middleware via its public alias).
            Entries that match nothing in the assembled stack raise
            `ValueError`, as does excluding scaffolding classes
            (`FilesystemMiddleware`, `SubAgentMiddleware`).
        subagents: Subagent specs available to the main agent.

            This collection supports three forms:

            - [`SubAgent`][deepagents.middleware.subagents.SubAgent]: A declarative synchronous subagent spec.
            - [`CompiledSubAgent`][deepagents.middleware.subagents.CompiledSubAgent]: A pre-compiled runnable subagent.
            - [`AsyncSubAgent`][deepagents.middleware.async_subagents.AsyncSubAgent]: A remote/background subagent spec.

            `SubAgent` entries are invoked through the `task` tool. They should
            provide `name`, `description`, and `system_prompt`, and may also
            override `tools`, `model`, `middleware`, `interrupt_on`, and
            `skills`. See `interrupt_on` below for inheritance and override
            behavior.

            `CompiledSubAgent` entries are also exposed through the `task` tool,
            but provide a pre-built `runnable` instead of a declarative prompt
            and tool configuration.

            `AsyncSubAgent` entries are identified by their async-subagent
            fields (`graph_id`, and optionally `url`/`headers`) and are routed
            into `AsyncSubAgentMiddleware` instead of `SubAgentMiddleware`.
            They should provide `name`, `description`, and `graph_id`, and may
            optionally include `url` and `headers`. These subagents run as
            background tasks and expose the async subagent tools for launching,
            checking, updating, cancelling, and listing tasks.

            If no subagent named `general-purpose` is provided, a default
            general-purpose synchronous subagent is added automatically unless
            the active harness profile disables it.

        skills: List of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes)
            and are relative to the backend's root. When using
            `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded
            from disk relative to the backend's `root_dir`. Later sources
            override earlier ones for skills with the same name (last one wins).
        memory: List of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.

            Memory is loaded at agent startup and added into the system prompt.
        permissions: List of `FilesystemPermission` rules for the main agent
            and its subagents.

            Rules are evaluated in declaration order; the first match wins.
            If no rule matches, the call is allowed.

            Subagents inherit these rules unless they specify their own
            `permissions` field, which replaces the parent's rules entirely.

            `FilesystemMiddleware` enforces these rules directly for the
            built-in filesystem tools. Subagents inherit the parent rules
            unless they specify their own `permissions`.
        backend: Optional backend for file storage and execution.

            Pass a `Backend` instance (e.g. `StateBackend()`).

            For execution support, use a backend that
            implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.

            Pass to pause agent execution at specified tool calls for human
            approval or modification.

            This config always applies to the main agent.

            For subagents:
            - Declarative `SubAgent` specs inherit the top-level `interrupt_on`
                config by default.
            - If a declarative `SubAgent` provides its own `interrupt_on`, that
                subagent-specific config overrides the inherited
                top-level config.
            - `CompiledSubAgent` runnables do not inherit top-level
                `interrupt_on`; configure human-in-the-loop behavior inside the
                compiled runnable itself.
            - Remote `AsyncSubAgent` specs do not inherit top-level
                `interrupt_on`; configure any approval behavior on the remote
                subagent itself.

            For example, `interrupt_on={"edit_file": True}` pauses before
            every edit.
        response_format: A structured output response format to use for the agent.
        context_schema: Schema class that defines immutable run-scoped context.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        checkpointer: Optional `Checkpointer` for persisting agent state
            between runs.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        store: Optional store for persistent storage (required if backend
            uses `StoreBackend`).

            Passed through to [`create_agent`][langchain.agents.create_agent].
        debug: Whether to enable debug mode.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        name: The name of the agent.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        cache: The cache to use for the agent.

            Passed through to [`create_agent`][langchain.agents.create_agent].

    Returns:
        A configured deep agent.

    Raises:
        ImportError: If a required provider package is missing or below the
            minimum supported version (e.g., `langchain-openrouter`).
    """
    _model_spec: str | None = model if isinstance(model, str) else None

    if model is None:
        warn_deprecated(
            since="0.5.3",
            removal="1.0.0",
            message=(
                "Passing `model=None` to `create_deep_agent` is deprecated "
                "and will be removed in deepagents==1.0.0. The `model` "
                "parameter type will change from `BaseChatModel | str | None` "
                "to `BaseChatModel | str`. Specify a model explicitly "
                "(e.g., `ChatAnthropic(model_name=...)`). See "
                "https://docs.langchain.com/oss/python/deepagents/models"
            ),
            package="deepagents",
        )
        # Use the un-decorated builder so we don't burn the dedupe flag on
        # `get_default_model` — direct user callers still see one warning.
        model = _build_default_model()
    else:
        model = resolve_model(model)
    _profile = _harness_profile_for_model(model, _model_spec)
    # Validate profile-level invariants (required scaffolding, private names)
    _validate_excluded_middleware_config(
        _profile,
        required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
        required_names=_REQUIRED_MIDDLEWARE_NAMES,
    )
    # Accumulate which entries matched across the main agent + general-purpose
    # subagent stacks (both use `_profile`). A profile-level entry only has to
    # match somewhere, not in every stack, so coverage is verified once after
    # all filters have run.
    _main_matched_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
    _main_matched_names: set[str] = set()

    # Copy of `tools` with any harness-specific description rewrites.
    # (Tool exclusion is handled by _ToolExclusionMiddleware which filters
    # all tools (user-supplied and middleware-injected) in one place.)
    _tools = _apply_tool_description_overrides(
        tools,
        _profile.tool_description_overrides,
    )

    backend = backend if backend is not None else StateBackend()

    # Process caller-supplied subagents first so the decision of whether to
    # auto-add the default general-purpose subagent can factor in an explicit
    # override, and so its middleware stack (including any factory-based
    # `extra_middleware`) isn't built and then discarded.
    inline_subagents: list[SubAgent | CompiledSubAgent] = []
    async_subagents: list[AsyncSubAgent] = []
    for spec in subagents or []:
        if "graph_id" in spec:
            # Then spec is an AsyncSubAgent
            async_subagents.append(cast("AsyncSubAgent", spec))
            continue
        if "runnable" in spec:
            # CompiledSubAgent - use as-is
            inline_subagents.append(spec)
        else:
            # SubAgent - fill in defaults and prepend base middleware
            raw_subagent_model = spec.get("model", model)
            subagent_model = resolve_model(raw_subagent_model)

            _subagent_spec = raw_subagent_model if isinstance(raw_subagent_model, str) else None
            _subagent_profile = _harness_profile_for_model(subagent_model, _subagent_spec)

            # Resolve permissions: subagent's own rules take priority, else inherit parent's
            subagent_permissions = spec.get("permissions", permissions)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                TodoListMiddleware(),
                FilesystemMiddleware(
                    backend=backend,
                    custom_tool_descriptions=_subagent_profile.tool_description_overrides,
                    permissions=subagent_permissions,
                ),
                create_summarization_middleware(subagent_model, backend),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            # Harness-profile middleware for this subagent's model
            subagent_middleware.extend(_subagent_profile.materialize_extra_middleware())
            if _subagent_profile.excluded_tools:
                subagent_middleware.append(_ToolExclusionMiddleware(excluded=_subagent_profile.excluded_tools))

            # Prompt caching
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))

            _subagent_matched_classes: set[type[AgentMiddleware[Any, Any, Any]]] = set()
            _subagent_matched_names: set[str] = set()
            _validate_excluded_middleware_config(
                _subagent_profile,
                required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
                required_names=_REQUIRED_MIDDLEWARE_NAMES,
            )
            subagent_middleware = _apply_excluded_middleware(
                subagent_middleware,
                _subagent_profile,
                matched_classes=_subagent_matched_classes,
                matched_names=_subagent_matched_names,
            )
            _verify_excluded_middleware_coverage(
                _subagent_profile,
                _subagent_matched_classes,
                _subagent_matched_names,
                required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
                required_names=_REQUIRED_MIDDLEWARE_NAMES,
            )

            subagent_interrupt_on = spec.get("interrupt_on", interrupt_on)

            # Inherit parent tools unless the subagent declares its own.
            # Descriptions are rewritten; exclusion is handled by middleware.
            raw_subagent_tools = spec.get("tools") if "tools" in spec else tools
            subagent_tools = _apply_tool_description_overrides(
                raw_subagent_tools,
                _subagent_profile.tool_description_overrides,
            )

            processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
                **spec,
                "model": subagent_model,
                "tools": subagent_tools or [],
                "middleware": subagent_middleware,
            }
            processed_spec["system_prompt"] = _apply_profile_prompt(_subagent_profile, spec["system_prompt"])
            if subagent_interrupt_on is not None:
                processed_spec["interrupt_on"] = subagent_interrupt_on
            inline_subagents.append(processed_spec)

    # Auto-add the default general-purpose subagent unless the harness profile
    # disables it or the caller already supplied their own — an explicit spec
    # is how callers override the default. Skipping in those cases also avoids
    # invoking factory-based `extra_middleware` whose output would be thrown
    # away.
    gp_profile = _profile.general_purpose_subagent or GeneralPurposeSubagentProfile()
    if gp_profile.enabled is not False and not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
        gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=_profile.tool_description_overrides,
            ),
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
        if skills is not None:
            gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))

        # Add harness-profile middleware, if any
        gp_middleware.extend(_profile.materialize_extra_middleware())

        # Strip excluded tools after all tool-injecting middleware has run
        if _profile.excluded_tools:
            gp_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
        # Prompt caching is unconditional: "ignore" silently skips non-Anthropic models
        gp_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))

        gp_middleware = _apply_excluded_middleware(
            gp_middleware,
            _profile,
            matched_classes=_main_matched_classes,
            matched_names=_main_matched_names,
        )

        general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
            **GENERAL_PURPOSE_SUBAGENT,
            "model": model,
            "tools": _tools or [],
            "middleware": gp_middleware,
        }
        if gp_profile.description is not None:
            general_purpose_spec["description"] = gp_profile.description
        if gp_profile.system_prompt is not None:
            # GP-specific override beats `profile.base_system_prompt`; only the
            # profile suffix layers on top.
            gp_prompt = gp_profile.system_prompt
            if _profile.system_prompt_suffix is not None:
                gp_prompt = gp_prompt + "\n\n" + _profile.system_prompt_suffix
            general_purpose_spec["system_prompt"] = gp_prompt
        else:
            general_purpose_spec["system_prompt"] = _apply_profile_prompt(_profile, GENERAL_PURPOSE_SUBAGENT["system_prompt"])
        if interrupt_on is not None:
            general_purpose_spec["interrupt_on"] = interrupt_on

        inline_subagents.insert(0, general_purpose_spec)

    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
    ]
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.append(
        FilesystemMiddleware(
            backend=backend,
            custom_tool_descriptions=_profile.tool_description_overrides,
            permissions=permissions,
        )
    )
    if inline_subagents:
        deepagent_middleware.append(
            SubAgentMiddleware(
                backend=backend,
                subagents=inline_subagents,
                task_description=_profile.tool_description_overrides.get("task"),
            )
        )
    deepagent_middleware.extend(
        [
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
    )

    if async_subagents:
        # Async here means that we run these subagents in a non-blocking manner.
        # Currently this supports agents deployed via LangSmith deployments.
        deepagent_middleware.append(AsyncSubAgentMiddleware(async_subagents=async_subagents))

    if middleware:
        deepagent_middleware.extend(middleware)
    # Harness-profile middleware goes between user middleware and memory so
    # that memory updates (which change the system prompt) don't invalidate the
    # Anthropic prompt cache prefix.
    deepagent_middleware.extend(_profile.materialize_extra_middleware())
    if _profile.excluded_tools:
        deepagent_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    # Unconditional prompt caching (see general-purpose subagent comment).
    deepagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    if memory is not None:
        # MemoryMiddleware applies the cache_control breakpoint only when the
        # request model is Anthropic, making it safe to enable unconditionally.
        deepagent_middleware.append(
            MemoryMiddleware(
                backend=backend,
                sources=memory,
                add_cache_control=True,
            )
        )
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    deepagent_middleware = _apply_excluded_middleware(
        deepagent_middleware,
        _profile,
        matched_classes=_main_matched_classes,
        matched_names=_main_matched_names,
    )
    # Verify every main-profile exclusion matched at least one middleware in
    # either the main agent stack or the GP subagent stack. An entry that
    # matched nothing across both is almost certainly a typo or a stale
    # profile.
    _verify_excluded_middleware_coverage(
        _profile,
        _main_matched_classes,
        _main_matched_names,
        required_classes=_REQUIRED_MIDDLEWARE_CLASSES,
        required_names=_REQUIRED_MIDDLEWARE_NAMES,
    )

    base_prompt = _apply_profile_prompt(_profile, BASE_AGENT_PROMPT)
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = base_prompt
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}])
    else:
        final_system_prompt = system_prompt + "\n\n" + base_prompt

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=_tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config(
        {
            "recursion_limit": 9_999,
            "metadata": {
                "ls_integration": "deepagents",
                "versions": {"deepagents": __version__},
                "lc_agent_name": name,
            },
        }
    )
