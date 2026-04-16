"""Primary graph assembly module for Deep Agents.

Provides `create_deep_agent`, the main entry point for constructing a fully
configured Deep Agent with planning, filesystem, subagent, and summarization
middleware.
"""

import logging
import warnings
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

from deepagents._models import get_model_identifier, get_model_provider, resolve_model
from deepagents._version import __version__
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.permissions import FilesystemPermission, _PermissionMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.middleware.tool_hooks import (
    AfterToolCallHook,
    AsyncAfterToolCallHook,
    AsyncBeforeToolCallHook,
    BeforeToolCallHook,
    ToolHooksMiddleware,
)
from deepagents.profiles import _get_harness_profile, _HarnessProfile

logger = logging.getLogger(__name__)

BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

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
"""Default base system prompt for every Deep Agent.

When a caller passes `system_prompt` to `create_deep_agent`, the custom prompt
is prepended and this base prompt is appended. When `system_prompt` is `None`,
this is used as the sole system prompt.
"""
# Replaceable via `_HarnessProfile.base_system_prompt` (internal)


def get_default_model() -> ChatAnthropic:
    """Get the default model for Deep Agents.

    Used as a fallback when `model=None` is passed to `create_deep_agent`.

    Requires `ANTHROPIC_API_KEY` to be set in the environment.

    Returns:
        `ChatAnthropic` instance configured with `claude-sonnet-4-6`.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-6",
    )


def _resolve_extra_middleware(
    profile: _HarnessProfile,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Materialize the `extra_middleware` from a provider profile.

    Args:
        profile: The provider profile to read from.

    Returns:
        A fresh list of middleware instances (may be empty).
    """
    extra = profile.extra_middleware
    if callable(extra):
        return list(extra())  # ty: ignore[call-top-callable]
    return list(extra)


def _harness_profile_for_model(model: BaseChatModel, spec: str | None) -> _HarnessProfile:
    """Look up the `_HarnessProfile` for an already-resolved model.

    If `spec` is provided (the original string the caller passed), it is used
    for registry lookup. Otherwise the model identifier is extracted from the
    instance (via `model_dump`) and used as a best-effort fallback.

    Args:
        model: Resolved chat model instance.
        spec: Original model spec string, or `None` for pre-built instances.

    Returns:
        The matching `_HarnessProfile`, or an empty default (null object).
    """
    if spec is not None:
        return _get_harness_profile(spec)
    identifier = get_model_identifier(model)
    if identifier is not None:
        profile = _get_harness_profile(identifier)
        if profile != _HarnessProfile():
            return profile
        logger.debug("No profile for identifier %r, trying provider fallback", identifier)
    # Bare model name (no colon) — fall back to provider from the model class.
    provider = get_model_provider(model)
    if provider is not None:
        return _get_harness_profile(provider)
    logger.debug("No harness profile found for pre-built model %s, using defaults", type(model).__name__)
    return _HarnessProfile()


def _tool_name(tool: BaseTool | Callable | dict[str, Any]) -> str | None:
    """Extract the tool name from any supported tool type.

    Args:
        tool: A tool in any of the forms accepted by `create_deep_agent`.

    Returns:
        The tool name, or `None` if it cannot be determined.
    """
    if isinstance(tool, dict):
        name = tool.get("name")  # ty: ignore[invalid-argument-type]  # Callable & dict intersection confuses ty
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _apply_tool_description_overrides(
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    overrides: dict[str, str],
) -> list[BaseTool | Callable | dict[str, Any]] | None:
    """Apply description overrides without mutating caller-owned tools.

    Only dict tools and `BaseTool` instances are rewritten. Plain callables are
    returned unchanged because safely replacing their descriptions would require
    wrapping them in new tool objects.

    Args:
        tools: User-supplied tools to copy and possibly rewrite.
        overrides: Description overrides keyed by tool name.

    Returns:
        A copied tool list with supported overrides applied, or `None`.
    """
    if tools is None:
        return None

    copied_tools: list[BaseTool | Callable | dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        override = overrides.get(name) if name is not None else None
        if override is None:
            copied_tools.append(tool)
            continue
        if isinstance(tool, dict):
            rewritten_tool = cast("dict[str, Any]", tool).copy()
            rewritten_tool["description"] = override
            copied_tools.append(rewritten_tool)
            continue
        if isinstance(tool, BaseTool):
            copied_tools.append(tool.model_copy(update={"description": override}))
            continue
        copied_tools.append(tool)
    return copied_tools


def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    before_tool_call_hooks: Sequence[BeforeToolCallHook] = (),
    after_tool_call_hooks: Sequence[AfterToolCallHook] = (),
    async_before_tool_call_hooks: Sequence[AsyncBeforeToolCallHook] = (),
    async_after_tool_call_hooks: Sequence[AsyncAfterToolCallHook] = (),
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]:  # ty: ignore[invalid-type-arguments]  # ty can't verify generic TypedDicts satisfy StateLike bound
    """Create a Deep Agent.

    !!! warning "Deep Agents require a LLM that supports tool calling!"

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
        system_prompt: Custom system instructions to prepend before the base
            Deep Agent prompt.

            If a string, it's concatenated with the base prompt.
        middleware: Additional middleware to apply after the base stack
            but before the tail middleware. The full ordering is:

            Base stack:

            - `TodoListMiddleware`
            - `SkillsMiddleware` (if `skills` is provided)
            - `FilesystemMiddleware`
            - `SubAgentMiddleware`
            - `SummarizationMiddleware`
            - `PatchToolCallsMiddleware`
            - `AsyncSubAgentMiddleware` (if async `subagents` are provided)
            - `ToolHooksMiddleware` (if any explicit tool-call hooks are provided)

            *User middleware is inserted here.*

        before_tool_call_hooks: Sync callbacks evaluated before each tool call.

            If a hook returns a `ToolMessage` or `Command`, the tool call is
            short-circuited and the returned value is used directly.
        after_tool_call_hooks: Sync callbacks evaluated after each tool call.

            Hooks can return a replacement `ToolMessage` or `Command` to
            override the original result.
        async_before_tool_call_hooks: Async callbacks evaluated before each
            tool call in async execution paths.
        async_after_tool_call_hooks: Async callbacks evaluated after each tool
            call in async execution paths.

            Tail stack:

            - Profile `extra_middleware` (provider-specific, if any)
            - `_ToolExclusionMiddleware` (if profile has `excluded_tools`)
            - `AnthropicPromptCachingMiddleware` (unconditional; no-ops for
                non-Anthropic models)
            - `MemoryMiddleware` (if `memory` is provided)
            - `HumanInTheLoopMiddleware` (if `interrupt_on` is provided)
            - `_PermissionMiddleware` (if permission rules are present, always last)
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
            general-purpose synchronous subagent is added automatically.

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
        response_format: A structured output response format to use for the agent.
        context_schema: Schema class that defines immutable run-scoped context.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        checkpointer: Optional `Checkpointer` for persisting agent state
            between runs.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        store: Optional store for persistent storage (required if backend
            uses `StoreBackend`).

            Passed through to [`create_agent`][langchain.agents.create_agent].
        backend: Optional backend for file storage and execution.

            Pass a `Backend` instance (e.g. `StateBackend()`).

            For execution support, use a backend that
            implements `SandboxBackendProtocol`.
        permissions: List of ``FilesystemPermission`` rules for the main agent
            and its subagents.

            Rules are evaluated in declaration order; the first match wins.
            If no rule matches, the call is allowed.

            Subagents inherit these rules unless they specify their own
            `permissions` field, which replaces the parent's rules entirely.

            `_PermissionMiddleware` is appended last in the stack so it sees
            all tools (including those injected by other middleware).
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
        debug: Whether to enable debug mode.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        name: The name of the agent.

            Passed through to [`create_agent`][langchain.agents.create_agent].
        cache: The cache to use for the agent.

            Passed through to [`create_agent`][langchain.agents.create_agent].

    Returns:
        A configured Deep Agent.

    Raises:
        ImportError: If a required provider package is missing or below the
            minimum supported version (e.g., `langchain-openrouter`).
    """
    _model_spec: str | None = model if isinstance(model, str) else None

    if model is None:
        warnings.warn(
            "Passing `model=None` to `create_deep_agent` is deprecated and "
            "will be removed in a future release. The `model` parameter type "
            "will change from `BaseChatModel | str | None` to "
            "`BaseChatModel | str`. Please specify a model explicitly. "
            "See https://docs.langchain.com/oss/python/deepagents/models",
            DeprecationWarning,
            stacklevel=2,
        )

    model = get_default_model() if model is None else resolve_model(model)
    _profile = _harness_profile_for_model(model, _model_spec)

    # Copy of `tools` with any provider-specific description rewrites.
    # (Tool exclusion is handled by _ToolExclusionMiddleware which filters
    # all tools (user-supplied and middleware-injected) in one place.)
    _tools = _apply_tool_description_overrides(
        tools,
        _profile.tool_description_overrides,
    )

    backend = backend if backend is not None else StateBackend()

    # Build general-purpose subagent with default middleware stack
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

    # Add provider-specific middleware, if any
    gp_middleware.extend(_resolve_extra_middleware(_profile))

    # Strip excluded tools after all tool-injecting middleware has run
    if _profile.excluded_tools:
        gp_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    # Prompt caching is unconditional: "ignore" silently skips non-Anthropic models
    gp_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))

    # Permissions must be last so they see all tools from prior middleware
    if permissions:
        gp_middleware.append(_PermissionMiddleware(rules=permissions, backend=backend))

    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": _tools or [],
        "middleware": gp_middleware,
    }
    if interrupt_on is not None:
        general_purpose_spec["interrupt_on"] = interrupt_on

    # Set up subagent middleware
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
                ),
                create_summarization_middleware(subagent_model, backend),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            # Provider-specific middleware for this subagent's model
            subagent_middleware.extend(_resolve_extra_middleware(_subagent_profile))
            if _subagent_profile.excluded_tools:
                subagent_middleware.append(_ToolExclusionMiddleware(excluded=_subagent_profile.excluded_tools))

            # Prompt caching
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
            if subagent_permissions:
                subagent_middleware.append(_PermissionMiddleware(rules=subagent_permissions, backend=backend))

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
            if subagent_interrupt_on is not None:
                processed_spec["interrupt_on"] = subagent_interrupt_on
            inline_subagents.append(processed_spec)

    # If an agent with general purpose name already exists in subagents, then don't add it
    # This is how you overwrite/configure general purpose subagent
    if not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
        # Add a general purpose subagent if it doesn't exist yet
        inline_subagents.insert(0, general_purpose_spec)

    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
    ]
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=_profile.tool_description_overrides,
            ),
            SubAgentMiddleware(
                backend=backend,
                subagents=inline_subagents,
                # Overrides the task tool description. Value should include
                # {available_agents} — a format placeholder replaced with the
                # subagent name/description list. Without it the model can't
                # see which subagents exist. None (default) uses the built-in
                # template. Stale keys silently no-op if the tool is renamed.
                task_description=_profile.tool_description_overrides.get("task"),
            ),
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
    )

    if async_subagents:
        # Async here means that we run these subagents in a non-blocking manner.
        # Currently this supports agents deployed via LangSmith deployments.
        deepagent_middleware.append(AsyncSubAgentMiddleware(async_subagents=async_subagents))

    if before_tool_call_hooks or after_tool_call_hooks or async_before_tool_call_hooks or async_after_tool_call_hooks:
        deepagent_middleware.append(
            ToolHooksMiddleware(
                before_call_hooks=before_tool_call_hooks,
                after_call_hooks=after_tool_call_hooks,
                async_before_call_hooks=async_before_tool_call_hooks,
                async_after_call_hooks=async_after_tool_call_hooks,
            )
        )

    if middleware:
        deepagent_middleware.extend(middleware)
    # Provider-specific middleware goes between user middleware and memory so
    # that memory updates (which change the system prompt) don't invalidate the
    # Anthropic prompt cache prefix.
    deepagent_middleware.extend(_resolve_extra_middleware(_profile))
    if _profile.excluded_tools:
        deepagent_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    # Unconditional prompt caching (see general-purpose subagent comment).
    deepagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    # _PermissionMiddleware must be last so it sees all tools from prior middleware
    if permissions:
        deepagent_middleware.append(_PermissionMiddleware(rules=permissions, backend=backend))

    # Assemble base prompt: use _profile.base_system_prompt if set, else
    # BASE_AGENT_PROMPT, then append profile suffix if present.
    # Finally prepend user system_prompt (handled below).
    base_prompt = _profile.base_system_prompt if _profile.base_system_prompt is not None else BASE_AGENT_PROMPT
    if _profile.system_prompt_suffix is not None:
        base_prompt = base_prompt + "\n\n" + _profile.system_prompt_suffix
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = base_prompt
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}])
    else:
        # String: simple concatenation
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
