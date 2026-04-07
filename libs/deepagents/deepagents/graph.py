"""Primary graph assembly module for Deep Agents.

Provides `create_deep_agent`, the main entry point for constructing a fully
configured Deep Agent with planning, filesystem, subagent, and summarization
middleware.  Also defines `BASE_AGENT_PROMPT` and the default model fallback.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ResponseT, _InputAgentState, _OutputAgentState
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.typing import ContextT

from deepagents._models import get_model_identifier, get_model_provider, resolve_model
from deepagents._profiles import ProviderProfile, get_provider_profile
from deepagents._version import __version__
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
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

BASE_AGENT_PROMPT = """You are a Deep Agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble (\"Sure!\", \"Great question!\", \"I'll now...\").
- Don't say \"I'll now do X\" — just do it.
- If the request is ambiguous, ask questions before acting.
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

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""  # noqa: E501
"""Default system prompt appended to every Deep Agent.

When a caller passes `system_prompt` to `create_deep_agent`, the custom prompt
is prepended and this base prompt is appended. When `system_prompt` is `None`,
this is used as the sole system prompt.
"""


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
    profile: ProviderProfile,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Materialize the `extra_middleware` from a provider profile.

    Handles both static sequences and zero-arg factories.

    Args:
        profile: The provider profile to read from.

    Returns:
        A fresh list of middleware instances (may be empty).
    """
    extra = profile.extra_middleware
    if callable(extra):
        return list(extra())  # ty: ignore[call-top-callable]  # Callable & Sequence union confuses ty
    return list(extra)


def _profile_for_model(model: BaseChatModel, spec: str | None) -> ProviderProfile:
    """Look up the `ProviderProfile` for an already-resolved model.

    If `spec` is provided (the original string the caller passed), it is used
    for registry lookup. Otherwise the model identifier is extracted from the
    instance and used as a best-effort fallback.

    For pre-built model instances (where `spec` is `None`), the identifier alone
    (e.g. `"claude-sonnet-4-6"`) has no provider prefix, so
    `get_provider_profile` cannot resolve the provider-level profile. As a
    fallback we use `get_model_provider` (backed by `_get_ls_params`) to recover
    the provider name and look up the profile directly.

    Args:
        model: Resolved chat model instance.
        spec: Original model spec string, or `None` for pre-built instances.

    Returns:
        The matching `ProviderProfile`, or an empty default.
    """
    if spec is not None:
        return get_provider_profile(spec)
    identifier = get_model_identifier(model)
    if identifier is not None:
        profile = get_provider_profile(identifier)
        if profile != ProviderProfile():
            return profile
    # Bare model name (no colon) — fall back to provider from the model class.
    provider = get_model_provider(model)
    if provider is not None:
        return get_provider_profile(provider)
    return ProviderProfile()


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
    return getattr(tool, "name", None)


def create_deep_agent(  # noqa: C901, PLR0912, PLR0915  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
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

            *User middleware is inserted here.*

            Tail stack:

            - Provider-specific middleware (from `ProviderProfile.extra_middleware`)
            - `MemoryMiddleware` (if `memory` is provided)
            - `HumanInTheLoopMiddleware` (if `interrupt_on` is provided)
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
    model_spec: str | None
    if isinstance(model, str):
        model_spec = model
    elif model is None:
        # Default model is always Anthropic; derive a spec so the profile
        # registry resolves correctly (bare model names lack a provider prefix).
        model_spec = "anthropic:claude-sonnet-4-6"
    else:
        model_spec = None
    model = get_default_model() if model is None else resolve_model(model)
    profile = _profile_for_model(model, model_spec)
    backend = backend if backend is not None else StateBackend()

    # Build general-purpose subagent with default middleware stack
    gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        create_summarization_middleware(model, backend),
        PatchToolCallsMiddleware(),
    ]
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    # Provider-specific middleware (e.g. Anthropic prompt caching).
    gp_middleware.extend(_resolve_extra_middleware(profile))
    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
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
            subagent_spec = raw_subagent_model if isinstance(raw_subagent_model, str) else None
            subagent_model = resolve_model(raw_subagent_model)
            subagent_profile = _profile_for_model(subagent_model, subagent_spec)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                create_summarization_middleware(subagent_model, backend),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))
            # Provider-specific middleware for this subagent's model.
            subagent_middleware.extend(_resolve_extra_middleware(subagent_profile))

            subagent_interrupt_on = spec.get("interrupt_on", interrupt_on)

            processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
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
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                backend=backend,
                subagents=inline_subagents,
            ),
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
    # Provider-specific middleware goes between user middleware and memory so
    # that memory updates (which change the system prompt) don't invalidate the
    # Anthropic prompt cache prefix.
    deepagent_middleware.extend(_resolve_extra_middleware(profile))
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Combine system_prompt with BASE_AGENT_PROMPT (+ optional profile suffix)
    base_prompt = BASE_AGENT_PROMPT
    if profile.system_prompt_suffix:
        base_prompt = base_prompt + "\n\n" + profile.system_prompt_suffix

    if system_prompt is None:
        final_system_prompt: str | SystemMessage = base_prompt
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}])
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + base_prompt

    # Apply tool exclusions and description overrides from the profile.
    # Copy the sequence so caller-owned objects are not mutated.
    filtered_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = list(tools) if tools is not None else None
    if profile.exclude_tools and filtered_tools:
        filtered_tools = [t for t in filtered_tools if _tool_name(t) not in profile.exclude_tools]
    if profile.tool_description_overrides and filtered_tools:
        for tool in filtered_tools:
            name = _tool_name(tool)
            if name is None:
                continue
            override = profile.tool_description_overrides.get(name)
            if override is not None:
                if isinstance(tool, dict):
                    tool["description"] = override  # ty: ignore[invalid-assignment]  # Callable & dict intersection
                else:
                    tool.description = override  # type: ignore[union-attr]

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=filtered_tools,
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
