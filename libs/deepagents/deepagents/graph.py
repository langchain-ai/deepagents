"""Deep Agents come with planning, filesystem, and subagents."""

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

from deepagents._models import resolve_model
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

            - `AnthropicPromptCachingMiddleware`
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
    model = get_default_model() if model is None else resolve_model(model)
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
    # "ignore" silently skips cache-control header injection for non-Anthropic
    # models, so this middleware can be added unconditionally.
    gp_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
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
            subagent_model = spec.get("model", model)
            subagent_model = resolve_model(subagent_model)

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
            # "ignore" skips caching for non-Anthropic models (see comment above).
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))

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
    # Caching + memory after all other middleware so memory updates don't
    # invalidate the Anthropic prompt cache prefix.
    # "ignore" skips caching for non-Anthropic models (see general-purpose
    # subagent comment above).
    deepagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"}])
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
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
