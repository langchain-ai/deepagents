"""Deep Agents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents._models import resolve_model
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.async_subagents import AsyncSubAgent
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
)
from deepagents.middleware_stack_factory import (
    DeepAgentBuildContext,
    DefaultMiddlewareStackFactory,
    MiddlewareStackFactory,
)

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


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.6.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-6",
    )


def create_deep_agent(  # Complex graph assembly logic with many conditional branches
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    middleware_factory: MiddlewareStackFactory | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

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

            Use the `provider:model` format (e.g., `openai:gpt-5`) to quickly switch between models.

            If an `openai:` model is used, the agent will use the OpenAI
            Responses API by default. To use OpenAI chat completions instead,
            initialize the model with
            `init_chat_model("openai:...", use_responses_api=False)` and pass
            the initialized model instance here. To disable data retention with
            the Responses API, use
            `init_chat_model("openai:...", use_responses_api=True, store=False, include=["reasoning.encrypted_content"])`
            and pass the initialized model instance here.
        tools: The tools the agent should have access to.

            In addition to custom tools you provide, deep agents include built-in tools for planning,
            file management, and subagent spawning.
        system_prompt: Custom system instructions to prepend before the base deep agent
            prompt.

            If a string, it's concatenated with the base prompt.
        middleware: Additional middleware to apply after the base stack
            (`TodoListMiddleware`, `FilesystemMiddleware`, `SubAgentMiddleware`,
            `SummarizationMiddleware`, `PatchToolCallsMiddleware`) but before
            `AnthropicPromptCachingMiddleware` and `MemoryMiddleware`.
        middleware_factory: Optional factory used to build the middleware stacks
            for the main agent, the default general-purpose subagent, and
            declarative subagents. If omitted, the current default composition
            is preserved.
        subagents: Optional subagent specs available to the main agent.

            This collection supports three forms:

            - `SubAgent`: A declarative synchronous subagent spec.
            - `CompiledSubAgent`: A pre-compiled runnable subagent.
            - `AsyncSubAgent`: A remote/background subagent spec.

            `SubAgent` entries are invoked through the `task` tool. They should
            provide `name`, `description`, and `system_prompt`, and may also
            override `tools`, `model`, `middleware`, `interrupt_on`, and
            `skills`.

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

        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes) and are relative
            to the backend's root. When using `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded from disk relative
            to the backend's `root_dir`. Later sources override earlier ones for skills with the
            same name (last one wins).
        memory: Optional list of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.

            Memory is loaded at agent startup and added into the system prompt.
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional `Checkpointer` for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses `StoreBackend`).
        backend: Optional backend for file storage and execution.

            Pass either a `Backend` instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.

            Pass to pause agent execution at specified tool calls for human approval or modification.

            Example: `interrupt_on={"edit_file": True}` pauses before every edit.
        debug: Whether to enable debug mode. Passed through to `create_agent`.
        name: The name of the agent. Passed through to `create_agent`.
        cache: The cache to use for the agent. Passed through to `create_agent`.

    Returns:
        A configured deep agent.
    """
    resolved_model = get_default_model() if model is None else resolve_model(model)
    resolved_tools = list(tools or [])
    resolved_backend = backend if backend is not None else StateBackend
    stack_factory = middleware_factory or DefaultMiddlewareStackFactory()

    build_ctx = DeepAgentBuildContext(
        model=resolved_model,
        tools=resolved_tools,
        backend=resolved_backend,
        skills=skills,
        memory=memory,
        interrupt_on=interrupt_on,
        user_middleware=middleware,
    )

    # Build default general-purpose subagent using the injected factory.
    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": resolved_model,
        "tools": resolved_tools,
        "middleware": list(stack_factory.build_general_purpose_subagent_stack(build_ctx)),
    }

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
            continue

        # SubAgent - fill in defaults and prepend base middleware
        declarative_spec = cast("SubAgent", spec)
        subagent_model = resolve_model(declarative_spec.get("model", resolved_model))
        subagent_middleware = list(
            stack_factory.build_subagent_stack(
                build_ctx,
                spec=declarative_spec,
                model=subagent_model,
            )
        )

        processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
            **declarative_spec,
            "model": subagent_model,
            "tools": declarative_spec.get("tools", resolved_tools),
            "middleware": subagent_middleware,
        }
        inline_subagents.append(processed_spec)

    # If an agent with general purpose name already exists in subagents, then don't add it
    # This is how you overwrite/configure general purpose subagent
    if not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
        # Add a general purpose subagent if it doesn't exist yet
        inline_subagents.insert(0, general_purpose_spec)

    # Build main agent middleware stack
    deepagent_middleware = list(
        stack_factory.build_main_stack(
            build_ctx,
            inline_subagents=inline_subagents,
            async_subagents=async_subagents,
        )
    )

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(
            content_blocks=[
                *system_prompt.content_blocks,
                {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
            ]
        )
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        resolved_model,
        system_prompt=final_system_prompt,
        tools=resolved_tools,
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
            "recursion_limit": 1000,
            "metadata": {
                "ls_integration": "deepagents",
            },
        }
    )
