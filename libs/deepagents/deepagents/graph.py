"""Deepagents come with planning, filesystem, and subagents."""

import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# Default context limits for known model families (used as fallback)
# These are checked in order - more specific patterns should come first
_DEFAULT_CONTEXT_LIMITS: list[tuple[str, int]] = [
    # OpenAI models (2025)
    ("gpt-4.1", 1000000),      # GPT-4.1, GPT-4.1-mini, GPT-4.1-nano: 1M tokens
    ("gpt-4o", 128000),        # GPT-4o, GPT-4o-mini: 128K tokens
    ("o1", 200000),            # o1, o1-pro: 200K tokens
    ("o3-mini", 200000),       # o3-mini: 200K tokens
    ("o3", 200000),            # o3: 200K tokens
    ("o4-mini", 128000),       # o4-mini: 128K tokens
    # Anthropic Claude models (2025)
    ("claude-sonnet-4", 200000),   # Claude Sonnet 4/4.5: 200K (1M in beta)
    ("claude-opus-4", 200000),     # Claude Opus 4/4.5: 200K
    ("claude-3", 200000),          # Claude 3.x family: 200K tokens
    ("claude", 200000),            # Generic Claude fallback: 200K tokens
    # Google Gemini models (2025)
    ("gemini-2.5", 1000000),   # Gemini 2.5 Pro/Flash: 1M tokens
    ("gemini-2.0", 1000000),   # Gemini 2.0 Flash: 1M tokens
    ("gemini-1.5", 1000000),   # Gemini 1.5 Pro/Flash: 1M tokens
    ("gemini", 1000000),       # Generic Gemini fallback: 1M tokens
    # DeepSeek models
    ("deepseek", 128000),      # DeepSeek models: 128K tokens
    # Mistral models
    ("mistral", 128000),       # Mistral models: typically 128K tokens
    # Llama models
    ("llama", 128000),         # Llama 3.x models: 128K tokens
]

# Fallback context limit for unknown models
_FALLBACK_CONTEXT_LIMIT = 128000

# Fraction of context to use before triggering summarization (85%)
_SUMMARIZATION_TRIGGER_FRACTION = 0.85

# Fraction of context to keep after summarization (10%)
_SUMMARIZATION_KEEP_FRACTION = 0.10


def _get_max_context_tokens(model: BaseChatModel) -> int | None:
    """Get the maximum context tokens for a model.

    Resolution order:
    1. DEEPAGENTS_MAX_CONTEXT_TOKENS environment variable (user override)
    2. Model's profile.max_input_tokens attribute
    3. Inference from model name (for known model families)

    Args:
        model: The language model instance.

    Returns:
        Maximum input tokens, or None if unable to determine.
    """
    # 1. Check environment variable first (highest priority - user override)
    env_limit = os.environ.get("DEEPAGENTS_MAX_CONTEXT_TOKENS")
    if env_limit:
        try:
            return int(env_limit)
        except ValueError:
            pass  # Invalid value, continue to other methods

    # 2. Check model profile
    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        return model.profile["max_input_tokens"]

    # 3. Try to infer from model name
    model_name = ""
    for attr in ["model_name", "model", "name"]:
        try:
            value = getattr(model, attr, None)
            if value and isinstance(value, str):
                model_name = value.lower()
                break
        except Exception:
            continue

    if model_name:
        for pattern, limit in _DEFAULT_CONTEXT_LIMITS:
            if pattern in model_name:
                return limit

    return None


def get_default_model() -> ChatAnthropic:
    """Get the default model for deep agents.

    Returns:
        ChatAnthropic instance configured with Claude Sonnet 4.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
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

    This agent will by default have access to a tool to write todos (write_todos),
    seven file and execution tools: ls, read_file, write_file, edit_file, glob, grep, execute,
    and a tool to call subagents.

    The execute tool allows running shell commands if the backend implements SandboxBackendProtocol.
    For non-sandbox backends, the execute tool will return an error message.

    Args:
        model: The model to use. Defaults to Claude Sonnet 4.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the
                  sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict
                  settings)
                - (optional) `middleware` (list of AgentMiddleware)
        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        backend: Optional backend for file storage and execution. Pass either a Backend instance
            or a callable factory like `lambda rt: StateBackend(rt)`. For execution support,
            use a backend that implements SandboxBackendProtocol.
        interrupt_on: Optional Dict[str, bool | InterruptOnConfig] mapping tool names to
            interrupt configs.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.

    Returns:
        A configured deep agent.
    """
    if model is None:
        model = get_default_model()

    # Determine context limit and configure summarization accordingly
    max_context = _get_max_context_tokens(model)

    if max_context is not None:
        # Use dynamic fraction-based triggers when we know the context limit
        trigger = ("tokens", int(max_context * _SUMMARIZATION_TRIGGER_FRACTION))
        keep = ("fraction", _SUMMARIZATION_KEEP_FRACTION)
    else:
        # Fallback to conservative fixed threshold for unknown models
        trigger = ("tokens", 170000)
        keep = ("messages", 6)

    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=model,
                    trigger=trigger,
                    keep=keep,
                    trim_tokens_to_summarize=None,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ],
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]
    if middleware:
        deepagent_middleware.extend(middleware)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
