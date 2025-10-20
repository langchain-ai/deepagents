"""Deepagents come with planning, filesystem, and subagents."""

from collections.abc import Callable, Sequence
from typing import Any
import os

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.agents.middleware import ShellToolMiddleware, HostExecutionPolicy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.local_filesystem import LocalFilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


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
    use_longterm_memory: bool = False,
    use_local_filesystem: bool = False,
    long_term_memory: bool = False,
    skills: list[dict[str, Any]] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    four file editing tools: write_file, ls, read_file, edit_file, and a tool to call
    subagents.

    Args:
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        model: The model to use.
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
        store: Optional store for persisting longterm memories.
        use_longterm_memory: Whether to use longterm memory - you must provide a store
            in order to use longterm memory.
        use_local_filesystem: If True, injects LocalFilesystemMiddleware (tools operate on disk).
            When True, longterm memory is not supported and `use_longterm_memory` must be False.
            Skills are automatically discovered from ~/.deepagents/skills/ and ./.deepagents/skills/.
            The agent_name for memory storage can be passed via config: {"configurable": {"agent_name": "myagent"}}.
        long_term_memory: If True, enables long-term memory features like agent.md persistence
            and memories folder. Only applies when use_local_filesystem=True.
        skills: Optional list of SkillDefinition for virtual filesystem mode. Only valid when
            use_local_filesystem=False. Skills are loaded into /skills/<name>/ in virtual filesystem.
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

    if use_local_filesystem and skills is not None:
        raise ValueError(
            "Cannot provide skill definitions with use_local_filesystem=True. "
            "Skills are automatically discovered from ~/.deepagents/skills/ and ./.deepagents/skills/. "
            "To use custom skills, set use_local_filesystem=False."
        )

    # Choose filesystem middleware kind
    def _fs_middleware() -> list[AgentMiddleware]:
        if use_local_filesystem:
            shell_middleware = ShellToolMiddleware(
                workspace_root=os.getcwd(),
                execution_policy=HostExecutionPolicy()
            )
            return [LocalFilesystemMiddleware(long_term_memory=long_term_memory), shell_middleware]
        return [FilesystemMiddleware(long_term_memory=use_longterm_memory, skills=skills)]

    deepagent_middleware = [
        TodoListMiddleware(),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                                   PatchToolCallsMiddleware(),
            ] + _fs_middleware(),
            default_interrupt_on=interrupt_on,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ]+ _fs_middleware()
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    if middleware is not None:
        deepagent_middleware.extend(middleware)

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
