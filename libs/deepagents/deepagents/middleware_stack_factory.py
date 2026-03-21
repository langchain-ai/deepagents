"""Middleware stack factory abstractions for deep agents."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware

__all__ = [
    "DeepAgentBuildContext",
    "DefaultMiddlewareStackFactory",
    "MiddlewareStackFactory",
]


@dataclass(frozen=True, slots=True)
class DeepAgentBuildContext:
    """Normalized inputs used while building middleware stacks."""

    model: BaseChatModel
    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    backend: BackendProtocol | BackendFactory
    skills: list[str] | None
    memory: list[str] | None
    interrupt_on: dict[str, bool | InterruptOnConfig] | None
    user_middleware: Sequence[AgentMiddleware[Any, Any, Any]]


class MiddlewareStackFactory(Protocol):
    """Factory used to assemble middleware stacks for deep agents and subagents."""

    def build_general_purpose_subagent_stack(
        self,
        ctx: DeepAgentBuildContext,
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the middleware stack for the auto-added general-purpose subagent."""
        ...

    def build_subagent_stack(
        self,
        ctx: DeepAgentBuildContext,
        *,
        spec: SubAgent,
        model: BaseChatModel,
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the middleware stack for a declarative SubAgent."""
        ...

    def build_main_stack(
        self,
        ctx: DeepAgentBuildContext,
        *,
        inline_subagents: Sequence[SubAgent | CompiledSubAgent],
        async_subagents: Sequence[AsyncSubAgent],
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the middleware stack for the main deep agent."""
        ...


class DefaultMiddlewareStackFactory:
    """Default factory that preserves the current DeepAgents middleware ordering."""

    def build_general_purpose_subagent_stack(
        self,
        ctx: DeepAgentBuildContext,
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the default stack for the auto-added general-purpose subagent."""
        stack: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=ctx.backend),
            create_summarization_middleware(ctx.model, ctx.backend),
            PatchToolCallsMiddleware(),
        ]

        if ctx.skills is not None:
            stack.append(SkillsMiddleware(backend=ctx.backend, sources=ctx.skills))

        stack.append(
            AnthropicPromptCachingMiddleware(
                unsupported_model_behavior="ignore",
            )
        )

        if ctx.interrupt_on is not None:
            stack.append(HumanInTheLoopMiddleware(interrupt_on=ctx.interrupt_on))

        return stack

    def build_subagent_stack(
        self,
        ctx: DeepAgentBuildContext,
        *,
        spec: SubAgent,
        model: BaseChatModel,
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the default stack for a declarative subagent."""
        stack: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=ctx.backend),
            create_summarization_middleware(model, ctx.backend),
            PatchToolCallsMiddleware(),
        ]

        subagent_skills = spec.get("skills")
        if subagent_skills:
            stack.append(
                SkillsMiddleware(
                    backend=ctx.backend,
                    sources=subagent_skills,
                )
            )

        stack.extend(spec.get("middleware", []))

        stack.append(
            AnthropicPromptCachingMiddleware(
                unsupported_model_behavior="ignore",
            )
        )

        return stack

    def build_main_stack(
        self,
        ctx: DeepAgentBuildContext,
        *,
        inline_subagents: Sequence[SubAgent | CompiledSubAgent],
        async_subagents: Sequence[AsyncSubAgent],
    ) -> Sequence[AgentMiddleware[Any, Any, Any]]:
        """Build the default stack for the main deep agent."""
        stack: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
        ]

        if ctx.skills is not None:
            stack.append(SkillsMiddleware(backend=ctx.backend, sources=ctx.skills))

        stack.extend(
            [
                FilesystemMiddleware(backend=ctx.backend),
                SubAgentMiddleware(
                    backend=ctx.backend,
                    subagents=inline_subagents,
                ),
                create_summarization_middleware(ctx.model, ctx.backend),
                PatchToolCallsMiddleware(),
            ]
        )

        if async_subagents:
            # Async here means that we run these subagents in a non-blocking manner.
            # Currently this supports agents deployed via LangSmith deployments.
            stack.append(AsyncSubAgentMiddleware(async_subagents=list(async_subagents)))

        if ctx.user_middleware:
            stack.extend(ctx.user_middleware)

        # Keep caching + memory after all other middleware so memory updates
        # don't invalidate the Anthropic prompt cache prefix.
        stack.append(
            AnthropicPromptCachingMiddleware(
                unsupported_model_behavior="ignore",
            )
        )

        if ctx.memory is not None:
            stack.append(MemoryMiddleware(backend=ctx.backend, sources=ctx.memory))

        if ctx.interrupt_on is not None:
            stack.append(HumanInTheLoopMiddleware(interrupt_on=ctx.interrupt_on))

        return stack
