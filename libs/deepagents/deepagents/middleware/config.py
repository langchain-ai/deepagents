"""Middleware configuration for DeepAgents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware


@dataclass
class MiddlewareConfig:
    """Configuration for built-in middleware in DeepAgents.

    Each field controls whether a specific built-in middleware is enabled and allows
    replacing it with a custom implementation.

    Field values can be:
        - ``True`` (default): Use the built-in middleware with default settings
        - ``False``: Disable this middleware entirely
        - ``AgentMiddleware`` instance: Replace with a custom implementation

    Example:
        Disable the todo list middleware::

            from deepagents import create_deep_agent, MiddlewareConfig

            agent = create_deep_agent(
                model="claude-sonnet-4-5-20250929",
                middleware_config=MiddlewareConfig(todo_list=False),
            )

        Replace the filesystem middleware with a custom implementation::

            from deepagents import create_deep_agent, MiddlewareConfig

            agent = create_deep_agent(
                model="claude-sonnet-4-5-20250929",
                middleware_config=MiddlewareConfig(
                    filesystem=MySecureFilesystemMiddleware(),
                ),
            )

        Create a minimal agent with only custom middleware::

            from deepagents import create_deep_agent, MiddlewareConfig

            agent = create_deep_agent(
                model="claude-sonnet-4-5-20250929",
                middleware_config=MiddlewareConfig(
                    todo_list=False,
                    filesystem=False,
                    subagents=False,
                    summarization=False,
                    prompt_caching=False,
                    patch_tool_calls=False,
                ),
                middleware=[MyOnlyMiddleware()],
            )
    """

    todo_list: bool | AgentMiddleware = True
    """Control the TodoListMiddleware.

    When enabled, provides the ``write_todos`` tool for task planning and tracking.
    """

    filesystem: bool | AgentMiddleware = True
    """Control the FilesystemMiddleware.

    When enabled, provides file tools: ``ls``, ``read_file``, ``write_file``,
    ``edit_file``, ``glob``, ``grep``, and optionally ``execute``.
    """

    subagents: bool | AgentMiddleware = True
    """Control the SubAgentMiddleware.

    When enabled, provides the ``task`` tool for spawning ephemeral subagents.
    Note: Disabling this also disables the general-purpose subagent.
    """

    summarization: bool | AgentMiddleware = True
    """Control the SummarizationMiddleware.

    When enabled, automatically summarizes older messages to manage context window.
    Disabling may cause context overflow errors on long conversations.
    """

    prompt_caching: bool | AgentMiddleware = True
    """Control the AnthropicPromptCachingMiddleware.

    When enabled, enables prompt caching for Anthropic models to reduce latency
    and costs. Automatically ignored for non-Anthropic models.
    """

    patch_tool_calls: bool | AgentMiddleware = True
    """Control the PatchToolCallsMiddleware.

    When enabled, patches dangling tool calls in message history to prevent
    errors. Disabling may cause issues with interrupted conversations.
    """


def resolve_middleware(
    config_value: bool | AgentMiddleware,
    default_factory: callable,
) -> AgentMiddleware | None:
    """Resolve a middleware config value to an actual middleware instance or None.

    Args:
        config_value: The configuration value (True, False, or a middleware instance).
        default_factory: A callable that returns the default middleware instance.

    Returns:
        The middleware instance to use, or None if disabled.
    """
    if config_value is False:
        return None
    if config_value is True:
        return default_factory()
    # It's a custom middleware instance
    return config_value
