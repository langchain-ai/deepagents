"""Middleware for loading agent memory/context from AGENTS.md files.

This module implements support for the AGENTS.md specification (https://agents.md/),
loading memory/context from configurable sources and injecting into the system prompt.

## Overview

AGENTS.md files provide project-specific context and instructions to help AI agents
work effectively. Unlike skills (which are on-demand workflows), memory is always
loaded and provides persistent context.

## Usage

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# Security: FilesystemBackend allows reading/writing from the entire filesystem.
# Either ensure the agent is running within a sandbox OR add human-in-the-loop (HIL)
# approval to file operations.
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",
        "./.deepagents/AGENTS.md",
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## Memory Sources

Sources are simply paths to AGENTS.md files that are loaded in order and combined.
Multiple sources are concatenated in order, with all content included.
Later sources appear after earlier ones in the combined prompt.

## File Format

AGENTS.md files are standard Markdown with no required structure.
Common sections include:
- Project overview
- Build/test commands
- Code style guidelines
- Architecture notes
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain.tools import ToolRuntime
from langgraph.runtime import Runtime

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """State schema for `MemoryMiddleware`.

    Attributes:
        memory_contents: Dict mapping source paths to their loaded content.
            Marked as private so it's not included in the final agent state.
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """State update for `MemoryMiddleware`."""

    memory_contents: dict[str, str]


MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    以上 <agent_memory> 内容是从你的文件系统中的文件加载的。随着你与用户的互动学习新信息，可以通过调用 `edit_file` 工具保存新知识。

    **学习反馈：**
    - 你的主要优先级之一是从与用户的互动中学习。这些学习可以是隐性的或显性的，这意味着你会在未来记住这些重要信息。
    - 当你需要记住某件事时，更新记忆必须是你的第一、立即行动——在回复用户之前、调用其他工具之前、做任何事之前，先立即更新记忆。
    - 当用户说某件事更好/更差时，记录原因并抽象成模式。
    - 每一次纠正都是永久改进的机会——不要只修复当下问题，要更新你的指令。
    - 当用户在工具调用过程中打断并提供反馈时，是更新记忆的好时机。在修改该工具调用之前先立即更新记忆。
    - 关注纠正背后的原则，而不仅是具体错误。
    - 用户可能不会明确要求你记住某事，但如果他们提供了对未来有用的信息，你应立即更新记忆。

    **询问信息：**
    - 如果你缺少执行操作所需的上下文（例如发送 Slack 私信需要用户 ID/邮箱），应明确向用户询问信息。
    - 优先选择提问，不要假设你不知道的信息！
    - 当用户提供了对未来有用的信息时，你应立即更新记忆。

    **何时更新记忆：**
    - 当用户明确要求你记住某事（例如“记住我的邮箱”“保存这个偏好”）
    - 当用户描述你的角色或你应如何行为（例如“你是网页研究员”“总是做 X”）
    - 当用户对你的工作给出反馈——记录哪里错了以及如何改进
    - 当用户提供了工具使用所需的信息（例如 Slack 频道 ID、邮箱地址）
    - 当用户提供了对未来任务有用的上下文，例如如何使用工具，或在特定情境下应采取哪些动作
    - 当你发现新的模式或偏好（编码风格、约定、工作流）

    **何时不要更新记忆：**
    - 当信息是临时或短暂的（例如“我要迟到了”“我现在在手机上”）
    - 当信息是一次性任务请求（例如“帮我找个菜谱”“25 * 4 是多少？”）
    - 当信息是简单问题且不体现长期偏好（例如“今天星期几？”“能解释 X 吗？”）
    - 当信息只是确认或闲聊（例如“听起来不错！”“你好”“谢谢你”）
    - 当信息在未来对话中陈旧或无关
    - 绝不要在任何文件、记忆或系统提示词中存储 API 密钥、访问令牌、密码或任何其他凭据。
    - 如果用户询问 API 密钥应放在哪里或提供了 API 密钥，不要复述或保存它。

    **示例：**
    示例 1（记住用户信息）：
    用户：你能连接我的 Google 账号吗？
    助手：好的，我会连接你的 Google 账号，你的 Google 账号邮箱是什么？
    用户：john@example.com
    助手：我把它记到我的记忆里。
    工具调用：edit_file(...) -> 记住用户的 Google 账号邮箱是 john@example.com

    示例 2（记住隐含偏好）：
    用户：能给我写一个在 LangChain 中创建 deep agent 的例子吗？
    助手：好的，我会写一个在 LangChain 中创建 deep agent 的示例 <Python 示例代码>
    用户：能用 JavaScript 写吗？
    助手：我把这个偏好记到记忆里。
    工具调用：edit_file(...) -> 记住用户更偏好 JavaScript 的 LangChain 代码示例
    助手：好的，下面是 JavaScript 示例 <JavaScript 示例代码>

    示例 3（不要记住短暂信息）：
    用户：我今晚要去打篮球，几个小时内会离线。
    助手：好的，我会在你的日历里添加一个时间占用。
    工具调用：create_calendar_event(...) -> 只是调用工具，不写入记忆，因为这是短暂信息
</memory_guidelines>
"""


class MemoryMiddleware(AgentMiddleware):
    """Middleware for loading agent memory from `AGENTS.md` files.

    Loads memory content from configured sources and injects into the system prompt.

    Supports multiple sources that are combined together.

    Args:
        backend: Backend instance or factory function for file operations.
        sources: List of `MemorySource` configurations specifying paths and names.
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        """Initialize the memory middleware.

        Args:
            backend: Backend instance or factory function that takes runtime
                     and returns a backend. Use a factory for StateBackend.
            sources: List of memory file paths to load (e.g., `["~/.deepagents/AGENTS.md",
                     "./.deepagents/AGENTS.md"]`).

                     Display names are automatically derived from the paths.

                     Sources are loaded in order.
        """
        self._backend = backend
        self.sources = sources

    def _get_backend(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            # Construct an artificial tool runtime to resolve backend factory
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """Format memory with locations and contents paired together.

        Args:
            contents: Dict mapping source paths to content.

        Returns:
            Formatted string with location+content pairs wrapped in <agent_memory> tags.
        """
        if not contents:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(未加载记忆)")

        sections = []
        for path in self.sources:
            if contents.get(path):
                sections.append(f"{path}\n{contents[path]}")

        if not sections:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(未加载记忆)")

        memory_body = "\n\n".join(sections)
        return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)

    async def _load_memory_from_backend(
        self,
        backend: BackendProtocol,
        path: str,
    ) -> str | None:
        """Load memory content from a backend path.

        Args:
            backend: Backend to load from.
            path: Path to the AGENTS.md file.

        Returns:
            File content if found, None otherwise.
        """
        results = await backend.adownload_files([path])
        # Should get exactly one response for one path
        if len(results) != 1:
            raise AssertionError(f"Expected 1 response for path {path}, got {len(results)}")
        response = results[0]

        if response.error is not None:
            # For now, memory files are treated as optional. file_not_found is expected
            # and we skip silently to allow graceful degradation.
            if response.error == "file_not_found":
                return None
            # Other errors should be raised
            raise ValueError(f"Failed to download {path}: {response.error}")

        if response.content is not None:
            return response.content.decode("utf-8")

        return None

    def _load_memory_from_backend_sync(
        self,
        backend: BackendProtocol,
        path: str,
    ) -> str | None:
        """Load memory content from a backend path synchronously.

        Args:
            backend: Backend to load from.
            path: Path to the AGENTS.md file.

        Returns:
            File content if found, None otherwise.
        """
        results = backend.download_files([path])
        # Should get exactly one response for one path
        if len(results) != 1:
            raise AssertionError(f"Expected 1 response for path {path}, got {len(results)}")
        response = results[0]

        if response.error is not None:
            # For now, memory files are treated as optional. file_not_found is expected
            # and we skip silently to allow graceful degradation.
            if response.error == "file_not_found":
                return None
            # Other errors should be raised
            raise ValueError(f"Failed to download {path}: {response.error}")

        if response.content is not None:
            return response.content.decode("utf-8")

        return None

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:
        """Load memory content before agent execution (synchronous).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # Skip if already loaded
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        for path in self.sources:
            content = self._load_memory_from_backend_sync(backend, path)
            if content:
                contents[path] = content
                logger.debug(f"Loaded memory from: {path}")

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:
        """Load memory content before agent execution.

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with memory_contents populated.
        """
        # Skip if already loaded
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        for path in self.sources:
            content = await self._load_memory_from_backend(backend, path)
            if content:
                contents[path] = content
                logger.debug(f"Loaded memory from: {path}")

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject memory content into the system message.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with memory injected into system message.
        """
        contents = request.state.get("memory_contents", {})
        agent_memory = self._format_agent_memory(contents)

        new_system_message = append_to_system_message(request.system_message, agent_memory)

        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
