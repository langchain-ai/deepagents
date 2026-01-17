"""Middleware for providing subagents to an agent via a `task` tool."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message


class SubAgent(TypedDict):
    """Specification for an agent.

    When specifying custom agents, the `default_middleware` from `SubAgentMiddleware`
    will be applied first, followed by any `middleware` specified in this spec.
    To use only custom middleware without the defaults, pass `default_middleware=[]`
    to `SubAgentMiddleware`.

    Required fields:
        name: Unique identifier for the subagent.

            The main agent uses this name when calling the `task()` tool.
        description: What this subagent does.

            Be specific and action-oriented. The main agent uses this to decide when to delegate.
        system_prompt: Instructions for the subagent.

            Include tool usage guidance and output format requirements.
        tools: Tools the subagent can use.

            Keep this minimal and include only what's needed.

    Optional fields:
        model: Override the main agent's model.

            Use the format `'provider:model-name'` (e.g., `'openai:gpt-4o'`).
        middleware: Additional middleware for custom behavior, logging, or rate limiting.
        interrupt_on: Configure human-in-the-loop for specific tools.

            Requires a checkpointer.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does. The main agent uses this to decide when to delegate."""

    system_prompt: str
    """Instructions for the subagent."""

    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    """Tools the subagent can use."""

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model. Use `'provider:model-name'` format."""

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec.

    !!! note

        The runnable's state schema must include a 'messages' key.

        This is required for the subagent to communicate results back to the main agent.

    When the subagent completes, the final message in the 'messages' list will be
    extracted and returned as a `ToolMessage` to the parent agent.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does."""

    runnable: Runnable
    """A custom agent implementation.

    Create a custom agent using either:

    1. LangChain's [`create_agent()`](https://docs.langchain.com/oss/python/langchain/quickstart)
    2. A custom graph using [`langgraph`](https://docs.langchain.com/oss/python/langgraph/quickstart)

    If you're creating a custom graph, make sure the state schema includes a 'messages' key.
    This is required for the subagent to communicate results back to the main agent.
    """


DEFAULT_SUBAGENT_PROMPT = "为了完成用户提出的目标，你可以使用一系列标准工具。"

# State keys that are excluded when passing state to subagents and when returning
# updates from subagents.
# When returning updates:
# 1. The messages key is handled explicitly to ensure only the final message is included
# 2. The todos and structured_response keys are excluded as they do not have a defined reducer
#    and no clear meaning for returning them from a subagent to the main agent.
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response"}

TASK_TOOL_DESCRIPTION = """启动一个短暂的子代理，用于处理复杂、多步骤且彼此独立的任务，并隔离上下文窗口。

可用的代理类型及其可用工具：
{available_agents}

使用 Task 工具时，你必须指定 subagent_type 参数来选择要使用的代理类型。

## 使用说明：
1. 尽可能并行启动多个代理以提升性能；为此，在一条消息中发起多个工具调用
2. 当代理完成后，它会返回一条消息给你。该结果对用户不可见。要向用户展示结果，你应发送一条简洁的文本消息总结结果。
3. 每次代理调用都是无状态的。你无法向代理发送追加消息，代理也无法在最终报告之外与你通信。因此，你的提示必须包含非常详细的任务描述，并明确它在最终且唯一的回复中应返回哪些信息。
4. 代理的输出通常可以信任
5. 清楚告诉代理你希望它创建内容、进行分析，还是仅做研究（搜索、文件读取、网页抓取等），因为它不知道用户意图
6. 如果代理描述提到应主动使用，那么你应尽量在用户未要求前就使用它。自行判断。
7. 当只提供通用代理时，你应使用它处理所有任务。它非常适合隔离上下文与 token 使用，并完成特定复杂任务，因为它拥有与主代理相同的能力。

### 通用代理示例：

<example_agent_descriptions>
"general-purpose": 用于通用任务的代理，拥有与主代理相同的全部工具。
</example_agent_descriptions>

<example>
用户："我想研究 LeBron James、Michael Jordan 和 Kobe Bryant 的成就，并进行对比。"
助手：*并行使用 task 工具分别研究三位球员*
助手：*综合三次独立研究的结果并回复用户*
<commentary>
研究本身就是一个复杂的多步骤任务。
对每位球员的研究彼此独立。
助手使用 task 工具将复杂目标拆成三个相互独立的任务。
每个研究任务只需关注一个球员的上下文与 token，并作为工具结果返回该球员的综合信息。
这样每个任务都可以深入研究各自对象，最终再综合结果，有助于节省对比时的 token。
</commentary>
</example>

<example>
用户："分析一个大型代码仓库的安全漏洞并生成报告。"
助手：*启动单个 `task` 子代理进行仓库分析*
助手：*收到报告并整合到最终摘要中*
<commentary>
即使只有一个任务，子代理也能隔离大型、上下文密集的任务，避免主线程被细节淹没。
如果用户随后追问，我们只需要引用简洁报告，而不是完整的分析历史与工具调用，这能节省时间与成本。
</commentary>
</example>

<example>
用户："帮我安排两场会议，并为每场准备议程。"
助手：*并行调用 task 工具启动两个 `task` 子代理（每场会议一个）来准备议程*
助手：*返回最终日程与议程*
<commentary>
单个任务不复杂，但子代理可以隔离每场会议的议程准备。
每个子代理只需关注一场会议的议程。
</commentary>
</example>

<example>
用户："我想从 Dominos 订披萨、从 McDonald's 订汉堡、从 Subway 订沙拉。"
助手：*直接并行调用工具完成三项订单*
<commentary>
助手没有使用 task 工具，因为目标简单清晰，只需少量工具调用。
这种情况更适合直接完成任务，而不是使用 `task` 工具。
</commentary>
</example>

### 自定义代理示例：

<example_agent_descriptions>
"content-reviewer": 在你完成重要内容或文档后使用该代理进行审阅
"greeting-responder": 当需要用友好笑话回应用户问候时使用该代理
"research-analyst": 用于对复杂主题进行深入研究的代理
</example_agent_description>

<example>
user: "请写一个判断整数是否为质数的函数"
assistant: 好的，我来写一个检查质数的函数
assistant: 先用 Write 工具写这个函数
assistant: 我将使用 Write 工具写入以下代码：
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
由于已经创建了重要内容并完成任务，现在使用 content-reviewer 代理来审阅成果
</commentary>
assistant: 现在使用 content-reviewer 代理审阅代码
assistant: 使用 Task 工具启动 content-reviewer 代理
</example>

<example>
user: "能帮我研究不同可再生能源的环境影响并生成一份综合报告吗？"
<commentary>
这是一个复杂研究任务，适合使用 research-analyst 代理进行深入分析
</commentary>
assistant: 我来帮你研究不同可再生能源的环境影响。让我用 research-analyst 代理进行全面研究。
assistant: 使用 Task 工具启动 research-analyst 代理，并提供详细研究要求与报告格式
</example>

<example>
user: "你好"
<commentary>
用户在打招呼，使用 greeting-responder 代理用友好的笑话回应
</commentary>
assistant: "我要使用 Task 工具启动 greeting-responder 代理"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task`（子代理启动器）

你可以使用 `task` 工具启动短生命周期的子代理来处理隔离任务。这些代理是临时的——只在任务期间存在，并返回单条结果。

何时使用 task 工具：
- 当任务复杂且多步骤，并且可以在隔离环境中完整委派
- 当任务与其他任务相互独立且可并行
- 当任务需要专注推理或大量 token/上下文，会使主线程膨胀
- 当沙箱能提升可靠性（例如代码执行、结构化搜索、数据格式化）
- 当你只关心子代理输出，而不关心中间步骤（例如进行大量研究后返回综合报告，或执行一系列计算/查询得到简洁答案）

子代理生命周期：
1. **启动** → 提供清晰角色、指令与期望输出
2. **执行** → 子代理自主完成任务
3. **返回** → 子代理提供单条结构化结果
4. **整合** → 将结果融入或综合到主线程

何时不要使用 task 工具：
- 如果你需要在子代理完成后看到中间推理或步骤（task 工具会隐藏它们）
- 如果任务很琐碎（少量工具调用或简单查询）
- 如果委派不能减少 token 使用、复杂度或上下文切换
- 如果拆分只会增加延迟而无收益

## 需要牢记的 Task 工具使用说明
- 只要可能就并行化你的工作。这既适用于工具调用，也适用于任务。只要步骤彼此独立，就用并行 tool calls 或启动并行任务。这会节省用户时间，这非常重要。
- 记得使用 `task` 工具来隔离多部分目标中的独立任务。
- 当你有复杂、多步骤且与其他任务独立的工作时，应该使用 `task` 工具。这些代理非常高效且能力强。"""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "通用代理，用于研究复杂问题、搜索文件与内容，以及执行多步骤任务。当你在搜索关键词或文件且不确定几次就能找到合适匹配时，让它替你进行搜索。该代理拥有与主代理相同的全部工具。"  # noqa: E501


def _get_subagents(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Create subagent instances from specifications.

    Args:
        default_model: Default model for subagents that don't specify one.
        default_tools: Default tools for subagents that don't specify tools.
        default_middleware: Middleware to apply to all subagents. If `None`,
            no default middleware is applied.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: List of agent specifications or pre-compiled agents.
        general_purpose_agent: Whether to include a general-purpose subagent.

    Returns:
        Tuple of (agent_dict, description_list) where agent_dict maps agent names
        to runnable instances and description_list contains formatted descriptions.
    """
    # Use empty list if None (no default middleware)
    default_subagent_middleware = default_middleware or []

    agents: dict[str, Any] = {}
    subagent_descriptions = []

    # Create general-purpose agent if enabled
    if general_purpose_agent:
        general_purpose_middleware = [*default_subagent_middleware]
        if default_interrupt_on:
            general_purpose_middleware.append(HumanInTheLoopMiddleware(interrupt_on=default_interrupt_on))
        general_purpose_subagent = create_agent(
            default_model,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            tools=default_tools,
            middleware=general_purpose_middleware,
            name="general-purpose",
        )
        agents["general-purpose"] = general_purpose_subagent
        subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    # Process custom subagents
    for agent_ in subagents:
        subagent_descriptions.append(f"- {agent_['name']}: {agent_['description']}")
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = [*default_subagent_middleware, *agent_["middleware"]] if "middleware" in agent_ else [*default_subagent_middleware]

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        agents[agent_["name"]] = create_agent(
            subagent_model,
            system_prompt=agent_["system_prompt"],
            tools=_tools,
            middleware=_middleware,
            name=agent_["name"],
        )
    return agents, subagent_descriptions


def _create_task_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool for invoking subagents.

    Args:
        default_model: Default model for subagents.
        default_tools: Default tools for subagents.
        default_middleware: Middleware to apply to all subagents.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: List of subagent specifications.
        general_purpose_agent: Whether to include general-purpose agent.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        # Validate that the result contains a 'messages' key
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        # Strip trailing whitespace to prevent API errors with Anthropic
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = subagent_graphs[subagent_type]
        # Create a new state dict to avoid mutating the original
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        # If custom description has placeholder, format with agent descriptions
        task_description = task_description.format(available_agents=subagent_description_str)

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents to an agent via a `task` tool.

    This  middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require a narrower
    subset of tools and focus.

    This middleware comes with a default general-purpose subagent that can be used to
    handle the same tasks as the main agent, but with isolated context.

    Args:
        default_model: The model to use for subagents.

            Can be a `LanguageModelLike` or a dict for `init_chat_model`.
        default_tools: The tools to use for the default general-purpose subagent.
        default_middleware: Default middleware to apply to all subagents.

            If `None`, no default middleware is applied.

            Pass a list to specify custom middleware.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent.

            These are also the fallback for any subagents that don't specify their own tool configs.
        subagents: A list of additional subagents to provide to the agent.
        system_prompt: Full system prompt override. When provided, completely replaces
            the agent's system prompt.
        general_purpose_agent: Whether to include the general-purpose agent.
        task_description: Custom description for the task tool.

            If `None`, uses the default description template.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        # Basic usage with defaults (no default middleware)
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    subagents=[],
                )
            ],
        )

        # Add custom middleware to subagents
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    default_middleware=[TodoListMiddleware()],
                    subagents=[],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()
        self.system_prompt = system_prompt
        task_tool = _create_task_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
