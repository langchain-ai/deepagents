"""Private subagent middleware with cleaner interface."""

from collections.abc import Awaitable, Callable
from typing import Annotated, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool, ToolRuntime
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
from deepagents.middleware.summarization import SummarizationMiddleware

# State keys excluded from subagent state transfer
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata"}

# General-purpose subagent constants
DEFAULT_GP_DESCRIPTION = (
    "General-purpose agent for researching complex questions, "
    "searching for files and content, and executing multi-step tasks. "
    "This agent has access to all tools as the main agent."
)

DEFAULT_GP_SYSTEM_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_description>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
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
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501


# Base spec for general-purpose subagent (caller adds model, tools, middleware)
GENERAL_PURPOSE_SUBAGENT = {
    "name": "general-purpose",
    "description": DEFAULT_GP_DESCRIPTION,
    "system_prompt": DEFAULT_GP_SYSTEM_PROMPT,
}


class SubAgentMiddleware(AgentMiddleware):
    """Middleware that provides subagents via a `task` tool.

    This middleware adds a `task` tool to the agent that can invoke subagents.
    Each subagent must be fully specified with its own model and tools.

    The middleware constructs a base middleware stack internally for all subagents:
    - TodoListMiddleware
    - FilesystemMiddleware
    - SummarizationMiddleware
    - AnthropicPromptCachingMiddleware
    - PatchToolCallsMiddleware

    Then any `middleware` specified in the SubAgent spec is appended.

    Args:
        backend: Backend for file operations and execution.
        subagents: List of fully-specified subagent configs. Each SubAgent
            must specify `model` and `tools`. Optional `interrupt_on` on
            individual subagents is respected.
        system_prompt: Instructions appended to main agent's system prompt
            about how to use the task tool.
        task_description: Custom description for the task tool.
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory,
        subagents: list[SubAgent | CompiledSubAgent],
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = TASK_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()

        if not subagents:
            raise ValueError("At least one subagent must be specified")

        self._backend = backend
        self._subagents = subagents

        # Build system prompt with available agents
        subagent_descriptions = [f"- {agent['name']}: {agent['description']}" for agent in subagents]
        if system_prompt and subagent_descriptions:
            agents_section = "\n\nAvailable subagent types:\n" + "\n".join(subagent_descriptions)
            self.system_prompt = system_prompt + agents_section
        else:
            self.system_prompt = system_prompt

        # Create the task tool
        self.tools = [self._create_task_tool(task_description)]

    def _build_base_middleware(self, model: BaseChatModel) -> list[AgentMiddleware]:
        """Construct the base middleware stack for a subagent.

        Args:
            model: The subagent's model, used for SummarizationMiddleware.

        Returns:
            List of middleware to apply to the subagent.
        """
        if (
            model.profile is not None
            and isinstance(model.profile, dict)
            and "max_input_tokens" in model.profile
            and isinstance(model.profile["max_input_tokens"], int)
        ):
            trigger = ("fraction", 0.85)
            keep = ("fraction", 0.10)
            truncate_args_settings = {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            }
        else:
            trigger = ("tokens", 170000)
            keep = ("messages", 6)
            truncate_args_settings = {
                "trigger": ("messages", 20),
                "keep": ("messages", 20),
            }

        return [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=self._backend),
            SummarizationMiddleware(
                model=model,
                backend=self._backend,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
                truncate_args_settings=truncate_args_settings,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]

    def _build_subagent_graphs(self) -> dict[str, Runnable]:
        """Create runnable agents from specs.

        Returns:
            Dict mapping subagent names to runnable agents.
        """
        agents: dict[str, Runnable] = {}

        for spec in self._subagents:
            if "runnable" in spec:
                # CompiledSubAgent - use as-is
                compiled = cast("CompiledSubAgent", spec)
                agents[compiled["name"]] = compiled["runnable"]
                continue

            # SubAgent - validate required fields
            if "model" not in spec:
                raise ValueError(f"SubAgent '{spec['name']}' must specify 'model'")
            if "tools" not in spec:
                raise ValueError(f"SubAgent '{spec['name']}' must specify 'tools'")

            # Resolve model if string
            model = spec["model"]
            if isinstance(model, str):
                model = init_chat_model(model)

            # Build middleware: base stack + user's middleware + interrupt_on
            middleware: list[AgentMiddleware] = [
                *self._build_base_middleware(model),
                *spec.get("middleware", []),
            ]
            if spec.get("interrupt_on"):
                middleware.append(HumanInTheLoopMiddleware(interrupt_on=spec["interrupt_on"]))

            agents[spec["name"]] = create_agent(
                model,
                system_prompt=spec["system_prompt"],
                tools=spec["tools"],
                middleware=middleware,
                name=spec["name"],
            )

        return agents

    def _create_task_tool(self, task_description: str | None) -> BaseTool:
        """Create the task tool for invoking subagents.

        Args:
            task_description: Custom description for the tool, or None for default.

        Returns:
            A StructuredTool that can invoke subagents.
        """
        subagent_graphs = self._build_subagent_graphs()

        # Build description
        agent_list = "\n".join(f"- {spec['name']}: {spec['description']}" for spec in self._subagents)
        description = task_description or f"Launch a subagent.\n\nAvailable agents:\n{agent_list}"

        # Format available_agents placeholder if present
        if "{available_agents}" in description:
            description = description.format(available_agents=agent_list)

        def _prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
            """Prepare state for subagent invocation."""
            subagent = subagent_graphs[subagent_type]
            state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
            state["messages"] = [HumanMessage(content=description)]
            return subagent, state

        def _make_result(result: dict, tool_call_id: str) -> Command:
            """Convert subagent result to Command with state update."""
            if "messages" not in result:
                raise ValueError(
                    "CompiledSubAgent must return a state containing a 'messages' key. "
                    "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                    "in their state schema to communicate results back to the main agent."
                )
            state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
            message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
            return Command(update={**state_update, "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)]})

        def task(
            description: Annotated[str, "A detailed description of the task for the subagent to perform autonomously."],
            subagent_type: Annotated[str, "The type of subagent to use."],
            runtime: ToolRuntime,
        ) -> str | Command:
            if subagent_type not in subagent_graphs:
                allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
                return f"Unknown subagent `{subagent_type}`. Available: {allowed}"
            subagent, state = _prepare_state(subagent_type, description, runtime)
            result = subagent.invoke(state)
            if not runtime.tool_call_id:
                raise ValueError("Tool call ID is required for subagent invocation")
            return _make_result(result, runtime.tool_call_id)

        async def atask(
            description: Annotated[str, "A detailed description of the task for the subagent to perform autonomously."],
            subagent_type: Annotated[str, "The type of subagent to use."],
            runtime: ToolRuntime,
        ) -> str | Command:
            if subagent_type not in subagent_graphs:
                allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
                return f"Unknown subagent `{subagent_type}`. Available: {allowed}"
            subagent, state = _prepare_state(subagent_type, description, runtime)
            result = await subagent.ainvoke(state)
            if not runtime.tool_call_id:
                raise ValueError("Tool call ID is required for subagent invocation")
            return _make_result(result, runtime.tool_call_id)

        return StructuredTool.from_function(
            name="task",
            func=task,
            coroutine=atask,
            description=description,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt:
            new_msg = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_msg))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt:
            new_msg = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_msg))
        return await handler(request)
