"""Middleware for providing subagents to an agent via a `task` tool."""

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.permissions import FilesystemPermission


class SubAgent(TypedDict):
    """Specification for an agent.

    When using `create_deep_agent`, subagents automatically receive a default middleware
    stack (TodoListMiddleware, FilesystemMiddleware, SummarizationMiddleware, etc.) before
    any custom `middleware` specified in this spec.

    Required fields:
        name: Unique identifier for the subagent.

            The main agent uses this name when calling the `task()` tool.
        description: What this subagent does.

            Be specific and action-oriented. The main agent uses this to decide when to delegate.
        system_prompt: Instructions for the subagent.

            Include tool usage guidance and output format requirements.

    Optional fields:
        tools: Tools the subagent can use.

            If not specified, inherits tools from the main agent via `default_tools`.
        model: Override the main agent's model.

            Use the format `'provider:model-name'` (e.g., `'openai:gpt-4o'`).
        middleware: Additional middleware for custom behavior, logging, or rate limiting.
        interrupt_on: Configure human-in-the-loop for specific tools.

            Requires a checkpointer.
        skills: Skill source paths for SkillsMiddleware.

            List of paths to skill directories (e.g., `["/skills/user/", "/skills/project/"]`).
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does. The main agent uses this to decide when to delegate."""

    system_prompt: str
    """Instructions for the subagent."""

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use. If not specified, inherits from main agent."""

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model. Use `'provider:model-name'` format."""

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""

    skills: NotRequired[list[str]]
    """Skill source paths for SkillsMiddleware."""

    permissions: NotRequired[list[FilesystemPermission]]
    """List of ``FilesystemPermission`` rules for this subagent.

    If omitted, inherits the parent agent's permissions. If specified, replaces
    the parent's permissions entirely for this subagent.

    Rules are evaluated in declaration order; the first match wins.
    ``_PermissionMiddleware`` is appended last in the middleware stack.
    """


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


DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# State keys that are excluded when passing state to subagents and when returning
# updates from subagents.
#
# When returning updates:
# 1. The messages key is handled explicitly to ensure only the final message is included
# 2. The todos and structured_response keys are excluded as they do not have a defined reducer
#    and no clear meaning for returning them from a subagent to the main agent.
# 3. The skills_metadata and memory_contents keys are automatically excluded from subagent output
#    via PrivateStateAttr annotations on their respective state schemas. However, they must ALSO
#    be explicitly filtered from runtime.state when invoking a subagent to prevent parent state
#    from leaking to child agents (e.g., the general-purpose subagent loads its own skills via
#    SkillsMiddleware).
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response", "skills_metadata", "memory_contents"}


class TaskToolSchema(BaseModel):
    """Input schema for the `task` tool."""

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )
    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))


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
It is better to just complete the task directly and NOT use the `task` tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_descriptions>

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

_TASK_ONLY_SYSTEM_PROMPT = """## `task` (subagent spawner)

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

Subagent limitations:
- A subagent has the same context window and tool limitations you do. If a file is too large for you to process in one pass, it's too large for a single subagent too — split the work across multiple subagents instead.

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501

_SWARM_SYSTEM_PROMPT_ADDITION = """

## `swarm` (parallel subagent execution)

Use `swarm` to fan out many independent tasks across multiple subagents and aggregate their results.

### When to use swarm

**Trigger condition**: Use swarm when the input contains too much data to process in a single pass. Indicators: the file or dataset contains hundreds of items that each need individual analysis, or exceeds ~500 lines. When in doubt, check the size and prefer swarm over attempting to process a large input inline.

Also use `swarm` when:
- A task requires applying intelligence to each item in a large collection
- Work can be decomposed into many independent, parallel subtasks

Use `task` instead when:
- You have a small number of independent subtasks
- Each subtask depends on the result of a previous one
- The work is exploratory or adaptive

### How to use swarm

Before calling swarm, understand what you're working with. Explore the data to learn its structure, format, and content using whatever tools are available. The goal is to write task descriptions detailed enough that each subagent can execute without needing to figure anything out on its own.

Once you understand the data:

1. **Generate tasks.** Write a generation script via `execute` that produces a `tasks.jsonl` file — one JSON object per line, each with `id`, `description`, and optional `subagent_type`. Each task should be a self-contained unit of work. **Prefer many small tasks over few large ones** — all tasks run in parallel, so 50 small tasks finish in roughly the same wall-clock time as 5 large ones. When splitting a file, aim for **30-60 lines** per chunk.
2. **Call swarm.** Pass the path to your `tasks.jsonl` file.
3. **Aggregate results.** Write an aggregation script via `execute` that reads `<results_dir>/results.jsonl` and combines the subagent outputs into a final answer.

### Task description quality

Each subagent receives **only its task description** — no other context. The quality of your descriptions determines the quality of swarm results. Invest time upfront to get them right.

Good task descriptions are **prescriptive**: they tell the subagent the data format, the processing logic, the exact range of data to work on, and the expected output format. The subagent should not need to explore or interpret — just execute.

When subagent results need to be aggregated (counting, classification, extraction), instruct each subagent to respond with **structured JSON only** — no explanations, no tables, just the JSON object. Include the exact output schema in the task description.

### Error handling

Each task runs exactly once — there are no automatic retries. If some tasks fail, the swarm summary includes a `failed_tasks` array with each failed task's ID and error message. Use this to decide:
- **Retry via swarm**: generate a new tasks.jsonl targeting just the failures (with modifications) and call swarm again.
- **Retry individually**: use `task` for a small number of failures.
- **Proceed with partial results**: aggregate what completed and skip the rest.

### Important: one swarm call per question

**Never re-run swarm to verify or cross-check results.** Swarm is expensive — treat the first run's per-task outputs as authoritative. If you need to validate, do it in the aggregation script (e.g., check that each chunk returned the expected number of items). Do not generate a second tasks.jsonl or call swarm again for the same question.

### Decomposition patterns

**Flat fan-out**: Split a dataset into equal chunks. All tasks are identical in structure.
Good for: large files, classification, extraction.

**One-per-item**: One task per discrete unit (file, document, URL).
Good for: summarizing collections, processing independent documents.

**Dimensional**: Multiple tasks examine the same input from different angles.
Good for: code review, multi-criteria evaluation."""  # noqa: E501

TASK_SYSTEM_PROMPT = _TASK_ONLY_SYSTEM_PROMPT + _SWARM_SYSTEM_PROMPT_ADDITION


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

# Base spec for general-purpose subagent (caller adds model, tools, middleware)
GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}


class _SubagentSpec(TypedDict):
    """Internal spec for building the task tool."""

    name: str
    description: str
    runnable: Runnable


def _build_task_tool(  # noqa: C901
    subagents: list[_SubagentSpec],
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool from pre-built subagent graphs.

    Args:
        subagents: List of subagent specs containing name, description, and runnable.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """
    # Build the graphs dict and descriptions from the unified spec list
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagents}
    subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=subagent_description_str)
    else:
        description = task_description

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

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = subagent.invoke(subagent_state)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)
        result = await subagent.ainvoke(subagent_state)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema,
    )


DEFAULT_SWARM_CONCURRENCY = 10
"""Default concurrency limit for swarm execution."""

MAX_SWARM_CONCURRENCY = 50
"""Maximum allowed concurrency."""

SWARM_TASK_TIMEOUT_SECONDS = 300
"""Per-task timeout in seconds."""

SWARM_TOOL_DESCRIPTION = """Execute a batch of independent tasks in parallel across multiple subagents.

## Workflow

1. Write a generation script via `execute` that produces a tasks.jsonl file with one JSON object per line:
   ```json
   {{"id": "chunk_0", "description": "Read lines 1-100 of data.txt. Process each item. Return JSON results.", "subagent_type": "general-purpose"}}
   {{"id": "chunk_1", "description": "Read lines 101-200 of data.txt. Process each item. Return JSON results.", "subagent_type": "general-purpose"}}
   ```
2. Call `swarm` with the path to the tasks.jsonl file.
3. The tool returns a JSON summary with `total`, `completed`, `failed`, and `results_dir`.
   Results are written to `<results_dir>/results.jsonl` — each line is the original task enriched with `status`, `result`, and/or `error` fields.
4. Write an aggregation script via `execute` that reads `<results_dir>/results.jsonl` and combines the outputs.

## tasks.jsonl fields

- "id" (string, required): unique task identifier
- "description" (string, required): complete, self-contained prompt — the subagent receives NOTHING else
- "subagent_type" (string, optional): which subagent to use (default: "general-purpose")

## After execution

The tool returns:
```json
{{"total": 20, "completed": 19, "failed": 1,
  "results_dir": "swarm_runs/<uuid>",
  "failed_tasks": [{{"id": "chunk_5", "error": "timed out"}}]}}
```

Each task runs exactly once — there are no automatic retries. Use the `failed_tasks` array to decide how to handle failures.

Available subagent types: {available_agents}
"""


class SwarmTaskSpec(TypedDict):
    """A single task line in a ``tasks.jsonl`` file.

    Fields:
        id: Unique task identifier (must be unique within the task list).
        description: Complete, self-contained prompt for the subagent.
        subagent_type: Which subagent to dispatch to. Defaults to
            ``"general-purpose"`` when omitted.
    """

    id: str
    description: str
    subagent_type: NotRequired[str]


class ParsedSwarmConfig(TypedDict):
    """Result of parsing a tasks JSONL file."""

    tasks: list[SwarmTaskSpec] | None
    error: str | None


def _parse_tasks_jsonl(content: str) -> ParsedSwarmConfig:
    """Parse and validate a ``tasks.jsonl`` string into task specs.

    Validates that each line is valid JSON with required fields, that all
    task IDs are unique, and that at least one task is present.

    Args:
        content: Raw JSONL string (one JSON object per line).

    Returns:
        Parsed config with task list or error message.
    """
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        return {"tasks": None, "error": "tasks.jsonl is empty. The generation script must write at least one task."}

    tasks: list[SwarmTaskSpec] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    adapter = TypeAdapter(SwarmTaskSpec)
    for idx, line in enumerate(lines):
        line_number = idx + 1
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"Line {line_number}: invalid JSON")
            continue

        try:
            task = adapter.validate_python(parsed)
        except ValidationError as e:
            messages = [issue["msg"] for issue in e.errors()]
            errors.append(f"Line {line_number}: {', '.join(messages)}")
            continue

        if task["id"] in seen_ids:
            errors.append(f'Line {line_number}: duplicate task id "{task["id"]}"')
            continue

        seen_ids.add(task["id"])
        tasks.append(task)

    if errors:
        return {"tasks": None, "error": "tasks.jsonl validation failed:\n" + "\n".join(errors)}

    return {"tasks": tasks, "error": None}


class SubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for providing subagents to an agent via a `task` tool.

    This middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require a narrower
    subset of tools and focus.

    Args:
        backend: Backend for file operations and execution.
        subagents: List of fully-specified subagent configs. Each SubAgent
            must specify `model` and `tools`. Optional `interrupt_on` on
            individual subagents is respected.
        system_prompt: Instructions appended to main agent's system prompt
            about how to use the task tool.
        task_description: Custom description for the task tool.

    Example:
        ```python
        from deepagents.middleware import SubAgentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    backend=my_backend,
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "Research agent",
                            "system_prompt": "You are a researcher.",
                            "model": "openai:gpt-4o",
                            "tools": [search_tool],
                        }
                    ],
                )
            ],
        )
        ```

    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
        enable_swarm: bool = False,
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()

        if not subagents:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)
        self._backend = backend
        self._subagents = subagents
        subagent_specs = self._get_subagents()

        task_tool = _build_task_tool(subagent_specs, task_description)

        # Use task-only prompt when swarm is disabled
        if not enable_swarm and system_prompt == TASK_SYSTEM_PROMPT:
            system_prompt = _TASK_ONLY_SYSTEM_PROMPT

        # Build system prompt with available agents
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        tools = [task_tool]
        if enable_swarm and backend is not None:
            tools.append(self._build_swarm_tool(subagent_specs))
        self.tools = tools

    def _build_swarm_tool(self, subagents: list[_SubagentSpec]) -> BaseTool:  # noqa: C901, PLR0915
        """Create the `swarm` tool.

        The `swarm` tool reads a JSON config file containing a list of task specs and
        runs those tasks in parallel across subagents.

        Each task result is written to `<output_dir>/<task_id>.txt` via the configured
        backend.

        Args:
            subagents: The available subagents that `swarm` may dispatch to.

        Returns:
            A structured tool named `swarm`.
        """
        backend = self._backend
        if backend is None:
            msg = "backend is required"
            raise ValueError(msg)

        subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagents}
        subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)
        description = SWARM_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)

        async def _run_single_task(
            task: SwarmTaskSpec,
            subagent: Runnable,
            parent_state: dict[str, Any],
        ) -> dict[str, Any]:
            """Run a single task with a timeout. Returns a lean result dict."""
            subagent_state = dict(parent_state)
            subagent_state["messages"] = [HumanMessage(content=task["description"])]
            subagent_type = task.get("subagent_type", "general-purpose")

            try:
                result = await asyncio.wait_for(
                    subagent.ainvoke(subagent_state),
                    timeout=SWARM_TASK_TIMEOUT_SECONDS,
                )
                messages = result.get("messages", [])
                text = messages[-1].text.rstrip() if messages and messages[-1].text else ""
                return {"id": task["id"], "subagent_type": subagent_type, "status": "completed", "result": text}
            except Exception as exc:  # noqa: BLE001
                return {"id": task["id"], "subagent_type": subagent_type, "status": "failed", "error": str(exc)}

        async def _execute_swarm(
            tasks: list[SwarmTaskSpec],
            parent_state: dict[str, Any],
            effective_concurrency: int,
        ) -> list[dict[str, Any]]:
            """Dispatch all tasks in parallel under a concurrency semaphore.

            Each task runs exactly once — there are no retries. The
            orchestrator owns error recovery.
            """
            semaphore = asyncio.Semaphore(effective_concurrency)

            for task in tasks:
                subagent_type = task.get("subagent_type", "general-purpose")
                if subagent_type not in subagent_graphs:
                    allowed = ", ".join(f'"{k}"' for k in subagent_graphs)
                    msg = f'Task "{task["id"]}" references unknown subagent_type "{subagent_type}". Available: {allowed}'
                    raise ValueError(msg)

            async def run_with_semaphore(task: SwarmTaskSpec) -> dict[str, Any]:
                subagent = subagent_graphs[task.get("subagent_type", "general-purpose")]
                async with semaphore:
                    return await _run_single_task(task, subagent, parent_state)

            results = await asyncio.gather(
                *[run_with_semaphore(t) for t in tasks],
                return_exceptions=True,
            )

            final: list[dict[str, Any]] = []
            for idx, raw_result in enumerate(results):
                if isinstance(raw_result, GraphInterrupt):
                    raise raw_result
                if isinstance(raw_result, BaseException):
                    subagent_type = tasks[idx].get("subagent_type", "general-purpose")
                    final.append({"id": tasks[idx]["id"], "subagent_type": subagent_type, "status": "failed", "error": str(raw_result)})
                else:
                    final.append(raw_result)
            return final

        async def aswarm(
            tasks_path: Annotated[str, "Path to the tasks.jsonl file produced by the generation script."],
            runtime: ToolRuntime,
            concurrency: Annotated[
                int | None,
                f"Maximum number of subagents running simultaneously. Default: {DEFAULT_SWARM_CONCURRENCY}, max: {MAX_SWARM_CONCURRENCY}.",
            ] = None,
        ) -> ToolMessage:
            """Run swarm tasks."""
            resolved_backend = backend(runtime) if callable(backend) else backend  # ty: ignore[call-top-callable]

            responses = await resolved_backend.adownload_files([tasks_path])
            response = responses[0]
            if response.error:
                return ToolMessage(
                    content=f'Failed to read tasks file at "{tasks_path}". '
                    f"Ensure the generation script writes the file to this exact path and try again.",
                    status="error",
                    tool_call_id=runtime.tool_call_id,
                )
            file_content = response.content
            if not isinstance(file_content, bytes):
                return ToolMessage(
                    content=f"Content was expected to be bytes. Got {type(file_content)}.",
                    status="error",
                    tool_call_id=runtime.tool_call_id,
                )
            parsed = _parse_tasks_jsonl(file_content.decode("utf-8"))
            if parsed["error"] is not None:
                return ToolMessage(content=parsed["error"], status="error", tool_call_id=runtime.tool_call_id)
            tasks = parsed["tasks"]
            if tasks is None:
                msg = "parsed swarm tasks unexpectedly missing"
                raise AssertionError(msg)

            parent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}

            effective_concurrency = max(1, min(concurrency or DEFAULT_SWARM_CONCURRENCY, MAX_SWARM_CONCURRENCY))

            task_results = await _execute_swarm(tasks, parent_state, effective_concurrency)

            artifacts_root = resolved_backend.artifacts_root if isinstance(resolved_backend, CompositeBackend) else "/"
            _root = artifacts_root.rstrip("/")
            results_dir = f"{_root}/swarm_runs/{uuid.uuid4()}"
            results_path = f"{results_dir}/results.jsonl"
            results_content = "\n".join(json.dumps(r) for r in task_results) + "\n"
            await resolved_backend.aupload_files([(results_path, results_content.encode("utf-8"))])

            completed_count = sum(1 for r in task_results if r["status"] == "completed")
            failed_results = [r for r in task_results if r["status"] == "failed"]
            summary: dict[str, Any] = {
                "total": len(task_results),
                "completed": completed_count,
                "failed": len(failed_results),
                "results_dir": results_dir,
                "failed_tasks": [{"id": r["id"], "error": r.get("error", "unknown")} for r in failed_results],
            }

            return ToolMessage(
                content=json.dumps(summary),
                status="success",
                tool_call_id=runtime.tool_call_id,
            )

        return StructuredTool.from_function(
            name="swarm",
            coroutine=aswarm,
            description=description,
        )

    def _get_subagents(self) -> list[_SubagentSpec]:
        """Create runnable agents from specs.

        Returns:
            List of subagent specs with name, description, and runnable.
        """
        specs: list[_SubagentSpec] = []

        for spec in self._subagents:
            if "runnable" in spec:
                # CompiledSubAgent - use as-is
                compiled = cast("CompiledSubAgent", spec)
                specs.append({"name": compiled["name"], "description": compiled["description"], "runnable": compiled["runnable"]})
                continue

            # SubAgent - validate required fields
            if "model" not in spec:
                msg = f"SubAgent '{spec['name']}' must specify 'model'"
                raise ValueError(msg)
            if "tools" not in spec:
                msg = f"SubAgent '{spec['name']}' must specify 'tools'"
                raise ValueError(msg)

            # Resolve model if string
            from deepagents._models import resolve_model  # noqa: PLC0415

            model = resolve_model(spec["model"])

            # Use middleware as provided (caller is responsible for building full stack)
            middleware: list[AgentMiddleware] = list(spec.get("middleware", []))

            interrupt_on = spec.get("interrupt_on")
            if interrupt_on:
                middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

            specs.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "runnable": create_agent(
                        model,
                        system_prompt=spec["system_prompt"],
                        tools=spec["tools"],
                        middleware=middleware,
                        name=spec["name"],
                    ),
                }
            )

        return specs

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
