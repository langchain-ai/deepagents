"""Middleware for providing subagents to an agent via a `task` tool."""

import asyncio
import json
import os
import warnings
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict, Unpack, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message


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

Subagent limitations (continued):
- A single subagent can handle large files for tasks like summarization. But when precise analysis is needed over a large file — counting, extraction, searching, or anything where accuracy matters — you MUST check the file size first and use `swarm` to split the work for any file over 500 lines. Do NOT send large files to a single `task` subagent for precise analysis, and do NOT try to process them with a single Python script via `execute` either.
- CRITICAL: When a task requires looping over items and applying intelligence to each one (classifying, labeling, categorizing, identifying, tagging), you MUST use `swarm` (not `task`) — regardless of file size. Do NOT classify items inline, do NOT delegate to a single `task` subagent, and do NOT write Python regex/heuristic scripts. The only acceptable approach is `swarm`.
- **Pre-pilot before full swarm fan-out:** Before launching the full swarm, run a calibration pilot:
  1. Take a small sample (~20-30 items) and classify them YOURSELF first — write out your classifications with reasoning.
  2. Send the SAME items to a single `task` subagent with your classification instructions.
  3. DIFF the two results. For every disagreement, determine who is right and WHY the subagent got it wrong.
  4. If you find systematic errors (e.g., the subagent consistently misclassifies a category boundary), add explicit corrective examples or rules to your instructions. For example: "IMPORTANT: 'What is [specific named thing]?' is entity, NOT description — unless the answer is an explanation/definition rather than a name."
  5. Re-run the pilot with updated instructions until the subagent matches your classifications on the sample.
  Only THEN launch the full swarm with the refined instructions.

## `swarm` (parallel subagent fan-out)

You also have access to a `swarm` tool for launching many subagents in parallel from a JSON config file. Use this when you need to process many chunks of data with the same (or similar) instructions. When specificity and precision is required over large files, you are best off using the `swarm` tool to split the work across many subagents.

When splitting work across subagents, figure out exactly how you would do the task yourself first — only after you have that clarity should you distribute to subagents. Before distributing work, write out the exact instructions you'll give to each worker. Be specific and leave no room for interpretation — workers will interpret ambiguity differently, and inconsistent results can't be aggregated reliably.

### When to use `swarm` instead of `task`:
- When you have multiple independent sub-tasks that follow a similar pattern
- When you can define all the subtasks upfront (e.g., chunk a file by line ranges)
- When you need to programmatically generate the task list

### When to use `task` instead of `swarm`:
- When the next subtask depends on results from a previous one
- When you need exploratory/adaptive work (e.g., grep first, then investigate)

### Workflow:
1. Write a JSON config file with all your tasks (or write a script to generate it)
2. Call `swarm(config_file="/path/to/config.json", output_dir="/path/to/results/")`
3. Read the result files from the output directory to aggregate"""  # noqa: E501

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


SWARM_TOOL_DESCRIPTION = """Launch many subagents in parallel from a JSON config file.

Use this when you need to fan out the same (or similar) work across many chunks of data — for example, processing sections of a large file, classifying batches of entries, or querying multiple documents.

## Workflow
1. Write a Python script (via `execute`) that reads your data, chunks it, and generates a JSON config file.
2. Call `swarm(config_file="/path/to/config.json", output_dir="/path/to/results/")`.
3. All tasks run in parallel. Each result is written to `<output_dir>/<task_id>.txt`.
4. Read the result files to aggregate.

## Config file format
```json
{{
  "tasks": [
    {{
      "id": "chunk_0",
      "description": "Read /tmp/context.txt lines 0-100. Classify each entry. Return JSON counts.",
      "subagent_type": "general-purpose"
    }},
    {{
      "id": "chunk_1",
      "description": "Read /tmp/context.txt lines 100-200. Classify each entry. Return JSON counts.",
      "subagent_type": "general-purpose"
    }}
  ]
}}
```

Fields:
- `id` (optional): Identifier for the task, used as the output filename. Defaults to the task index.
- `description` (required): The task description, same as what you'd pass to `task(description=...)`.
- `subagent_type` (required): Which subagent to use, same as `task(subagent_type=...)`.

Available subagent types:
{available_agents}
"""


def _sanitize_task_id(task_id: str, output_dir: str, fallback: str) -> str:
    """Sanitize a task ID so it cannot escape the output directory.

    Applies ``os.path.basename`` to strip directory components, then resolves
    the resulting path and verifies it is still inside *output_dir*.

    Args:
        task_id: Raw task identifier (may contain path separators or ``..``).
        output_dir: The directory that output files must stay within.
        fallback: Value to use when *task_id* is empty after sanitization.

    Returns:
        A safe filename component (without extension).

    Raises:
        ValueError: If the resolved path escapes *output_dir*.
    """
    name = Path(task_id).name
    safe = fallback if (not name or name in (".", "..")) else name
    resolved = os.path.realpath(Path(output_dir) / f"{safe}.txt")
    real_output_dir = os.path.realpath(output_dir)
    if not resolved.startswith(real_output_dir + os.sep):
        msg = f"task id {task_id!r} resolves outside output directory"
        raise ValueError(msg)
    return safe


class SwarmTaskSpec(TypedDict):
    """A single task entry in a swarm config.

    Only `description`, `subagent_type`, and `id` are used by the swarm implementation.
    Additional keys may be present but are ignored.

    Fields:
        description: The instruction passed to the subagent as a single `HumanMessage`.
        subagent_type: The subagent name to dispatch to.
        id: Optional identifier used as the output filename. If omitted, defaults to the
            task index. This value is stringified when writing `<output_dir>/<id>.txt`.
    """

    description: str
    subagent_type: str
    id: NotRequired[str | int]


class ParsedSwarmConfig(TypedDict):
    """Result of parsing a swarm config file."""

    tasks: list[SwarmTaskSpec] | None
    error: str | None


async def _load_swarm_tasks(
    resolved_backend: BackendProtocol,
    config_file: str,
) -> ParsedSwarmConfig:
    responses = await resolved_backend.adownload_files([config_file])
    response = responses[0]
    if response.error:
        return {"tasks": None, "error": f"Error reading config file '{config_file}': {response.error}"}

    content = response.content
    if not isinstance(content, bytes):
        return {"tasks": None, "error": f"Content was expected to be bytes. Got {type(content)}."}

    try:
        config_data = json.loads(content.decode("utf-8"))
    except UnicodeDecodeError as e:
        return {"tasks": None, "error": f"Error reading config file '{config_file}': {e}"}

    raw_tasks = config_data.get("tasks", [])
    if not raw_tasks:
        return {"tasks": None, "error": f"Error file '{config_file}' contains no tasks!"}

    adapter = TypeAdapter(list[SwarmTaskSpec])
    try:
        tasks = adapter.validate_python(raw_tasks)
    except ValidationError as e:
        return {"tasks": None, "error": json.dumps(e.errors(), indent=2)}

    return {"tasks": tasks, "error": None}


class _DeprecatedKwargs(TypedDict, total=False):
    """TypedDict for deprecated SubAgentMiddleware keyword arguments.

    These arguments are deprecated and will be removed in version 0.5.0.
    Use `backend` and fully-specified `subagents` instead.
    """


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
        enable_swarm: bool = True,
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

        async def _run_swarm_tasks(
            subagents: dict[str, Runnable],
            parent_state: dict[str, Any],
            tasks: list[SwarmTaskSpec],
        ) -> list[tuple[str, str] | Exception]:
            """Run tasks."""

            async def run_task(task_spec: dict[str, Any], idx: int) -> tuple[str, str]:
                task_id = str(task_spec.get("id", idx))
                desc = task_spec["description"]
                subagent_type = task_spec["subagent_type"]

                subagent = subagents.get(subagent_type)
                if subagent is None:
                    allowed = ", ".join(subagents.keys())
                    return task_id, f"Error: unknown subagent_type '{subagent_type}'. Allowed: {allowed}"

                subagent_state = dict(parent_state)
                subagent_state["messages"] = [HumanMessage(content=desc)]

                result = await subagent.ainvoke(subagent_state)
                messages = result.get("messages", [])
                if messages:
                    text = messages[-1].text or ""
                    return task_id, text.rstrip()
                return task_id, ""

            coros = [run_task(task_spec, idx) for idx, task_spec in enumerate(tasks)]
            return await asyncio.gather(*coros, return_exceptions=True)

        async def aswarm(
            config_file: Annotated[str, "Absolute path to the JSON config file containing the task definitions."],
            output_dir: Annotated[
                str,
                "Absolute path to the directory where result files will be written. Each task result is written to <output_dir>/<task_id>.txt.",
            ],
            runtime: ToolRuntime,
        ) -> ToolMessage:
            """Run swarm tasks."""
            resolved_backend = backend(runtime) if callable(backend) else backend  # ty: ignore[call-top-callable]

            parsed = await _load_swarm_tasks(resolved_backend, config_file)
            if parsed["error"] is not None:
                return ToolMessage(content=parsed["error"], status="error", tool_call_id=runtime.tool_call_id)
            tasks = parsed["tasks"]
            if tasks is None:
                msg = "parsed swarm tasks unexpectedly missing"
                raise AssertionError(msg)

            parent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}

            results = await _run_swarm_tasks(subagent_graphs, parent_state, tasks)

            interrupts: list[Interrupt] = []
            interrupt_excs: list[GraphInterrupt] = []

            for item in results:
                if isinstance(item, GraphInterrupt):
                    interrupt_excs.append(item)
                    if not item.args:
                        continue
                    seq = item.args[0]
                    if isinstance(seq, tuple | list) and all(isinstance(x, Interrupt) for x in seq):
                        interrupts.extend(seq)

            if interrupt_excs:
                if interrupts:
                    raise GraphInterrupt(interrupts)
                raise interrupt_excs[0]

            output_dir_clean = output_dir.rstrip("/")

            outputs: list[tuple[str, bytes]] = []
            summaries: list[str] = []
            for item in results:
                if isinstance(item, Exception):
                    summaries.append(f"Error: {item}")
                    continue
                task_id, result_text = item
                task_id = _sanitize_task_id(task_id, output_dir_clean, fallback=str(results.index(item)))
                output_path = f"{output_dir_clean}/{task_id}.txt"
                outputs.append((output_path, result_text.encode("utf-8")))

            upload_responses = await resolved_backend.aupload_files(outputs)
            for upload in upload_responses:
                if upload.error is not None:
                    summaries.append(f"✗ {upload.path}: error writing result: {upload.error}")
                else:
                    result_len = next((len(b) for p, b in outputs if p == upload.path), 0)
                    summaries.append(f"✓ {upload.path}: wrote {result_len} chars")

            completed = sum(1 for s in summaries if s.startswith("✓"))
            summary = f"{completed}/{len(tasks)} tasks completed. Results in {output_dir_clean}/\n"
            summary += "\n".join(summaries)
            return ToolMessage(
                content=summary,
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
