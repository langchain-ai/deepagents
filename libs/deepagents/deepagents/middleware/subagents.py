"""Middleware for providing subagents to an agent via a `task` tool."""

import asyncio
import json
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict, Unpack, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command

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
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.

## `swarm` (parallel subagent fan-out)

You also have access to a `swarm` tool for launching many subagents in parallel from a JSON config file. Use this when you need to process many chunks of data with the same (or similar) instructions.

### When to use `swarm` instead of `task`:
- When you have **10+ independent sub-tasks** that follow a similar pattern
- When you want to **programmatically generate** the task list (write a script to create the config)
- When you need to **process a large file in chunks** — write a script to split it, generate the config, then swarm

### Workflow:
1. Write a Python script that generates a JSON config file with all your tasks
2. Call `swarm(config_file="/path/to/config.json", output_dir="/path/to/results/")`
3. Read the result files from the output directory to aggregate"""  # noqa: E501


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


def _get_subagents_legacy(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> list[_SubagentSpec]:
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
        List of subagent specs containing name, description, and runnable.
    """
    # Use empty list if None (no default middleware)
    default_subagent_middleware = default_middleware or []

    specs: list[_SubagentSpec] = []

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
        specs.append(
            {
                "name": "general-purpose",
                "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
                "runnable": general_purpose_subagent,
            }
        )

    # Process custom subagents
    for agent_ in subagents:
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            specs.append(
                {
                    "name": custom_agent["name"],
                    "description": custom_agent["description"],
                    "runnable": custom_agent["runnable"],
                }
            )
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = [*default_subagent_middleware, *agent_["middleware"]] if "middleware" in agent_ else [*default_subagent_middleware]

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        specs.append(
            {
                "name": agent_["name"],
                "description": agent_["description"],
                "runnable": create_agent(
                    subagent_model,
                    system_prompt=agent_["system_prompt"],
                    tools=_tools,
                    middleware=_middleware,
                    name=agent_["name"],
                ),
            }
        )

    return specs


def _build_task_tool(  # noqa: C901
    subagents: list[_SubagentSpec],
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool from pre-built subagent graphs.

    This is the shared implementation used by both the legacy API and new API.

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
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",  # noqa: E501
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
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
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",  # noqa: E501
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
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
        description=description,
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


def _build_swarm_tool(
    subagents: list[_SubagentSpec],
    backend: "BackendProtocol | BackendFactory | None" = None,
) -> BaseTool:
    """Create a swarm tool that fans out many subagent tasks in parallel.

    Args:
        subagents: List of subagent specs containing name, description, and runnable.
        backend: Backend for reading config and writing results.

    Returns:
        A StructuredTool that reads a JSON config and runs all tasks in parallel.
    """
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in subagents}
    subagent_description_str = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)
    description = SWARM_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)

    async def _run_one_task(
        task_spec: dict,
        parent_state: dict,
        idx: int,
    ) -> tuple[str, str]:
        """Run a single swarm sub-task. Returns (task_id, result_text)."""
        task_id = str(task_spec.get("id", idx))
        desc = task_spec["description"]
        subagent_type = task_spec["subagent_type"]

        if subagent_type not in subagent_graphs:
            allowed = ", ".join(subagent_graphs.keys())
            return task_id, f"Error: unknown subagent_type '{subagent_type}'. Allowed: {allowed}"

        subagent = subagent_graphs[subagent_type]
        subagent_state = dict(parent_state)
        subagent_state["messages"] = [HumanMessage(content=desc)]

        try:
            result = await subagent.ainvoke(subagent_state)
            messages = result.get("messages", [])
            if messages:
                text = messages[-1].text or ""
                return task_id, text.rstrip()
            return task_id, ""
        except Exception as e:
            return task_id, f"Error: {e}"

    async def _execute_swarm(
        config_file: str,
        output_dir: str,
        runtime: ToolRuntime,
    ) -> str:
        """Core implementation of swarm."""
        # Read config file from disk
        try:
            with open(config_file) as f:
                config_data = json.load(f)
        except Exception as e:
            return f"Error reading config file '{config_file}': {e}"

        tasks = config_data.get("tasks", [])
        if not tasks:
            return "Error: config file contains no tasks."

        # Prepare parent state (stripped of excluded keys)
        parent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}

        # Run all tasks in parallel
        coros = [_run_one_task(task_spec, parent_state, idx) for idx, task_spec in enumerate(tasks)]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Write results to output files
        import os
        output_dir_clean = output_dir.rstrip("/")
        os.makedirs(output_dir_clean, exist_ok=True)
        summaries = []
        for item in results:
            if isinstance(item, Exception):
                summaries.append(f"Error: {item}")
                continue
            task_id, result_text = item
            output_path = f"{output_dir_clean}/{task_id}.txt"
            try:
                with open(output_path, "w") as f:
                    f.write(result_text)
                summaries.append(f"✓ {task_id}: wrote {len(result_text)} chars to {output_path}")
            except Exception as e:
                summaries.append(f"✗ {task_id}: error writing result: {e}")

        completed = sum(1 for s in summaries if s.startswith("✓"))
        summary = f"{completed}/{len(tasks)} tasks completed. Results in {output_dir_clean}/\n"
        summary += "\n".join(summaries)
        return summary

    def swarm(
        config_file: Annotated[str, "Absolute path to the JSON config file containing the task definitions."],
        output_dir: Annotated[str, "Absolute path to the directory where result files will be written. Each task result is written to <output_dir>/<task_id>.txt."],
        runtime: ToolRuntime,
    ) -> str:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _execute_swarm(config_file, output_dir, runtime)).result()

    async def aswarm(
        config_file: Annotated[str, "Absolute path to the JSON config file containing the task definitions."],
        output_dir: Annotated[str, "Absolute path to the directory where result files will be written. Each task result is written to <output_dir>/<task_id>.txt."],
        runtime: ToolRuntime,
    ) -> str:
        return await _execute_swarm(config_file, output_dir, runtime)

    return StructuredTool.from_function(
        name="swarm",
        func=swarm,
        coroutine=aswarm,
        description=description,
    )


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
        backend: Backend for file operations and execution. Required for the new API.
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

    .. deprecated::
        The following arguments are deprecated and will be removed in version 0.5.0:
        `default_model`, `default_tools`, `default_middleware`,
        `default_interrupt_on`, `general_purpose_agent`. Use `backend` and `subagents` instead.
    """

    # Valid deprecated kwarg names for runtime validation
    _VALID_DEPRECATED_KWARGS = frozenset(
        {
            "default_model",
            "default_tools",
            "default_middleware",
            "default_interrupt_on",
            "general_purpose_agent",
        }
    )

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
        **deprecated_kwargs: Unpack[_DeprecatedKwargs],
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()

        # Validate that only known deprecated kwargs are passed
        unknown_kwargs = set(deprecated_kwargs.keys()) - self._VALID_DEPRECATED_KWARGS
        if unknown_kwargs:
            msg = f"SubAgentMiddleware got unexpected keyword argument(s): {', '.join(sorted(unknown_kwargs))}"
            raise TypeError(msg)

        # Handle deprecated kwargs for backward compatibility
        default_model = deprecated_kwargs.get("default_model")
        default_tools = deprecated_kwargs.get("default_tools")
        default_middleware = deprecated_kwargs.get("default_middleware")
        default_interrupt_on = deprecated_kwargs.get("default_interrupt_on")
        # general_purpose_agent defaults to True if not specified
        general_purpose_agent = deprecated_kwargs.get("general_purpose_agent", True)

        # Warn about any deprecated kwargs that were provided
        provided_deprecated = [key for key in deprecated_kwargs if key != "general_purpose_agent"]
        if "general_purpose_agent" in deprecated_kwargs and not general_purpose_agent:
            provided_deprecated.append("general_purpose_agent")

        if provided_deprecated:
            warnings.warn(
                f"The following SubAgentMiddleware arguments are deprecated and will be removed "
                f"in version 0.5.0: {', '.join(provided_deprecated)}. "
                f"Use `backend` and fully-specified `subagents` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Detect which API is being used
        using_new_api = backend is not None
        using_old_api = default_model is not None

        if using_old_api and not using_new_api:
            # Legacy API - build subagents from deprecated args
            subagent_specs = _get_subagents_legacy(
                default_model=default_model,  # ty: ignore[invalid-argument-type]
                default_tools=default_tools or [],
                default_middleware=default_middleware,
                default_interrupt_on=default_interrupt_on,
                subagents=subagents or [],
                general_purpose_agent=general_purpose_agent,
            )
        elif using_new_api:
            if not subagents:
                msg = "At least one subagent must be specified when using the new API"
                raise ValueError(msg)
            self._backend = backend
            self._subagents = subagents
            subagent_specs = self._get_subagents()
        else:
            msg = "SubAgentMiddleware requires either `backend` (new API) or `default_model` (deprecated API)"
            raise ValueError(msg)

        task_tool = _build_task_tool(subagent_specs, task_description)
        swarm_tool = _build_swarm_tool(subagent_specs, backend=backend)

        # Build system prompt with available agents
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagent_specs)
            self.system_prompt = system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
        else:
            self.system_prompt = system_prompt

        self.tools = [task_tool, swarm_tool]

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
            model = spec["model"]
            if isinstance(model, str):
                model = init_chat_model(model)

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
