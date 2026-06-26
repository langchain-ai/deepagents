"""Middleware for declarative, multi-agent workflows via a `workflow` tool.

`workflow_mode` lets the main agent author a *declarative orchestration plan*
in a single tool call instead of round-tripping through the `task` tool stage
by stage. The plan is a small DAG of subagent invocations grouped into phases:

- Phases run **sequentially** — phase ``N + 1`` starts only after phase ``N``
    finishes.
- Steps **within a phase** run **concurrently** — this is the fan-out.
- A step in a later phase may consume the output of any earlier step via
    ``{{step_id}}`` templating in its prompt — this is the fan-in / pipeline.

The engine executes the plan autonomously and returns only the final phase's
output(s) to the orchestrator, keeping the orchestrator's context small. It
reuses the same compiled subagent runnables and result semantics as the
[`task` tool][deepagents.middleware.subagents.SubAgentMiddleware], so a
workflow can delegate to any subagent the agent already has.

This is the *declarative* engine. It deliberately does not execute
model-authored code; control flow is limited to the phase/step DAG plus
templated data passing, which covers fan-out, fan-in, and pipelines safely.
"""

import asyncio
import contextvars
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable, RunnableConfig

from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import (
    CompiledSubAgent,
    SubAgent,
    _subagent_tracing_context,
    compile_subagent_spec,
    extract_subagent_text,
    prepare_subagent_state,
    subagent_state_delta,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 25
"""Maximum number of steps allowed in a single workflow (runaway guard)."""

_TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_-]+)(?:\.output)?\s*\}\}")
"""Matches ``{{step_id}}`` and ``{{step_id.output}}`` placeholders."""


def _default_max_concurrency() -> int:
    """Default cap on subagents running at once within a phase."""
    return min(8, max(1, os.cpu_count() or 4))


class WorkflowStep(BaseModel):
    """A single subagent invocation within a workflow phase."""

    id: str = Field(
        description="Unique identifier for this step. Referenced by later steps via `{{id}}` in their prompt.",
    )
    subagent_type: str = Field(
        description="The subagent to run for this step. Must be one of the available agent types listed in the tool description.",
    )
    description: str | None = Field(
        default=None,
        description="Short human-readable summary of what this step does (a few words). Shown in the plan preview before the workflow runs.",
    )
    prompt: str = Field(
        description=(
            "The complete, self-contained task for the subagent. Include all context the subagent needs. "
            "To consume the output of an earlier step, embed `{{step_id}}` where that step's result should appear."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of earlier-phase steps whose output this step consumes. Every `{{id}}` used in `prompt` must be listed here.",
    )

    @field_validator("id", "subagent_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            msg = "must be a non-empty string"
            raise ValueError(msg)
        return value


class WorkflowPhase(BaseModel):
    """A group of steps that run concurrently. Phases run in declaration order."""

    title: str = Field(description="Short human-readable label for this phase (shown in progress output).")
    steps: list[WorkflowStep] = Field(description="Steps to run concurrently in this phase. Must contain at least one step.")

    @field_validator("steps")
    @classmethod
    def _non_empty_steps(cls, value: list[WorkflowStep]) -> list[WorkflowStep]:
        if not value:
            msg = "each phase must contain at least one step"
            raise ValueError(msg)
        return value


class WorkflowSpec(BaseModel):
    """A declarative workflow: an ordered list of phases.

    Used internally to validate a workflow before execution. The `workflow`
    tool itself advertises a deliberately looser argument schema
    ([`WorkflowToolArgs`][deepagents.middleware.workflow.WorkflowToolArgs]) so
    that a slightly-malformed plan reaches the engine and is answered with an
    actionable error message instead of an opaque tool-call rejection.
    """

    phases: list[WorkflowPhase] = Field(description="Ordered list of phases. Phases run sequentially; steps within a phase run in parallel.")

    @field_validator("phases")
    @classmethod
    def _non_empty_phases(cls, value: list[WorkflowPhase]) -> list[WorkflowPhase]:
        if not value:
            msg = "a workflow must contain at least one phase"
            raise ValueError(msg)
        return value


class WorkflowToolArgs(BaseModel):
    """Permissive argument schema advertised by the `workflow` tool.

    Kept loose on purpose: full structural and semantic validation runs inside
    the tool via [`WorkflowSpec`][deepagents.middleware.workflow.WorkflowSpec]
    and [`validate_workflow`][deepagents.middleware.workflow.validate_workflow],
    so the model receives an actionable error message it can correct from,
    rather than an opaque tool-call rejection, when a plan is slightly off. The
    exact expected shape is spelled out in the tool description.
    """

    # `list[Any]` (not `list[dict]`) on purpose: a stricter element type makes
    # LangChain reject a slightly-off plan at the tool boundary with an opaque
    # error before the engine's friendly validation can run. Accepting any list
    # lets `WorkflowSpec` coerce + report precise, correctable errors.
    phases: list[Any] = Field(
        description=(
            "Ordered list of phase objects, each shaped like "
            "{title, steps: [{id, subagent_type, prompt, depends_on}]}. Phases run "
            "sequentially; steps within a phase run in parallel. See the tool "
            "description for the full schema and an example."
        )
    )


def _template_refs(prompt: str) -> set[str]:
    """Return the set of step ids referenced via `{{...}}` in `prompt`."""
    return set(_TEMPLATE_RE.findall(prompt))


def validate_workflow(  # noqa: C901, PLR0911
    spec: WorkflowSpec,
    *,
    available_subagents: frozenset[str],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> str | None:
    """Validate a workflow spec against the structural rules.

    Returns an error message describing the first problem found, or ``None``
    when the spec is valid. Returning a message (rather than raising) lets the
    `workflow` tool hand the model an actionable correction instead of
    crashing the run.

    Rules enforced:

    - Step ids are unique across the whole workflow.
    - Every `subagent_type` names an available subagent.
    - Every `depends_on` id refers to a step in a strictly earlier phase
        (no same-phase or forward dependencies — same-phase steps run in
        parallel and cannot see each other).
    - Every `{{id}}` referenced in a prompt is declared in that step's
        `depends_on`.
    - The total step count does not exceed `max_steps`.
    """
    total_steps = sum(len(phase.steps) for phase in spec.phases)
    if total_steps > max_steps:
        return f"Workflow has {total_steps} steps, which exceeds the limit of {max_steps}. Split the work or use fewer, broader steps."

    seen_ids: set[str] = set()
    # Maps each step id to the phase index it was declared in.
    id_phase: dict[str, int] = {}
    for phase_idx, phase in enumerate(spec.phases):
        for step in phase.steps:
            if step.id in seen_ids:
                return f"Duplicate step id '{step.id}'. Every step id must be unique across the whole workflow."
            seen_ids.add(step.id)
            id_phase[step.id] = phase_idx

    for phase_idx, phase in enumerate(spec.phases):
        for step in phase.steps:
            if step.subagent_type not in available_subagents:
                allowed = ", ".join(f"`{name}`" for name in sorted(available_subagents))
                return f"Step '{step.id}' references unknown subagent '{step.subagent_type}'. Available subagents: {allowed}."
            for dep in step.depends_on:
                if dep not in id_phase:
                    return f"Step '{step.id}' depends on unknown step '{dep}'."
                if id_phase[dep] >= phase_idx:
                    return (
                        f"Step '{step.id}' depends on '{dep}', which is in the same or a later phase. "
                        "Dependencies must point to steps in an earlier phase."
                    )
            missing = _template_refs(step.prompt) - set(step.depends_on)
            if missing:
                missing_str = ", ".join(sorted(missing))
                return f"Step '{step.id}' references {{{{{missing_str}}}}} in its prompt but does not list them in `depends_on`."
    return None


def _render_prompt(prompt: str, results: Mapping[str, str]) -> str:
    """Substitute `{{step_id}}` / `{{step_id.output}}` with prior step outputs."""

    def _replace(match: re.Match[str]) -> str:
        return results.get(match.group(1), match.group(0))

    return _TEMPLATE_RE.sub(_replace, prompt)


def _merge_state_delta(acc: dict[str, Any], new: Mapping[str, Any]) -> None:
    """Merge a subagent's state delta into the accumulator in place.

    Dict-valued keys (e.g. the filesystem map) are shallow-merged so writes to
    distinct files from parallel steps all survive; for the same key, the later
    write wins. Non-dict values are replaced wholesale (last write wins).
    """
    for key, value in new.items():
        existing = acc.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = {**existing, **value}
            acc[key] = merged
        else:
            acc[key] = value


def _aggregate_final_output(spec: WorkflowSpec, results: Mapping[str, str]) -> str:
    """Build the orchestrator-facing result from the final phase's outputs."""
    final_steps = spec.phases[-1].steps
    if len(final_steps) == 1:
        return results.get(final_steps[0].id, "")
    return "\n\n".join(f"## {step.id}\n{results.get(step.id, '')}" for step in final_steps)


def workflow_plan_payload(spec: WorkflowSpec) -> dict[str, Any]:
    """Build the `plan` progress event emitted before a workflow runs.

    Consumers (e.g. a CLI renderer) can use this to show the full plan — phases,
    steps, each step's subagent, description, and dependencies — before any work
    starts. Emitted once, ahead of the first `phase_start` event.
    """
    return {
        "event": "plan",
        "phase_count": len(spec.phases),
        "step_count": sum(len(phase.steps) for phase in spec.phases),
        "phases": [
            {
                "index": phase_idx,
                "title": phase.title,
                "steps": [
                    {
                        "id": step.id,
                        "subagent_type": step.subagent_type,
                        "description": step.description,
                        "depends_on": list(step.depends_on),
                    }
                    for step in phase.steps
                ],
            }
            for phase_idx, phase in enumerate(spec.phases)
        ],
    }


WORKFLOW_TOOL_DESCRIPTION = """Run a declarative, multi-agent workflow in a single call.

A workflow is an ordered list of **phases**; each phase contains **steps** that each delegate to a subagent.

- Phases run **sequentially** (phase N+1 starts only after phase N finishes).
- Steps **within a phase** run **concurrently** — this is how you fan out work.
- A step in a later phase can consume the output of any earlier step by embedding `{{step_id}}` in its `prompt`. List every referenced id in that step's `depends_on`.
- Give every step a short `description` (a few words). It is shown in a plan preview before the workflow runs, so the user can see what will happen.

The engine runs the whole plan autonomously and returns only the **final phase's output** to you — intermediate step outputs and tool calls are not surfaced. This keeps your context small for large multi-stage tasks.

Available subagent types:
{available_agents}

## When to use `workflow` instead of `task`
- Multi-stage work where later stages depend on earlier stages (research → verify → synthesize).
- Fan-out then fan-in: run many independent subtasks in parallel, then combine their results.
- Any time you would otherwise call `task`, read the results, then call `task` again — author it as one workflow so you don't burn context round-tripping each stage.

## When NOT to use it
- A single delegated task — use `task`.
- Trivial work doable with a few direct tool calls.
- Steps that need to loop or branch on intermediate results in ways a fixed DAG can't express — drive those with `task` yourself.

## Example
Research two topics in parallel, then synthesize:
```json
{
  "phases": [
    {
      "title": "Research",
      "steps": [
        {"id": "a", "subagent_type": "general-purpose", "description": "Research topic A", "prompt": "Research topic A and report key findings."},
        {"id": "b", "subagent_type": "general-purpose", "description": "Research topic B", "prompt": "Research topic B and report key findings."}
      ]
    },
    {
      "title": "Synthesize",
      "steps": [
        {"id": "synth", "subagent_type": "general-purpose", "description": "Synthesize A and B", "depends_on": ["a", "b"],
         "prompt": "Compare these two research findings and write a synthesis.\\n\\nTopic A:\\n{{a}}\\n\\nTopic B:\\n{{b}}"}
      ]
    }
  ]
}
```
"""  # noqa: E501

WORKFLOW_SYSTEM_PROMPT = """## `workflow` (multi-agent orchestration)

You have access to a `workflow` tool that runs a declarative, multi-stage plan of subagents in a single call. Use it for complex objectives that decompose into stages where later stages build on earlier ones, or where many independent subtasks should run in parallel and then be combined.

Author a workflow as an ordered list of phases:

- Phases run **sequentially**; steps within a phase run **in parallel**.
- Put independent subtasks as parallel steps in the same phase (fan-out).
- Put a combine/synthesize/verify step in a later phase and reference upstream outputs with `{{step_id}}` (fan-in). List referenced ids in that step's `depends_on`.

Prefer `workflow` over repeated `task` calls whenever the orchestration is fixed in advance — it runs autonomously and returns only the final result, saving your context. Keep using `task` for one-off delegation and for orchestration that must branch on intermediate results."""  # noqa: E501


def _build_workflow_tool(  # noqa: C901, PLR0915
    subagents: Sequence[SubAgent | CompiledSubAgent],
    *,
    description: str | None = None,
    private_state_keys: frozenset[str] = frozenset(),
    state_schema: type | None = None,
    model: "BaseChatModel | None" = None,
    max_concurrency: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> BaseTool:
    """Create the `workflow` tool backed by the given subagent specs.

    When `model` is provided, raw `SubAgent` step runners are compiled to run on
    it instead of their default (the main agent's) model; `CompiledSubAgent`
    entries keep the model they were built with.
    """

    def _with_model(spec: SubAgent | CompiledSubAgent) -> SubAgent | CompiledSubAgent:
        if model is None or "runnable" in spec:
            return spec
        return {**spec, "model": model}

    compiled = [compile_subagent_spec(_with_model(spec), state_schema=state_schema) for spec in subagents]
    runnables: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in compiled}
    available = frozenset(runnables)
    concurrency = max_concurrency if max_concurrency is not None else _default_max_concurrency()

    subagent_description_str = "\n".join(f"- {spec['name']}: {spec['description']}" for spec in compiled)
    # Use str.replace (not str.format) because the description embeds a literal
    # JSON example with `{`/`}` braces and `{{step_id}}` template placeholders.
    source = WORKFLOW_TOOL_DESCRIPTION if description is None else description
    tool_description = source.replace("{available_agents}", subagent_description_str)

    def _emit(runtime: ToolRuntime, payload: dict[str, Any]) -> None:
        """Best-effort progress emission via the stream writer."""
        writer = getattr(runtime, "stream_writer", None)
        if writer is None:
            return
        try:
            writer({"workflow": payload})
        except Exception:  # noqa: BLE001 - progress emission must never break execution
            logger.debug("WorkflowMiddleware stream_writer raised; ignoring", exc_info=True)

    def _run_step_sync(step: WorkflowStep, base_state: Mapping[str, Any], results: dict[str, str]) -> tuple[str, dict]:
        rendered = _render_prompt(step.prompt, results)
        state = prepare_subagent_state(base_state, rendered, private_state_keys=private_state_keys)
        config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = runnables[step.subagent_type].invoke(state, config)
        return extract_subagent_text(result), subagent_state_delta(result)

    async def _run_step_async(step: WorkflowStep, base_state: Mapping[str, Any], results: dict[str, str]) -> tuple[str, dict]:
        rendered = _render_prompt(step.prompt, results)
        state = prepare_subagent_state(base_state, rendered, private_state_keys=private_state_keys)
        config: RunnableConfig = {"configurable": {"ls_agent_type": "subagent"}}
        with _subagent_tracing_context():
            result = await runnables[step.subagent_type].ainvoke(state, config)
        return extract_subagent_text(result), subagent_state_delta(result)

    def _prepare(phases: list[WorkflowPhase], runtime: ToolRuntime) -> tuple[WorkflowSpec | None, str | None]:
        """Validate inputs; return (spec, error_message)."""
        if not runtime.tool_call_id:
            msg = "Tool call ID is required for workflow invocation"
            raise ValueError(msg)
        try:
            spec = WorkflowSpec(phases=phases)
        except Exception as exc:  # noqa: BLE001 - surface validation errors to the model
            return None, f"Invalid workflow spec: {exc}"
        error = validate_workflow(spec, available_subagents=available, max_steps=max_steps)
        if error is not None:
            return None, error
        return spec, None

    def _execute_sync(spec: WorkflowSpec, runtime: ToolRuntime) -> Command:
        results: dict[str, str] = {}
        working_state: dict[str, Any] = dict(runtime.state)
        state_delta: dict[str, Any] = {}
        _emit(runtime, workflow_plan_payload(spec))

        def _run_in_context(step: WorkflowStep) -> tuple[str, dict]:
            # Run each step in a copy of the current context so worker threads
            # inherit the parent's callbacks and tracing (LangSmith, token usage).
            # A fresh copy per step avoids re-entering a single Context.
            return contextvars.copy_context().run(_run_step_sync, step, working_state, results)

        for phase_idx, phase in enumerate(spec.phases):
            _emit(runtime, {"event": "phase_start", "index": phase_idx, "title": phase.title})
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {step.id: pool.submit(_run_in_context, step) for step in phase.steps}
                for step_id, future in futures.items():
                    try:
                        text, delta = future.result()
                    except Exception as exc:  # isolate step failure; logged below
                        logger.exception("Workflow step '%s' failed", step_id)
                        results[step_id] = f"[step '{step_id}' failed: {exc}]"
                        _emit(runtime, {"event": "step_error", "id": step_id, "error": str(exc)})
                        continue
                    results[step_id] = text
                    _merge_state_delta(state_delta, delta)
                    _emit(runtime, {"event": "step_done", "id": step_id})
            _merge_state_delta(working_state, state_delta)
        final = _aggregate_final_output(spec, results)
        return Command(update={**state_delta, "messages": [ToolMessage(final, tool_call_id=runtime.tool_call_id)]})

    async def _execute_async(spec: WorkflowSpec, runtime: ToolRuntime) -> Command:
        results: dict[str, str] = {}
        working_state: dict[str, Any] = dict(runtime.state)
        state_delta: dict[str, Any] = {}
        _emit(runtime, workflow_plan_payload(spec))
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(step: WorkflowStep) -> tuple[str, dict]:
            async with semaphore:
                return await _run_step_async(step, working_state, results)

        for phase_idx, phase in enumerate(spec.phases):
            _emit(runtime, {"event": "phase_start", "index": phase_idx, "title": phase.title})
            outcomes = await asyncio.gather(
                *(_guarded(step) for step in phase.steps),
                return_exceptions=True,
            )
            for step, outcome in zip(phase.steps, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    logger.exception("Workflow step '%s' failed", step.id, exc_info=outcome)
                    results[step.id] = f"[step '{step.id}' failed: {outcome}]"
                    _emit(runtime, {"event": "step_error", "id": step.id, "error": str(outcome)})
                    continue
                text, delta = outcome
                results[step.id] = text
                _merge_state_delta(state_delta, delta)
                _emit(runtime, {"event": "step_done", "id": step.id})
            _merge_state_delta(working_state, state_delta)
        final = _aggregate_final_output(spec, results)
        return Command(update={**state_delta, "messages": [ToolMessage(final, tool_call_id=runtime.tool_call_id)]})

    def workflow(phases: list[WorkflowPhase], runtime: ToolRuntime) -> str | Command:
        # Catch-all so a bad plan or a step blowup always returns a correctable
        # message to the model instead of an opaque "Error invoking tool".
        try:
            spec, error = _prepare(phases, runtime)
            if spec is None:
                return error or "Invalid workflow."
            return _execute_sync(spec, runtime)
        except Exception as exc:
            logger.exception("workflow tool failed")
            return f"Workflow failed: {exc}"

    async def aworkflow(phases: list[WorkflowPhase], runtime: ToolRuntime) -> str | Command:
        try:
            spec, error = _prepare(phases, runtime)
            if spec is None:
                return error or "Invalid workflow."
            return await _execute_async(spec, runtime)
        except Exception as exc:
            logger.exception("workflow tool failed")
            return f"Workflow failed: {exc}"

    return StructuredTool.from_function(
        name="workflow",
        func=workflow,
        coroutine=aworkflow,
        description=tool_description,
        infer_schema=False,
        args_schema=WorkflowToolArgs,
    )


class WorkflowMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that exposes a declarative multi-agent `workflow` tool.

    Enabled by `create_deep_agent(..., workflow_mode=True)`. The tool lets the
    main agent author a phase/step DAG of subagent invocations that the engine
    runs autonomously — fan-out within a phase, fan-in / pipeline across
    phases via `{{step_id}}` templating — returning only the final result.

    It draws on the same subagent specs as
    [`SubAgentMiddleware`][deepagents.middleware.subagents.SubAgentMiddleware],
    so any subagent reachable through `task` is also reachable through a
    workflow step.

    Args:
        subagents: Subagent specs available as workflow steps. At least one is
            required.
        system_prompt: Instructions appended to the main agent's system prompt
            about when and how to author workflows.
        description: Custom description for the `workflow` tool. Supports the
            `{available_agents}` placeholder (replaced with the subagent
            name/description list).
        private_state_keys: State keys marked `PrivateStateAttr` that are
            stripped from parent state before invoking subagents.
        state_schema: Base graph state schema forwarded to raw `SubAgent`
            specs when their runnables are compiled.
        model: Optional model that the workflow's step runners use instead of
            their default (the main agent's) model. Applies to raw `SubAgent`
            steps only; `CompiledSubAgent` runnables keep the model they were
            built with. Leave `None` to inherit the main model.
        max_concurrency: Maximum subagents running at once within a phase.
            Defaults to a CPU-derived cap.
        max_steps: Maximum total steps allowed in a single workflow.
    """

    def __init__(
        self,
        *,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = WORKFLOW_SYSTEM_PROMPT,
        description: str | None = None,
        private_state_keys: frozenset[str] | None = None,
        state_schema: type | None = None,
        model: "BaseChatModel | None" = None,
        max_concurrency: int | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        """Initialize the `WorkflowMiddleware`."""
        super().__init__()
        if not subagents:
            msg = "At least one subagent must be specified for workflow mode"
            raise ValueError(msg)
        self._subagents = subagents
        self._description = description
        self._private_state_keys = private_state_keys or frozenset()
        self._state_schema = state_schema
        self._model = model
        self._max_concurrency = max_concurrency
        self._max_steps = max_steps
        self.subagent_names: frozenset[str] = frozenset(spec["name"] for spec in subagents)
        self.system_prompt = system_prompt
        self.tools = [self._build_tool()]

    def _build_tool(self) -> BaseTool:
        return _build_workflow_tool(
            self._subagents,
            description=self._description,
            private_state_keys=self._private_state_keys,
            state_schema=self._state_schema,
            model=self._model,
            max_concurrency=self._max_concurrency,
            max_steps=self._max_steps,
        )

    @property
    def private_state_keys(self) -> frozenset[str]:
        """State keys stripped from parent state before invoking subagents."""
        return self._private_state_keys

    @private_state_keys.setter
    def private_state_keys(self, value: frozenset[str]) -> None:
        self._private_state_keys = value
        self.tools = [self._build_tool()]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Append workflow-usage instructions to the system message."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Append workflow-usage instructions to the system message."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
