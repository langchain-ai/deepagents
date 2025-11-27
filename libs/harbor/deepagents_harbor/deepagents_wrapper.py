"""A wrapper for DeepAgents to run in Harbor environments."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deepagents import create_deep_agent
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from langchain.chat_models import init_chat_model
from langchain.messages import UsageMetadata
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents_harbor.backend import HarborSandboxFallback
from deepagents_harbor.tracing import create_example_id_from_instruction

SYSTEM_MESSAGE = """
You are an autonomous agent executing tasks in a sandboxed environment. Follow these instructions carefully.

## WORKING DIRECTORY
Work in the /app directory unless explicitly instructed otherwise.

## TASK UNDERSTANDING & REQUIREMENTS

1. **Extract Complete Requirements First**:
   - Before starting ANY implementation, extract ALL fields, conditions, constraints, and criteria into an explicit checklist
   - Parse conditional clauses (e.g., "if X then Y") as hard constraints, not suggestions
   - Parse temporal phrases literally: "as of date X" means use data/state from that specific date
   - Read field names, file paths, boundaries, and specifications EXACTLY as written - do not abbreviate or substitute "equivalent" alternatives

2. **Decompose Complex Logic**:
   - Break down temporal statements: "currently enrolled as of 2023" = (enrolled by 2023) AND (not graduated by 2023)
   - Identify multi-part requirements that need multiple strategies, not parameter tuning
   - When requirements mention "iterative," "refinement," or "feedback," implement actual loops with convergence criteria

## OPERATION SAFETY

3. **Irreversibility Guard**:
   - BEFORE executing operations that cannot be undone (database access, file deletion, git operations), verify all prerequisites
   - NEVER access files marked as "corrupted" or "encrypted" without first examining their actual state
   - For git operations requiring temporal snapshots (e.g., "as of commit X"), follow deterministic checkout procedures

4. **Sequencing Discipline**:
   - Plan the complete sequence before starting - wrong ordering can cause irreversible errors
   - Do not adopt "fix it as I go" mentality for operations with dependencies

## TOOL & APPROACH SELECTION

5. **Respect Explicit Specifications**:
   - If user mentions specific package/library/version → use exactly that (not alternatives you prefer)
   - If user specifies a tool or method → use it even if you think another is better
   - Version specifications are hard constraints, not suggestions

6. **Choose Appropriate Tools**:
   - Structured data (JSON, CSV, XML) → use proper parsers, not regex
   - Pattern matching across variations → use regex, not string literals
   - Programmatic images/graphics → consider algorithmic approaches (ray tracing, fractals, procedural generation)
   - Identification tasks → use APIs/databases first (e.g., PDB for proteins, external validators for keys/formats)

## DOMAIN-SPECIFIC CONSIDERATIONS

7. **Preprocessing Detection**:
   - Check if task requires domain-specific transformations BEFORE processing:
     * Unit conversions (wavelength ↔ frequency, temperature scales)
     * Scientific notation conversions (wavelength ↔ wavenumber)
     * Date format standardizations
     * Coordinate system transformations
   - For specialized domains (molecular biology, emulation, signal processing), research required conventions and components first

8. **External Validation Requirements**:
   - Do NOT validate your implementation only against tests you create yourself
   - Validate against: external specifications, standard conventions, known reference values, third-party validators
   - Distrust success indicators from self-created tests

## VERIFICATION ANTI-PATTERNS TO AVOID

9. **Do NOT Assume**:
   - "It compiles/runs, therefore it's correct" - execution is not correctness
   - "This version should work" - use exact versions specified
   - "Parameter tuning will solve this" - may mask fundamental strategy problems
   - "Working on demo data" without algorithmic verification
   - Fast convergence or round numbers without checking against ground truth

10. **Complete Implementation**:
    - Implement ALL specified algorithm components, not simplified versions
    - For multi-objective problems, consider multiple strategies
    - Read requirements for completeness: if "generate report including X, Y, Z" - include all three

## EXECUTION APPROACH
- Extract and verify complete requirements checklist FIRST
- Identify irreversible operations and plan around them
- Select tools based on specification and data type
- Research domain conventions before specialized implementations
- Validate against external sources, not just self-created tests
"""


class DeepAgentsWrapper(BaseAgent):
    """Harbor agent implementation using LangChain DeepAgents.

    Wraps DeepAgents to execute tasks in Harbor environments.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        temperature: float = 0.0,
        verbose: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize DeepAgentsWrapper."""
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            # Use DeepAgents default
            model_name = "anthropic:claude-sonnet-4-5-20250929"

        self._model_name = model_name
        self._temperature = temperature
        self._verbose = verbose
        self._model = init_chat_model(model_name, temperature=temperature)

        # LangSmith run tracking for feedback
        self._langsmith_run_id: str | None = None
        self._task_name: str | None = None

    @staticmethod
    def name() -> str:
        return "deepagent-harbor"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent with the given environment.

        Args:
            environment: Harbor environment (Docker, Modal, etc.)
        """
        pass

    def version(self) -> str | None:
        return "0.0.1"

    async def _enhance_instruction_with_context(
        self, instruction: str, backend: HarborSandboxFallback
    ) -> str:
        """Enhance instruction with current directory and file listing context.

        Args:
            instruction: Original task instruction
            backend: Harbor sandbox backend to query for directory information

        Returns:
            Enhanced instruction with appended context
        """
        # Get directory information from backend
        ls_info = await backend.als_info(".")
        current_dir = (await backend.aexecute("pwd")).output

        # Get first 10 files
        first_10_files = ls_info[:10] if ls_info else []
        has_more = len(ls_info) > 10 if ls_info else False

        # Build file listing
        if first_10_files:
            file_listing = "\n".join(f"{i + 1}. {file}" for i, file in enumerate(first_10_files))
            if has_more:
                file_listing += f"\n\n(Output truncated: {len(ls_info) - 10} more files available in current directory)"
        else:
            file_listing = "(no files found)"

        # Build enhanced instruction
        enhanced_instruction = f"""{instruction}
        
----

Current directory: 
{current_dir.strip() if current_dir else "unknown"}

First 10 files in current directory:
{file_listing}
"""

        return enhanced_instruction

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Execute the DeepAgent on the given instruction.

        Args:
            instruction: The task to complete
            environment: Harbor environment (Docker, Modal, etc.)
            context: Context to populate with metrics
        """
        # Track token usage and cost for this run
        total_prompt_tokens = 0
        total_completion_tokens = 0

        configuration = json.loads(environment.trial_paths.config_path.read_text())
        job_id = configuration["job_id"]

        backend = HarborSandboxFallback(environment)

        # Enhance instruction with directory context
        enhanced_instruction = await self._enhance_instruction_with_context(instruction, backend)

        deep_agent = create_deep_agent(
            model=self._model, backend=backend, system_prompt=SYSTEM_MESSAGE
        )

        # Build metadata with experiment tracking info
        metadata = {
            "task_instruction": instruction,
            "model": self._model_name,
            # This is a harbor-specific session ID for the entire task run
            # It's different from the LangSmith experiment ID (called session_id)
            "harbor_session_id": environment.session_id,
            "job_id": job_id,
        }

        # Compute example_id from instruction for deterministic linking
        # This uses the same hashing as create_langsmith_dataset.py
        example_id = create_example_id_from_instruction(instruction)
        metadata["reference_example_id"] = example_id

        config: RunnableConfig = {
            "run_name": f"{environment.session_id}",
            "tags": [self._model_name, environment.session_id],
            "metadata": metadata,
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        }

        # Invoke deep agent with LangSmith tracing
        result = await deep_agent.ainvoke(
            {"messages": [{"role": "user", "content": enhanced_instruction}]},  # type: ignore
            config=config,
        )
        # Create trajectory
        steps = [
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="system",
                message="Agent initialized and ready to execute the task.",
            ),
            Step(
                step_id=2,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=instruction,
            ),
        ]

        observations = []
        pending_step: Step | None = None

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                # Extract usage metadata from AIMessage
                usage: UsageMetadata = msg.usage_metadata
                if usage:
                    total_prompt_tokens += usage["input_tokens"]
                    total_completion_tokens += usage["output_tokens"]
                # If there's a pending step with tool calls, add it now with observations
                if pending_step is not None:
                    if pending_step.tool_calls and observations:
                        # Add observations to the pending step
                        pending_step.observation = Observation(results=observations)
                        observations = []
                    steps.append(pending_step)
                    pending_step = None

                # Extract content and tool calls from current AIMessage
                atf_tool_calls = []
                message = ""
                for cb in msg.content_blocks:
                    if cb["type"] == "text":
                        message += cb["text"]
                    elif cb["type"] == "reasoning":
                        message += cb["reasoning"]
                    elif cb["type"] == "tool_call":
                        atf_tool_calls.append(
                            ToolCall(
                                tool_call_id=cb["id"],
                                function_name=cb["name"],
                                arguments=cb["args"],
                            )
                        )
                    else:
                        # TODO: Add server side tool call results.
                        continue

                # Create new step
                new_step = Step(
                    step_id=steps[-1].step_id + 1 if steps else 0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    message=message,
                    tool_calls=atf_tool_calls if atf_tool_calls else None,
                )

                # If this AIMessage has tool calls, make it pending (wait for observations)
                # Otherwise, add it immediately
                if atf_tool_calls:
                    pending_step = new_step
                else:
                    steps.append(new_step)

            elif isinstance(msg, ToolMessage):
                # Collect observations for the pending step
                observations.append(
                    ObservationResult(
                        source_call_id=msg.tool_call_id,
                        content=str(msg.content),
                    )
                )
            elif isinstance(msg, HumanMessage):
                pass
            else:
                raise NotImplementedError(
                    f"Message type {type(msg)} not supported for step conversion"
                )

        # Add any remaining pending step
        if pending_step is not None:
            if pending_step.tool_calls and observations:
                pending_step.observation = Observation(results=observations)
            steps.append(pending_step)

        # Build and save trajectory
        metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens or None,
            total_completion_tokens=total_completion_tokens or None,
            total_steps=len(steps),
        )
        self._save_trajectory(environment, steps, metrics)

    def _save_trajectory(
        self, environment: BaseEnvironment, steps: list[Step], metrics: FinalMetrics
    ) -> None:
        """Save current trajectory to logs directory."""
        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id=environment.session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra={
                    "framework": "deepagents",
                    "langchain_version": "1.0+",
                },
            ),
            steps=steps,
            final_metrics=metrics,
        )
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))
