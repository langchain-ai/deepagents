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
        deep_agent = create_deep_agent(model=self._model, backend=backend)

        # Build metadata with experiment tracking info
        metadata = {
            "task_instruction": instruction,
            "model": self._model_name,
            # This is a harbor-specific session ID for the entire task run
            # It's different from the LangSmith experiment ID (called session_id)
            "harbor_session_id": environment.session_id,
            "job_id": job_id,
        }

        # Add LangSmith experiment tracking metadata if available
        if self._experiment_session_id:
            # Experiment session ID for langsmith integration
            metadata["session_id"] = self._experiment_session_id
            # Compute example_id from instruction for deterministic linking
            # This uses the same hashing as create_langsmith_dataset.py
            example_id = create_example_id_from_instruction(instruction)
            metadata["reference_example_id"] = example_id

        config: RunnableConfig = {
            "run_name": f"harbor-deepagent-{environment.session_id}",
            "tags": [self._model_name, environment.session_id],
            "metadata": metadata,
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        }

        # Invoke deep agent with LangSmith tracing
        result = await deep_agent.ainvoke(
            {"messages": [{"role": "user", "content": instruction}]},  # type: ignore
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
