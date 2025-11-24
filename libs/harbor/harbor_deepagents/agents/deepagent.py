"""Harbor agent implementation using LangChain DeepAgents."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

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
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from deepagents import create_deep_agent
from harbor_deepagents.agents.backend import HarborSandbox, HarborSandboxFallback


class DeepAgentsWrapper(BaseAgent):
    """Harbor agent implementation using LangChain DeepAgents.

    Wraps DeepAgents to execute tasks in Harbor environments.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_iterations: int = 50,
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
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._verbose = verbose
        self._session_id = str(uuid.uuid4())
        self._environment: BaseEnvironment | None = None
        self._model = init_chat_model(model_name, temperature=temperature)

        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

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
        self._environment = environment

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
        backend = HarborSandboxFallback(environment)
        deep_agent = create_deep_agent(
            model=self._model,
            backend=backend
        )

        config: RunnableConfig = {
            "run_name": f"harbor-deepagent-{self._session_id[:8]}",
            "tags": ["harbor", "deepagent", self._model_name, self._session_id],
            "metadata": {
                "task_instruction": instruction,
                "model": self._model_name,
                "session_id": self._session_id,
            },
            "recursion_limit": self._max_iterations,
        }

        # Invoke deep agent with LangSmith tracing
        result = await deep_agent.ainvoke(
            {"messages": [{"role": "user", "content": instruction}]}, # type: ignore
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
                message=instruction
            )
        ]

        observations = []
        pending_step: Step | None = None

        for msg in result['messages']:
            if isinstance(msg, AIMessage):
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
                    if cb['type'] == "text":
                        message += cb['text']
                    elif cb['type'] == "reasoning":
                        message += cb['reasoning']
                    elif cb["type"] == "tool_call":
                        atf_tool_calls.append(
                            ToolCall(
                                tool_call_id=cb['id'],
                                function_name=cb['name'],
                                arguments=cb['args']
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



        # # Store task name for feedback
        # self._task_name = instruction[:100]  # Truncate for readability
        #
        # # Query LangSmith to get the run_id by searching for our unique run_name
        # self._langsmith_run_id = self._session_id  # Default fallback
        #
        # if os.getenv("LANGCHAIN_TRACING_V2"):
        #     client = Client()
        #     project_name = os.getenv("LANGCHAIN_PROJECT", "default")
        #     run_name = f"harbor-deepagent-{self._session_id[:8]}"
        #     # Search by exact run_name which is unique
        #     runs = list(client.list_runs(
        #         project_name=project_name,
        #         filter=f'eq(name, "{run_name}")',
        #         limit=1,
        #     ))
        #     if runs:
        #         self._langsmith_run_id = str(runs[0].id)
        #         if self._verbose:
        #             print(f"✓ Found LangSmith run_id: {self._langsmith_run_id}")
        #

        ### Extract messages from result
        ###
        ### # Extract final output
        ### final_message = self._extract_final_message(messages) or "Task completed"
        ### last_step = self._trajectory_steps[-1] if self._trajectory_steps else None
        ### if not (
        ###     last_step
        ###     and last_step.source == "agent"
        ###     and (last_step.message or "").strip() == final_message.strip()
        ### ):
        ###     self._add_step(
        ###         source="agent",
        ###         message=final_message,
        ###     )

        # Build and save trajectory

        final_metrics = FinalMetrics(
            total_prompt_tokens=self._total_prompt_tokens or None,
            total_completion_tokens=self._total_completion_tokens or None,
            total_cost_usd=self._total_cost or None,
            total_steps=len(steps),
        )

        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id=self._session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self._model_name,
                extra={
                    "framework": "deepagents",
                    "langchain_version": "1.0+",
                    "langsmith_run_id": self._langsmith_run_id,  # Store for feedback
                },
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))

        if self._verbose:
            print(f"✓ Trajectory saved to {trajectory_path}")

        # Populate context with metrics
        if trajectory.final_metrics:
            context.n_input_tokens = trajectory.final_metrics.total_prompt_tokens
            context.n_output_tokens = trajectory.final_metrics.total_completion_tokens
            context.cost_usd = trajectory.final_metrics.total_cost_usd

    def _update_token_usage(self, usage_metadata: UsageMetadata) -> None:
        """Update token usage from message metadata."""
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)

        self._total_prompt_tokens += input_tokens
        self._total_completion_tokens += output_tokens

        # Estimate cost based on model provider
        if self._model_name.startswith("openai/") or self._model_name.startswith("gpt-"):
            # OpenAI pricing (approximate, as of Nov 2025)
            if "gpt-5-mini" in self._model_name or "gpt-4o-mini" in self._model_name:
                # Mini models: $0.15 per 1M input, $0.60 per 1M output
                input_cost = input_tokens * 0.00000015
                output_cost = output_tokens * 0.0000006
            elif "gpt-5" in self._model_name or "gpt-4o" in self._model_name:
                # Standard models: $2.50 per 1M input, $10 per 1M output
                input_cost = input_tokens * 0.0000025
                output_cost = output_tokens * 0.00001
            else:
                # Default OpenAI pricing
                input_cost = input_tokens * 0.0000015
                output_cost = output_tokens * 0.000006
        else:
            # Anthropic pricing (Claude Sonnet)
            # $3 per 1M input, $15 per 1M output
            input_cost = input_tokens * 0.000003
            output_cost = output_tokens * 0.000015

        self._total_cost += input_cost + output_cost