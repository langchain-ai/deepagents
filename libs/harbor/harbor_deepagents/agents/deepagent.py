"""Harbor agent implementation using LangChain DeepAgents."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langsmith import Client

from deepagents import create_deep_agent
from harbor_deepagents.agents.harbor_backend import HarborSandbox
from harbor_deepagents.agents.tracing import send_harbor_feedback


class DeepAgentHarbor(BaseAgent):
    """Harbor agent that uses LangChain DeepAgents for intelligent task completion."""

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

        # Trajectory tracking (ATIF format)
        self._trajectory_steps: list[Step] = []
        self._step_counter = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

        # LangSmith run tracking for feedback
        self._langsmith_run_id: str | None = None
        self._task_name: str | None = None

    @staticmethod
    def name() -> str:
        return "deepagent-harbor"

    def version(self) -> str | None:
        return "1.0.0"

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
        raise ValueError(BaseEnvironment)
        self._add_step(
            source="user",
            message=instruction,
        )
        backend = HarborSandbox(environment)

        # Create DeepAgent with:
        # - Harbor's bash tool (for command execution)
        # - FilesystemBackend for real file I/O (no mock filesystem)
        # - Built-in planning (write_todos) and subagent (task) tools
        deep_agent = create_deep_agent(
            model=self._model,
            backend=backend
        )

        # Invoke deep agent with LangSmith tracing
        result = await deep_agent.ainvoke(
            {"messages": [{"role": "user", "content": instruction}]},
            config={
                "run_name": f"harbor-deepagent-{self._session_id[:8]}",
                "tags": ["harbor", "deepagent", self._model_name, self._session_id],
                "metadata": {
                    "task_instruction": instruction,
                    "model": self._model_name,
                    "session_id": self._session_id,
                },
                "recursion_limit": self._max_iterations,
            },
        )

        # Store task name for feedback
        self._task_name = instruction[:100]  # Truncate for readability

        # Query LangSmith to get the run_id by searching for our unique run_name
        self._langsmith_run_id = self._session_id  # Default fallback

        if os.getenv("LANGCHAIN_TRACING_V2"):
            client = Client()
            project_name = os.getenv("LANGCHAIN_PROJECT", "default")
            run_name = f"harbor-deepagent-{self._session_id[:8]}"
            # Search by exact run_name which is unique
            runs = list(client.list_runs(
                project_name=project_name,
                filter=f'eq(name, "{run_name}")',
                limit=1,
            ))
            if runs:
                self._langsmith_run_id = str(runs[0].id)
                if self._verbose:
                    print(f"✓ Found LangSmith run_id: {self._langsmith_run_id}")

        # Extract messages from result
        messages = result.get("messages", [])

        # Process messages into ATIF steps
        self._process_messages_to_steps(messages)

        # Extract final output
        final_message = self._extract_final_message(messages) or "Task completed"
        last_step = self._trajectory_steps[-1] if self._trajectory_steps else None
        if not (
            last_step
            and last_step.source == "agent"
            and (last_step.message or "").strip() == final_message.strip()
        ):
            self._add_step(
                source="agent",
                message=final_message,
            )

        # Build and save trajectory
        trajectory = self._build_trajectory()
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))

        if self._verbose:
            print(f"✓ Trajectory saved to {trajectory_path}")

        # Populate context with metrics
        if trajectory.final_metrics:
            context.n_input_tokens = trajectory.final_metrics.total_prompt_tokens
            context.n_output_tokens = trajectory.final_metrics.total_completion_tokens
            context.cost_usd = trajectory.final_metrics.total_cost_usd

    def _process_message(self, msg: BaseMessage) -> None:
        """Process LangChain messages into ATIF trajectory steps."""
        if isinstance(msg, AIMessage):
            tool_calls = msg.tool_calls
            for tc in tool_calls:
                tool_call_id = tc.get("id", f"call_{self._step_counter + 1}")
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})

                self._add_step(
                    source="agent",
                    message=f"Using tool: {tool_name}",
                    tool_calls=[
                        ToolCall(
                            tool_call_id=tool_call_id,
                            function_name=tool_name,
                            arguments=tool_args,
                        )
                    ],
                )
            else:
                self._add_step(
                    source="agent",
                    message=str(msg.content_blocks),
                )

            # Extract usage metadata if available
            usage_metadata = getattr(msg, "usage_metadata", None)
            if usage_metadata:
                self._update_token_usage(usage_metadata)
        elif isinstance(msg, ToolMessage):
            content_blocks = msg.content_blocks
            tool_call_id = msg.tool_call_id
            # Find the corresponding step and add observation
            if self._trajectory_steps:
                last_step = self._trajectory_steps[-1]
                if last_step.tool_calls:
                    # Add observation to the last tool call step
                    last_step.observation = Observation(
                        results=[
                            ObservationResult(
                                source_call_id=tool_call_id,
                                content=str(msg.content_blocks),
                            )
                        ]
                    )
        else:
            raise NotImplementedError(
                f"Message type {type(msg)} not supported for step conversion"
            )

    def _extract_final_message(self, messages: list[BaseMessage]) -> str:
        """Extract the final agent message."""
        for msg in reversed(messages):
            if msg.type == "ai":
                content = getattr(msg, "content", "")
                if content:
                    return str(content)
        return ""

    def _update_token_usage(self, usage_metadata: dict[str, Any]) -> None:
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

    def _add_step(
        self,
        source: Literal["system", "user", "agent"],
        message: str,
        tool_calls: list[ToolCall] | None = None,
        observation: Observation | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        """Add a step to the ATIF trajectory."""
        self._step_counter += 1
        step = Step(
            step_id=self._step_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            message=message,
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
        )

        self._trajectory_steps.append(step)

    def _build_trajectory(self) -> Trajectory:
        """Build final ATIF trajectory."""
        final_metrics = FinalMetrics(
            total_prompt_tokens=self._total_prompt_tokens or None,
            total_completion_tokens=self._total_completion_tokens or None,
            total_cost_usd=self._total_cost or None,
            total_steps=len(self._trajectory_steps),
        )

        return Trajectory(
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
            steps=self._trajectory_steps,
            final_metrics=final_metrics,
        )

    def send_verification_feedback(self, reward: float) -> None:
        """Send Harbor verification results to LangSmith as feedback.

        This should be called after Harbor's verifier runs to push the
        reward score to LangSmith, making it visible in the trace UI.

        Args:
            reward: Reward score from Harbor verifier (0.0 to 1.0)

        Example:
            >>> agent = DeepAgentHarbor(...)
            >>> await agent.run(instruction, environment, context)
            >>> # After Harbor verifies the task:
            >>> agent.send_verification_feedback(reward=1.0)
        """
        if not self._langsmith_run_id or not self._task_name:
            if self._verbose:
                print("Warning: No run_id or task_name available for feedback")
            return

        send_harbor_feedback(
            run_id=self._langsmith_run_id,
            task_name=self._task_name,
            reward=reward,
            agent_cost_usd=self._total_cost,
            total_steps=len(self._trajectory_steps),
        )

        if self._verbose:
            print(f"✓ Sent feedback to LangSmith: reward={reward * 100:.0f}%")
