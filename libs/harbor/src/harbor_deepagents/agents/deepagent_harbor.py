"""Harbor agent implementation using LangChain DeepAgents."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langsmith import Client
from pydantic import BaseModel

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

from .harbor_tools import create_harbor_environment_tools
from .langsmith_integration import send_harbor_feedback
from .prompts import HARBOR_SYSTEM_PROMPT


class DeepAgentHarbor(BaseAgent):
    """Harbor agent that uses LangChain DeepAgents for intelligent task completion.

    DeepAgents provides:
    - Planning via write_todos tool
    - Real filesystem access via FilesystemBackend (ls, read_file, write_file, edit_file, glob_search, grep_search)
    - Subagent spawning for complex tasks
    - Built on LangChain 1.0 + LangGraph

    Harbor provides:
    - bash tool for command execution in sandboxed environment
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_iterations: int = 50,
        temperature: float = 0.0,
        verbose: bool = True,
        system_prompt: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            # Use DeepAgents default
            model_name = "claude-sonnet-4-5-20250929"

        self._model_name = model_name
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._verbose = verbose
        self._session_id = str(uuid.uuid4())
        self._environment: BaseEnvironment | None = None
        self._system_prompt = system_prompt or HARBOR_SYSTEM_PROMPT

        # Initialize LLM based on provider prefix (LangSmith will trace if LANGCHAIN_TRACING_V2=true)
        if model_name.startswith("openai/") or model_name.startswith("gpt-"):
            # OpenAI models: openai/gpt-4o or gpt-4o-mini
            actual_model = model_name.split("/")[-1] if "/" in model_name else model_name
            self._llm = ChatOpenAI(
                model=actual_model,
                temperature=temperature,
            )
        elif model_name.startswith("anthropic/") or model_name.startswith("claude-"):
            # Anthropic models: anthropic/claude-sonnet-4-5-20250929 or claude-sonnet-4-5-20250929
            actual_model = model_name.split("/")[-1] if "/" in model_name else model_name
            self._llm = ChatAnthropic(
                model=actual_model,
                temperature=temperature,
            )
        else:
            # Default to Anthropic for backward compatibility
            self._llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
            )

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

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup phase - store environment reference."""
        self._environment = environment

        # Add system step to trajectory
        self._add_step(
            source="system",
            message=self._system_prompt,
        )

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
        self._environment = environment

        # Add user instruction as a step
        self._add_step(
            source="user",
            message=instruction,
        )

        # Create Harbor-specific tools that interact with the environment
        harbor_tools = create_harbor_environment_tools(environment)

        # Extract only bash tool - DeepAgents will handle filesystem via backend
        bash_tool = next((tool for tool in harbor_tools if tool.name == "bash"), None)
        if not bash_tool:
            raise RuntimeError("bash tool not found in harbor_tools")

        # Configure FilesystemBackend for real filesystem access in Harbor's /app directory
        filesystem_backend = FilesystemBackend(
            root_dir="/app",  # Terminal Bench working directory
            virtual_mode=False,  # Use real filesystem, not virtual/mock
        )

        # Create DeepAgent with:
        # - Harbor's bash tool (for command execution)
        # - FilesystemBackend for real file I/O (no mock filesystem)
        # - Built-in planning (write_todos) and subagent (task) tools
        deep_agent = create_deep_agent(
            model=self._llm,
            tools=[bash_tool],  # Only bash from Harbor
            system_prompt=self._system_prompt,
            backend=filesystem_backend,  # Real filesystem backend
        )

        try:
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
                try:
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
                except Exception as e:
                    if self._verbose:
                        print(f"Warning: Could not query LangSmith run_id: {e}")

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

        except Exception as e:
            error_msg = f"DeepAgent execution failed: {str(e)}"
            self._add_step(
                source="system",
                message=error_msg,
            )
            if self._verbose:
                print(f"✗ Error: {error_msg}")
            raise

    def _process_messages_to_steps(self, messages: list[BaseMessage]) -> None:
        """Process LangChain messages into ATIF trajectory steps.

        Args:
            messages: List of messages from agent execution
        """
        for msg in messages:
            if msg.type == "ai":
                # AI message with potential tool calls
                message_content = getattr(msg, "content", "")

                # Check for tool calls
                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    # Has tool calls
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
                elif message_content:
                    # Regular AI message
                    self._add_step(
                        source="agent",
                        message=str(message_content),
                    )

                # Extract usage metadata if available
                usage_metadata = getattr(msg, "usage_metadata", None)
                if usage_metadata:
                    self._update_token_usage(usage_metadata)

            elif msg.type == "tool":
                # Tool result
                content = getattr(msg, "content", "")
                tool_call_id = getattr(msg, "tool_call_id", None)

                # Find the corresponding step and add observation
                if self._trajectory_steps:
                    last_step = self._trajectory_steps[-1]
                    if last_step.tool_calls:
                        # Add observation to the last tool call step
                        last_step.observation = Observation(
                            results=[
                                ObservationResult(
                                    source_call_id=tool_call_id,
                                    content=str(content),
                                )
                            ]
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
        source: str,
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
