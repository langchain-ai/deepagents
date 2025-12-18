"""A wrapper for DeepAgents to run in Harbor environments."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Load .env file if present
load_dotenv()
from deepagents_cli.agent import create_cli_agent
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
from langsmith import trace

from deepagents_harbor.backend import HarborSandbox
from deepagents_harbor.tracing import create_example_id_from_instruction

SYSTEM_MESSAGE = """
You are an autonomous agent executing tasks in a sandboxed environment. Follow these instructions carefully.

## WORKING DIRECTORY & ENVIRONMENT CONTEXT

Your current working directory is:
{current_directory}

{file_listing_header}
{file_listing}

**IMPORTANT**: This directory information is provided for your convenience at the start of the task. You should:
- Use this information to understand the initial environment state
- Avoid redundantly calling `ls` or similar commands just to list the same directory
- Only use file listing commands if you need updated information (after creating/deleting files) or need to explore subdirectories
- Work in the /app directory unless explicitly instructed otherwise
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
        use_cli_agent: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize DeepAgentsWrapper.

        Args:
            logs_dir: Directory for storing logs
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the model
            verbose: Enable verbose output
            use_cli_agent: If True, use create_cli_agent from deepagents-cli (default).
                          If False, use create_deep_agent from SDK.
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            # Use DeepAgents default
            model_name = "anthropic:claude-sonnet-4-5-20250929"

        self._model_name = model_name
        self._temperature = temperature
        self._verbose = verbose
        self._use_cli_agent = use_cli_agent
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
        """The version of the agent."""
        return "0.0.1"

    async def _get_formatted_system_prompt(self, backend: HarborSandbox) -> str:
        """Format the system prompt with current directory and file listing context.

        Args:
            backend: Harbor sandbox backend to query for directory information

        Returns:
            Formatted system prompt with directory context
        """
        # Get directory information from backend
        ls_info = await backend.als_info(".")
        current_dir = (await backend.aexecute("pwd")).output

        # Get first 10 files
        total_files = len(ls_info) if ls_info else 0
        first_10_files = ls_info[:10] if ls_info else []
        has_more = total_files > 10

        # Build file listing header based on actual count
        if total_files == 0:
            file_listing_header = "Current directory is empty."
            file_listing = ""
        elif total_files <= 10:
            # Show actual count when 10 or fewer
            file_count_text = "1 file" if total_files == 1 else f"{total_files} files"
            file_listing_header = f"Files in current directory ({file_count_text}):"
            file_listing = "\n".join(f"{i + 1}. {file}" for i, file in enumerate(first_10_files))
        else:
            # Show "First 10 of N" when more than 10
            file_listing_header = f"Files in current directory (showing first 10 of {total_files}):"
            file_listing = "\n".join(f"{i + 1}. {file}" for i, file in enumerate(first_10_files))

        # Format the system prompt with context
        formatted_prompt = SYSTEM_MESSAGE.format(
            current_directory=current_dir.strip() if current_dir else "/app",
            file_listing_header=file_listing_header,
            file_listing=file_listing,
        )

        return formatted_prompt

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
        configuration = json.loads(environment.trial_paths.config_path.read_text())
        if not isinstance(configuration, dict):
            raise AssertionError(
                f"Unexpected configuration format. Expected a dict got {type(configuration)}."
            )

        backend = HarborSandbox(environment)

        # Create agent based on mode (CLI vs SDK)
        if self._use_cli_agent:
            # Get Harbor's system prompt with directory context
            harbor_system_prompt = await self._get_formatted_system_prompt(backend)

            # Use CLI agent with auto-approve mode
            deep_agent, _ = create_cli_agent(
                model=self._model,
                assistant_id=environment.session_id,
                sandbox=backend,
                sandbox_type=None,
                system_prompt=harbor_system_prompt,  # Use Harbor's custom prompt
                auto_approve=True,  # Skip HITL in Harbor
                enable_memory=False,
                enable_skills=False,  # Disable CLI skills for now
                enable_shell=False,  # Sandbox provides execution
            )
        else:
            # Use SDK agent
            # Get formatted system prompt with directory context
            system_prompt = await self._get_formatted_system_prompt(backend)

            deep_agent = create_deep_agent(
                model=self._model, backend=backend, system_prompt=system_prompt
            )

        # Build metadata with experiment tracking info
        metadata = {
            "task_instruction": instruction,
            "model": self._model_name,
            # This is a harbor-specific session ID for the entire task run
            # It's different from the LangSmith experiment ID (called session_id)
            "harbor_session_id": environment.session_id,
            # Tag to indicate which agent implementation is being used
            "agent_mode": "cli" if self._use_cli_agent else "sdk",
        }
        metadata.update(configuration)

        # Compute example_id from instruction for deterministic linking
        # This uses the same hashing as create_langsmith_dataset.py
        example_id = create_example_id_from_instruction(instruction)

        config: RunnableConfig = {
            "run_name": f"{environment.session_id}",
            "tags": [
                self._model_name,
                environment.session_id,
                "cli-agent" if self._use_cli_agent else "sdk-agent",
            ],
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        }

        # If LANGSMITH_EXPERIMENT is set, wrap in trace context.
        # This will link runs to the given experiment in LangSmith.
        langsmith_experiment_name = os.environ.get("LANGSMITH_EXPERIMENT", "").strip() or None

        if langsmith_experiment_name:
            with trace(
                name=environment.session_id,
                reference_example_id=example_id,
                inputs={"instruction": instruction},
                project_name=langsmith_experiment_name,
                metadata=metadata,
            ) as run_tree:
                # Invoke deep agent with LangSmith tracing
                result = await deep_agent.ainvoke(
                    {"messages": [{"role": "user", "content": instruction}]},  # type: ignore
                    config=config,
                )
                # Extract last AI message and add as output
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    run_tree.end(outputs={"last_message": last_message.text})
        else:
            config["metadata"] = metadata
            result = await deep_agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},  # type: ignore
                config=config,
            )

        self._save_trajectory(environment, instruction, result)

    def _save_trajectory(
        self, environment: BaseEnvironment, instruction: str, result: dict
    ) -> None:
        """Save current trajectory to logs directory."""
        # Track token usage and cost for this run
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Create trajectory
        steps = [
            Step(
                step_id=1,
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
