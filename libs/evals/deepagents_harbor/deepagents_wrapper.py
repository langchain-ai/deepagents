"""A wrapper for Deep Agents to run in Harbor environments."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.graph import get_default_model
from deepagents.middleware.summarization import SummarizationStrategy
from deepagents_cli.agent import create_cli_agent
from dotenv import load_dotenv
from harbor.agents.base import BaseAgent
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import trace
from langsmith.client import Client

if TYPE_CHECKING:
    from pathlib import Path

    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from langchain.messages import UsageMetadata
    from langchain_core.runnables import RunnableConfig

from deepagents_harbor.backend import HarborSandbox
from deepagents_harbor.metadata import InfraMetadata, collect_sandbox_metadata

logger = logging.getLogger(__name__)

# Load .env file if present
load_dotenv()

_MAX_FILE_LISTING = 10  # maximum files shown in the system prompt directory context

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
    """Harbor agent implementation using LangChain Deep Agents.

    Wraps Deep Agents to execute tasks in Harbor environments.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        temperature: float = 0.0,
        verbose: bool = True,
        use_cli_agent: bool = True,
        openrouter_provider: str | None = None,
        summarization_strategy: SummarizationStrategy = "default",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize Deep AgentsWrapper.

        Args:
            logs_dir: Directory for storing logs
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the model
            verbose: Enable verbose output
            use_cli_agent: If True, use create_cli_agent from deepagents-cli (default).
                If False, use create_deep_agent from SDK.
            openrouter_provider: Pin OpenRouter routing to a single provider
                (e.g. `"MiniMax"`).

                Requires an `openrouter:` model prefix.
            summarization_strategy: Context management strategy for summarization.
                ``"default"`` uses threshold-driven summarization.
                ``"cat"`` uses proactive compaction with structured summaries.
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if openrouter_provider and (model_name is None or not model_name.startswith("openrouter:")):
            msg = "openrouter_provider requires an openrouter: model prefix"
            raise ValueError(msg)

        if model_name is None:
            # Keep Harbor default aligned with the SDK default model.
            model = get_default_model()
            # Apply Harbor's runtime temperature knob to the SDK default when supported.
            if hasattr(model, "temperature"):
                model = model.model_copy(update={"temperature": temperature})
            self._model = model
            self._model_name = model.model
        else:
            self._model_name = model_name
            model_kwargs: dict[str, Any] = {}
            if openrouter_provider:
                model_kwargs["openrouter_provider"] = {
                    "only": [openrouter_provider],
                    "allow_fallbacks": False,
                }
            self._model = init_chat_model(model_name, temperature=temperature, **model_kwargs)

        self._temperature = temperature
        self._verbose = verbose
        self._use_cli_agent = use_cli_agent
        self._summarization_strategy = summarization_strategy

        # LangSmith run tracking for feedback
        self._langsmith_run_id: str | None = None
        self._task_name: str | None = None

        # Build instruction->example_id mapping if LANGSMITH_EXPERIMENT is set
        self._instruction_to_example_id: dict[str, str] = {}
        langsmith_experiment_name = os.environ.get("LANGSMITH_EXPERIMENT", "").strip() or None
        if langsmith_experiment_name:
            try:
                client = Client()
                experiment = client.read_project(project_name=langsmith_experiment_name)
                examples = list(client.list_examples(dataset_id=experiment.reference_dataset_id))

                # Build mapping from instruction to example ID
                for example in examples:
                    instruction = example.inputs.get("instruction") if example.inputs else None
                    if instruction:
                        self._instruction_to_example_id[instruction] = str(example.id)
            except Exception:  # noqa: BLE001  # gracefully degrade when LangSmith is unavailable
                logger.warning("Failed to build instruction->example_id mapping", exc_info=True)

    @staticmethod
    def name() -> str:
        """Return the agent name identifier."""
        return "deepagent-harbor"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent with the given environment.

        Args:
            environment: Harbor environment (Docker, Modal, etc.)
        """

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
        ls_result = await backend.als(".")
        current_dir = (await backend.aexecute("pwd")).output

        if ls_result.error:
            logger.warning("Failed to list working directory: %s", ls_result.error)

        entries = ls_result.entries or []
        total_files = len(entries)
        first_files = entries[:_MAX_FILE_LISTING]

        # Build file listing header based on actual count
        if total_files == 0:
            file_listing_header = "Current directory is empty."
            file_listing = ""
        elif total_files <= _MAX_FILE_LISTING:
            # Show actual count when 10 or fewer
            file_count_text = "1 file" if total_files == 1 else f"{total_files} files"
            file_listing_header = f"Files in current directory ({file_count_text}):"
            file_listing = "\n".join(f"{i + 1}. {file}" for i, file in enumerate(first_files))
        else:
            file_listing_header = (
                f"Files in current directory (showing first {_MAX_FILE_LISTING} of {total_files}):"
            )
            file_listing = "\n".join(f"{i + 1}. {file}" for i, file in enumerate(first_files))

        # Format the system prompt with context
        return SYSTEM_MESSAGE.format(
            current_directory=current_dir.strip() if current_dir else "/app",
            file_listing_header=file_listing_header,
            file_listing=file_listing,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,  # noqa: ARG002  # required by BaseAgent interface
    ) -> None:
        """Execute the Deep Agent on the given instruction.

        Args:
            instruction: The task to complete
            environment: Harbor environment (Docker, Modal, etc.)
            context: Context to populate with metrics
        """
        configuration = json.loads(environment.trial_paths.config_path.read_text())
        if not isinstance(configuration, dict):
            msg = f"Unexpected configuration format. Expected a dict got {type(configuration)}."
            raise TypeError(msg)

        backend = HarborSandbox(environment)

        # Infrastructure metadata for noise analysis
        try:
            infra_meta = await collect_sandbox_metadata(backend)
        except Exception:  # noqa: BLE001  # metadata is supplementary; never abort a trial
            logger.warning("Failed to collect infrastructure metadata", exc_info=True)
            infra_meta = None

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
                model=self._model,
                backend=backend,
                system_prompt=system_prompt,
                summarization_strategy=self._summarization_strategy,
            )

        # Build metadata with experiment tracking info
        try:
            sdk_version = importlib.metadata.version("deepagents")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = "unknown"

        metadata = {
            "task_instruction": instruction,
            # "model" is the legacy key; "model_name" is the canonical key
            # used for LangSmith experiment filtering.
            "model": self._model_name,
            "model_name": self._model_name,
            "sdk_version": sdk_version,
            # Harbor's per-task session ID, distinct from the LangSmith
            # TracerSession UUID also called "session_id" in the API.
            "harbor_session_id": environment.session_id,
            # Tag to indicate which agent implementation is being used
            "agent_mode": "cli" if self._use_cli_agent else "sdk",
            "summarization_strategy": self._summarization_strategy,
        }
        metadata.update(configuration)

        # Look up example_id from instruction using the mapping built at initialization
        example_id = self._instruction_to_example_id.get(instruction)

        config: RunnableConfig = {
            "run_name": f"{environment.session_id}",
            "tags": [
                self._model_name,
                environment.session_id,
                "cli-agent" if self._use_cli_agent else "sdk-agent",
                f"summarization:{self._summarization_strategy}",
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
                    {"messages": [{"role": "user", "content": instruction}]},
                    config=config,
                )
                # Extract last AI message and add as output
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    run_tree.end(outputs={"last_message": last_message.text})
        else:
            config["metadata"] = metadata
            result = await deep_agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config=config,
            )

        self._save_trajectory(environment, instruction, result, infra_meta)

    def _save_trajectory(
        self,
        environment: BaseEnvironment,
        instruction: str,
        result: dict,
        infra_meta: InfraMetadata | None = None,
    ) -> None:
        """Save current trajectory to logs directory.

        Args:
            environment: Harbor environment with trial paths.
            instruction: The task instruction given to the agent.
            result: Agent invocation result containing messages.
            infra_meta: Infrastructure metadata collected at trial start,
                if available.
        """
        # Track token usage and cost for this run
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Create trajectory
        steps = [
            Step(
                step_id=1,
                timestamp=datetime.now(UTC).isoformat(),
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
                    timestamp=datetime.now(UTC).isoformat(),
                    source="agent",
                    message=message,
                    tool_calls=atf_tool_calls or None,
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
                err_msg = f"Message type {type(msg)} not supported for step conversion"
                raise NotImplementedError(err_msg)

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
                    "langchain_version": importlib.metadata.version("langchain"),
                    "langchain_core_version": importlib.metadata.version("langchain-core"),
                    **({"infrastructure": infra_meta.to_dict()} if infra_meta else {}),
                },
            ),
            steps=steps,
            final_metrics=metrics,
        )
        trajectory_path = self.logs_dir / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory.to_json_dict(), indent=2))
